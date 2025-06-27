from sentence_transformers import SentenceTransformer
import torch
import random
from src.utils import cos_similarity, load_json, progress_bar
import nanogcg
from nanogcg import GCGConfig
from loguru import logger


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(
    averaged_grad, embedding_matrix, increase_loss=False, num_candidates=1, filter=None
):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(embedding_matrix, averaged_grad)
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


class Attacker:
    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.attack_method = args.attack_method
        self.adv_per_query = args.adv_per_query
        self.adv_a_position = args.adv_a_position
        self.model = kwargs.get("model", None)
        self.c_model = kwargs.get("c_model", None)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.get_emb = kwargs.get("get_emb", None)
        if "hotflip" in args.attack_method:
            self.max_seq_length = kwargs.get("max_seq_length", 128)
            self.pad_to_max_length = kwargs.get("pad_to_max_length", True)
            self.per_gpu_eval_batch_size = kwargs.get("per_gpu_eval_batch_size", 64)
            self.num_adv_passage_tokens = kwargs.get("num_adv_passage_tokens", 30)
            self.num_cand = kwargs.get("num_cand", 100)
            self.num_iter = kwargs.get("num_iter", 30)
            self.gold_init = kwargs.get("gold_init", True)
            self.early_stop = kwargs.get("early_stop", False)
        self.all_adv_texts = load_json(
            f"results/adv_targeted_results/{args.eval_dataset}.json"
        )
        logger.info(f"Initializing attacker with method: {args.attack_method}")

    def get_attack(self, target_queries) -> list:
        """
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        """
        adv_text_groups = []
        if self.attack_method == "LM_targeted":
            # blackbox attack
            for i in range(len(target_queries)):
                question = target_queries[i]["query"]
                id = target_queries[i]["id"]
                adv_texts_b = self.all_adv_texts[id]["adv_texts"][: self.adv_per_query]
                adv_text_a = question + "."
                # 🛠️ Here we choose order based on adv_a_position
                if self.adv_a_position == "start":
                    adv_texts = [adv_text_a + i for i in adv_texts_b]
                elif self.adv_a_position == "end":
                    adv_texts = [i + adv_text_a for i in adv_texts_b]
                adv_text_groups.append(adv_texts)
        elif self.attack_method == "hotflip":
            adv_text_groups = self.hotflip(target_queries)
        elif self.attack_method == "gcg":
            adv_text_groups = self.gcg(target_queries)
        else:
            raise NotImplementedError
        return adv_text_groups

    def hotflip(self, target_queries, **kwargs) -> list:
        device = "cuda"
        adv_text_groups = []
        for query_score in progress_bar(
            target_queries, desc="Processing HotFlip queries"
        ):
            query = query_score["query"]
            # top1_score = query_score["top1_score"]
            id = query_score["id"]
            adv_texts_b = self.all_adv_texts[id]["adv_texts"]

            adv_texts = []
            for j in range(self.adv_per_query):
                adv_b = adv_texts_b[j]
                adv_b = self.tokenizer(
                    adv_b,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding=False,
                )["input_ids"]
                if self.gold_init:
                    adv_a = query
                    adv_a = self.tokenizer(
                        adv_a,
                        max_length=self.max_seq_length,
                        truncation=True,
                        padding=False,
                    )["input_ids"]
                else:
                    adv_a = [self.tokenizer.mask_token_id] * self.num_adv_passage_tokens

                embeddings = get_embeddings(self.c_model)
                embedding_gradient = GradientStorage(embeddings)

                # 🛠️ Here we choose order based on adv_a_position
                if self.adv_a_position == "start":
                    adv_passage = adv_a + adv_b
                elif self.adv_a_position == "end":
                    adv_passage = adv_b + adv_a
                else:
                    raise ValueError(f"Invalid adv_a_position: {self.adv_a_position}")

                adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0)
                adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
                adv_passage_token_type = torch.zeros_like(
                    adv_passage_ids, device=device
                )

                q_sent = self.tokenizer(
                    query,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding="max_length" if self.pad_to_max_length else False,
                    return_tensors="pt",
                )
                q_sent = {key: value.cuda() for key, value in q_sent.items()}
                q_emb = self.get_emb(self.model, q_sent).detach()

                for it_ in range(self.num_iter):
                    grad = None
                    self.c_model.zero_grad()

                    p_sent = {
                        "input_ids": adv_passage_ids,
                        "attention_mask": adv_passage_attention,
                        "token_type_ids": adv_passage_token_type,
                    }
                    p_emb = self.get_emb(self.c_model, p_sent)

                    if self.args.score_function == "dot":
                        sim = torch.mm(p_emb, q_emb.T)
                    elif self.args.score_function == "cos_sim":
                        sim = cos_similarity(p_emb, q_emb)
                    else:
                        raise KeyError

                    loss = sim.mean()
                    # if self.early_stop and sim.item() > top1_score + 0.1:
                    #     break
                    loss.backward()

                    temp_grad = embedding_gradient.get()
                    if grad is None:
                        grad = temp_grad.sum(dim=0)
                    else:
                        grad += temp_grad.sum(dim=0)

                    token_to_flip = random.randrange(len(adv_a))
                    if self.adv_a_position == "end":
                        token_to_flip += len(adv_b)
                    candidates = hotflip_attack(
                        grad[token_to_flip],
                        embeddings.weight,
                        increase_loss=True,
                        num_candidates=self.num_cand,
                        filter=None,
                    )
                    current_score = loss.sum().cpu().item()
                    candidate_scores = torch.zeros(self.num_cand, device=device)

                    for i, candidate in enumerate(candidates):
                        temp_adv_passage = adv_passage_ids.clone()
                        temp_adv_passage[:, token_to_flip] = candidate
                        temp_p_sent = {
                            "input_ids": temp_adv_passage,
                            "attention_mask": adv_passage_attention,
                            "token_type_ids": adv_passage_token_type,
                        }
                        temp_p_emb = self.get_emb(self.c_model, temp_p_sent)
                        with torch.no_grad():
                            if self.args.score_function == "dot":
                                temp_sim = torch.mm(temp_p_emb, q_emb.T)
                            elif self.args.score_function == "cos_sim":
                                temp_sim = cos_similarity(temp_p_emb, q_emb)
                            else:
                                raise KeyError
                            can_loss = temp_sim.mean()
                            candidate_scores[i] = can_loss.sum().cpu().item()

                    # if find a better one, update
                    if (candidate_scores > current_score).any():
                        best_candidate_idx = candidate_scores.argmax()
                        adv_passage_ids[:, token_to_flip] = candidates[
                            best_candidate_idx
                        ]
                    else:
                        continue

                adv_text = self.tokenizer.decode(
                    adv_passage_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                adv_texts.append(adv_text)
            adv_text_groups.append(adv_texts)

        return adv_text_groups

    def aggd(self, target_queries, **kwargs) -> list:
        """
        This function implements the AGGD attack.
        It returns adv_text_groups, which contains adv_texts for M queries.
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text.
        """
        embeddings = get_embeddings(self.c_model)
        print("Model embedding", embeddings)
        embedding_gradient = GradientStorage(embeddings)
        for it_ in range(self.num_iter):

            print(f"Iteration: {it_}")
            print(f"Accumulating Gradient {args.num_grad_iter}")
            self.c_model.zero_grad()
            pbar = range(args.num_grad_iter)
            train_iter = iter(dataloader)
            grad = None

            for _ in pbar:
                try:
                    data = next(train_iter)
                    data = data_collator(data)  # [bsz, 3, max_len]
                except:
                    print("Insufficient data!")
                    break

                q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
                q_emb = self.get_emb(self.model, q_sent).detach()

                p_sent = {
                    "input_ids": adv_passage_ids,
                    "attention_mask": adv_passage_attention,
                    "token_type_ids": adv_passage_token_type,
                }
                p_emb = self.get_emb(self.c_model, p_sent)

                # Compute loss
                sim = torch.mm(q_emb, p_emb.T)  # [b x k]

                loss = sim.mean()
                print("loss", loss.cpu().item())
                loss.backward()

                temp_grad = embedding_gradient.get()  # (1, 50, 768)
                if grad is None:
                    grad = (
                        temp_grad.sum(dim=0) / args.num_grad_iter
                    )  # temp_grad.sum(dim=0): (50, 768)
                else:
                    grad += temp_grad.sum(dim=0) / args.num_grad_iter

            print("Evaluating Candidates")
            pbar = range(args.num_grad_iter)
            train_iter = iter(dataloader)
            candidates = AGGD_candidate(args, grad, embeddings, depth)
            candidate_scores, candidate_acc_rates, current_score, current_acc_rate = (
                AGGD_candidate_score(
                    args,
                    it_,
                    candidates,
                    pbar,
                    train_iter,
                    data_collator,
                    get_emb,
                    model,
                    c_model,
                    adv_passage_ids,
                    adv_passage_attention,
                    adv_passage_token_type,
                )
            )

            if (candidate_scores > current_score).any() or (
                candidate_acc_rates > current_acc_rate
            ).any():
                print("Better adv_passage detected. Depth=", depth)

                best_candidate_idx = candidate_scores.argmax()
                adv_passage_ids[:, candidates[best_candidate_idx][0]] = torch.tensor(
                    candidates[best_candidate_idx][1]
                ).to(device)
                cur_acc = evaluate_acc(
                    model,
                    c_model,
                    get_emb,
                    valid_dataloader,
                    adv_passage_ids,
                    adv_passage_attention,
                    adv_passage_token_type,
                    data_collator,
                )
                if cur_acc < best_acc:
                    depth = 1
                    best_acc = cur_acc
                    best_adv_passage_ids = adv_passage_ids.clone()
                    print("!!! Updated best adv_passage, Depth=", depth)
                    print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))
                else:
                    depth += 1
                    print("Depth = ", depth)
                if args.output_file is not None:
                    with open(args.output_file, "a") as f:
                        result = {
                            "it": it_,
                            "current_best_acc": best_acc,  # The best accancy achieved until iter i
                            "best_adv_text": tokenizer.convert_ids_to_tokens(
                                best_adv_passage_ids[0]
                            ),
                        }

                        json.dump(result, f)
                        f.write("\n")

            else:
                print("No improvement detected! Depth=", depth)
                depth += 1
                if args.output_file is not None:
                    with open(args.output_file, "a") as f:
                        result = {
                            "it": it_,
                            "current_best_acc": best_acc,  # The best accancy achieved until iter i
                            "best_adv_text": tokenizer.convert_ids_to_tokens(
                                best_adv_passage_ids[0]
                            ),
                        }

                        json.dump(result, f)
                        f.write("\n")

    def gcg(self, target_queries, **kwargs) -> list:
        adv_text_groups = []
        gcg_config = GCGConfig()
        for query_score in progress_bar(target_queries, desc="Processing GCG queries"):
            query = query_score["query"]
            # top1_score = query_score["top1_score"]
            id = query_score["id"]
            adv_texts_b = self.all_adv_texts[id]["adv_texts"]
            gcg_config.optim_str_init = query
            adv_texts = []
            for j in range(self.adv_per_query):
                adv_b = adv_texts_b[j]
                gcg_result = nanogcg.run(
                    self.c_model, self.tokenizer, query, adv_b, gcg_config
                )
                adv_text = gcg_result.best_string
                adv_texts.append(adv_text)
            adv_text_groups.append(adv_texts)

        return adv_text_groups
