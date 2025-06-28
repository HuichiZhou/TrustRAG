from sentence_transformers import SentenceTransformer
import torch
import random
from src.utils import cos_similarity, load_json, progress_bar
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
        self.max_seq_length = kwargs.get("max_seq_length", 128)
        self.pad_to_max_length = kwargs.get("pad_to_max_length", True)
        self.per_gpu_eval_batch_size = kwargs.get("per_gpu_eval_batch_size", 64)
        self.num_adv_passage_tokens = kwargs.get("num_adv_passage_tokens", 30)
        self.num_cand = kwargs.get("num_cand", 100)
        if "hotflip" in args.attack_method:
            self.num_iter = kwargs.get("num_iter", 30)
            self.gold_init = kwargs.get("gold_init", True)
            self.early_stop = kwargs.get("early_stop", False)
        if "aggd" in args.attack_method:
            self.num_iter = kwargs.get("num_iter", 100)
        self.all_adv_texts = load_json(
            f"results/adv_targeted_results/{args.eval_dataset}.json"
        )
        logger.info(f"Initializing attacker with method: {args.attack_method}")

    def get_attack(self, target_queries) -> list:
        """
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        """
        if self.attack_method == "LM_targeted":
            adv_text_groups = self.pia(target_queries)
        elif self.attack_method == "hotflip":
            adv_text_groups = self.hotflip(target_queries)
        elif self.attack_method == "aggd":
            adv_text_groups = self.aggd(target_queries)
        else:
            raise NotImplementedError
        return adv_text_groups

    def pia(self, target_queries):
        adv_text_groups = []
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
        return adv_text_groups

    def _prepare_adversarial_passage(self, query, adv_b, use_gold_init=False):
        """Prepare initial adversarial passage tokens."""
        device = "cuda"

        # Tokenize adv_b
        adv_b = self.tokenizer(
            adv_b,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
        )["input_ids"]

        # Prepare adv_a
        if use_gold_init:
            adv_a = self.tokenizer(
                query,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
            )["input_ids"]
            # Ensure adv_a has the correct number of tokens
            adv_a = adv_a[: self.num_adv_passage_tokens]
            # todo - remove
            if len(adv_a) < self.num_adv_passage_tokens:
                adv_a += [self.tokenizer.mask_token_id] * (
                    self.num_adv_passage_tokens - len(adv_a)
                )
            assert (
                len(adv_a) == self.num_adv_passage_tokens
            ), f"adv_a length {len(adv_a)} does not match num_adv_passage_tokens {self.num_adv_passage_tokens}"
        else:
            adv_a = [self.tokenizer.mask_token_id] * self.num_adv_passage_tokens

        # Choose order based on adv_a_position
        if self.adv_a_position == "start":
            adv_passage = adv_a + adv_b
        elif self.adv_a_position == "end":
            adv_passage = adv_b + adv_a
        else:
            raise ValueError(f"Invalid adv_a_position: {self.adv_a_position}")

        # Convert to tensors
        adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0)
        adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
        adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

        return (
            adv_passage_ids,
            adv_passage_attention,
            adv_passage_token_type,
            len(adv_a),
            len(adv_b),
        )

    def _prepare_query_embedding(self, query):
        """Prepare query embedding."""
        q_sent = self.tokenizer(
            query,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False,
            return_tensors="pt",
        )
        q_sent = {key: value.cuda() for key, value in q_sent.items()}
        return self.get_emb(self.model, q_sent).detach()

    def _compute_similarity(self, p_emb, q_emb):
        """Compute similarity based on score function."""
        if self.args.score_function == "dot":
            return torch.mm(p_emb, q_emb.T)
        elif self.args.score_function == "cos_sim":
            return cos_similarity(p_emb, q_emb)
        else:
            raise KeyError(f"Unknown score function: {self.args.score_function}")

    def _hotflip_iteration(
        self,
        adv_passage_ids,
        adv_passage_attention,
        adv_passage_token_type,
        q_emb,
        embeddings,
        embedding_gradient,
        adv_a_len,
        adv_b_len,
    ):
        """Single HotFlip iteration."""
        device = adv_passage_ids.device

        self.c_model.zero_grad()

        p_sent = {
            "input_ids": adv_passage_ids,
            "attention_mask": adv_passage_attention,
            "token_type_ids": adv_passage_token_type,
        }
        p_emb = self.get_emb(self.c_model, p_sent)

        sim = self._compute_similarity(p_emb, q_emb)
        loss = sim.mean()
        loss.backward()

        temp_grad = embedding_gradient.get()
        grad = temp_grad.sum(dim=0)

        # Select token to flip randomly from adv_a
        token_to_flip = random.randrange(adv_a_len)
        if self.adv_a_position == "end":
            token_to_flip += adv_b_len

        candidates = self._hotflip_attack(
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
                temp_sim = self._compute_similarity(temp_p_emb, q_emb)
                can_loss = temp_sim.mean()
                candidate_scores[i] = can_loss.sum().cpu().item()

        # Update if better candidate found
        if (candidate_scores > current_score).any():
            best_candidate_idx = candidate_scores.argmax()
            adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
            return True
        return False

    def _aggd_iteration(
        self,
        adv_passage_ids,
        adv_passage_attention,
        adv_passage_token_type,
        q_emb,
        embeddings,
        embedding_gradient,
        depth,
        best_sim,
        adv_a_len,
        adv_b_len,
    ):
        """Single AGGD iteration."""
        self.c_model.zero_grad()

        p_sent = {
            "input_ids": adv_passage_ids,
            "attention_mask": adv_passage_attention,
            "token_type_ids": adv_passage_token_type,
        }
        p_emb = self.get_emb(self.c_model, p_sent)

        # Compute loss (note: different from hotflip)
        sim = self._compute_similarity(q_emb, p_emb)
        loss = sim.mean()
        loss.backward()

        temp_grad = embedding_gradient.get()
        grad = temp_grad.sum(dim=0)

        # Calculate the offset for adv_a tokens based on position
        if self.adv_a_position == "start":
            adv_a_start_idx = 0
        elif self.adv_a_position == "end":
            adv_a_start_idx = adv_b_len
        else:
            raise ValueError(f"Invalid adv_a_position: {self.adv_a_position}")

        # Extract gradient for adv_a tokens only
        adv_a_grad = grad[adv_a_start_idx : adv_a_start_idx + adv_a_len]

        candidates = self._aggd_candidate(
            self.num_cand,
            self.num_adv_passage_tokens,
            adv_a_grad,  # Pass only the gradient for adv_a tokens
            embeddings,
            depth,
        )

        # Adjust candidate positions to match the actual positions in the full passage
        adjusted_candidates = []
        for pos, token in candidates:
            adjusted_pos = pos + adv_a_start_idx
            adjusted_candidates.append((adjusted_pos, token))

        (
            candidate_scores,
            current_score,
        ) = self._aggd_candidate_score(
            self.num_cand,
            adjusted_candidates,  # Use adjusted candidates
            q_emb,
            self.get_emb,
            self.c_model,
            adv_passage_ids,
            adv_passage_attention,
            adv_passage_token_type,
        )

        if (candidate_scores > current_score).any():

            best_candidate_idx = candidate_scores.argmax()
            adv_passage_ids[:, adjusted_candidates[best_candidate_idx][0]] = (
                adjusted_candidates[best_candidate_idx][1].to(adv_passage_ids.device)
            )
            # print(
            #     "Better adv_passage detected. Depth=",
            #     depth,
            #     "new best_sim",
            #     candidate_scores.max(),
            # )
        else:
            # print("No improvement detected! Depth=", depth)
            depth = depth + 1

        return depth, candidate_scores.max(), adv_passage_ids.clone()

    def _run_adversarial_attack(self, attack_type, target_queries, **kwargs):
        """Unified method for running adversarial attacks."""
        adv_text_groups = []

        desc = f"Processing {attack_type.upper()} queries"
        for query_score in progress_bar(target_queries, desc=desc):
            query = query_score["query"]
            id = query_score["id"]
            adv_texts_b = self.all_adv_texts[id]["adv_texts"]

            # AGGD-specific variables

            adv_texts = []
            for j in range(self.adv_per_query):
                adv_b = adv_texts_b[j]

                # Prepare adversarial passage
                use_gold_init = attack_type == "hotflip" and self.gold_init
                (
                    adv_passage_ids,
                    adv_passage_attention,
                    adv_passage_token_type,
                    adv_a_len,
                    adv_b_len,
                ) = self._prepare_adversarial_passage(query, adv_b, use_gold_init)

                # Prepare query embedding
                q_emb = self._prepare_query_embedding(query)

                # Initialize embeddings and gradient storage
                embeddings = get_embeddings(self.c_model)
                embedding_gradient = GradientStorage(embeddings)

                # AGGD-specific initialization
                if attack_type == "aggd":
                    depth = 1
                    best_adv_passage_ids = adv_passage_ids.clone()
                    best_sim = 0

                # Run iterations
                for _ in range(self.num_iter):
                    if attack_type == "hotflip":
                        self._hotflip_iteration(
                            adv_passage_ids,
                            adv_passage_attention,
                            adv_passage_token_type,
                            q_emb,
                            embeddings,
                            embedding_gradient,
                            adv_a_len,
                            adv_b_len,
                        )
                    elif attack_type == "aggd":
                        (depth, best_sim, best_adv_passage_ids) = self._aggd_iteration(
                            adv_passage_ids,
                            adv_passage_attention,
                            adv_passage_token_type,
                            q_emb,
                            embeddings,
                            embedding_gradient,
                            depth,
                            best_sim,
                            adv_a_len,
                            adv_b_len,
                        )

                # Decode final adversarial text
                final_ids = (
                    best_adv_passage_ids if attack_type == "aggd" else adv_passage_ids
                )
                adv_text = self.tokenizer.decode(
                    final_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                adv_texts.append(adv_text)

            adv_text_groups.append(adv_texts)

        return adv_text_groups

    def hotflip(self, target_queries, **kwargs) -> list:
        """HotFlip attack method."""
        return self._run_adversarial_attack("hotflip", target_queries, **kwargs)

    def aggd(self, target_queries, **kwargs) -> list:
        """AGGD attack method."""
        return self._run_adversarial_attack("aggd", target_queries, **kwargs)

    def _hotflip_attack(
        self,
        averaged_grad,
        embedding_matrix,
        increase_loss=False,
        num_candidates=1,
        filter=None,
    ):
        """Returns the top candidate replacements."""
        with torch.no_grad():
            gradient_dot_embedding_matrix = self._compute_similarity(
                embedding_matrix, averaged_grad.unsqueeze(dim=0)
            ).squeeze(dim=1)
            if filter is not None:
                gradient_dot_embedding_matrix -= filter
            if not increase_loss:
                gradient_dot_embedding_matrix *= -1
            _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

        return top_k_ids

    def _aggd_candidate(
        self,
        num_cand,
        num_adv_passage_tokens,
        grad,
        embeddings,
        depth,
        increase_loss=True,
    ):

        top_k_start_row = num_cand * (depth - 1) // num_adv_passage_tokens
        top_k_start_col = num_cand * (depth - 1) % num_adv_passage_tokens
        top_k_end_row = num_cand * depth // num_adv_passage_tokens + 1
        top_k_end_col = num_cand * depth % num_adv_passage_tokens
        with torch.no_grad():
            gradient_dot_embedding_matrix = self._compute_similarity(
                embeddings.weight, grad  # shape: (50265,, 768)  # shape: 768
            )
            if not increase_loss:
                gradient_dot_embedding_matrix *= -1
            _, top_k_ids = gradient_dot_embedding_matrix.topk(top_k_end_row, axis=0)
        top_k_ids = top_k_ids[top_k_start_row:top_k_end_row, :num_adv_passage_tokens]
        top_k_ids = top_k_ids.reshape(1, -1).T
        pos = (
            torch.tile(
                torch.arange(num_adv_passage_tokens),
                (top_k_end_row - top_k_start_row, 1),
            )
            .view(1, -1)
            .T.to("cuda" if torch.cuda.is_available() else "cpu")
        )
        candidates = torch.cat([pos, top_k_ids], dim=1)
        candidates = candidates[
            top_k_start_col : top_k_end_col - num_adv_passage_tokens
        ]
        return candidates

    def _aggd_candidate_score(
        self,
        num_cand,
        candidates,
        q_emb,
        get_emb,
        c_model,
        adv_passage_ids,
        adv_passage_attention,
        adv_passage_token_type,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        candidate_scores = torch.zeros(num_cand, device=device)

        device = device
        p_sent = {
            "input_ids": adv_passage_ids,
            "attention_mask": adv_passage_attention,
            "token_type_ids": adv_passage_token_type,
        }
        p_emb = get_emb(c_model, p_sent)

        # Compute loss
        sim = self._compute_similarity(q_emb, p_emb)  # [b x k]

        loss = sim.mean()  # sim.shape: (64, 1)

        current_score = loss.sum().cpu().item()

        for i, candidate in enumerate(candidates):
            temp_adv_passage = adv_passage_ids.clone()
            temp_adv_passage[:, candidate[0]] = candidate[1]
            p_sent = {
                "input_ids": temp_adv_passage,
                "attention_mask": adv_passage_attention,
                "token_type_ids": adv_passage_token_type,
            }
            p_emb = get_emb(c_model, p_sent)
            with torch.no_grad():
                sim = self._compute_similarity(q_emb, p_emb)
                can_loss = sim.mean()
                temp_score = can_loss.sum().cpu().item()

                candidate_scores[i] += temp_score
        return candidate_scores, current_score
