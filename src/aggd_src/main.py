import logging
import os
import torch
import json
from transformers import set_seed
from utils.load_data import load_data
from utils.AGGD import AGGD_candidate, AGGD_candidate_score
import argparse
from utils.load_model import load_models, get_embeddings
from utils.AGGD import GradientStorage
from utils.evaluate import evaluate_acc

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def main():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--dataset", type=str, default="nq", help="BEIR dataset to evaluate"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_code", type=str, default="contriever")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", default=True)
    parser.add_argument("--num_adv_passage_tokens", default=50, type=int)
    parser.add_argument("--num_cand", default=30, type=int)
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for indexing.",
    )
    parser.add_argument("--num_iter", default=1000, type=int)
    parser.add_argument("--num_grad_iter", default=1, type=int)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--kmeans_split", default=0, type=int)
    parser.add_argument("--do_kmeans", default=False, action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()
    print(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    output_dir_name = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    set_seed(args.seed)
    device = args.device
    model, c_model, tokenizer, get_emb = load_models(args.model_code)
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)
    embeddings = get_embeddings(c_model)
    print("Model embedding", embeddings)
    embedding_gradient = GradientStorage(embeddings)

    adv_passage_ids = [tokenizer.mask_token_id] * args.num_adv_passage_tokens

    adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(
        0
    )  # shape: [1, 50]
    adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
    adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)
    best_adv_passage_ids = adv_passage_ids.clone()
    data_collator, dataloader, valid_dataloader = load_data(
        args, tokenizer, model, get_emb
    )
    best_acc = evaluate_acc(
        model,
        c_model,
        get_emb,
        valid_dataloader,
        best_adv_passage_ids,
        adv_passage_attention,
        adv_passage_token_type,
        data_collator,
    )
    print(best_acc)
    depth = 1
    cur_acc = 1

    for it_ in range(args.num_iter):

        print(f"Iteration: {it_}")
        print(f"Accumulating Gradient {args.num_grad_iter}")
        c_model.zero_grad()
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
            q_emb = get_emb(model, q_sent).detach()

            gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
            gold_emb = get_emb(c_model, gold_pass).detach()

            sim_to_gold = torch.bmm(
                q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)
            ).squeeze()
            sim_to_gold_mean = sim_to_gold.mean().cpu().item()
            print("Avg sim to gold p =", sim_to_gold_mean)

            p_sent = {
                "input_ids": adv_passage_ids,
                "attention_mask": adv_passage_attention,
                "token_type_ids": adv_passage_token_type,
            }
            p_emb = get_emb(c_model, p_sent)

            # Compute loss
            sim = torch.mm(q_emb, p_emb.T)  # [b x k]
            print(
                it_,
                _,
                "Avg sim to adv =",
                sim.mean().cpu().item(),
                "sim to gold =",
                sim_to_gold_mean,
            )
            suc_att = (
                ((sim - sim_to_gold.unsqueeze(-1)) >= 0).float().mean().cpu().item()
            )
            print("Attack on train: %d " % suc_att, "best_acc", best_acc)
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


if __name__ == "__main__":
    main()
