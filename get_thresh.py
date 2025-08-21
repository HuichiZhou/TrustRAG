import argparse
import json
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import torch

from src.utils import load_beir_datasets, load_cached_data, load_models, save_outputs
from src.defend_module import drift_filtering_question_aware, drift_filtering_uniform
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run drift filtering on a BEIR dataset"
    )
    parser.add_argument(
        "--eval_dataset", type=str, required=True, help="Dataset name, e.g., nq"
    )
    parser.add_argument(
        "--eval_model_code",
        type=str,
        required=True,
        help="Model code, e.g., contriever",
    )
    parser.add_argument(
        "--score_function",
        type=str,
        choices=["dot", "cos_sim"],
        default="dot",
        help="Scoring function",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Top-k passages to consider"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["drift", "drift_uniform", "perplexity", "drift_perplexity"],
        required=True,
        help="Filtering method to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split = "test"
    device = "cuda"

    # Load model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.to(device).eval()

    # Load dataset
    corpus, queries, qrels = load_cached_data(
        f"data_cache/{args.eval_dataset}_{split}.pkl",
        load_beir_datasets,
        args.eval_dataset,
        split,
    )

    log_args = SimpleNamespace(
        log_name=f"temp_extract_val_{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}_{args.method}",
        score_function=args.score_function,
    )

    # Load retrieval results
    result_path = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}"
    if args.score_function == "cos_sim":
        result_path += "-cos"
    result_path += ".json"

    with open(result_path, "r") as f:
        results = json.load(f)

    # Run filtering based on the selected method
    if args.method == "drift":
        drift_start_all = []
        drift_end_all = []
        i = 0
        for query_id, topk_dict in tqdm(
            results.items(), desc=f"Evaluating {args.method} impact"
        ):
            i += 1
            if i % 1000 == 0:
                break
            question = queries[query_id]
            topk_indices = list(topk_dict.keys())[: args.top_k]
            original_docs = [corpus[doc_id]["text"] for doc_id in topk_indices]

            try:
                _, diff_start, diff_end = drift_filtering_question_aware(
                    args=log_args,
                    tokenizer=tokenizer,
                    model=model,
                    get_emb=get_emb,
                    question=question,
                    contents=original_docs,
                    score=args.score_function,
                )
            except Exception as e:
                print(f"Skipping {query_id} due to error: {e}")
                continue

            drift_start_all = np.concatenate((drift_start_all, diff_start))
            drift_end_all = np.concatenate((drift_end_all, diff_end))

        drift_start_all = np.array(drift_start_all)
        drift_start_all = drift_start_all[~np.isnan(drift_start_all)]
        drift_end_all = np.array(drift_end_all)
        drift_end_all = drift_end_all[~np.isnan(drift_end_all)]

        percentiles = [95, 90, 85]
        thresh_start = np.percentile(drift_start_all, percentiles)
        thresh_end = np.percentile(drift_end_all, percentiles)
        for pct, t_start, t_end in zip(percentiles, thresh_start, thresh_end):
            print(f"{pct}th percentile drift (start): {t_start:.4f}")
            print(f"{pct}th percentile drift (end): {t_end:.4f}")

        save_outputs(
            thresh_start,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-drift_start",
        )
        save_outputs(
            thresh_end,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-drift_end",
        )
    elif args.method == "drift_uniform":
        # Implement drift uniform filtering
        drift_all = []
        i = 0
        for query_id, topk_dict in tqdm(
            results.items(), desc=f"Evaluating {args.method} impact"
        ):
            i += 1
            if i % 1000 == 0:
                break
            question = queries[query_id]
            topk_indices = list(topk_dict.keys())[: args.top_k]
            original_docs = [corpus[doc_id]["text"] for doc_id in topk_indices]

            try:
                _, self_scores = drift_filtering_uniform(
                    args=args,
                    tokenizer=tokenizer,
                    model=model,
                    get_emb=get_emb,
                    question=question,
                    contents=original_docs,
                    iter_idx=i,
                    num_tokens_to_remove=30,
                    is_attack_list=[1 for _ in range(len(original_docs))],
                )
            except Exception as e:
                print(f"Skipping {query_id} due to error: {e}")
                continue
            diff = np.max(self_scores, axis=1) - np.min(self_scores, axis=1)
            drift_all = np.concatenate((drift_all, diff))

        drift_all = np.array(drift_all)
        drift_all = drift_all[~np.isnan(drift_all)]

        percentiles = [99, 95, 90, 85]
        thresh = np.percentile(drift_all, percentiles)
        for pct, thr in zip(percentiles, thresh):
            print(f"{pct}th percentile drift (end): {thr:.4f}")

        save_outputs(
            thresh,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-drift_uniform",
        )

    elif args.method == "perplexity":
        # You may want to add these as arguments
        gen_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        # gen_model_name = "meta-llama/Llama-2-7b-chat-hf"
        print(f"Loading generation model: {gen_model_name}")
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

        gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
        gen_model.eval()

        all_nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        i = 0
        for query_id, topk_dict in tqdm(
            results.items(), desc="Calculating NLLs for perplexity threshold"
        ):
            i += 1
            if i % 1000 == 0:
                break
            topk_indices = list(topk_dict.keys())[: args.top_k]
            original_docs = [corpus[doc_id]["text"] for doc_id in topk_indices]

            try:
                # Get NLLs for each doc (do not filter, just collect NLLs)
                input_ids = gen_tokenizer(
                    original_docs, return_tensors="pt", padding=True, truncation=True
                ).input_ids.to(device)
                with torch.no_grad():
                    logits = gen_model(input_ids, labels=input_ids).logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    # Calculate loss for each token, then mean for each sequence
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    losses = losses.view(shift_labels.size())
                    nlls = losses.mean(dim=1).cpu().numpy()
                    all_nlls.extend(nlls)
                    del logits
            except Exception as e:
                print(f"Skipping {query_id} due to error: {e}")
                continue

        all_nlls = np.array(all_nlls)
        print(f"\nCollected {len(all_nlls)} NLL values.")
        percentiles = [99.5, 99, 95, 90, 85]
        all_threshs = np.percentile(all_nlls, percentiles)
        for pct, thresh in zip(percentiles, all_threshs):
            print(f"{pct}th percentile NLL: {thresh:.4f}")

        # Optionally save all NLLs for further analysis
        save_outputs(
            all_threshs,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-nlls_all",
        )
    elif args.method == "drift_perplexity":
        gen_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"Loading generation model: {gen_model_name}")
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
        gen_model.eval()

        all_drift_scores_start = []
        all_drift_scores_end = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        window_size = 30  # or set via argument
        i = 0
        for query_id, topk_dict in tqdm(
            results.items(), desc="Calculating drift perplexity scores"
        ):
            i += 1
            if i % 1000 == 0:
                break
            topk_indices = list(topk_dict.keys())[: args.top_k]
            original_docs = [corpus[doc_id]["text"] for doc_id in topk_indices]

            for doc in original_docs:
                if not doc:
                    continue
                input_ids = gen_tokenizer(
                    doc, return_tensors="pt", truncation=True, padding=True
                ).input_ids.to(device)
                with torch.no_grad():
                    logits = gen_model(input_ids, labels=input_ids).logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    orig_ppl = losses.nanmean().item()

                seq_len = input_ids.shape[1]
                # Start truncation
                if seq_len > int(window_size * 1.5):
                    truncated_start = input_ids[:, window_size:]
                    with torch.no_grad():
                        logits = gen_model(
                            truncated_start, labels=truncated_start
                        ).logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = truncated_start[:, 1:].contiguous()
                        losses = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        drift_ppl_start = losses.nanmean().item()
                else:
                    drift_ppl_start = orig_ppl

                # End truncation
                if seq_len > window_size:
                    truncated_end = input_ids[:, : seq_len - window_size]
                    with torch.no_grad():
                        logits = gen_model(truncated_end, labels=truncated_end).logits
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = truncated_end[:, 1:].contiguous()
                        losses = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        drift_ppl_end = losses.nanmean().item()
                else:
                    drift_ppl_end = orig_ppl

                drift_score_start = abs(drift_ppl_start - orig_ppl)
                drift_score_end = abs(drift_ppl_end - orig_ppl)
                all_drift_scores_start.append(drift_score_start)
                all_drift_scores_end.append(drift_score_end)

        all_drift_scores_start = np.array(all_drift_scores_start)
        all_drift_scores_start = all_drift_scores_start[
            ~np.isnan(all_drift_scores_start)
        ]
        all_drift_scores_end = np.array(all_drift_scores_end)
        all_drift_scores_end = all_drift_scores_end[~np.isnan(all_drift_scores_end)]
        percentiles = [95, 90, 85]
        thresh_start = np.percentile(all_drift_scores_start, percentiles)
        thresh_end = np.percentile(all_drift_scores_end, percentiles)
        for pct, t_start, t_end in zip(percentiles, thresh_start, thresh_end):
            print(f"{pct}th percentile drift perplexity (start): {t_start:.4f}")
            print(f"{pct}th percentile drift perplexity (end): {t_end:.4f}")

        save_outputs(
            thresh_start,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-drift_perplexity_start",
        )
        save_outputs(
            thresh_end,
            log_args.log_name,
            f"{args.eval_dataset}-{split}-{args.eval_model_code}-{args.score_function}-{args.top_k}-drift_perplexity_end",
        )
    else:
        raise ValueError("Unknown method")


if __name__ == "__main__":
    main()
