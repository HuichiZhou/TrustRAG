import os
from .contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import numpy as np
import pickle
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
from loguru import logger

import sys

import time

model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-ms": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "roberta": "sentence-transformers/all-distilroberta-v1",
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-ms": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "roberta": "sentence-transformers/all-distilroberta-v1",
}


def cos_similarity(a, b):
    return torch.mm(
        F.normalize(a, dim=1),
        F.normalize(b, dim=1).T,
    )


def load_cached_data(cache_file, load_function, *args, **kwargs):
    if os.path.exists(cache_file):
        logger.info(f"Cache file {cache_file} exists. Loading data...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        logger.info(f"Cache file {cache_file} does not exist. Generating data...")
        data = load_function(*args, **kwargs)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        return data


def setup_experiment_logging(experiment_name=None, log_dir="logs"):
    """
    Configure logging for experiments with both console and file output.

    Args:
        experiment_name: Name of the experiment for the log file
        log_dir: Directory to store log files
    """
    # Remove any existing handlers
    logger.remove()

    # Add console handler with a simple format
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file handler if experiment_name is provided
    if experiment_name:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="INFO",
        )

    return logger


def contriever_get_emb(model, input):
    return model(**input)


def dpr_get_emb(model, input):
    return model(**input).pooler_output


def ance_get_emb(model, input):
    input.pop("token_type_ids", None)
    return model(input)["sentence_embedding"]


def load_models(model_code):
    assert (
        model_code in model_code_to_qmodel_name
        and model_code in model_code_to_cmodel_name
    ), f"Model code {model_code} not supported!"
    if "contriever" in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert (
            model_code_to_cmodel_name[model_code]
            == model_code_to_qmodel_name[model_code]
        )
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif any(
        substring in model_code for substring in ["ance", "minilm", "mpnet", "roberta"]
    ):
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert (
            model_code_to_cmodel_name[model_code]
            == model_code_to_qmodel_name[model_code]
        )
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    # model: 用于生成query的embedding
    # c_model: 用于生成context的embedding
    return model, c_model, tokenizer, get_emb


def load_beir_datasets(dataset_name, split):
    assert dataset_name in ["nq", "msmarco", "hotpotqa", "mirage"]
    if dataset_name == "msmarco":
        split = "train"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name
    )
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        print(f"Downloading {dataset_name} ...")
        print("out_dir: ", out_dir)
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if "-train" in data_path:
        split = "train"
    corpus, queries, qrels = data.load(split=split)
    # corpus: 文档集合
    # queries: 查询集合
    # qrels: 查询-文档关系集合

    return corpus, queries, qrels


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_outputs(outputs, dir, file_name):
    json_dict = json.dumps(outputs, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f"data_cache/outputs/{dir}"):
        os.makedirs(f"data_cache/outputs/{dir}", exist_ok=True)
    with open(
        os.path.join(f"data_cache/outputs/{dir}", f"{file_name}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dict_from_str, f, indent=4)


def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f"results/query_results/{dir}"):
        os.makedirs(f"results/query_results/{dir}", exist_ok=True)
    with open(
        os.path.join(f"results/query_results/{dir}", f"{file_name}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dict_from_str, f, indent=4)


def load_results(file_name):
    with open(os.path.join("results", file_name)) as file:
        results = json.load(file)
    return results


def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dict_from_str, f, indent=4)


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_str(s):
    try:
        s = str(s)
    except (ValueError, TypeError) as e:
        print(e)
        print("Error: the output cannot be converted to a string")
    s = s.strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def f1_score(precision, recall):
    f1_scores = np.divide(
        2 * precision * recall, precision + recall, where=(precision + recall) != 0
    )
    return f1_scores


class LoguruProgress:
    def __init__(self, iterable=None, desc=None, total=None, **kwargs):
        self.iterable = iterable
        self.desc = desc
        self.total = len(iterable) if iterable is not None else total
        self.n = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info(f"Starting {desc}: 0/{self.total}")

    def update(self, n=1):
        self.n += n
        current_time = time.time()
        # Log every second or at completion
        if current_time - self.last_log_time > 1 or self.n >= self.total:
            elapsed = current_time - self.start_time
            rate = self.n / elapsed if elapsed > 0 else 0
            logger.info(
                f"{self.desc}: {self.n}/{self.total} "
                f"[{elapsed:.1f}s elapsed, {rate:.1f} it/s]"
            )
            self.last_log_time = current_time

    def __iter__(self):
        if self.iterable is None:
            raise ValueError("Iterable not provided")
        for obj in self.iterable:
            yield obj
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            elapsed = time.time() - self.start_time
            rate = self.n / elapsed if elapsed > 0 else 0
            logger.info(
                f"Completed {self.desc}: {self.n}/{self.total} "
                f"[{elapsed:.1f}s elapsed, {rate:.1f} it/s]"
            )


def progress_bar(iterable=None, desc=None, total=None, **kwargs):
    """
    A wrapper function that returns either LoguruProgress or tqdm based on whether we want
    logging output or standard tqdm output
    """
    return LoguruProgress(iterable=iterable, desc=desc, total=total, **kwargs)


def get_tokenized_sentence_sliced(tokenized, i, slice_order):
    keys = ["input_ids", "token_type_ids", "attention_mask"]
    tensors = {k: tokenized[k] for k in tokenized.keys() if k in keys}

    valid_token_count = tokenized["attention_mask"].sum(dim=1)
    if valid_token_count - 2 <= i:
        print(
            f"Valid token count {valid_token_count} is less than the number of indexes {i} to truncate.\nReturning original input."
        )
        return tensors
    eos = {
        k: v[:, valid_token_count - 1 : valid_token_count] for k, v in tensors.items()
    }
    sos = {k: v[:, :1] for k, v in tensors.items()}
    if slice_order == "end":
        main = {k: v[:, : valid_token_count - i - 1] for k, v in tensors.items()}
        out = {
            k: torch.cat((main[k], eos[k]), dim=1) for k in tensors.keys() if k in keys
        }
    elif slice_order == "start":
        main = {k: v[:, i + 1 :] for k, v in tensors.items()}
        out = {
            k: torch.cat((sos[k], main[k]), dim=1) for k in tensors.keys() if k in keys
        }
    else:
        raise ValueError(f"Unsupported slice_order: {slice_order}")

    return out


def get_tokenized_sentence_block_removed(tokenized, block_start: int, block_size: int):
    """
    Removes a fixed block of tokens from the tokenized input while preserving the SOS (start-of-sequence) and EOS (end-of-sequence) tokens.

    Args:
        tokenized (dict): Dictionary containing "input_ids", "token_type_ids", and "attention_mask".
        block_start (int): Index (exclusive of SOS) at which to start removing tokens.
        block_size (int): Number of tokens to remove starting from block_start.

    Returns:
        dict: Modified tokenized input with the block removed.
    """
    keys = ["input_ids", "token_type_ids", "attention_mask"]
    tensors = {k: tokenized[k] for k in tokenized.keys() if k in keys}

    valid_token_count = tokenized["attention_mask"].sum(dim=1)
    # Ensure there is room to remove a block between SOS and EOS
    available_tokens = valid_token_count - 2  # exclude SOS and EOS
    if block_start >= available_tokens:
        print(
            f"block_start {block_start} is beyond available tokens ({available_tokens}). Returning original."
        )
        return tensors
    block_end = min(block_start + block_size, available_tokens)

    sos = {k: v[:, :1] for k, v in tensors.items()}
    eos = {
        k: v[:, valid_token_count - 1 : valid_token_count] for k, v in tensors.items()
    }
    before = {k: v[:, 1 + 0 : 1 + block_start] for k, v in tensors.items()}  # after SOS
    after = {
        k: v[:, 1 + block_end : valid_token_count - 1] for k, v in tensors.items()
    }  # before EOS

    out = {
        k: torch.cat([sos[k], before[k], after[k], eos[k]], dim=1)
        for k in tensors.keys()
    }
    return out


def get_tokenized_sentence_uniform(tokenized, num_tokens_to_remove):
    """Remove ``k`` tokens uniformly across the sequence.

    Parameters
    ----------
    tokenized : dict
        Tokenized input containing ``input_ids``, ``token_type_ids`` and
        ``attention_mask``.
    k : int
        Number of tokens to remove.

    Returns
    -------
    dict
        Tokenized input with ``k`` tokens removed uniformly. If ``k`` is
        larger than the available number of tokens (excluding the special
        tokens), the original tokenization is returned unchanged.
    """

    keys = ["input_ids", "token_type_ids", "attention_mask"]
    valid_token_count = int(tokenized["attention_mask"].sum(dim=1).item())
    tensors = {
        k: tokenized[k][:, :valid_token_count] for k in tokenized.keys() if k in keys
    }
    # ensure num_tokens_to_remove is int
    num_tokens_to_remove = int(num_tokens_to_remove)
    if valid_token_count - 2 <= num_tokens_to_remove:
        print(
            f"Valid token count {valid_token_count} is less than the number of indexes {num_tokens_to_remove} to truncate.\nReturning original input."
        )
        return tensors

    # Choose positions evenly across the sequence excluding the first/last token
    remove_positions = (
        torch.randperm(valid_token_count - 2, device=tensors["input_ids"].device)[
            :num_tokens_to_remove
        ]
        + 1
    )

    mask = torch.ones(
        valid_token_count, dtype=torch.bool, device=tensors["input_ids"].device
    )
    mask[remove_positions] = False

    out = {k: tensors[k][:, mask] for k in tensors.keys() if k in keys}
    return out
