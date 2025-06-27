import os
from functools import partial
from beir.datasets.data_loader import GenericDataLoader
from utils.load_model import kmeans_split
from torch.utils.data import DataLoader
import numpy as np
import random
from datasets import Dataset
from transformers import default_data_collator
from beir import util
def tokenization(examples, tokenizer, max_seq_length, pad_to_max_length):
    q_feat = tokenizer(examples["sent0"], max_length=max_seq_length, truncation=True, padding="max_length" if pad_to_max_length else False)
    c_feat = tokenizer(examples["sent1"], max_length=max_seq_length, truncation=True, padding="max_length" if pad_to_max_length else False)

    ret = {}
    for key in q_feat:
        ret[key] = [(q_feat[key][i], c_feat[key][i]) for i in range(len(examples["sent0"]))]
    return ret

def load_data(args, tokenizer, model, get_emb):

    # Load datasets
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)
    l = list(qrels.items()) 
    random.shuffle(l) 
    qrels = dict(l)

    data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = queries[q]
        for c in qrels[q]:
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            data_dict["sent0"].append(q_ctx)
            data_dict["sent1"].append(c_ctx)

    if args.do_kmeans:
        data_dict = kmeans_split(data_dict, model, get_emb, tokenizer, k=args.k, split=args.kmeans_split)

    datasets = {"train": Dataset.from_dict(data_dict)}



    print('Train data size = %d'%(len(datasets["train"])))
    num_valid = min(1000, int(len(datasets["train"]) * 0.3))
    datasets["subset_valid"] = Dataset.from_dict(datasets["train"][:num_valid])
    datasets["subset_train"] = Dataset.from_dict(datasets["train"][num_valid:])

    train_dataset = datasets["subset_train"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names, fn_kwargs={"tokenizer":  tokenizer, "max_seq_length": args.max_seq_length, "pad_to_max_length": args.pad_to_max_length})
    dataset = datasets["subset_valid"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names, fn_kwargs={"tokenizer":  tokenizer, "max_seq_length": args.max_seq_length, "pad_to_max_length": args.pad_to_max_length})


    data_collator = default_data_collator
    dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=True, collate_fn=lambda x: x )
    valid_dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x )
    return data_collator, dataloader, valid_dataloader