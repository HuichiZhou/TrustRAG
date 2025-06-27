import sys
sys.path.append("./contriever")
sys.path.append("./contriever/src")
import os
import torch
from collections import Counter
from contriever import Contriever

from transformers import AutoTokenizer

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-ctx_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'dpr' in model_code:
        model = DPRQuestionEncoder.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = DPRContextEncoder.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = dpr_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb
def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings
def kmeans_split(data_dict, model, get_emb, tokenizer, k, split):
    """Get all query embeddings and perform kmeans"""
    
    # get query embs
    q_embs = []
    for q in tqdm(data_dict["sent0"]):
        query_input = tokenizer(q, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input) # query_emb.shape: [1, 768]
        q_embs.append(query_emb[0].cpu().numpy()) # query_emb[0].shape: [768]
    q_embs = np.array(q_embs)
    print("q_embs", q_embs.shape) # (132803, 768)


    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    print(Counter(kmeans.labels_))

    ret_dict = {"sent0": [], "sent1": []}
    for i in range(len(data_dict["sent0"])):
        if kmeans.labels_[i] == split:
            ret_dict["sent0"].append(data_dict["sent0"][i])
            ret_dict["sent1"].append(data_dict["sent1"][i])
    print("K = %d, split = %d, tot num = %d"%(k, split, len(ret_dict["sent0"])))
    return ret_dict