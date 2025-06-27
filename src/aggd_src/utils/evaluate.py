import torch
from tqdm import tqdm
def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention,adv_passage_token_type, data_collator, device='cuda'):
    """Returns the 2-way classification accuracy (used during training)"""
    model.eval()
    c_model.eval()
    acc = 0
    tot = 0
    for idx, (data) in tqdm(enumerate(dataloader)):
        data = data_collator(data) # [bsz, 2, max_len]

        # Get query embeddings
        q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
        q_emb = get_emb(model, q_sent)  # [b x d] # [16, 768]

        gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
        gold_emb = get_emb(c_model, gold_pass) # [b x d]

        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze() #[16, 1, 1]
        # q_emb.unsqueeze(dim=1): [16, 1, 768]; gold_emb.unsqueeze(dim=2): [16, 768, 1]
        p_sent = {'input_ids': adv_passage_ids, 
                  'attention_mask': adv_passage_attention,
                  'token_type_ids': adv_passage_token_type
                  }
        p_emb = get_emb(c_model, p_sent)  # [k x d] # [1, 768]

        sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]

        acc += (sim_to_gold > sim).sum().cpu().item() # sim_to_gold.shape: [16], sim.shape: [16]
        tot += q_emb.shape[0]
    
    print(f'Acc = {acc / tot * 100} ({acc} / {tot})')
    return acc / tot