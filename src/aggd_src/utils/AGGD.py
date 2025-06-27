import torch
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


def AGGD_candidate(args, grad, embeddings, depth, increase_loss= True):
    
    top_k_start_row = args.num_cand * (depth -1) // args.num_adv_passage_tokens
    top_k_start_col = args.num_cand * (depth -1) % args.num_adv_passage_tokens
    top_k_end_row = args.num_cand * depth // args.num_adv_passage_tokens + 1
    top_k_end_col = args.num_cand * depth % args.num_adv_passage_tokens
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embeddings.weight, # shape: (50265,, 768)
            grad.T # shape: 768
        )
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(top_k_end_row, axis = 0)
    top_k_ids = top_k_ids[top_k_start_row: top_k_end_row, :args.num_adv_passage_tokens]
    top_k_ids = top_k_ids.reshape(1, -1).T
    pos = torch.tile(torch.arange(args.num_adv_passage_tokens), (top_k_end_row- top_k_start_row, 1)).view(1,-1).T.to(args.device)
    candidates = torch.cat([pos, top_k_ids], dim =1)
    candidates = candidates[top_k_start_col: top_k_end_col - args.num_adv_passage_tokens]
    return candidates
    
def AGGD_candidate_score(args, it_, candidates, pbar, train_iter, data_collator, get_emb, model, c_model, adv_passage_ids, adv_passage_attention, adv_passage_token_type):
    current_score = 0
    candidate_scores = torch.zeros(args.num_cand, device=args.device)
    current_acc_rate = 0
    candidate_acc_rates = torch.zeros(args.num_cand, device=args.device)
    device = args.device
    for step in pbar:
        try:
            data = next(train_iter)
            data = data_collator(data) # [bsz, 3, max_len]
        except:
            print('Insufficient data!')
            break
            
        q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
        q_emb = get_emb(model, q_sent).detach()

        gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
        gold_emb = get_emb(c_model, gold_pass).detach()

        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
        sim_to_gold_mean = sim_to_gold.mean().cpu().item()
        print('Avg sim to gold p =', sim_to_gold_mean)

        p_sent = {'input_ids': adv_passage_ids, 
                'attention_mask': adv_passage_attention, 
                'token_type_ids': adv_passage_token_type}
        p_emb = get_emb(c_model, p_sent)

        # Compute loss
        sim = torch.mm(q_emb, p_emb.T)  # [b x k]
        print(it_, 'Avg sim to adv =', sim.mean().cpu().item(), 'sim to gold =', sim_to_gold_mean)
        suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).float().mean().cpu().item()
        loss = sim.mean() #sim.shape: (64, 1)
        temp_score = loss.sum().cpu().item()

        current_score += temp_score
        current_acc_rate += suc_att

        for i, candidate in enumerate(candidates):
            temp_adv_passage = adv_passage_ids.clone()
            temp_adv_passage[:, candidate[0]] = candidate[1]
            p_sent = {'input_ids': temp_adv_passage, 
                'attention_mask': adv_passage_attention, 
                'token_type_ids': adv_passage_token_type}
            p_emb = get_emb(c_model, p_sent)
            with torch.no_grad():
                sim = torch.mm(q_emb, p_emb.T)
                can_suc_att = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).float().mean().cpu().item()
                can_loss = sim.mean()
                temp_score = can_loss.sum().cpu().item()

                candidate_scores[i] += temp_score
                candidate_acc_rates[i] += can_suc_att
    print(current_score, max(candidate_scores).cpu().item())
    print(current_acc_rate, max(candidate_acc_rates).cpu().item())
    return candidate_scores, candidate_acc_rates, current_score, current_acc_rate
    

    
