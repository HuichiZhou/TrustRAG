import os
import re

def run(test_params):
    log_name = get_log_name(test_params)
    if log_name + '.log' in os.listdir('logs'):
        print(f"log {log_name}.log already exists")
        # load the log file
        with open(f'logs/{log_name}.log', 'r') as f:
            lines = f.readlines()
        # search if "Incorrect Answer Percentage:" in the log file with re
        if len(lines) > 0:
            incorrect_answer_percentage = re.search(r'Incorrect Answer Percentage:', lines[-1])
            if not incorrect_answer_percentage:
                print(f"log {log_name}.log is not complete, remove it")
                os.remove(f'logs/{log_name}.log')
            else:
                print(f"log {log_name}.log is complete")
                return
        else:
            print(f"log {log_name}.log is not complete, remove it")
            os.remove(f'logs/{log_name}.log')

    cmd = f"python3 -u main_trustrag.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --top_k {test_params['top_k']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --log_name {log_name} \
        --defend_method {test_params['defend_method']}\
        --removal_method {test_params['removal_method']}"    
    os.system(cmd)

def get_log_name(test_params):
    log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}"
    if test_params['removal_method'] != None:
        log_name += f"-removal-{test_params['removal_method']}"
    if test_params['defend_method'] != None:
        log_name += f"-defend-{test_params['defend_method']}"
    log_name += f"-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"
    if test_params['note'] != None:
        log_name = test_params['note']
    return log_name



test_params = {
    'eval_model_code': "contriever",
    'eval_dataset': "nq", # ['nq','hotpotqa', 'msmarco']
    'split': "test",
    'query_results_dir': 'main',
    'model_name': 'llama7b', 
    'top_k': 5,
    'gpu_id': 0,
    'attack_method': 'LM_targeted', # ['none', 'LM_targeted', 'hotflip', 'pia']
    'defend_method': 'conflict', # ['none', 'conflict', 'astute', 'instruct']
    'removal_method': 'none', # ['kmeans', 'kmeans_ngram', 'ragdefender', 'none']
    'adv_per_query': 3, # poison rate = adv_per_query / top_k
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10, # number of queries
    'seed': 12,
    'note': None
}


for dataset in ['hotpotqa', 'nq', 'msmarco']:
    for model in ["mistralai/Mistral-Nemo-Instruct-2407"]:
        for number_of_adv in [1, 2, 3, 4, 5]:
            test_params['eval_dataset'] = dataset
            test_params['adv_per_query']=number_of_adv
            test_params['model_name'] = model
            test_params['attack_method'] = 'hotflip'
            test_params['defend_method'] = 'conflict'
            test_params['removal_method'] = 'kmeans_ngram'
            run(test_params)