import json
import numpy as np
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--human_fname",
        type=str,
        default="outputs_human",
        help="File name of human code detection results",
    )
    parser.add_argument(
        "--machine_fname",
        type=str,
        default="outputs",
        help="File name of machine code detection results",
    )
    parser.add_argument(
        "--attack_res_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--gen_method",
        type=str,
        default='kgw',
    )
    parser.add_argument(
        "--detect_method",
        type=str,
        default=None,
        help='None, orig, ewd, by, dip, ub'
    )
    return parser.parse_args()

def read_file(filename):
    with open(filename) as f:
        return json.load(f)

def collect_detect_res(human_path, machine_path, attack_res_file):
    human_res = read_file(human_path)
    machine_res = read_file(machine_path)
    
    if attack_res_file is not None:
        attack_res = read_file(attack_res_file)
        orig_ids = attack_res['orig_ids']
        human_corr_res = [human_res[idx] for idx in orig_ids]
    return human_corr_res, machine_res

def calculate_f1_score(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    except:
        return -100

def f1_under_fpr(pos_data, neg_data,fpr):
    pos_data=np.array(pos_data)
    neg_data=np.array(neg_data)
    neg_data=np.sort(neg_data)
    threshold=0
    for i in range(len(neg_data)):
        threshold=neg_data[i]
        tp = np.sum(pos_data > threshold)
        fp = np.sum(neg_data > threshold)
        fn = np.sum(pos_data <= threshold)
        if fp/len(neg_data) <=fpr:
            break
    print('threshold',threshold)
    tp = np.sum(pos_data > threshold)
    fp = np.sum(neg_data > threshold)
    fn = np.sum(pos_data <= threshold)
    f1_score = calculate_f1_score(tp, fp, fn)
    print('fpr:',fp/len(neg_data))
    return tp/(tp+fn),f1_score

def find_best_threshold(pos_data, neg_data):
    thresholds = np.arange(-10, 20, 0.1)
    best_f1 = 0
    best_threshold = None

    for threshold in thresholds:
        tp = np.sum(pos_data > threshold)
        fp = np.sum(neg_data > threshold)
        fn = np.sum(pos_data <= threshold)
        f1_score = calculate_f1_score(tp, fp, fn)

        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

    return best_threshold, best_f1

def f1(pos_data, neg_data,threshold):
    pos_data=np.array(pos_data)
    neg_data=np.array(neg_data)
    tp = np.sum(pos_data > threshold)
    fp = np.sum(neg_data > threshold)
    fn = np.sum(pos_data <= threshold)
    f1_score = calculate_f1_score(tp, fp, fn)
    return tp/(tp+fn),f1_score

def analyze_one(method, args):    
    if method == 'orig':
        if args.gen_method == 'kgw':
            res_key = 'wllm_detection_results'
        elif args.gen_method == 'sweet':
            res_key = 'sweet_detection_results'
    else:
        res_key = f'{method}_detection_results'

    human_results = read_file(args.human_fname)[res_key]
    machine_results = read_file(args.machine_fname)[res_key]
    human_final = []
    machine_final = []

    for human, machine in zip(human_results, machine_results):
        if human["num_tokens_detect"]>=args.min_length and machine['num_tokens_detect']>=args.min_length:
            human_final.append(human)
            machine_final.append(machine)

    score_col = 'score' if method in ['by', 'dip', 'ub'] else 'z_score'
    print(f'{method}******************************')
    human_score = [r[score_col] for r in human_final]
    machine_score = [r[score_col] for r in machine_final]
    print(len(human_score),len(machine_score))

    print('TPR (FPR = 0%)') 
    tpr_value,f1 = f1_under_fpr(machine_score, human_score, 0.0)
    print(tpr_value)
    print(f1)
    print('TPR (FPR = 1%)') 
    tpr_value,f1 = f1_under_fpr(machine_score, human_score, 0.01)
    print(tpr_value)
    print(f1)
    print('TPR (FPR = 5%)') 
    tpr_value,f1 = f1_under_fpr(machine_score, human_score, 0.05)
    print(tpr_value)
    print(f1)

    _, best_f1 = find_best_threshold(np.array(machine_score), np.array(human_score))
    print('best f1:',best_f1)


def main():
    args = parse_args()
    if args.detect_method is not None:
        analyze_one(args.detect_method, args)
        return

    human_results = read_file(args.human_fname)
    orig_detection_col = "wllm_detection_results" if args.gen_method == 'kgw' else "sweet_detection_results"
    orig_human = human_results[orig_detection_col]
    # wllm_human = human_results["wllm_detection_results"]
    # sweet_human = human_results["sweet_detection_results"]
    ewd_human = human_results["ewd_detection_results"]
    by_human = human_results["by_detection_results"]

    machine_results = read_file(args.machine_fname)
    orig_machine = machine_results[orig_detection_col]
    # wllm_machine = machine_results["wllm_detection_results"]
    # sweet_machine = machine_results["sweet_detection_results"]
    ewd_machine = machine_results["ewd_detection_results"]
    by_machine = machine_results["by_detection_results"]
    orig_human_final,ewd_human_final,by_human_final=[],[],[]
    orig_machine_final,ewd_machine_final,by_machine_final=[],[],[]
    for human,i0,i1,machine,j0,j1 in zip(orig_human,ewd_human,by_human,orig_machine,ewd_machine,by_machine):
        if human["num_tokens_detect"]>=args.min_length and machine['num_tokens_detect']>=args.min_length:
            orig_human_final.append(human)
            ewd_human_final.append(i0)
            by_human_final.append(i1)
            orig_machine_final.append(machine)
            ewd_machine_final.append(j0)
            by_machine_final.append(j1)
    print('Original******************************')
    human_z = [r['z_score'] for r in orig_human_final]
    machine_z = [r['z_score'] for r in orig_machine_final]
    print(len(human_z),len(machine_z))

    print('TPR (FPR = 1%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.01)
    print(tpr_value0)
    print(f1)
    print('TPR (FPR = 5%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.05)
    print(tpr_value0)
    print(f1)

    best_threshold, best_f1 = find_best_threshold(np.array(machine_z), np.array(human_z))
    print('best f1:',best_f1)

    print('EWD******************************')
    human_z = [r['z_score'] for r in ewd_human_final]
    machine_z = [r['z_score'] for r in ewd_machine_final]

    print('TPR (FPR = 1%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.01)
    print(tpr_value0)
    print(f1)

    print('TPR (FPR = 5%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.05)
    print(tpr_value0)
    print(f1)

    best_threshold, best_f1 = find_best_threshold(np.array(machine_z), np.array(human_z))
    print('best f1:',best_f1)
    
    print('bayesian******************************')
    human_z = [r['score'] for r in by_human_final]
    machine_z = [r['score'] for r in by_machine_final]

    print('TPR (FPR = 1%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.01)
    print(tpr_value0)
    print(f1)

    print('TPR (FPR = 5%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.05)
    print(tpr_value0)
    print(f1)

    best_threshold, best_f1 = find_best_threshold(np.array(machine_z), np.array(human_z))
    print('best f1:',best_f1)



if __name__ == "__main__":
    main()