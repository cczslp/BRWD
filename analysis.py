import torch
import pickle
import json
import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datasets import load_dataset
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import tasks
from lm_eval.tasks import ALL_TASKS
from lm_eval.base import Task
from argparse import ArgumentParser
from matplotlib.ticker import FormatStrFormatter


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default="approx",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mbpp",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ewd",
    )
    parser.add_argument(
        "--logp_cache_path",
        type=str,
        default="pkl/mbpp_gr_logp.pkl",
    )
    parser.add_argument(
        "--wii_cache_path",
        type=str,
        default="pkl/wii.pkl",
    )
    parser.add_argument(
        "--ent_cache_path",
        type=str,
        default="pkl/ent.pkl"
    )
    parser.add_argument(
        "--generations_dir",
        type=str,
        default="outputs/mbpp_kgw_outputs",
    )
    parser.add_argument(
        "--model",
        default="/data/starcoder2-7b",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--plt_lb",
        type=float,
        default=None
    )
    parser.add_argument(
        "--plt_ub",
        type=float,
        default=None
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma for WLLM",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Delta for WLLM",
    )
    parser.add_argument(
        "--by_topk",
        type=int,
        default=2000,
        help='bayesian detection hyperparameter'
    )
    parser.add_argument(
        "--detection_temp",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def read_file(filename):
    with open(filename) as f:
        return json.load(f)


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save_pickle(pkl_path, obj):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(obj, fp)


def seed_rng(rng, input_ids: torch.LongTensor, hash_key: int = 15485917) -> None:
    prev_token = input_ids[-1].item()
    rng.manual_seed(hash_key * prev_token)


def get_green_ids(rng, vocab_size, gamma, input_ids: torch.LongTensor):
    seed_rng(rng, input_ids)
    greenlist_size = int(vocab_size * gamma)
    vocab_permutation = torch.randperm(vocab_size, generator=rng)
    return vocab_permutation[:greenlist_size]


def tokenize(example, tokenizer):
    inputs = tokenizer(
        example,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=2048
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }


@torch.no_grad()
def calculate_penalty_and_prob(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        # trust_remote_code=args.trust_remote_code,
        # use_auth_token=args.use_auth_token,
        truncation_side="left",
        padding_side="right",
    )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dict_precisions[args.precision],
    ).cuda()

    target_tasks = ['humaneval', 'mbpp']
    pr_settings = [True, False]
    rng = torch.Generator()
    all_penalties = []
    all_prob_human = []

    print('================calculate penalty and prob-gap===============')
    for task in target_tasks:
        task = tasks.get_task(task)
        ds = task.get_dataset()
        for use_pr in pr_settings:
            print(f'----------{task} {"with" if use_pr else "without"} prompt----------')
            ds_penalties, ds_green_prob = calculate_ds_penalty_and_prob(task, ds, model, tokenizer,
                                                                        rng, args, use_pr)
            all_penalties.append(ds_penalties)
            all_prob_human.append(ds_green_prob)

    return all_penalties, all_prob_human

@torch.no_grad()
def calculate_ds_penalty_and_prob(task, ds, model, tokenizer, rng, args, use_pr=True):
    vocab_size = len(tokenizer.get_vocab())
    const_exp = math.exp(args.delta / args.detection_temp)
    
    def swap_prefix(text, old_prefix, new_prefix)->str:
        return new_prefix + text[len(old_prefix):]
    
    ds_penalties = []
    ds_green_prob = []
    general_prompt = task.get_general_prompt()
    for sample in tqdm(ds):
        prompt = task.get_prompt(sample)
        full_text = task.get_full_data(sample)
        if not use_pr:
            orig_pr = prompt
            prompt = general_prompt
            full_text = swap_prefix(full_text, orig_pr, prompt)

        tokenized_prompt = tokenize(prompt, tokenizer)['input_ids'].to(model.device)
        tokenized_full_text = tokenize(full_text, tokenizer)['input_ids'].to(model.device)
        
        prompt_len, full_len = tokenized_prompt.size(1), tokenized_full_text.size(1)
        tokenized_completion = tokenized_full_text[0, prompt_len:]

        output = model(tokenized_full_text, return_dict=True)
        logits = output.logits.to(dtype=torch.float32).squeeze()[prompt_len:]
        _, topk_ids = torch.topk(logits, args.by_topk, dim=-1, sorted=False)
        topk_masks = torch.zeros_like(logits)
        topk_masks.scatter_(1, topk_ids, torch.ones_like(logits)) # modified
        green_masks = torch.zeros_like(logits)
        for i in range(prompt_len, full_len):
            green_ids = get_green_ids(rng, vocab_size, args.gamma, tokenized_full_text[0, :i]).to(model.device)
            green_masks[i - prompt_len, green_ids] = 1
        green_masks = green_masks[1:, :] # !!! align prob, mask

        # print('topk_masks_sum:', topk_masks.sum(dim=-1)[:5], 'green_masks:', green_masks[:5, :10], sep='\n')
        probs = torch.softmax(logits / args.detection_temp, dim=-1)[:-1, :] # !!! align prob, mask 
        top_probs = probs * topk_masks[:-1, :]
        top_green_masses = (top_probs * green_masks).sum(dim=-1)
        top_red_masses = (top_probs * (1 - green_masks)).sum(dim=-1)
        # print('top_green_mass:', top_green_masses[:5], 'top_red_mass:', top_red_masses[:5], sep='\n')

        penalty = ((const_exp * top_green_masses + top_red_masses) / (top_green_masses + top_red_masses)).log()
        green_indicators = torch.gather(green_masks, dim=1, index=tokenized_completion[1:].unsqueeze(1)).squeeze()
        
        # wm_green_probs = top_green_masses * const_exp / (top_green_masses * const_exp + top_red_masses)
        # green_prob_gaps = wm_green_probs - green_indicators

        # print('penalty:', penalty[:5], 'gap:', green_prob_gaps[:5], sep='\n')
        ds_penalties.extend(penalty.tolist())
        ds_green_prob.extend(green_indicators.tolist())
    return ds_penalties, ds_green_prob

@torch.no_grad()
def calculate_reward_and_prob(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        # trust_remote_code=args.trust_remote_code,
        # use_auth_token=args.use_auth_token,
        truncation_side="left",
        padding_side="right",
    )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dict_precisions[args.precision],
    ).cuda()

    target_task = tasks.get_task('mbpp')
    ds = target_task.get_dataset()

    gamma, delta = args.gamma, args.delta
    vocab_size = len(tokenizer.get_vocab())
    alpha = math.exp(delta)
    tau = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))
    rng = torch.Generator()

    # This function is based on the authors' original implementation
    def compute_ewd_reward(probs):
        denoms = 1 + tau * probs
        renormed_probs = probs / denoms
        spike_ents = renormed_probs.sum(dim=-1)
        SE = torch.sub(spike_ents, torch.min(spike_ents))
        reward = SE * (1 - gamma)
        return reward.cpu().tolist()
    
    def compute_green_prob(green_masks, probs):
        green_probs = (probs * green_masks).sum(dim=-1)
        return green_probs.cpu().tolist()

    def compute_by_reward(green_masks, probs):
        _, topk_ids = torch.topk(probs, args.by_topk, dim=-1, sorted=False)
        topk_masks = torch.zeros_like(probs)
        topk_masks.scatter_(1, topk_ids, torch.ones_like(probs))
        top_probs = topk_masks * probs
        top_green_masses = (top_probs * green_masks).sum(dim=-1)
        top_red_masses = (top_probs * (1 - green_masks)).sum(dim=-1)
        reward = delta - ((alpha * top_green_masses + top_red_masses) / (top_green_masses + top_red_masses)).log()
        return reward.cpu().tolist()

    ewd_reward_list = []
    by_reward_list = []
    green_prob_list = []
    for sample in tqdm(ds):
        prompt = target_task.get_prompt(sample)
        full_text = target_task.get_full_data(sample)
        tokenized_prompt = tokenize(prompt, tokenizer)['input_ids'].to(model.device)
        tokenized_full_text = tokenize(full_text, tokenizer)['input_ids'].to(model.device)
        prompt_len, full_len = tokenized_prompt.size(1), tokenized_full_text.size(1)

        output = model(tokenized_full_text, return_dict=True)
        logits = output.logits.to(dtype=torch.float32).squeeze()[prompt_len-1:-1]
        probs = torch.softmax(logits, dim=-1)

        green_masks = torch.zeros_like(logits)
        for i in range(prompt_len, full_len):
            green_ids = get_green_ids(rng, vocab_size, gamma, tokenized_full_text[0, :i]).to(model.device)
            green_masks[i - prompt_len, green_ids] = 1

        ewd_rewards = compute_ewd_reward(probs)
        by_rewards = compute_by_reward(green_masks, probs)
        green_probs = compute_green_prob(green_masks, probs)
        ewd_reward_list.extend(ewd_rewards)
        by_reward_list.extend(by_rewards)
        green_prob_list.extend(green_probs)

    return green_prob_list, [ewd_reward_list, by_reward_list]


@torch.no_grad()
def calculate_text_green_prob(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        # trust_remote_code=args.trust_remote_code,
        # use_auth_token=args.use_auth_token,
        truncation_side="left",
        padding_side="right",
    )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dict_precisions[args.precision],
    ).cuda()

    target_task = tasks.get_task('mbpp')
    ds = target_task.get_dataset()
    generations_path = f'{args.generations_dir}/machine_code.json'
    with open(generations_path) as fp:
        generations = json.load(fp)
    prompt_contents = [target_task.get_prompt(ds[sid]) for sid in range(len(generations))]

    gamma = args.gamma
    vocab_size = len(tokenizer.get_vocab())
    rng = torch.Generator()

    human_avg_green_logprobs = []
    wm_avg_green_logprobs = []

    def cal_green_logprob(prompt, full_text):
        tokenized_prompt = tokenize(prompt, tokenizer)['input_ids'].to(model.device)
        tokenized_full_text = tokenize(full_text, tokenizer)['input_ids'].to(model.device)
        prompt_len, full_len = tokenized_prompt.size(1), tokenized_full_text.size(1)
        if full_len <= prompt_len:
            return None

        output = model(tokenized_full_text, return_dict=True)
        logits = output.logits.to(dtype=torch.float32).squeeze()[prompt_len-1:-1]
        probs = torch.softmax(logits, dim=-1)

        green_masks = torch.zeros_like(logits)
        for i in range(prompt_len, full_len):
            green_ids = get_green_ids(rng, vocab_size, gamma, tokenized_full_text[0, :i]).to(model.device)
            green_masks[i - prompt_len, green_ids] = 1

        green_probs = (probs * green_masks).sum(dim=-1)
        avg_logprob = green_probs.log().mean().item()        
        return avg_logprob

    # calculate average green log prob for human written code
    for sample in tqdm(ds):
        prompt = target_task.get_prompt(sample)
        full_text = target_task.get_full_data(sample)
        avg_gr_logprob = cal_green_logprob(prompt, full_text)
        if avg_gr_logprob is not None:
            human_avg_green_logprobs.append(avg_gr_logprob)
    
    # calculate average green log prob for watermarked code
    for idx in range(len(generations)):
        prompt = prompt_contents[idx]
        gen = generations[idx][0]
        avg_gr_logprob = cal_green_logprob(prompt, gen)
        if avg_gr_logprob is not None:
            wm_avg_green_logprobs.append(avg_gr_logprob)

    return human_avg_green_logprobs, wm_avg_green_logprobs


@torch.no_grad()
def calculate_wii_and_ent(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        # trust_remote_code=args.trust_remote_code,
        # use_auth_token=args.use_auth_token,
        truncation_side="left",
        padding_side="right",
    )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dict_precisions[args.precision],
    ).cuda()

    task = tasks.get_task(args.task_name)
    ds = task.get_dataset()
    ds_num = len(ds)
    prompt_contents = [task.get_prompt(ds[sample]) for sample in range(ds_num)]
    load_gen_path = f'{args.generations_dir}/machine_code.json'
    generations = read_file(load_gen_path)
    vocab_size = len(tokenizer.get_vocab())
    rng = torch.Generator()

    green_wii_list, green_ent_list = [], []
    red_wii_list, red_ent_list = [], []
    for idx, gens in tqdm(enumerate(generations)):
        gen = gens[0]
        prefix = prompt_contents[idx]
        tokenized_prefix = tokenize(prefix, tokenizer)['input_ids'].squeeze()
        tokenized_text = tokenize(gen, tokenizer)['input_ids'].squeeze().to(model.device)
        prefix_len = len(tokenized_prefix)
        full_len = len(tokenized_text)

        model_logits = model(torch.unsqueeze(tokenized_text, 0), return_dict=True).logits[0, prefix_len-1:-1, :]
        probs = torch.softmax(model_logits, dim=-1)
        info_entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)

        green_masks = torch.zeros_like(probs)
        for i in range(prefix_len, full_len):
            green_ids = get_green_ids(rng, vocab_size, args.gamma, tokenized_text[:i]).to(model.device)
            green_masks[i - prefix_len, green_ids] = 1
        green_indicators = torch.gather(green_masks, dim=1, index=tokenized_text[prefix_len:].unsqueeze(1)).squeeze()
        green_indices = torch.where(green_indicators > 0)[0]
        red_indices = torch.where(green_indicators == 0)[0]
        
        green_prob = (probs * green_masks).sum(dim=-1)
        red_prob = 1 - green_prob
        green_wii_list.extend(red_prob[green_indices].cpu().tolist())
        green_ent_list.extend(info_entropy[green_indices].cpu().tolist())
        red_wii_list.extend(green_prob[red_indices].cpu().tolist())
        red_ent_list.extend(info_entropy[red_indices].cpu().tolist())

    wii_res = {
        'green':green_wii_list,
        'red':red_wii_list
    }
    ent_res = {
        'green':green_ent_list,
        'red':red_ent_list
    }
    return wii_res, ent_res    


def plot_avg_corr(x, y, fig_path):
    task_names = ['HumanEval', 'MBPP']
    colors = ['purple', 'red', 'green', 'black']
    suffixes = ['w/ prompt', 'w/o prompt']

    bins = np.arange(0, 2.2, 0.2)
    bin_centers = bins[:-1] + 0.1  # group centers of token penalty
    
    _, ax = plt.subplots(figsize=(6, 9))
    for i in range(2):
        for j in range(2):
            label = f'{task_names[i]} {suffixes[j]}'
            x_cur, y_cur = np.array(x[2*i+j]), np.array(y[2*i+j])
            bin_indices = np.digitize(x_cur, bins)
            # toks_bin_cnt = np.bincount(bin_indices)
            # print(toks_bin_cnt)
            mean_y_per_bin = [np.mean(y_cur[bin_indices == b]) for b in range(1, len(bins))]
            ax.plot(bin_centers, mean_y_per_bin, marker='o', alpha=0.8, color=colors[2*i+j], label=label)

    ax.legend()
    ax.set_xlabel('Bayesian Inference Penalty on Token')
    ax.set_ylabel('Probability of Token Being Green') 
    ax.grid(False)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')


def plot_reward_penalty_prob_corr(reward, penalty, prob, method, fig_path):
    bins = np.arange(0, 1.1, 0.1)
    bin_centers = bins[:-1] + 0.05
    reward, penalty, prob = np.array(reward), np.array(penalty), np.array(prob)

    reward_label = f'{method} Reward'
    penalty_label = f'{method} Penalty'
    _, ax = plt.subplots(figsize=(6, 4.5))
    bin_indices = np.digitize(prob, bins)

    def plot_y_bin_avg(y, color, label):
        mean_y_per_bin = [np.mean(y[bin_indices == b]) for b in range(1, len(bins))]
        ax.plot(bin_centers, mean_y_per_bin, marker='o', alpha=0.8, color=color, label=label)

    plot_y_bin_avg(reward, 'green', reward_label)
    plot_y_bin_avg(penalty, 'red', penalty_label)

    ax.legend(prop={'family': 'serif', 'size': 12})
    ax.set_xlabel('probability of token being green', fontfamily='serif', fontsize=16)
    ax.set_ylabel('token specific reward or penalty', fontfamily='serif', fontsize=16) 
    ax.grid(False)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')


def plot_score_dist(human_score, wm_score, method, fig_path, lower_bound=None, upper_bound=None):
    plt.figure(figsize=(6, 4.5))

    # Determine and mark the 1% FPR threshold line for human scores
    human_score_sorted = np.sort(human_score)[::-1]
    threshold_index = int(len(human_score_sorted) * 0.05)
    threshold_value = human_score_sorted[threshold_index]

    # Plot the KDE plots for both datasets
    sns.kdeplot(human_score, fill=True, color="skyblue", alpha=0.5, label="Human Code Score")
    sns.kdeplot(wm_score, fill=True, color="red", alpha=0.5, label="Watermarked Code Score")
    plt.axvline(threshold_value, color='black', linestyle='--', label=f'5% FPR Threshold (score={threshold_value:.2f})')
    
    # x = np.linspace(min(wm_score), max(wm_score), 1000)
    kde = sns.kdeplot(wm_score, fill=False, color="red", alpha=0.5).get_lines()[-1].get_data()
    x_kde, y_kde = kde
    mask = x_kde > threshold_value
    plt.fill_betweenx(y_kde, x_kde, threshold_value, where=mask, label='TPR at 5% FPR',
                      color="none", hatch="//", edgecolor="black", alpha=0.3)

    # Set the x-axis limits if specified
    if lower_bound is not None or upper_bound is not None:
        plt.xlim(lower_bound, upper_bound)
    
    # Format y-axis to show three decimal places
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Set the x-axis label based on the method
    if method == 'ewd':
        xlabel = 'scores given by EWD detector'
    else:
        xlabel = 'scores given by BIWD detector'

    plt.legend(prop={'family': 'serif', 'size': 8}, loc='upper left')
    plt.xlabel(xlabel, fontsize=16, fontfamily='serif')
    plt.ylabel("probability density", fontsize=16, fontfamily='serif')
    plt.tight_layout()

    # Save the plot to the specified file path
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

# def plot_score_dist(human_score, human_x, wm_score, wm_x, 
#                           fig_path, method):
#     _, ax = plt.subplots(figsize=(6, 4.5))
#     ax.scatter(human_x, human_score, alpha=0.3, color='orange', label='Human Code')    
#     ax.scatter(wm_x, wm_score, alpha=0.3, color='blue', label='Watermarked Code')    
#     ax.set_xlabel("average log probability of token being green", fontsize=16, fontfamily='serif')
    
#     y_label = 'z-score given by EWD' if method == 'ewd' else 'score given by our detector'
#     ax.set_ylabel(y_label, fontsize=16, fontfamily='serif')
#     ax.legend(prop={'family': 'serif', 'size': 12})
    
#     plt.tight_layout()
#     plt.savefig(fig_path, dpi=300, bbox_inches='tight')


def plot_prob_gt_penalty(penalty, human_green, model_green_prob, fig_path):
    pass


def plot_wii_ent_scatter(wii, ent, fig_path):
    wii_green, ent_green = wii['green'], ent['green']
    wii_red, ent_red = wii['red'], ent['red']
    
    _, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(wii_green, ent_green, s=4, alpha=0.7, color='green', label='Green Tokens')
    ax.scatter(wii_red, ent_red, s=4, alpha=0.7, color='#FF6347', label='Red Tokens')
    ax.set_xlabel("watermark information score of tokens", fontsize=16, fontfamily='serif')
    
    y_label = 'information entropy of tokens'
    ax.set_ylabel(y_label, fontsize=16, fontfamily='serif')
    ax.legend(prop={'family': 'serif', 'size': 10})
    
    rect_x = 0.6
    rect_y = 0.2
    rect_width = 0.35
    rect_height = 1.2
    rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
        linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)

    # Adding the arrow and text
    ax.annotate(
        "high WIS but low-entropy",
        xy=(rect_x + rect_width / 2, rect_y + rect_height),
        xytext=(0.8, 4),                         
        arrowprops=dict(facecolor='blue', arrowstyle='->', lw=1),
        fontsize=10, fontfamily='serif', fontweight='bold', ha='center', va='center', color='black'
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    

def analyze_by_approximation(args):
    cache_penalty_path = 'penalty.pkl'
    cache_human_path = 'human_green.pkl'
    if os.path.exists(cache_penalty_path) and os.path.exists(cache_human_path):
        with open(cache_penalty_path, 'rb') as penalty_file:
            penalty = pickle.load(penalty_file)
        with open(cache_human_path, 'rb') as human_file:
            human_green = pickle.load(human_file)
    else:
        penalty, human_green = calculate_penalty_and_prob(args)
        with open(cache_penalty_path, 'wb') as penalty_file:
            pickle.dump(penalty, penalty_file)
        with open(cache_human_path, 'wb') as human_file:
            pickle.dump(human_green, human_file)

    fig_path = 'imgs/by_analysis.png'
    plot_avg_corr(penalty, human_green, fig_path)


def compare_ewd_mechanism(args):
    cache_green_prob_path = 'green_prob.pkl'
    cache_reward_path = 'detection_reward.pkl'
    ewd_fig_path = 'imgs/ewd_reward_penalty.pdf'
    bayes_fig_path = 'imgs/by_reward_penalty.pdf'
    ewd_penalty_reward_ratio = args.gamma / (1 - args.gamma)
    if os.path.exists(cache_green_prob_path) and os.path.exists(cache_reward_path):
        with open(cache_green_prob_path, 'rb') as prob_file:
            green_prob = pickle.load(prob_file)
        with open(cache_reward_path, 'rb') as reward_file:
            reward = pickle.load(reward_file)
    else:
        green_prob, reward = calculate_reward_and_prob(args)
        with open(cache_green_prob_path, 'wb') as prob_file:
            pickle.dump(green_prob, prob_file)
        with open(cache_reward_path, 'wb') as reward_file:
            pickle.dump(reward, reward_file)
    ewd_penalty = [(ewd_penalty_reward_ratio * r) for r in reward[0]]
    by_penalty = [(args.delta - r) for r in reward[1]]
    plot_reward_penalty_prob_corr(reward[0], ewd_penalty, green_prob, 'EWD', ewd_fig_path)
    plot_reward_penalty_prob_corr(reward[1], by_penalty, green_prob, 'BIWD', bayes_fig_path)


def analyze_score_dist(args):
    res_key = f'{args.method}_detection_results'
    human_res = read_file(f'{args.generations_dir}/human_results.json')[res_key]
    wm_res = read_file(f'{args.generations_dir}/evaluation_results.json')[res_key]
        
    score_key = 'z_score' if args.method == 'ewd' else 'score'
    human_score = [res[score_key] for res in human_res]
    wm_score = [res[score_key] for res in wm_res]

    # test_green_prob_dist(args)
    fig_path = f'imgs/{args.method}_score_dist.pdf'
    # print(human_avg_logp, wm_avg_logp)
    # print(len(human_score), len(human_avg_logp), len(wm_score), len(wm_avg_logp))
    plot_score_dist(human_score, wm_score, args.method, fig_path, args.plt_lb, args.plt_ub)


def analyze_penalty_validity(args):
    cache_penalty_path = 'penalty.pkl'
    cache_human_path = 'human_green.pkl'
    if os.path.exists(cache_penalty_path) and os.path.exists(cache_human_path):
        penalty = read_pickle(cache_penalty_path)
        human_green = read_pickle(cache_human_path)
    else:
        penalty, human_green = calculate_penalty_and_prob(args)
        save_pickle(cache_penalty_path, penalty)        
        save_pickle(cache_human_path, human_green)

    penalty = [ds_penalty[0] for ds_penalty in penalty]
    human_green = [ds_green[0] for ds_green in human_green]
    def reverse_green_prob(pnt):
        pass


    plot_prob_gt_penalty(penalty, human_green, 'imgs/penalty_validity.png')


# analyze the correlation between watermark information indicator and entropy
def analyze_wii_entropy(args):
    if os.path.exists(args.wii_cache_path) and os.path.exists(args.ent_cache_path):
        wii = read_pickle(args.wii_cache_path)
        ent = read_pickle(args.ent_cache_path)
    else:
        wii, ent = calculate_wii_and_ent(args)
        save_pickle(args.wii_cache_path, wii)
        save_pickle(args.ent_cache_path, ent)
    
    img_path = 'imgs/wis_ent_scatter.pdf'
    plot_wii_ent_scatter(wii, ent, fig_path=img_path)


@torch.no_grad()
def main():
    args = parse_args()

    if args.target == 'approx':
        analyze_by_approximation(args)
    elif args.target == 'comp':
        compare_ewd_mechanism(args)
    elif args.target == 'score':
        analyze_score_dist(args)
    elif args.target == 'wii':
        analyze_wii_entropy(args)


def test_ds():
    # try: 
    #     task:Task = tasks.get_task('humaneval')
    # except UserWarning:
    #     pass
    # print(type(task), isinstance(task, HumanEval))
    ds = load_dataset(path='humaneval-lk')
    print(ds)

    for sp in ds:
        print(sp)


def test_calculation_res():
    cache_penalty_path = 'penalty.npy'
    cache_gap_path = 'gap.npy'
    penalty:np.ndarray = np.load(cache_penalty_path, allow_pickle=True)
    gap:np.ndarray = np.load(cache_gap_path, allow_pickle=True)
    print(penalty.shape, gap.shape)
    print(penalty[:5])
    print(gap[:5])


def test_green_prob_dist(args):
    human_avg_logp, wm_avg_logp = read_pickle(args.logp_cache_path)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(human_avg_logp, color='blue', shade=True, label='human avg logp')
    sns.kdeplot(wm_avg_logp, color='red', shade=True, label='wm avg logp')
    plt.title('human / watermarked avg_logp kernel density')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('imgs/green_logp_dist.png')



if __name__ == '__main__':
    main()