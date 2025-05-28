# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt
import pdb
import scipy.stats
import math

import torch
from torch import Tensor
from transformers import LogitsProcessor

# from nltk.util import ngrams
# from normalizers import normalization_strategy_lookup

class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485917,  # just a large prime number to create a rng seed with sufficient bit width
        select_green_tokens: bool = True,
        entropy_threshold: float = 0.0,
    ):
        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.entropy_threshold = entropy_threshold
        alpha = torch.exp(torch.tensor(self.delta)).item()
        self.z_value = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))
        self.cached_probs = None

    def _seed_rng(self, input_ids: torch.LongTensor, hash_key: int, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(hash_key * prev_token) ### newly change self.hash_key to hash_key ###
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        self._seed_rng(input_ids, self.hash_key)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        if self.select_green_tokens: # directly
            greenlist_ids = vocab_permutation[:greenlist_size] # new
        else: # select green via red
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]  # legacy behavior
        return greenlist_ids

    @torch.no_grad()
    def calculate_spike_entropy(self, model, tokenized_text) -> list[float]:
        """Calculate the entropy of the tokenized text using the model."""
        if self.cached_probs is not None:
            probs = self.cached_probs
        else:
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            self.cached_probs = probs
        denoms = 1+(self.z_value * probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum(dim=-1)
        entropy=sum_renormed_probs[0].cpu().tolist()
        entropy.insert(0, -10000.0)
        return entropy[:-1]

    @torch.no_grad()
    def calculate_info_entropy(self, model, tokenized_text) -> list[float]:
        if self.cached_probs is not None:
            probs = self.cached_probs
        else:
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            self.cached_probs = probs
        information_entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        information_entropy = information_entropy[0].cpu().tolist()
        information_entropy.insert(0, -10000.0)
        return information_entropy[:-1]

    @torch.no_grad()
    def calculate_info_entropy_tensor(self, model, tokenized_text) -> list[float]:
        if self.cached_probs is not None:
            probs = self.cached_probs
        else:
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            self.cached_probs = probs
        information_entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        return information_entropy[0]

    def clear_probs_cache(self):
        self.cached_probs=None


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator()

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores


class SweetLogitsProcessor(WatermarkLogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator()

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # get entropy
        raw_probs = torch.softmax(scores, dim=-1)  # batch_size, vocab_size
        ent = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
        entropy_mask = (ent > self.entropy_threshold).view(-1, 1)
        
        green_tokens_mask = green_tokens_mask * entropy_mask

        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        tokenizer: None,
        ignore_repeated_bigrams: bool = False,
        type: str = 'wllm', # wllm, sweet, ewd, sweet_ewd
        model:None,
        acc:None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        self.tokenizer = tokenizer
        self.rng = torch.Generator()
        self.type=type
        if model:
            self.model = model.to(acc.device)
            self.acc = acc

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams: 
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_z_score(self, observed_count, w):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * torch.sum(w, dim=0)
        denom = sqrt(torch.sum(torch.square(w), dim=0) * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _score_sequence(
        self,
        input_ids: Tensor,
        prefix_len: int,
        info_entropy = None, # prefix is removed
        spike_entropy = None
    ):
        score_dict = dict()
        if self.ignore_repeated_bigrams:
            raise NotImplementedError("not used")
        else:
            prefix_len = max(self.min_prefix_len, prefix_len)

            num_tokens_scored = len(input_ids) - prefix_len
            if self.type in ['sweet', 'sweet_ewd']:
                assert len(info_entropy) == num_tokens_scored, 'information entropy length should be the same as num_tokens_scored'          
            if self.type in ['ewd', 'sweet_ewd']:
                assert len(spike_entropy) == num_tokens_scored, 'spike entropy length should be the same as num_tokens_scored'
            if num_tokens_scored < 1:
                print(f"only {num_tokens_scored} scored : cannot score.")
                score_dict["invalid"] = True
                return score_dict
            score_dict['num_tokens_detect'] = num_tokens_scored

            green_token_count, flag_all= 0, []
            for idx in range(prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                flag_all.append(curr_token in greenlist_ids)
        if self.type=='wllm':
            score_dict.update(dict(z_score=self._compute_z_score(len([i for i in flag_all if i]), torch.tensor([1.0 for _ in range(num_tokens_scored)])).item()))
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
            score_dict.update(dict(num_green_tokens=len([i for i in flag_all if i])))
        elif self.type=='sweet':
            num_token_sweet=len([i for i in info_entropy if i>self.entropy_threshold])
            green = len([i for i,j in zip(info_entropy,flag_all) if i>self.entropy_threshold and j])
            score_dict.update(dict(z_score=self._compute_z_score(green, torch.tensor([1.0 for _ in range(num_token_sweet)])).item()))
            score_dict.update(dict(num_tokens_scored=num_token_sweet))
            score_dict.update(dict(num_green_tokens=green))
        elif self.type == 'ewd':
            SE=torch.sub(torch.tensor(spike_entropy), torch.min(torch.tensor(spike_entropy)))
            tensor_list = [tensor for tensor, flag in zip(SE, flag_all) if flag]
            if not tensor_list:
                green_token_count = torch.tensor([0], device=SE.device)
            else:
                green_token = torch.stack(tensor_list)
                green_token_count = torch.sum(green_token, dim=0)
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, SE).item()))
            score_dict.update(dict(num_green_tokens=green_token_count.item()))
            score_dict.update(dict(spike_entropy=spike_entropy))
        elif self.type == 'sweet_ewd':
            high_ent_ids = [idx for idx, ent in enumerate(info_entropy) if ent>self.entropy_threshold]
            s_ents = []
            flag_final = []
            for tid in high_ent_ids:
                s_ents.append(spike_entropy[tid])
                flag_final.append(flag_all[tid]) 
            
            SE=torch.sub(torch.tensor(s_ents), torch.min(torch.tensor(s_ents)))
            green_token=torch.stack([tensor for tensor, flag in zip(SE, flag_final) if flag])
            green_token_count = torch.sum(green_token, dim=0)
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, SE).item()))
            score_dict.update(dict(num_green_tokens=green_token_count.item()))
            score_dict.update(dict(spike_entropy=spike_entropy))
        # score_dict.update(dict(flag=flag_all))
        return score_dict


    def detect(
        self,
        tokenized_text: torch.Tensor = None,
        tokenized_prefix: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        assert tokenized_text is not None, "Must pass either tokenized string"
        info_entropy, spike_entropy = None, None
        if self.type in ['sweet','sweet_ewd']:
            #calculate information entropy
            info_entropy = self.calculate_info_entropy(self.model, tokenized_text.to(self.acc.device))
            info_entropy = info_entropy[len(tokenized_prefix):]
        if self.type in ['ewd', 'sweet_ewd']:
            spike_entropy = self.calculate_spike_entropy(self.model, tokenized_text.to(self.acc.device))
            spike_entropy = spike_entropy[len(tokenized_prefix):]

        score_dict = self._score_sequence(input_ids=tokenized_text, prefix_len=len(tokenized_prefix),
                                          info_entropy=info_entropy, spike_entropy=spike_entropy)
        self.clear_probs_cache()
        return score_dict


class BayesianWMDetector(WatermarkBase):
    
    def __init__(
        self,
        *args,
        tokenizer: None,
        ignore_repeated_bigrams: bool = False,
        type: str = 'kgw_by', # kgw_by, sweet_by
        model:None,
        acc:None,
        topk: int = 2000,
        detection_temp: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        self.tokenizer = tokenizer
        self.rng = torch.Generator()
        self.type=type
        self.topk = topk
        self.exp_delta = math.exp(self.delta)
        self.detection_temp = detection_temp
        if model:
            self.model = model.to(acc.device)
            self.acc = acc

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []        
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams: 
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    @torch.no_grad()
    def detect(
        self,
        tokenized_text: torch.Tensor = None,
        tokenized_prefix: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        if self.ignore_repeated_bigrams:
            raise NotImplementedError("not used")

        score_dict = dict()

        tokenized_text = tokenized_text.to(self.acc.device)
        tokenized_prefix = tokenized_prefix.to(self.acc.device)

        prefix_len = max(self.min_prefix_len, len(tokenized_prefix))
        gen_len = len(tokenized_text)

        model_logits = self.model(torch.unsqueeze(tokenized_text, 0), return_dict=True).logits
        self.cached_probs = torch.softmax(model_logits, dim=-1)

        if self.type == 'sweet_by':
            info_entropy = self.calculate_info_entropy_tensor(self.model, tokenized_text)[prefix_len-1:-1]

        model_logits = model_logits[0].to(torch.float32)
        probs = torch.softmax(model_logits / self.detection_temp, dim=-1)[(prefix_len - 1):-1, :]

        _, topk_ids = torch.topk(probs, self.topk, dim=-1, sorted=False)
        topk_masks = torch.zeros_like(probs)
        topk_masks.scatter_(1, topk_ids, torch.ones_like(probs))

        green_masks = torch.zeros_like(probs)
        for i in range(prefix_len, gen_len):
            green_ids = self._get_greenlist_ids(tokenized_text[:i]).to(probs.device)
            green_masks[i-prefix_len, green_ids] = 1

        top_probs = probs * topk_masks
        top_green_masses = (top_probs * green_masks).sum(dim=-1)
        top_red_masses = (top_probs * (1 - green_masks)).sum(dim=-1)
        green_indicators = torch.gather(green_masks, dim=1, index=tokenized_text[prefix_len:].unsqueeze(1)).squeeze()
        penalty = ((self.exp_delta * top_green_masses + top_red_masses) / (top_green_masses + top_red_masses)).log()
        by_scores = self.delta * green_indicators - penalty
        
        tok_valid = torch.ones_like(by_scores)
        if self.type == 'sweet_by':
            tok_valid = torch.where(info_entropy > self.entropy_threshold, 1, 0).to(dtype=torch.int)
        
        num_tokens_scored = int(tok_valid.sum().item())
        if num_tokens_scored == 0:
            score_dict['invalid'] = True
            return score_dict            
        num_green_tokens = (green_indicators * tok_valid).sum().item()

        bayes_score = ((by_scores * tok_valid).sum()).item()

        # bayes_score = 0
        # num_tokens_scored = 0
        # num_green_tokens = 0
        # for i in range(prefix_len, gen_len):
        #     greenlist_ids = set(self._get_greenlist_ids(tokenized_text[:i]).tolist())
        #     # print(min(greenlist_ids), max(greenlist_ids))            
        #     tok_valid = info_entropy[i] > self.entropy_threshold if self.type == 'sweet_by' else True
        #     if not tok_valid:
        #         continue

        #     cur_id = tokenized_text[i].item()
        #     num_green_tokens += (cur_id in greenlist_ids)
        #     num_tokens_scored += 1
            
        #     _, topk_ids = torch.topk(model_logits[i - 1], self.topk)
        #     topk_ids = topk_ids.tolist()
        #     # print(topk_ids)
        #     top_green_ids = torch.tensor([tid for tid in topk_ids if tid in greenlist_ids], dtype=int, device=self.acc.device)
        #     # print(len(top_green_ids))
        #     top_red_ids = torch.tensor([tid for tid in topk_ids if tid not in greenlist_ids], dtype=int, device=self.acc.device)
        #     top_green_mass, top_red_mass = probs[i - 1, top_green_ids].sum().item(), probs[i - 1, top_red_ids].sum().item()
            
        #     bayes_score -= self.penalty * math.log((self.exp_delta * top_green_mass + top_red_mass) / (top_green_mass + top_red_mass))
        #     if cur_id in greenlist_ids:
        #         bayes_score += self.delta

        score_dict.update({
            'score':bayes_score,
            'num_tokens_scored':num_tokens_scored,
            'num_green_tokens':int(num_green_tokens),
            'num_tokens_detect':gen_len-prefix_len
        })

        self.clear_probs_cache()
        return score_dict


class DipMarkDetector(WatermarkBase):
    def __init__(self, inject_type, tokenizer, model=None, acc=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inject_type = inject_type
        self.rng = torch.Generator()
        self.tokenizer = tokenizer
        if model:
            self.model = model.to(acc.device)
        self.acc = acc
        self.min_prefix_len = 1

    def _dip_score(self, green, total):
        return green / total - self.gamma

    @torch.no_grad()
    def detect(
        self,
        tokenized_text: torch.Tensor = None,
        tokenized_prefix: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        tokenized_prefix = tokenized_prefix.to(self.acc.device)
        tokenized_text = tokenized_text.to(self.acc.device)

        info_entropy = None
        if self.inject_type == 'sweet':
            info_entropy = self.calculate_info_entropy(self.model, tokenized_text)
            info_entropy = info_entropy[len(tokenized_prefix):]
        
        gen_len = len(tokenized_text)
        prefix_len = max(self.min_prefix_len, len(tokenized_prefix))
        num_tokens_detect = gen_len - prefix_len
        
        green_flags = []
        for idx in range(prefix_len, gen_len):
            curr_token = tokenized_text[idx]
            greenlist_ids = self._get_greenlist_ids(tokenized_text[:idx]).to(self.acc.device)
            green_flags.append(curr_token in greenlist_ids)
        
        if self.inject_type == 'kgw':
            num_tokens_scored = num_tokens_detect
            num_tokens_green = len([i for i in green_flags if i])
        elif self.inject_type == 'sweet':
            num_tokens_scored = len([i for i in info_entropy if i>self.entropy_threshold])
            num_tokens_green = len([i for i,j in zip(info_entropy, green_flags) if i>self.entropy_threshold and j])
        else:
            raise NotImplementedError('other watermark injecting method not implemented here')

        if num_tokens_scored == 0:
            score_dict = {'invalid':True}
        else:
            score_dict = {
                'score':self._dip_score(num_tokens_green, num_tokens_scored),
                'num_tokens_scored':num_tokens_scored,
                'num_green_tokens':num_tokens_green,
                'num_tokens_detect':num_tokens_detect
            }
        self.clear_probs_cache()
        return score_dict