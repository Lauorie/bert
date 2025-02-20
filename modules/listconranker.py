# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
# and associated documentation files (the “Software”), to deal in the Software without 
# restriction, including without limitation the rights to use, copy, modify, merge, publish, 
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import math
import torch
import numpy as np
from transformers import AutoTokenizer, is_torch_npu_available
from typing import Union, List
from .modeling import CrossEncoder

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ListConRanker:
    def __init__(
            self,
            model_name_or_path: str = None,
            use_fp16: bool = False,
            cache_dir: str = None,
            device: Union[str, int] = None,
            list_transformer_layer = None
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = CrossEncoder.from_pretrained_for_eval(model_name_or_path, list_transformer_layer)

        if device and isinstance(device, str):
            self.device = torch.device(device)
            if device == 'cpu':
                use_fp16 = False
        else:
            if torch.cuda.is_available():
                if device is not None:
                    self.device = torch.device(f"cuda:{device}")
                else:
                    self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        if use_fp16:
            self.model.half()

        self.model = self.model.to(self.device)

        self.model.eval()

        if device is None:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                print(f"----------using {self.num_gpus}*GPUs----------")
                self.model = torch.nn.DataParallel(self.model)
        else:
            self.num_gpus = 1

    @torch.no_grad()
    def compute_score(self, sentence_pairs: List[List[str]], max_length: int = 512) -> List[List[float]]:
        pair_nums = [len(pairs) - 1 for pairs in sentence_pairs]
        sentences_batch = sum(sentence_pairs, [])
        inputs = self.tokenizer(
            sentences_batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length,
        ).to(self.device)
        inputs['pair_num'] = torch.LongTensor(pair_nums)
        scores = self.model(inputs).float()
        all_scores = scores.cpu().numpy().tolist()

        if isinstance(all_scores, float):
            return [all_scores]
        result = []
        curr_idx = 0
        for i in range(len(pair_nums)):
            result.append(all_scores[curr_idx: curr_idx + pair_nums[i]])
            curr_idx += pair_nums[i]
        # return all_scores
        return result

    @torch.no_grad()
    def iterative_inference(self, sentence_pairs: List[str], max_length: int = 512) -> List[float]:
        query = sentence_pairs[0]
        passage = sentence_pairs[1:]

        filter_times = 0
        passage2score = {}
        while len(passage) > 20:
            batch = [[query] + passage]
            pred_scores = self.compute_score(batch, max_length)[0]
             # Sort in increasing order
            pred_scores_argsort = np.argsort(pred_scores).tolist()
            passage_len = len(passage)
            to_filter_num = math.ceil(passage_len * 0.2)
            if to_filter_num < 10:
                to_filter_num = 10

            have_filter_num = 0
            while have_filter_num < to_filter_num:
                idx = pred_scores_argsort[have_filter_num]
                if passage[idx] in passage2score:
                    passage2score[passage[idx]].append(pred_scores[idx] + filter_times)
                else:
                    passage2score[passage[idx]] = [pred_scores[idx] + filter_times]
                have_filter_num += 1
            while pred_scores[pred_scores_argsort[have_filter_num - 1]] == pred_scores[pred_scores_argsort[have_filter_num]]:
                idx = pred_scores_argsort[have_filter_num]
                if passage[idx] in passage2score:
                    passage2score[passage[idx]].append(pred_scores[idx] + filter_times)
                else:
                    passage2score[passage[idx]] = [pred_scores[idx] + filter_times]
                have_filter_num += 1
            next_passage = []
            next_passage_idx = have_filter_num
            while next_passage_idx < len(passage):
                idx = pred_scores_argsort[next_passage_idx]
                next_passage.append(passage[idx])
                next_passage_idx += 1
            passage = next_passage
            filter_times += 1

        batch = [[query] + passage]
        pred_scores = self.compute_score(batch, max_length)[0]
        cnt = 0
        while cnt < len(passage):
            if passage[cnt] in passage2score:
                passage2score[passage[cnt]].append(pred_scores[cnt] + filter_times)
            else:
                passage2score[passage[cnt]] = [pred_scores[cnt] + filter_times]
            cnt += 1

        passage = sentence_pairs[1:]
        final_score = []
        for i in range(len(passage)):
            p = passage[i]
            final_score += passage2score[p]
        return final_score
