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

import logging
import numpy as np
from mteb import RerankingEvaluator, AbsTaskReranking
from tqdm import tqdm
import math

logger = logging.getLogger(__name__)


class ChineseRerankingEvaluator(RerankingEvaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.
    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP is compute to measure the quality of the ranking.
    :param samples: Must be a list and each element is of the form:
        - {'query': '', 'positive': [], 'negative': []}. Query is the search query, positive is a list of positive
        (relevant) documents, negative is a list of negative (irrelevant) documents.
        - {'query': [], 'positive': [], 'negative': []}. Where query is a list of strings, which embeddings we average
        to get the query embedding.
    """

    def __call__(self, model):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model):
        return (
            self.compute_metrics_batched(model)
            if self.use_batched_encoding
            else self.compute_metrics_individual(model)
        )

    def compute_metrics_batched(self, model):
        """
        Computes the metrices in a batched way, by batching all queries and
        all documents together
        """

        if hasattr(model, 'compute_score'):
            return self.compute_metrics_batched_from_crossencoder(model)
        else:
            return self.compute_metrics_batched_from_biencoder(model)

    def compute_metrics_batched_from_crossencoder(self, model):
        all_ap_scores = []
        all_mrr_1_scores = []
        all_mrr_5_scores = []
        all_mrr_10_scores = []

        for sample in tqdm(self.samples, desc="Evaluating"):
            query = sample['query']
            pos = sample['positive']
            neg = sample['negative']
            passage = pos + neg
            passage2label = {}
            for p in pos:
                passage2label[p] = True
            for p in neg:
                passage2label[p] = False

            filter_times = 0
            passage2score = {}
            while len(passage) > 20:
                batch = [[query] + passage]
                pred_scores = model.compute_score(batch)[0]
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
            pred_scores = model.compute_score(batch)[0]
            cnt = 0
            while cnt < len(passage):
                if passage[cnt] in passage2score:
                    passage2score[passage[cnt]].append(pred_scores[cnt] + filter_times)
                else:
                    passage2score[passage[cnt]] = [pred_scores[cnt] + filter_times]
                cnt += 1

            passage = list(set(pos + neg))
            is_relevant = []
            final_score = []
            for i in range(len(passage)):
                p = passage[i]
                is_relevant += [passage2label[p]] * len(passage2score[p])
                final_score += passage2score[p]

            ap = self.ap_score(is_relevant, final_score)

            pred_scores_argsort = np.argsort(-(np.array(final_score)))
            mrr_1 = self.mrr_at_k_score(is_relevant, pred_scores_argsort, 1)
            mrr_5 = self.mrr_at_k_score(is_relevant, pred_scores_argsort, 5)
            mrr_10 = self.mrr_at_k_score(is_relevant, pred_scores_argsort, 10)

            all_ap_scores.append(ap)
            all_mrr_1_scores.append(mrr_1)
            all_mrr_5_scores.append(mrr_5)
            all_mrr_10_scores.append(mrr_10)

        mean_ap = np.mean(all_ap_scores)
        mean_mrr_1 = np.mean(all_mrr_1_scores)
        mean_mrr_5 = np.mean(all_mrr_5_scores)
        mean_mrr_10 = np.mean(all_mrr_10_scores)

        return {"map": mean_ap, "mrr_1": mean_mrr_1, 'mrr_5': mean_mrr_5, 'mrr_10': mean_mrr_10}

    def compute_metrics_batched_from_biencoder(self, model):
        all_mrr_scores = []
        all_ap_scores = []
        logger.info("Encoding queries...")
        if isinstance(self.samples[0]["query"], str):
            if hasattr(model, 'encode_queries'):
                all_query_embs = model.encode_queries(
                    [sample["query"] for sample in self.samples],
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                )
            else:
                all_query_embs = model.encode(
                    [sample["query"] for sample in self.samples],
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                )
        elif isinstance(self.samples[0]["query"], list):
            # In case the query is a list of strings, we get the most similar embedding to any of the queries
            all_query_flattened = [q for sample in self.samples for q in sample["query"]]
            if hasattr(model, 'encode_queries'):
                all_query_embs = model.encode_queries(all_query_flattened, convert_to_tensor=True,
                                                      batch_size=self.batch_size)
            else:
                all_query_embs = model.encode(all_query_flattened, convert_to_tensor=True, batch_size=self.batch_size)
        else:
            raise ValueError(f"Query must be a string or a list of strings but is {type(self.samples[0]['query'])}")

        logger.info("Encoding candidates...")
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample["positive"])
            all_docs.extend(sample["negative"])

        all_docs_embs = model.encode(all_docs, convert_to_tensor=True, batch_size=self.batch_size)

        # Compute scores
        logger.info("Evaluating...")
        query_idx, docs_idx = 0, 0
        for instance in self.samples:
            num_subqueries = len(instance["query"]) if isinstance(instance["query"], list) else 1
            query_emb = all_query_embs[query_idx: query_idx + num_subqueries]
            query_idx += num_subqueries

            num_pos = len(instance["positive"])
            num_neg = len(instance["negative"])
            docs_emb = all_docs_embs[docs_idx: docs_idx + num_pos + num_neg]
            docs_idx += num_pos + num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            is_relevant = [True] * num_pos + [False] * num_neg

            scores = self._compute_metrics_instance(query_emb, docs_emb, is_relevant)
            all_mrr_scores.append(scores["mrr"])
            all_ap_scores.append(scores["ap"])

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {"map": mean_ap, "mrr": mean_mrr}


def evaluate(self, model, split="test", **kwargs):
    if not self.data_loaded:
        self.load_data()

    data_split = self.dataset[split]

    evaluator = ChineseRerankingEvaluator(data_split, **kwargs)
    scores = evaluator(model)

    return dict(scores)


AbsTaskReranking.evaluate = evaluate


class T2Reranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2Reranking',
            'hf_hub_name': "C-MTEB/T2Reranking",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class T2RerankingZh2En(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2RerankingZh2En',
            'hf_hub_name': "C-MTEB/T2Reranking_zh2en",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh2en'],
            'main_score': 'map',
        }


class T2RerankingEn2Zh(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2RerankingEn2Zh',
            'hf_hub_name': "C-MTEB/T2Reranking_en2zh",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en2zh'],
            'main_score': 'map',
        }


class MMarcoReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'MMarcoReranking',
            'hf_hub_name': "C-MTEB/Mmarco-reranking",
            'description': 'mMARCO is a multilingual version of the MS MARCO passage ranking dataset',
            "reference": "https://github.com/unicamp-dl/mMARCO",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class CMedQAv1(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'CMedQAv1',
            "hf_hub_name": "C-MTEB/CMedQAv1-reranking",
            'description': 'Chinese community medical question answering',
            "reference": "https://github.com/zhangsheng93/cMedQA",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class CMedQAv2(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'CMedQAv2',
            "hf_hub_name": "C-MTEB/CMedQAv2-reranking",
            'description': 'Chinese community medical question answering',
            "reference": "https://github.com/zhangsheng93/cMedQA2",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }
