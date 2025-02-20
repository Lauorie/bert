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
import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class ListTransformer(nn.Module):
    def __init__(self, num_layer, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.list_transformer_layer = nn.TransformerEncoderLayer(1792, self.config.num_attention_heads, batch_first=True, activation=F.gelu, norm_first=False)
        self.list_transformer = nn.TransformerEncoder(self.list_transformer_layer, num_layer)
        self.relu = nn.ReLU()
        self.query_embedding = QueryEmbedding(config, device)

        self.linear_score3 = nn.Linear(1792 * 2, 1792)
        self.linear_score2 = nn.Linear(1792 * 2, 1792)
        self.linear_score1 = nn.Linear(1792 * 2, 1)

    def forward(self, pair_features, pair_nums):
        pair_nums = [x + 1 for x in pair_nums]
        batch_pair_features = pair_features.split(pair_nums)

        pair_feature_query_passage_concat_list = []
        for i in range(len(batch_pair_features)):
            pair_feature_query = batch_pair_features[i][0].unsqueeze(0).repeat(pair_nums[i] - 1, 1)
            pair_feature_passage = batch_pair_features[i][1:]
            pair_feature_query_passage_concat_list.append(torch.cat([pair_feature_query, pair_feature_passage], dim=1))
        pair_feature_query_passage_concat = torch.cat(pair_feature_query_passage_concat_list, dim=0)

        batch_pair_features = nn.utils.rnn.pad_sequence(batch_pair_features, batch_first=True)

        query_embedding_tags = torch.zeros(batch_pair_features.size(0), batch_pair_features.size(1), dtype=torch.long, device=self.device)
        query_embedding_tags[:, 0] = 1
        batch_pair_features = self.query_embedding(batch_pair_features, query_embedding_tags)

        mask = self.generate_attention_mask(pair_nums)
        query_mask = self.generate_attention_mask_custom(pair_nums)
        pair_list_features = self.list_transformer(batch_pair_features, src_key_padding_mask=mask, mask=query_mask)

        output_pair_list_features = []
        output_query_list_features = []
        pair_features_after_transformer_list = []
        for idx, pair_num in enumerate(pair_nums):
            output_pair_list_features.append(pair_list_features[idx, 1:pair_num, :])
            output_query_list_features.append(pair_list_features[idx, 0, :])
            pair_features_after_transformer_list.append(pair_list_features[idx, :pair_num, :])

        pair_features_after_transformer_cat_query_list = []
        for idx, pair_num in enumerate(pair_nums):
            query_ft = output_query_list_features[idx].unsqueeze(0).repeat(pair_num - 1, 1)
            pair_features_after_transformer_cat_query = torch.cat([query_ft, output_pair_list_features[idx]], dim=1)
            pair_features_after_transformer_cat_query_list.append(pair_features_after_transformer_cat_query)
        pair_features_after_transformer_cat_query = torch.cat(pair_features_after_transformer_cat_query_list, dim=0)

        pair_feature_query_passage_concat = self.relu(self.linear_score2(pair_feature_query_passage_concat))
        pair_features_after_transformer_cat_query = self.relu(self.linear_score3(pair_features_after_transformer_cat_query))
        final_ft = torch.cat([pair_feature_query_passage_concat, pair_features_after_transformer_cat_query], dim=1)
        logits = self.linear_score1(final_ft).squeeze()

        return logits, torch.cat(pair_features_after_transformer_list, dim=0)

    def generate_attention_mask(self, pair_num):
        max_len = max(pair_num)
        batch_size = len(pair_num)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        for i, length in enumerate(pair_num):
            mask[i, length:] = True
        return mask

    def generate_attention_mask_custom(self, pair_num):
        max_len = max(pair_num)

        mask = torch.zeros(max_len, max_len, dtype=torch.bool, device=self.device)
        mask[0, 1:] = True

        return mask


class QueryEmbedding(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.query_embedding = nn.Embedding(2, 1792)
        self.layerNorm = nn.LayerNorm(1792)

    def forward(self, x, tags):
        query_embeddings = self.query_embedding(tags)
        x += query_embeddings
        x = self.layerNorm(x)
        return x


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, list_transformer_layer_4eval: int=None):
        super().__init__()
        self.hf_model = hf_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()

        self.config = self.hf_model.config
        self.config.output_hidden_states = True

        self.linear_in_embedding = nn.Linear(1024, 1792)
        self.list_transformer_layer = list_transformer_layer_4eval
        self.list_transformer = ListTransformer(self.list_transformer_layer, self.config, self.device)

    def forward(self, batch):
        if 'pair_num' in batch:
            pair_nums = batch.pop('pair_num').tolist()

        if self.training:
            pass
        else:
            split_batch = 400
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            if sum(pair_nums) > split_batch:
                last_hidden_state_list = []
                input_ids_list = input_ids.split(split_batch)
                attention_mask_list = attention_mask.split(split_batch)
                for i in range(len(input_ids_list)):
                    last_hidden_state = self.hf_model(input_ids=input_ids_list[i], attention_mask=attention_mask_list[i], return_dict=True).hidden_states[-1]
                    last_hidden_state_list.append(last_hidden_state)
                last_hidden_state = torch.cat(last_hidden_state_list, dim=0)
            else:
                ranker_out = self.hf_model(**batch, return_dict=True)
                last_hidden_state = ranker_out.last_hidden_state

            pair_features = self.average_pooling(last_hidden_state, attention_mask)
            pair_features = self.linear_in_embedding(pair_features)

            logits, pair_features_after_list_transformer = self.list_transformer(pair_features, pair_nums)
            logits = self.sigmoid(logits)

            return logits

    @classmethod
    def from_pretrained_for_eval(cls, model_name_or_path, list_transformer_layer):
        hf_model = AutoModel.from_pretrained(model_name_or_path)
        reranker = cls(hf_model, list_transformer_layer)
        reranker.linear_in_embedding.load_state_dict(torch.load(model_name_or_path + '/linear_in_embedding.pt',weights_only=True))
        reranker.list_transformer.load_state_dict(torch.load(model_name_or_path + '/list_transformer.pt',weights_only=True))
        return reranker

    def average_pooling(self, hidden_state, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).to(dtype=hidden_state.dtype)
        masked_hidden_state = hidden_state * extended_attention_mask
        sum_embeddings = torch.sum(masked_hidden_state, dim=1)
        sum_mask = extended_attention_mask.sum(dim=1)
        return sum_embeddings / sum_mask
