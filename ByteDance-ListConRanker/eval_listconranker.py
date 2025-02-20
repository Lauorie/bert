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

import argparse
from modules.Reranking import *
from mteb import MTEB
from modules.listconranker import ListConRanker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="./", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = ListConRanker(args.model_name_or_path, use_fp16=True, list_transformer_layer=2)
    dir_name = args.model_name_or_path.split('/')[-2]
    if 'checkpoint-' in args.model_name_or_path:
        save_name = "_".join(args.model_name_or_path.split('/')[-2:])
        dir_name = args.model_name_or_path.split('/')[-3]
    else:
        save_name = "_".join(args.model_name_or_path.split('/')[-1:])
        dir_name = args.model_name_or_path.split('/')[-2]

    evaluation = MTEB(task_types=["Reranking"], task_langs=['zh'])
    evaluation.run(model, output_folder="reranker_results/{}/{}".format(dir_name, save_name))
