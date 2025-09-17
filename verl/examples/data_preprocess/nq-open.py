# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the NQ-open.train.jsonl dataset to parquet format
"""

import argparse
import os

import datasets

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/webquestions")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/home/zwt/Dev/webquestions/trainmodel.json"
    # data_source = "data/nq_open/NQ-open.train.jsonl"
    dataset = datasets.load_dataset("json", data_files=data_source)
    train_dataset = dataset[list(dataset.keys())[0]].select(range(2834))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("qText")
            answer = example.pop("answers")
            answer = ", ".join(answer)

            prompt = "Answer the following question with one or few words. If you do not know the answer, answer 'Unknown'. Question: " + question
            
            data = {
                "data_source": "NQ-open",
                "prompt": [{
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train_first10k.parquet"))

