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
Preprocess the UI100 dataset to parquet format
"""

import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import pandas as pd

PROMPT = """<image>You are specialize in GUI grounding. This is an interface to a GUI and you are going to perform click action to {instruction}.
Output exactly one line containing a single JSON object in the following format:
{{"point_2d": [x, y], "label": "object name/description"}}"""

def get_prompt(instruction):
    instruction = PROMPT.format(instruction=instruction)
    messages = [
        {
            "role": "user",
            "content": instruction
        },
    ]
    return messages


def image_to_bytes(image_path):
    """读取图像并转换为字节流"""
    with open(image_path, 'rb') as file:
        return file.read()
    

def make_map_fn(data_source, split):
    
    def process_fn(example, idx):
        instruction = example.get('instruction', '')
        bbox = example.get('bbox', '')
        image_path = example.get('img_filename', '')

        # 读取图像为字节流
        image_bytes = image_to_bytes(image_path)
        prompt = get_prompt(instruction)
        
        data = {
            "data_source": data_source,
            "prompt": prompt,
            "images": [{
                "bytes": image_bytes
            }],
            "ability": "gui grounding",
            "reward_model": {
                "style": "rule",
                "ground_truth": bbox
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': bbox
            }
        }
        return data

    return process_fn


def load_json_to_parquet(json_path, local_dir, split):
    # 读取本地JSON文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 将每个JSON项处理成DataFrame格式
    processed_data = [make_map_fn('ui100', split)(item, idx) for idx, item in enumerate(data)]
    df = pd.DataFrame(processed_data)
    
    # 保存为Parquet文件
    parquet_path = os.path.join(local_dir, f'{split}.parquet')
    df.to_parquet(parquet_path)
    print(f"数据已成功转换为Parquet格式：{parquet_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/inspire/hdd/global_user/liupengfei-24025/yhhe/code/V-RFT/verl/data/ui100')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    json_path = "/inspire/hdd/global_user/liupengfei-24025/yhhe/code/V-RFT/data/UI128/train.json"

    # 创建保存目录
    os.makedirs(args.local_dir, exist_ok=True)

    # 转换JSON为Parquet
    load_json_to_parquet(json_path, args.local_dir, "train")
    load_json_to_parquet(json_path, args.local_dir, "test")
