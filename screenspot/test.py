import torch
import json
import os
import logging
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from grounding import Grounding

# 日志级别可以调整
logging.basicConfig(level=logging.ERROR)
torch.manual_seed(1234)

client = OpenAI(
    api_key="123",
    base_url="http://localhost:9000/v1",
)
model = None
model_name = model.split("/")[-1].lower() if model else "qwen2.5-vl-7b-instruct"
grounding = Grounding(client, model)

prefix = "/inspire/hdd/global_user/liupengfei-24025/yhhe/code/V-RFT/screenspot/"

# 路径定义
screenspot_imgs = os.path.join(prefix, 'SeeClick/data/ScreenSpot/screenspot_imgs')
screenspot_test = os.path.join(prefix, 'SeeClick/data/ScreenSpot')

result_file = os.path.join(prefix, f'SeeClick/data/ScreenSpot/result_{model_name}.txt')
response_file = os.path.join(prefix, f'SeeClick/data/ScreenSpot/response_{model_name}.jsonl')
checkpoint_file = os.path.join(prefix, f'SeeClick/data/ScreenSpot/checkpoint_{model_name}.json')

tasks = ["mobile", "desktop", "web"]

# 加载 checkpoint，如果文件存在则读取各数据集已经处理的索引
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as cf:
        checkpoint = json.load(cf)
else:
    checkpoint = {}

if __name__ == "__main__":
    tasks_result = []
    for task in tasks:
        dataset = "screenspot_" + task + ".json"
        dataset_path = os.path.join(screenspot_test, dataset)
        with open(dataset_path, 'r') as f:
            screenspot_data = json.load(f)
        total_samples = len(screenspot_data)
        print(f"Task {task}: Num of sample: {total_samples}")

        # 获取断点位置，若无记录则从 0 开始
        start_index = checkpoint.get(dataset, 0)

        # tqdm显示时从start_index开始计数
        for j, item in tqdm(enumerate(screenspot_data), initial=start_index, total=total_samples):
            if j < start_index:
                continue  # 跳过已处理的记录
            filename = item["img_filename"]
            img_path = os.path.join(screenspot_imgs, filename)
            if not os.path.exists(img_path):
                logging.error(f"Image not found: {img_path}")
                continue

            # 调整bbox格式：[x_min, y_min, x_max, y_max]
            bbox = item["bbox"]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            instruction = item["instruction"]

            # 获取预测结果
            x, y = grounding.call(instruction, img_path)

            # 判断是否在 bbox 内
            if x is not None and y is not None and (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3]):
                success = True
            else:
                success = False

            # 记录每条响应（包括成功与否），同时增加 task 字段便于后续统计
            response_data = {
                "task": task,
                "img_path": img_path,
                "instruction": instruction,
                "bbox": bbox,
                "pred": [x, y],
                "data_type": item["data_type"],
                "data_source": item.get("data_source", ""),
                "success": success
            }
            with open(response_file, 'a') as rf:
                json.dump(response_data, rf)
                rf.write('\n')

            # 每处理完一个样本就更新断点文件
            checkpoint[dataset] = j + 1
            with open(checkpoint_file, 'w') as cf:
                json.dump(checkpoint, cf)
            logging.info(f"Completed: {j + 1} / {total_samples}")

    # 所有任务处理完毕后，从 response file 中统计结果
    task_results = {}
    total_cnt = 0
    total_correct = 0
    with open(response_file, 'r') as rf:
        for line in rf:
            data = json.loads(line)
            t = data.get("task", "unknown")
            if t not in task_results:
                task_results[t] = {
                    "total": 0,
                    "correct": 0,
                    "wrong_format": 0,
                    "text_total": 0,
                    "text_correct": 0,
                    "icon_total": 0,
                    "icon_correct": 0,
                }
            task_results[t]["total"] += 1
            total_cnt += 1
            if data.get("success", False):
                task_results[t]["correct"] += 1
                total_correct += 1
            else:
                # 若预测结果均为 None，则认为格式错误
                x, y = data.get("pred", [None, None])
                if x is None and y is None:
                    task_results[t]["wrong_format"] += 1
            # 针对不同数据类型的统计
            if data.get("data_type") == "text":
                task_results[t]["text_total"] += 1
                if data.get("success", False):
                    task_results[t]["text_correct"] += 1
            elif data.get("data_type") == "icon":
                task_results[t]["icon_total"] += 1
                if data.get("success", False):
                    task_results[t]["icon_correct"] += 1

    # 将统计结果写入 result file
    with open(result_file, 'w') as rf:
        for t, stats in task_results.items():
            rf.write(f"Task: {t}\n")
            rf.write("Action Acc: " + str(stats["correct"] / stats["total"] if stats["total"] > 0 else 0) + "\n")
            rf.write("Total num: " + str(stats["total"]) + "\n")
            rf.write("Wrong format num: " + str(stats["wrong_format"]) + "\n")
            rf.write("Text Acc: " + str(stats["text_correct"] / stats["text_total"] if stats["text_total"] > 0 else 0) + "\n")
            rf.write("Icon Acc: " + str(stats["icon_correct"] / stats["icon_total"] if stats["icon_total"] > 0 else 0) + "\n\n")
        
        rf.write("Total Action Acc: " + str(total_correct / total_cnt if total_cnt > 0 else 0) + "\n")
    
    logging.info("All tasks have been processed. Results are written to the result file.")
