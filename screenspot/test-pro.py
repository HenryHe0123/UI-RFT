import torch
import json
import os
import logging
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from grounding import Grounding

# 设置日志级别和随机种子
logging.basicConfig(level=logging.ERROR)
torch.manual_seed(1234)

# 初始化 OpenAI client 和 Grounding 模型
client = OpenAI(
    api_key="123",
    base_url="http://localhost:9006/v1",
)
model = None
model_name = model.split("/")[-1].lower() if model else "qwen2.5-vl-7b-instruct"
grounding = Grounding(client, model)

# 定义 ScreenSpot-Pro 数据路径
prefix = "/inspire/hdd/global_user/liupengfei-24025/yhhe/code/V-RFT/"
base_dir = os.path.join(prefix, "ScreenSpot-Pro")
annotations_dir = os.path.join(base_dir, "annotations")
images_dir = os.path.join(base_dir, "images")

result_file = os.path.join(base_dir, f'result_{model_name}.txt')
response_file = os.path.join(base_dir, f'response_{model_name}.jsonl')

def evaluate_sample(item):
    """
    对单个样本进行评估：
    - 根据 img_filename 构造完整图片路径
    - 将 bbox 从 [x, y, w, h] 转换为 [x_min, y_min, x_max, y_max]
    - 调用 grounding.call 获取预测点 (x, y)
    - 判断预测点是否在 bbox 内
    """
    img_filename = item.get("img_filename")
    if not img_filename:
        logging.error("样本中缺少 img_filename")
        return None
    # 为 img_filename 添加 images 目录前缀
    img_path = os.path.join(images_dir, img_filename)
    if not os.path.exists(img_path):
        logging.error(f"未找到图片：{img_path}")
        return None
    
    bbox = item.get("bbox")
    if not bbox or len(bbox) != 4:
        logging.error(f"图片 {img_path} 的 bbox 格式不正确")
        return None

    instruction = item.get("instruction", "")
    
    # 调用 grounding 获取预测结果
    x, y = grounding.call(instruction, img_path)
    
    # 判断预测点是否在 bbox 内
    if x is not None and y is not None and (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3]):
        success = True
    else:
        success = False
    
    response_data = {
        "img_path": img_path,
        "instruction": instruction,
        "bbox": bbox,
        "pred": [x, y],
        "success": success
    }
    return response_data


if __name__ == "__main__":
    all_samples = []
    # 从 annotations 目录下读取所有 JSON 文件中的样本
    for file_name in os.listdir(annotations_dir):
        if file_name.endswith(".json"):
            json_path = os.path.join(annotations_dir, file_name)
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_samples.extend(data)
                    else:
                        logging.error(f"{json_path} 的 JSON 格式异常，期望为列表")
                except Exception as e:
                    logging.error(f"读取 {json_path} 时出错: {e}")
    
    total_samples = len(all_samples)
    print(f"总样本数：{total_samples}")
    
    responses = []
    max_workers = 16 # 可根据实际情况调整线程数
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(evaluate_sample, sample): sample for sample in all_samples}
        for future in tqdm(as_completed(future_to_sample), total=total_samples, desc="评估中"):
            result = future.result()
            if result is not None:
                responses.append(result)
    
    # 将评估结果写入响应文件（jsonl 格式）
    with open(response_file, 'w') as rf:
        for resp in responses:
            rf.write(json.dumps(resp) + "\n")
    
    # 统计总体评估结果
    total_cnt = len(responses)
    total_correct = sum(1 for r in responses if r.get("success"))
    accuracy = total_correct / total_cnt if total_cnt > 0 else 0
    
    with open(result_file, 'w') as rf:
        rf.write(f"总样本数: {total_cnt}\n")
        rf.write(f"预测正确数: {total_correct}\n")
        rf.write(f"准确率: {accuracy}\n")
    
    logging.info("所有样本评估完成，结果已写入结果文件。")
