import json
import re

def parse_coordinates(response: str):
    """
        Parse JSON format response to extract coordinates. Example:
        {"point_2d": [x, y], "label": "object name/description"}
        
        Returns:    
            tuple: (x, y) coordinates
    """
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        data = json.loads(cleaned)
        coords = data.get("point_2d")
        if isinstance(coords, list) and len(coords) == 2:
            return tuple(coords)
        else:
            return None, None
    except Exception:
        return None, None


def acc_reward(predict_str: str, ground_truth: list) -> float:
    """校验坐标是否在目标区域内"""
    x, y = parse_coordinates(predict_str)
    if x is None or y is None:
        return 0.0
    
    x_min, y_min, x_max, y_max = ground_truth
    return 1.0 if (x_min <= x <= x_max) and (y_min <= y <= y_max) else 0.0


def compute_score(predict_str: str, ground_truth: list) -> float:
    """综合评分（是否准确100%）"""
    if predict_str.endswith("<|im_end|>"):
        predict_str = predict_str[:-10]
    
    score = acc_reward(predict_str, ground_truth)
    # print(f"score: {score}")
    return acc_reward(predict_str, ground_truth)