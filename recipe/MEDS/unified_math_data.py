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
Unified math dataset preprocessing following Qwen-Math template
Combines DAPO-Math-17K-processed and MATH levels 3-5 into unified format
"""

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd

from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed


def extract_solution(solution_str: str) -> str:
    """Extract solution from boxed format"""
    return remove_boxed(last_boxed_only_string(solution_str))


def qwen_math_template(problem: str) -> List[Dict[str, str]]:
    """
    Create Qwen-Math template format
    
    Template:
    <|im_start|>system
    Please reason step by step, and put your final answer within \boxed{}. <|im_end|>
    <|im_start|>user
    {problem}
    <|im_end|>
    <|im_start|>assistant
    """
    return [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}."
        },
        {
            "role": "user",
            "content": problem
        },
        {
            "role": "assistant",
            "content": ""  # Assistant response will be generated during training
        }
    ]


def process_dapo_math_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process DAPO-Math-17K-processed format to unified format
    
    Input format:
    {
        "prompt": str,           # Original problem
        "solution": str,          # Full solution
        "data_source": "math_dapo",
        "source_prompt": List[Dict],  # Structured prompt
        "ability": "MATH",
        "reward_model": {"ground_truth": str, "style": str},
        "extra_info": Dict
    }
    """
    problem = item["prompt"]
    solution = item.get("solution", "")
    ground_truth = item["reward_model"]["ground_truth"]
    
    # Create Qwen-Math template format
    prompt_template = qwen_math_template(problem)
    
    return {
        "data_source": "DigitalLearningGmbH/MATH-lighteval",
        "prompt": prompt_template,
        "ability": "MATH",
        "reward_model": {
            "ground_truth": ground_truth,
            "style": "rule-Qwen-Math"
        },
        "extra_info": item.get("extra_info", {}),
        "solution": solution,  # Keep original solution for reference
    }


def process_math_lighteval_item(item: Dict[str, Any], level_filter: List[int] = [3, 4, 5]) -> Dict[str, Any]:
    """
    Process MATH-lighteval format to unified format
    
    Input format:
    {
        "problem": str,
        "level": str (e.g., "Level 5"),
        "type": str,
        "solution": str
    }
    """
    # Extract level number
    level_str = item.get("level", "")
    level_num = None
    try:
        level_num = int(level_str.split()[-1]) if level_str else None
    except:
        pass
    
    # Filter by level (only levels 3-5)
    if level_num is not None and level_num not in level_filter:
        return None
    
    problem = item["problem"]
    solution = item["solution"]
    
    # Extract ground truth from solution
    ground_truth = extract_solution(solution)
    
    # Create Qwen-Math template format
    prompt_template = qwen_math_template(problem)
    
    return {
        "data_source": "DigitalLearningGmbH/MATH-lighteval",
        "prompt": prompt_template,
        "ability": "MATH",
        "reward_model": {
            "ground_truth": ground_truth,
            "style": "rule-Qwen-Math"
        },
        "extra_info": {
            "level": level_str,
            "type": item.get("type", ""),
            "original_solution": solution
        }
    }


def load_dapo_math(file_path: str) -> List[Dict[str, Any]]:
    """Load DAPO-Math-17K-processed data"""
    print(f"Loading DAPO-Math data from {file_path}...")
    
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
        data = df.to_dict('records')
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded {len(data)} items from DAPO-Math")
    return data


def load_math_lighteval(file_path: str, level_filter: List[int] = [3, 4, 5]) -> List[Dict[str, Any]]:
    """Load MATH-lighteval data (levels 3-5 only)"""
    print(f"Loading MATH-lighteval data from {file_path}...")
    
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
        data = df.to_dict('records')
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Filter by level
    filtered_data = []
    for item in data:
        processed = process_math_lighteval_item(item, level_filter)
        if processed is not None:
            filtered_data.append(processed)
    
    print(f"Loaded {len(filtered_data)} items from MATH-lighteval (levels {level_filter})")
    return filtered_data


def merge_and_save(data_list: List[List[Dict[str, Any]]], output_path: str):
    """Merge multiple datasets and save to parquet"""
    # Flatten list of lists
    all_data = []
    for data in data_list:
        all_data.extend(data)
    
    print(f"Total items after merging: {len(all_data)}")
    
    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_parquet(output_path, index=False)
    print(f"Saved unified dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Unify math datasets using Qwen-Math template")
    parser.add_argument('--dapo_path', type=str, required=True,
                       help='Path to DAPO-Math-17K-processed file (parquet or jsonl)')
    parser.add_argument('--math_path', type=str, required=True,
                       help='Path to MATH-lighteval file (parquet or jsonl)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for unified dataset (parquet)')
    parser.add_argument('--level_filter', type=int, nargs='+', default=[3, 4, 5],
                       help='MATH levels to include (default: 3 4 5)')
    
    args = parser.parse_args()
    
    # Load DAPO-Math data
    dapo_data = load_dapo_math(args.dapo_path)
    processed_dapo = [process_dapo_math_item(item) for item in dapo_data]
    
    # Load MATH-lighteval data
    math_data = load_math_lighteval(args.math_path, args.level_filter)
    
    # Merge datasets
    merge_and_save([processed_dapo, math_data], args.output_path)
    
    print("\nDone! Dataset statistics:")
    print(f"  DAPO-Math items: {len(processed_dapo)}")
    print(f"  MATH-lighteval items: {len(math_data)}")
    print(f"  Total items: {len(processed_dapo) + len(math_data)}")


if __name__ == '__main__':
    main()

