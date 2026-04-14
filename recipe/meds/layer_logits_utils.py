# Copyright 2026 OpenMOSS
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
import hashlib
import json
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from verl import DataProto
import hdbscan
import re

def _extract_question(prompt_str: str) -> str:
    extraction_patterns = [
        (
            r"Solve the following math problem step by step[^\n]*\n\n(.*?)\n\nRemember to put your answer",
            re.DOTALL
        ),
        (
            r"<\|User\|>(.*?)<\|Assistant\|>",
            re.DOTALL
        ),
        (
            r"\[INST\](.*?)\[/INST\]",
            re.DOTALL
        ),
    ]
    
    for pattern, flags in extraction_patterns:
        match = re.search(pattern, prompt_str, flags)
        if match:
            return match.group(1).strip()
    
    start_marker = "Solve the following math problem step by step"
    end_marker = "Remember to put your answer"
    
    if start_marker in prompt_str:
        parts = prompt_str.split(start_marker, 1)
        if len(parts) > 1:
            question_part = parts[1]
            if end_marker in question_part:
                return question_part.split(end_marker)[0].strip()
            return question_part.strip()
    
    cleaned = re.sub(r'<[^>]+>', '', prompt_str)
    cleaned = re.sub(r'\[/?INST\]', '', cleaned)
    cleaned = cleaned.strip()
    
    if len(cleaned) < 10:
        return prompt_str.strip()
    
    return cleaned

def _normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text

def _compute_prompt_hash(prompt_text: str) -> str:
    question_text = _extract_question(prompt_text)
    normalized = _normalize_text(question_text)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


def ensure_step_dir(config, global_steps: int, epoch_idx: int) -> tuple[str, str]:
    root_dir = config.trainer.get(
        "knn_analysis_dir", os.path.join(config.trainer.default_local_dir, "knn_analysis")
    )
    epoch_dir = os.path.join(root_dir, f"epoch_{epoch_idx + 1:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    step_dir = os.path.join(epoch_dir, f"step_{global_steps:06d}")
    os.makedirs(step_dir, exist_ok=True)
    return epoch_dir, step_dir


def _save_knn_state(knn_state: dict, output_path: str) -> None:
    serializable_state = {}
    for pid, state in knn_state.items():
        serializable_state[str(pid)] = {
            "vectors": [
                vec.tolist() if isinstance(vec, np.ndarray) else list(vec) 
                for vec in state.get("vectors", [])
            ],
            "labels": [
                int(label) if isinstance(label, (np.integer, int)) else label
                for label in state.get("labels", [])
            ],
        }
        for key, value in state.items():
            if key not in ["vectors", "labels"]:
                if isinstance(value, (np.integer, int)):
                    serializable_state[str(pid)][key] = int(value)
                elif isinstance(value, (np.floating, float)):
                    serializable_state[str(pid)][key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_state[str(pid)][key] = value.tolist()
                else:
                    serializable_state[str(pid)][key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, ensure_ascii=False, indent=2)


def cluster_and_log(
    tokenizer,
    config,
    global_steps: int,
    batch: DataProto,
    epoch_idx: int,
    knn_state: dict,
    knn_prompt_examples: dict,
    old_log_prob: DataProto = None,
    layer_logits_by_batch_idx: dict = None,
) -> None:
    """
    Analyze the layer logits of the current step and save the results to a JSON file.
    
    Args:
        tokenizer: The tokenizer for decoding
        config: Trainer config
        global_steps: Current global step number
        batch: DataProto containing batch data
        epoch_idx: Current epoch index (0-based)
        knn_state: Dictionary to store KNN state (will be modified)
        knn_prompt_examples: Dictionary to store prompt examples (will be modified)
        old_log_prob: DataProto returned from compute_log_prob, containing layer_logits (optional)
        layer_logits_by_batch_idx: Dictionary mapping batch indices to layer logits (optional, will be extracted from old_log_prob if not provided)
    """
    epoch_dir, step_dir = ensure_step_dir(config, global_steps, epoch_idx)

    if layer_logits_by_batch_idx is None and old_log_prob is not None:
        layer_logits_by_batch_idx = {}
        if hasattr(old_log_prob, 'non_tensor_batch') and old_log_prob.non_tensor_batch is not None:
            layer_logits_array = old_log_prob.non_tensor_batch.get("layer_logits", None)
            if layer_logits_array is not None:
                for batch_idx in range(len(layer_logits_array)):
                    vec = layer_logits_array[batch_idx]
                    if vec is not None:
                        if isinstance(vec, (list, np.ndarray)):
                            layer_logits_by_batch_idx[batch_idx] = [float(x) for x in vec]
                        else:
                            layer_logits_by_batch_idx[batch_idx] = [float(x) for x in list(vec)]


    prompts_ids = batch.batch.get("prompts", None)
    if prompts_ids is None:
        return
    responses_ids = batch.batch["responses"]

    prompts_text = tokenizer.batch_decode(prompts_ids, skip_special_tokens=True)
    responses_text = tokenizer.batch_decode(responses_ids, skip_special_tokens=True)

    ground_truths = []
    for item in batch:
        gt = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
        ground_truths.append(gt)

    acc_arr = batch.non_tensor_batch.get("acc", None)
    if acc_arr is None:
        is_correct = [None] * len(batch)
    else:
        is_correct = [bool(float(x) >= 0.5) for x in acc_arr.tolist()]

    brace_pos_in_response_raw = batch.non_tensor_batch.get("brace_pos_in_response", None)
    if brace_pos_in_response_raw is None:
        brace_pos_in_response_raw = np.array([-1] * len(batch), dtype=np.int64)
    brace_pos_in_response = [
        int(x) if x >= 0 else None 
        for x in brace_pos_in_response_raw.tolist()
    ]

    layer_logits_by_batch_idx = layer_logits_by_batch_idx if layer_logits_by_batch_idx is not None else {}

    info_path = os.path.join(step_dir, "step_info.json")
    
    prompt_index_map = {}
    prompt_indices = []
    for idx, prompt_text in enumerate(prompts_text):
        if prompt_text not in prompt_index_map:
            prompt_index_map[prompt_text] = _compute_prompt_hash(prompt_text)
        prompt_indices.append(prompt_index_map[prompt_text])
    
    batch_size = len(batch)
    cluster_size_per_sample = np.zeros(batch_size, dtype=np.int64)
    
    batch_idx_to_label = {}
    
    cluster_method = config.trainer.get("cluster_method", "kmeans")
    cluster_config = config.trainer.get("clustering", {})
    use_last_n_layers = cluster_config.get("use_last_n_layers", None)
    use_layer_diff = cluster_config.get("use_layer_diff", False)
    
    prompt_batch_samples = {}
    
    for batch_idx, vec in layer_logits_by_batch_idx.items():
        if batch_idx >= len(prompt_indices) or batch_idx >= batch_size:
            continue
        
        pid = prompt_indices[batch_idx]
        if pid is None:
            continue

        corr = is_correct[batch_idx] if batch_idx < len(is_correct) else None
        if corr is True:
            corr_group = 1
        elif corr is False:
            corr_group = 0
        else:
            corr_group = -1

        pid_key = (pid, corr_group)
        
        if pid_key not in knn_prompt_examples:
            knn_prompt_examples[pid_key] = prompts_text[batch_idx]
        
        if pid_key not in knn_state:
            knn_state[pid_key] = {"vectors": [], "labels": [], "k": 0}
        
        vec_np = np.array(vec, dtype=float)
        
        if use_layer_diff:
            if len(vec_np) < 2:
                continue
            vec_np = np.diff(vec_np)
        
        if use_last_n_layers is not None:
            if len(vec_np) >= use_last_n_layers:
                vec_np = vec_np[-use_last_n_layers:]
            elif len(vec_np) < use_last_n_layers:
                pass
        
        if pid_key not in prompt_batch_samples:
            prompt_batch_samples[pid_key] = []
        prompt_batch_samples[pid_key].append((batch_idx, vec_np))
    
    for pid_key, samples in prompt_batch_samples.items():
        state = knn_state[pid_key]
        
        
        if cluster_method == "hdbscan":
            has_labels_before = _has_clustered_labels(state)
            labels_before = np.array(state.get("labels", []), dtype=int) if has_labels_before else None
            n_historical = len(state.get("vectors", []))

            predicted_labels_step = []
            if has_labels_before:
                knn_metric = cluster_config.get("knn_metric", "cosine")
                for batch_idx, vec_np in samples:
                    pred = _classify_with_knn(state, vec_np, k=3, metric=knn_metric)
                    predicted_labels_step.append((batch_idx, pred))
            else:
                for batch_idx, vec_np in samples:
                    predicted_labels_step.append((batch_idx, None))

            historical_labels = labels_before if labels_before is not None else np.array([], dtype=int)
            for batch_idx, pred in predicted_labels_step:
                if pred is None or n_historical == 0 or historical_labels.size == 0:
                    cluster_size = 0
                else:
                    cluster_size = int(np.sum(historical_labels == pred))
                cluster_size_per_sample[batch_idx] = cluster_size
                batch_idx_to_label[batch_idx] = pred

            for batch_idx, vec_np in samples:
                _collect_sample(state, vec_np)

            vectors = state.get("vectors", [])
            use_l2_normalize = cluster_config.get("use_l2_normalize", True)
            hdbscan_metric = cluster_config.get("hdbscan_metric", "euclidean")
            min_cluster_size = 2

            _cluster_with_hdbscan(
                state,
                min_cluster_size=min_cluster_size,
                use_l2_normalize=use_l2_normalize,
                metric=hdbscan_metric,
            )

        elif cluster_method == "single":
            labels_before = np.array(state.get("labels", []), dtype=int) if _has_clustered_labels(state) else None
            n_historical = len(state.get("vectors", []))

            for batch_idx, vec_np in samples:
                cluster_size_per_sample[batch_idx] = n_historical
                batch_idx_to_label[batch_idx] = 0

            for batch_idx, vec_np in samples:
                _collect_sample(state, vec_np)

            total_samples = len(state.get("vectors", []))
            state["labels"] = [0] * total_samples
            state["k"] = 1
    
    penalty_target = cluster_config.get("cluster_penalty_target", "both")
    if acc_arr is not None:
        for i in range(len(batch)):
            c = is_correct[i]
            if penalty_target == "wrong":
                if c is True:
                    cluster_size_per_sample[i] = 0
            elif penalty_target == "right":
                if c is False:
                    cluster_size_per_sample[i] = 0
            elif penalty_target == "none":
                cluster_size_per_sample[i] = 0
            
    batch.non_tensor_batch["cluster_size_per_sample"] = cluster_size_per_sample
    
    correctness_arr = np.zeros(len(batch), dtype=np.int8)
    for i in range(len(batch)):
        if is_correct[i] is True:
            correctness_arr[i] = 1
        elif is_correct[i] is False:
            correctness_arr[i] = 0
        else:
            correctness_arr[i] = -1
    batch.non_tensor_batch["cluster_penalty_correctness"] = correctness_arr 
    
    records = []
    for i in range(len(batch)):
        pid = prompt_indices[i] if i < len(prompt_indices) else None

        rec = {
            "global_step": int(global_steps),
            "epoch": int(epoch_idx + 1),
            "prompt_index(pid)": pid,
            "prompt": prompts_text[i],
            "ground_truth": ground_truths[i],
            "response": responses_text[i],
            "is_correct": is_correct[i],
            "brace_pos_in_response": (
                int(brace_pos_in_response[i]) if brace_pos_in_response[i] is not None else None
            ),
            "cluster_label": batch_idx_to_label.get(i, None),
            "cluster_size": int(cluster_size_per_sample[i]) if i < len(cluster_size_per_sample) else None,
            "layer_logits": layer_logits_by_batch_idx.get(i, None),
        }
        records.append(rec)
    
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    knn_state_path = os.path.join(step_dir, "knn_state.json")
    _save_knn_state(knn_state, knn_state_path)


def _has_clustered_labels(state: dict) -> bool:
    vectors = state.get("vectors", [])
    labels = state.get("labels", [])
    return len(labels) > 0 and len(labels) == len(vectors)


def _collect_sample(state: dict, vec_np: np.ndarray) -> None:
    if "vectors" not in state:
        state["vectors"] = []
    state["vectors"].append(vec_np)


def _classify_with_knn(state: dict, vec_np: np.ndarray, k: int = 3, metric: str = "cosine") -> int:
    X = np.stack(state["vectors"], axis=0)
    labels = np.array(state["labels"], dtype=int)
    
    if len(X) == 0:
        return 0
    
    k_actual = min(k, len(X))
    nn = NearestNeighbors(n_neighbors=k_actual, metric=metric, algorithm='brute')
    nn.fit(X)
    
    distances, indices = nn.kneighbors(vec_np.reshape(1, -1))
    nn_labels = labels[indices[0]]
    
    unique, counts = np.unique(nn_labels, return_counts=True)
    return int(unique[np.argmax(counts)])


def _cluster_with_hdbscan(
    state: dict,
    min_cluster_size: int = 2,
    use_l2_normalize: bool = True,
    metric: str = "euclidean",
) -> None:
    """
    Use HDBSCAN density-based clustering.
    
    Args:
        state: KNN state dictionary with 'vectors' key
        min_cluster_size: Minimum cluster size for HDBSCAN
    """
    vectors = state.get("vectors", [])
    if len(vectors) < 2:
        state["labels"] = [-1] * len(vectors)
        state["k"] = 0
        return
    
    X = np.stack(vectors, axis=0)
    X_used = normalize(X, norm='l2')

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric=metric,
    )
    labels = clusterer.fit_predict(X_used)
    
    state["labels"] = labels.tolist()
    unique_labels = np.unique(labels)
    non_noise_labels = unique_labels[unique_labels != -1]
    state["k"] = int(len(non_noise_labels))
