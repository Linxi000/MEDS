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


from typing import List, Optional, Sequence
import numpy as np
import torch
from verl import DataProto

def prepare_layer_logits_analysis(tokenizer, batch: DataProto) -> None:
    responses_ids = batch.batch["responses"]
    
    brace_pos_in_response = find_brace_positions_in_responses(tokenizer, responses_ids)

    batch.non_tensor_batch["brace_pos_in_response"] = np.array(
        [int(x) if x is not None else -1 for x in brace_pos_in_response], 
        dtype=np.int64
    )
    
    input_ids = batch.batch["input_ids"]
    seq_len = int(input_ids.shape[1])
    resp_len = int(responses_ids.shape[1])
    prompt_len = seq_len - resp_len
    
    prompt_len = max(0, prompt_len)
    
    brace_abs = []
    for i in range(len(batch)):
        if brace_pos_in_response[i] is not None and brace_pos_in_response[i] >= 0:
            abs_pos = int(prompt_len + brace_pos_in_response[i])
            if abs_pos < seq_len:
                brace_abs.append(abs_pos)
            else:
                brace_abs.append(-1)
        else:
            brace_abs.append(-1)
    
    batch.non_tensor_batch["brace_pos_absolute"] = np.array(brace_abs, dtype=np.int64)

def find_brace_positions_in_responses(tokenizer, responses: torch.Tensor) -> List[Optional[int]]:
    """
    In the token sequence of each response, find the index of the token where "{" in "\\boxed{" is located (relative to the starting position of the response).
    If not found, return None.

    Args:
        tokenizer: The tokenizer to use for encoding/decoding
        responses: Tensor of shape (batch_size, response_length) containing response token IDs

    Returns:
        List[Optional[int]]: List of positions (relative to response start) or None if not found
    """
    positions = []
    responses_cpu = responses.cpu()

    boxed_tokens = tokenizer.encode("\\boxed{", add_special_tokens=False)

    for row in responses_cpu:
        ids = row.tolist()
        pos = None

        if boxed_tokens:
            L = len(boxed_tokens)
            for j in range(len(ids) - L, -1, -1):
                if ids[j : j + L] == boxed_tokens:
                    pos = j + L - 1
                    break

        if pos is None:
            tokens = tokenizer.convert_ids_to_tokens(ids)
            for idx in range(len(tokens) - 1, -1, -1):
                tok = tokens[idx]
                if tok and isinstance(tok, str) and "boxed" in tok.lower():
                    for m in range(idx, min(idx + 10, len(tokens))):
                        if tokens[m] and isinstance(tokens[m], str) and "{" in tokens[m]:
                            pos = m
                            break
                    if pos is not None:
                        break

        positions.append(pos)

    return positions


def extract_layer_logits_from_output(
    output,
    actor_module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    predict_positions: Optional[Sequence[int]],
    indices_in_micro: Optional[Sequence[int]],
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Optional[List[List[float]]]:
    """
    Unified function: Extract layer_logits from output, supporting both remove_padding and non-remove_padding modes.
    
    Args:
        output: Model output containing hidden_states
        actor_module: The actor module (used to get lm_head)
        input_ids: Input token IDs tensor of shape (batch_size, seqlen)
        attention_mask: Attention mask tensor of shape (batch_size, seqlen)
        predict_positions: List of positions to extract logits for
        indices_in_micro: List of sample indices in micro_batch (for subset extraction)
        cu_seqlens: Cumulative sequence lengths for remove_padding mode (optional)
    
    Returns:
        List[List[float]]: layer_logits for each sample, or None if extraction fails
    """
    if predict_positions is None or len(predict_positions) == 0:
        return None
    

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        return None

    model = getattr(actor_module, "module", actor_module)
    lm_head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    if lm_head is None:
        return None

    if cu_seqlens is not None:
        batch_size = input_ids.shape[0]
        predict_positions_rmpad = []
        target_token_ids_list = []
        
        if indices_in_micro is None or len(indices_in_micro) == 0:
            return None
        
        for idx, sample_idx in enumerate(indices_in_micro):
            if sample_idx >= batch_size:
                continue
            orig_pos = predict_positions[idx]
            if orig_pos >= attention_mask.shape[1]:
                continue
            if attention_mask[sample_idx, orig_pos] <= 0:
                continue
            if orig_pos + 1 >= input_ids.shape[1]:
                continue
            
            sample_attention = attention_mask[sample_idx, :orig_pos + 1]
            num_non_pad_tokens = sample_attention.sum().item()
            relative_pos_in_unpad = num_non_pad_tokens - 1
            
            start_pos = cu_seqlens[sample_idx].item() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens[sample_idx]
            rmpad_pos = start_pos + relative_pos_in_unpad
            predict_positions_rmpad.append(rmpad_pos)
            target_token_ids_list.append(input_ids[sample_idx, orig_pos + 1].item())
        
        if len(target_token_ids_list) == 0:
            return None
        
        hidden_states_at_pos = []
        for layer_idx in range(len(hidden_states)):
            hs = hidden_states[layer_idx]

            assert len(hs.shape) == 3, f"Expected 3D hidden_states in remove_padding mode, got shape {hs.shape}"

            max_pos = hs.shape[1]
            picked_hs_list = [hs[0, p, :].unsqueeze(0) for p in predict_positions_rmpad if p < max_pos]
            
            if len(picked_hs_list) != len(predict_positions_rmpad):
                return None
            hidden_states_at_pos.append(torch.cat(picked_hs_list, dim=0))
        
        target_token_ids = torch.tensor(target_token_ids_list, device=input_ids.device, dtype=torch.long)
        return extract_layer_logits_from_hidden_states(
            hidden_states=tuple(hidden_states_at_pos),
            lm_head=lm_head,
            target_token_ids=target_token_ids,
        )
    else:

        indices_tensor = torch.tensor(indices_in_micro, device=input_ids.device, dtype=torch.long)
        hidden_states_subset = tuple(hs[indices_tensor] for hs in hidden_states)
        input_ids_subset = input_ids[indices_tensor]
        predict_pos_t = torch.tensor(predict_positions, device=input_ids.device, dtype=torch.long)
        batch_idx = torch.arange(len(indices_in_micro), device=input_ids.device)
        target_token_ids = input_ids_subset[batch_idx, predict_pos_t + 1]
        
        hidden_states_at_pos = []
        for hs in hidden_states_subset:
            picked = hs[batch_idx, predict_pos_t, :]
            hidden_states_at_pos.append(picked)
        
        return extract_layer_logits_from_hidden_states(
            hidden_states=tuple(hidden_states_at_pos),
            lm_head=lm_head,
            target_token_ids=target_token_ids,
        )


def extract_layer_logits_from_hidden_states(
    hidden_states: tuple,
    lm_head,
    target_token_ids: torch.Tensor,
) -> List[List[float]]:
    """
    Extract layer_logits from the existing hidden_states to avoid repeated forward propagation.
    The hidden_states should be in the pre-extracted format of (bs, hidden_size), skipping embedding layer 0.

    Args:
        hidden_states: Tuple of hidden states from model output (already extracted, shape: (bs, hidden_size))
        lm_head: The language model head (lm_head or embed_out)
        target_token_ids: Target token IDs tensor of shape (bs,)

    Returns:
        List[List[float]]: shape = [bs, num_layers]
    """
    bs = target_token_ids.shape[0]
    if len(hidden_states) == 0:
        return [[] for _ in range(bs)]


    if len(hidden_states) > 1:
        hidden_states = hidden_states[1:]

    layer_logits_per_sample: List[List[float]] = [[] for _ in range(bs)]
    batch_idx = torch.arange(bs, device=target_token_ids.device)

    for layer_idx, hs in enumerate(hidden_states):
        if hs.shape[0] != bs:
            raise ValueError(f"hidden_states[{layer_idx}] batch size {hs.shape[0]} != target_token_ids batch size {bs}")
        
        logits = lm_head(hs)  # (bs, vocab)
        logits = logits.float()
        picked_logits = logits[batch_idx, target_token_ids]  # (bs,)
        vals = picked_logits.detach().cpu().tolist()
        for i in range(bs):
            layer_logits_per_sample[i].append(float(vals[i]))

    return layer_logits_per_sample
