#!/usr/bin/env python3
"""
HuggingFace Transformers attention bridge for llm-visuals.

Loads a causal LM, runs generation with output_attentions=True, and streams
JSON lines of per-layer/per-head attention activity to stdout.

Usage:
  python python_bridge.py --model <hf_id_or_path> --prompt "..." [--max-tokens N]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream HF attention weights as JSONL")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--dtype", default="auto", help="float16 / bfloat16 / float32 / auto")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        emit({"type": "error", "message": f"Missing dependency: {exc}"})
        return 1

    emit({"type": "status", "message": f"Loading model {args.model}"})

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=args.device,
            trust_remote_code=True,
            attn_implementation="eager",  # required for output_attentions
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001 - surface load errors to Rust
        emit({"type": "error", "message": f"Failed to load model: {exc}"})
        return 1

    num_layers = getattr(model.config, "num_hidden_layers", None)
    num_heads = getattr(model.config, "num_attention_heads", None)
    ctx_max = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", None
    )
    emit(
        {
            "type": "model_info",
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ctx_max": ctx_max,
            "model": args.model,
        }
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    emit({"type": "status", "message": f"Prompt tokens: {prompt_len}"})

    generated = inputs["input_ids"]
    past_key_values = None

    with torch.no_grad():
        for step in range(args.max_tokens):
            out = model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            query_pos = generated.shape[1]  # position of the new token

            # attentions: tuple(num_layers) of (batch, heads, q_len, kv_len)
            if out.attentions is not None:
                for layer_idx, attn in enumerate(out.attentions):
                    # Take last query position only (streaming step)
                    # attn shape: [1, heads, q_len, kv_len]
                    layer_attn = attn[0, :, -1, :]  # [heads, kv_len]
                    # Intensity = max attention over key positions per head
                    max_per_head = layer_attn.max(dim=-1).values  # [heads]
                    for head_idx, intensity in enumerate(max_per_head.tolist()):
                        emit(
                            {
                                "type": "attention",
                                "layer": layer_idx,
                                "head": head_idx,
                                "query_pos": query_pos,
                                "key_pos": -1,
                                "weight": float(intensity),
                            }
                        )

            token_id = int(next_token.item())
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            emit(
                {
                    "type": "token",
                    "token_index": query_pos,
                    "token_id": token_id,
                    "text": token_text,
                }
            )

            generated = torch.cat([generated, next_token], dim=-1)

            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

    emit({"type": "done", "tokens_generated": generated.shape[1] - prompt_len})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
