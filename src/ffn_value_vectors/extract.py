"""Extract FFN value vectors and project to vocab tokens."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ffn_value_vectors.action_tokenizer import ActionTokenizer


@dataclass
class ProjectionConfig:
    model_name: str = "openvla/openvla-7b-finetuned-libero-10"
    output_dir: str = "ffn_value_vectors/artifacts/ffn_value_projection"
    batch_size: int = 2048
    top_k: int = 10
    save_logits: bool = False
    use_action_tokens: bool = True
    action_bins: int = 256
    action_min: float = -1.0
    action_max: float = 1.0
    dtype: torch.dtype = torch.bfloat16

    @property
    def value_vectors_path(self) -> str:
        return os.path.join(self.output_dir, "value_vectors.pkl")

    @property
    def logits_dir(self) -> str:
        return os.path.join(self.output_dir, "logit_batches")

    @property
    def top_tokens_path(self) -> str:
        return os.path.join(self.output_dir, "top_tokens.pkl")

    @property
    def top_tokens_txt(self) -> str:
        return os.path.join(self.output_dir, "top_tokens_output.txt")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizers(cfg: ProjectionConfig):
    processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model_name,
        torch_dtype=cfg.dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(_device()).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    return model, processor, tokenizer


def extract_value_vectors(cfg: ProjectionConfig, model) -> torch.Tensor:
    if os.path.exists(cfg.value_vectors_path):
        with open(cfg.value_vectors_path, "rb") as f:
            return pickle.load(f)

    print("Generating value vectors via down_proj...")
    device = _device()
    vectors: List[torch.Tensor] = []
    decoder_layers = model.language_model.model.layers

    with torch.no_grad():
        for layer in tqdm(decoder_layers, desc="Extracting layers"):
            intermediate_size = layer.mlp.down_proj.in_features
            one_hot = torch.eye(intermediate_size, dtype=cfg.dtype, device=device)
            down_proj = layer.mlp.down_proj.to(device=device, dtype=cfg.dtype)
            projected = down_proj(one_hot)
            vectors.append(projected.cpu())

    value_vectors = torch.cat(vectors, dim=0)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(cfg.value_vectors_path, "wb") as f:
        pickle.dump(value_vectors, f)
    print(f"Saved value vectors to {cfg.value_vectors_path}. Shape: {value_vectors.shape}")
    return value_vectors


def project_to_vocab_batched(cfg: ProjectionConfig, model, value_vectors: torch.Tensor) -> None:
    batch_dir = cfg.logits_dir
    if os.path.exists(batch_dir) and any(name.endswith(".pkl") for name in os.listdir(batch_dir)):
        print(f"Found existing logits in {batch_dir}. Skipping projection.")
        return

    os.makedirs(batch_dir, exist_ok=True)
    device = _device()
    value_vectors = value_vectors.to(device=device, dtype=cfg.dtype)
    embedding_matrix = model.language_model.lm_head.weight.detach().to(device=device, dtype=cfg.dtype)

    num_batches = (value_vectors.shape[0] + cfg.batch_size - 1) // cfg.batch_size
    print("Projecting value vectors to vocab space in batches...")
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Projecting"):
            batch = value_vectors[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            batch_logits = (batch @ embedding_matrix.T).float().cpu()
            batch_file = os.path.join(batch_dir, f"logits_{i:03d}.pkl")
            with open(batch_file, "wb") as f:
                pickle.dump(batch_logits, f)

    print(f"Saved {num_batches} batched logits to {batch_dir}")


def _decode_token(
    token_id: int,
    tokenizer,
    action_tokenizer: ActionTokenizer | None,
    action_token_start: int,
    action_token_end: int,
) -> str:
    if action_tokenizer and action_token_start <= token_id < action_token_end:
        action_val = action_tokenizer.decode_token_ids_to_actions(np.array([token_id]))[0]
        return f"[action: {action_val:.3f}]"
    return repr(tokenizer.decode([token_id]))


def top_tokens_from_batches(cfg: ProjectionConfig, tokenizer, processor) -> Tuple[torch.Tensor, List[List[str]]]:
    if os.path.exists(cfg.top_tokens_path):
        with open(cfg.top_tokens_path, "rb") as f:
            return pickle.load(f)

    print(f"Finding top-{cfg.top_k} tokens per vector from batched logits...")
    top_token_ids: List[List[int]] = []
    top_token_texts: List[List[str]] = []

    vocab_size = processor.tokenizer.vocab_size
    action_token_start = vocab_size - cfg.action_bins
    action_token_end = vocab_size
    action_tokenizer = None
    if cfg.use_action_tokens:
        action_tokenizer = ActionTokenizer(
            vocab_size=vocab_size,
            bins=cfg.action_bins,
            min_action=cfg.action_min,
            max_action=cfg.action_max,
        )

    batch_files = sorted([f for f in os.listdir(cfg.logits_dir) if f.endswith(".pkl")])

    for file in tqdm(batch_files, desc="Processing batches"):
        with open(os.path.join(cfg.logits_dir, file), "rb") as f:
            logits = pickle.load(f)
        logits = logits.to(_device())
        top_ids = torch.topk(logits, k=cfg.top_k, dim=1).indices.cpu()

        for row_ids in top_ids:
            token_strs = [
                _decode_token(token_id, tokenizer, action_tokenizer, action_token_start, action_token_end)
                for token_id in row_ids.tolist()
            ]
            top_token_ids.append(row_ids.tolist())
            top_token_texts.append(token_strs)

    top_token_ids_tensor = torch.tensor(top_token_ids)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(cfg.top_tokens_path, "wb") as f:
        pickle.dump((top_token_ids_tensor, top_token_texts), f)

    print(f"Saved top-{cfg.top_k} tokens to {cfg.top_tokens_path}")
    return top_token_ids_tensor, top_token_texts


def project_to_vocab_top_tokens_streaming(
    cfg: ProjectionConfig,
    model,
    value_vectors: torch.Tensor,
    tokenizer,
    processor,
) -> Tuple[torch.Tensor, List[List[str]]]:
    if os.path.exists(cfg.top_tokens_path):
        with open(cfg.top_tokens_path, "rb") as f:
            return pickle.load(f)

    print("Projecting value vectors and streaming top tokens...")
    device = _device()
    value_vectors = value_vectors.to(device=device, dtype=cfg.dtype)
    embedding_matrix = model.language_model.lm_head.weight.detach().to(device=device, dtype=cfg.dtype)

    vocab_size = processor.tokenizer.vocab_size
    action_token_start = vocab_size - cfg.action_bins
    action_token_end = vocab_size
    action_tokenizer = None
    if cfg.use_action_tokens:
        action_tokenizer = ActionTokenizer(
            vocab_size=vocab_size,
            bins=cfg.action_bins,
            min_action=cfg.action_min,
            max_action=cfg.action_max,
        )

    num_batches = (value_vectors.shape[0] + cfg.batch_size - 1) // cfg.batch_size
    top_token_ids: List[List[int]] = []
    top_token_texts: List[List[str]] = []

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Projecting"):
            batch = value_vectors[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            batch_logits = (batch @ embedding_matrix.T).float()
            top_ids = torch.topk(batch_logits, k=cfg.top_k, dim=1).indices.cpu()

            for row_ids in top_ids:
                token_strs = [
                    _decode_token(token_id, tokenizer, action_tokenizer, action_token_start, action_token_end)
                    for token_id in row_ids.tolist()
                ]
                top_token_ids.append(row_ids.tolist())
                top_token_texts.append(token_strs)

    top_token_ids_tensor = torch.tensor(top_token_ids)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(cfg.top_tokens_path, "wb") as f:
        pickle.dump((top_token_ids_tensor, top_token_texts), f)

    print(f"Saved top-{cfg.top_k} tokens to {cfg.top_tokens_path}")
    write_top_tokens_txt(cfg, top_token_texts)
    return top_token_ids_tensor, top_token_texts


def write_top_tokens_txt(cfg: ProjectionConfig, top_token_texts: List[List[str]]) -> None:
    with open(cfg.top_tokens_txt, "w") as f:
        for i, tokens in enumerate(top_token_texts):
            line = f"[{i:04d}] {', '.join(tokens)}\n"
            f.write(line)
    print(f"Wrote top tokens to {cfg.top_tokens_txt}")


def run(cfg: ProjectionConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    model, processor, tokenizer = load_model_and_tokenizers(cfg)
    value_vectors = extract_value_vectors(cfg, model)
    if cfg.save_logits:
        project_to_vocab_batched(cfg, model, value_vectors)
        _, top_token_texts = top_tokens_from_batches(cfg, tokenizer, processor)
        write_top_tokens_txt(cfg, top_token_texts)
    else:
        project_to_vocab_top_tokens_streaming(cfg, model, value_vectors, tokenizer, processor)


def build_config_from_args(args) -> ProjectionConfig:
    return ProjectionConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        top_k=args.top_k,
        save_logits=args.save_logits,
        use_action_tokens=not args.no_action_tokens,
        action_bins=args.action_bins,
        action_min=args.action_min,
        action_max=args.action_max,
    )
