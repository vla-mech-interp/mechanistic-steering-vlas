"""Extract pi0 FFN value vectors and project to vocab tokens."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from tqdm import tqdm

from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import download
from openpi.training import config as _config


@dataclass
class ProjectionConfig:
    checkpoint_name: str = "pi0_fast_droid"
    output_dir: str = "openpi/ffn_value_vectors/artifacts/ffn_value_projection"
    batch_size: int = 5000
    top_k: int = 10
    max_batches: int | None = None
    save_logits: bool = False
    save_value_vectors: bool = False
    save_params: bool = False

    @property
    def params_path(self) -> str:
        return os.path.join(self.output_dir, "openpi_params.pkl")

    @property
    def value_vectors_path(self) -> str:
        return os.path.join(self.output_dir, "openpi_value_vectors.pkl")

    @property
    def logits_dir(self) -> str:
        return os.path.join(self.output_dir, "logits_batches")

    @property
    def top_tokens_dir(self) -> str:
        return os.path.join(self.output_dir, "top_tokens_batches")

    @property
    def top_tokens_txt(self) -> str:
        return os.path.join(self.top_tokens_dir, "top_tokens.txt")


def load_checkpoint_params(cfg: ProjectionConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    if os.path.exists(cfg.params_path):
        with open(cfg.params_path, "rb") as f:
            params = pickle.load(f)
        print(f"Loaded checkpoint params from {cfg.params_path}")
        return params

    train_config = _config.get_config(cfg.checkpoint_name)
    checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{cfg.checkpoint_name}")
    params = _model.restore_params(Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
    if cfg.save_params:
        with open(cfg.params_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Saved checkpoint params to {cfg.params_path}")
    return params


def _get_embedding_matrix(flat_params: dict) -> np.ndarray:
    embedding = np.array(flat_params["PaliGemma.llm.embedder.input_embedding"])
    return embedding


def extract_value_vectors(cfg: ProjectionConfig, params) -> np.ndarray:
    if os.path.exists(cfg.value_vectors_path):
        with open(cfg.value_vectors_path, "rb") as f:
            return pickle.load(f)

    flat_params = traverse_util.flatten_dict(params, sep=".")
    tensor = flat_params.get("PaliGemma.llm.layers.mlp.linear")
    if tensor is None:
        raise KeyError("Could not find expected key 'PaliGemma.llm.layers.mlp.linear' in parameter tree.")

    weight_np = np.array(tensor)
    embedding = _get_embedding_matrix(flat_params)
    if weight_np.shape[-1] != embedding.shape[1]:
        weight_np = np.transpose(weight_np, (0, 2, 1))

    value_matrix = weight_np.reshape(-1, weight_np.shape[-1])
    print(f"Extracted value vectors: {value_matrix.shape}")

    if cfg.save_value_vectors:
        with open(cfg.value_vectors_path, "wb") as f:
            pickle.dump(value_matrix, f)
        print(f"Saved value vectors to {cfg.value_vectors_path}")
    return value_matrix


def project_to_vocab(cfg: ProjectionConfig, params, value_matrix: np.ndarray, tokenizer) -> None:
    os.makedirs(cfg.logits_dir, exist_ok=True)
    os.makedirs(cfg.top_tokens_dir, exist_ok=True)

    flat_params = traverse_util.flatten_dict(params, sep=".")
    embedding_matrix = _get_embedding_matrix(flat_params)
    embedding_matrix_t = jax.device_put(embedding_matrix.T)
    value_matrix = jax.device_put(value_matrix)

    num_vectors = value_matrix.shape[0]
    print(f"Projecting in GPU batches of {cfg.batch_size}...")
    for i, start in enumerate(range(0, num_vectors, cfg.batch_size)):
        if cfg.max_batches is not None and i >= cfg.max_batches:
            break
        logits_path = os.path.join(cfg.logits_dir, f"logits_batch_{i:04d}.pkl")
        top_tokens_path = os.path.join(cfg.top_tokens_dir, f"logits_batch_{i:04d}_top_tokens.pkl")
        if os.path.exists(top_tokens_path) and (not cfg.save_logits or os.path.exists(logits_path)):
            print(f"Skipping existing batch {i:04d}")
            continue
        end = min(start + cfg.batch_size, num_vectors)
        batch = value_matrix[start:end]
        batch_logits = jnp.matmul(batch, embedding_matrix_t)
        logits_cpu = np.array(batch_logits, dtype=np.float32)
        if cfg.save_logits:
            with open(logits_path, "wb") as f:
                pickle.dump(logits_cpu, f)
            print(f"Saved logits batch {start}-{end} to {logits_path}")

        top_token_ids = jax.lax.top_k(batch_logits, k=cfg.top_k)[1]
        top_token_ids_cpu = np.array(top_token_ids)
        flat_ids = top_token_ids_cpu.flatten().tolist()
        decoded_flat = [tokenizer._tokenizer.decode_ids([tok_id]) for tok_id in flat_ids]
        top_token_texts = np.array(decoded_flat).reshape(top_token_ids_cpu.shape).tolist()
        with open(top_tokens_path, "wb") as f:
            pickle.dump((top_token_ids_cpu, top_token_texts), f)
        print(f"Saved top tokens for batch {start}-{end} to {top_tokens_path}")


def load_top_tokens_from_cache(cfg: ProjectionConfig) -> Tuple[List[List[int]], List[List[str]]]:
    batch_files = sorted(Path(cfg.top_tokens_dir).glob("logits_batch_*_top_tokens.pkl"))
    all_top_ids: List[List[int]] = []
    all_top_tokens: List[List[str]] = []
    for batch_path in tqdm(batch_files, desc="Top tokens"):
        with open(batch_path, "rb") as f:
            top_ids, top_texts = pickle.load(f)
        all_top_ids.extend(top_ids.tolist())
        all_top_tokens.extend(top_texts)
    return all_top_ids, all_top_tokens


def write_top_tokens_txt(cfg: ProjectionConfig, top_token_texts: List[List[str]]) -> None:
    os.makedirs(cfg.top_tokens_dir, exist_ok=True)
    with open(cfg.top_tokens_txt, "w", encoding="utf-8") as f:
        for i, tokens in enumerate(top_token_texts):
            line = f"[{i:06}] {', '.join(tokens)}\n"
            f.write(line)
    print(f"Wrote ALL top token sets ({len(top_token_texts)}) to {cfg.top_tokens_txt}")


def run(cfg: ProjectionConfig) -> None:
    params = load_checkpoint_params(cfg)
    tokenizer = PaligemmaTokenizer()
    value_matrix = extract_value_vectors(cfg, params)
    project_to_vocab(cfg, params, value_matrix, tokenizer)
    _, top_token_texts = load_top_tokens_from_cache(cfg)
    write_top_tokens_txt(cfg, top_token_texts)


def build_config_from_args(args) -> ProjectionConfig:
    return ProjectionConfig(
        checkpoint_name=args.checkpoint_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        top_k=args.top_k,
        max_batches=args.max_batches,
        save_logits=args.save_logits,
        save_value_vectors=args.save_value_vectors,
        save_params=args.save_params,
    )
