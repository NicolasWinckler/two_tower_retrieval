#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import click
import torch
from torch import distributed as dist
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.inference.state_dict_transform import state_dict_gather
from torchrec.inference import modules as trec_infer_modules

# --- local imports: keep consistent with your training code layout ---
from retrieval.modules.two_tower import TwoTower
from retrieval.data.booksdataset import DEFAULT_RATINGS_COLUMN_NAMES
from retrieval.data.schema import UID_KEY, ITEMID_KEY  # "User-ID", "ISBN"


# ----------------------------- helpers -----------------------------

def _init_pg_once() -> None:
    """Initialize a 1-process PG so ShardedTensor can deserialize."""
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="file:///tmp/torchrec_infer_pg",
            rank=0,
            world_size=1,
        )


def _norm_if_needed(x: torch.Tensor, metric: str) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1) if metric.lower() == "l2" else x


def _load_id_maps(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _invert_map(m: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in m.items()}


def _load_titles(books_csv: Optional[str]) -> Dict[str, str]:
    if not books_csv or not os.path.exists(books_csv):
        return {}
    title_by_isbn: Dict[str, str] = {}
    with open(books_csv, newline="", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        _ = next(r, None)  # header
        for row in r:
            if not row:
                continue
            isbn = row[0]
            title = row[1] if len(row) > 1 else ""
            title_by_isbn[isbn] = title
    return title_by_isbn


def _tensor_shape(obj) -> Tuple[int, int]:
    """Return (rows, dim) for Tensor or ShardedTensor from a checkpoint."""
    # Tensor / Parameter
    if hasattr(obj, "size"):
        s = obj.size()
        if isinstance(s, torch.Size):
            s = tuple(s)
        return int(s[0]), int(s[1])
    # ShardedTensor
    if hasattr(obj, "metadata"):
        md = obj.metadata()  # type: ignore[attr-defined]
        sz = getattr(md, "size", None)
        if sz is not None:
            return int(sz[0]), int(sz[1])
    raise RuntimeError("Unsupported checkpoint tensor type for shape inference.")


def _infer_table_sizes_from_ckpt(ckpt_path: str) -> Dict[str, int]:
    """
    Robustly infer (num_users, num_items, emb_dim) from the checkpoint by searching for
    keys that contain 't_<UID_KEY>' and 't_<ITEMID_KEY>' and end with 'qweight' or 'weight'.
    Works for quantized/FP and sharded/non-sharded states.
    """
    sd = torch.load(ckpt_path, map_location="cpu")

    def pick_shape(token: str) -> Tuple[str, Tuple[int, int]]:
        candidates = []
        for k, v in sd.items():
            if token in k and (k.endswith("qweight") or k.endswith("weight")):
                candidates.append((k, v))
        if not candidates:
            raise RuntimeError(f"No embedding weights found in checkpoint for token '{token}'.")
        # Prefer qweight if present
        candidates.sort(key=lambda kv: 0 if kv[0].endswith("qweight") else 1)
        k, obj = candidates[0]
        shape = _tensor_shape(obj)
        return k, shape

    u_key, u_shape = pick_shape(f"t_{UID_KEY}")
    i_key, i_shape = pick_shape(f"t_{ITEMID_KEY}")

    if u_shape[1] != i_shape[1]:
        print(f"[WARN] User/item dims differ in ckpt: {u_shape} vs {i_shape}; using user dim.")

    return {"users": u_shape[0], "items": i_shape[0], "dim": u_shape[1]}


def _build_two_tower_cpu(embedding_dim: int, num_users: int, num_items: int, layer_sizes: List[int]) -> TwoTower:
    keys = DEFAULT_RATINGS_COLUMN_NAMES[:2]
    assert keys[0] == UID_KEY and keys[1] == ITEMID_KEY, "Schema mismatch."
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{UID_KEY}",
            embedding_dim=embedding_dim,
            num_embeddings=num_users,
            feature_names=[UID_KEY],
        ),
        EmbeddingBagConfig(
            name=f"t_{ITEMID_KEY}",
            embedding_dim=embedding_dim,
            num_embeddings=num_items,
            feature_names=[ITEMID_KEY],
        ),
    ]
    ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))
    model = TwoTower(embedding_bag_collection=ebc, layer_sizes=layer_sizes, device=torch.device("cpu"))
    return model.to_empty(device="cpu").eval()


def _load_state_flex(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load sharded + (quantized|fp) TwoTower/T2TTask checkpoints into a plain TwoTower model.
    - auto-detects quantized vs. fp (qweight vs weight)
    - strips common prefixes
    - tries gather() first; falls back to load_state_dict if src isn't sharded
    """
    sd_src = torch.load(ckpt_path, map_location="cpu")
    saved_is_quant = any(k.endswith("qweight") for k in sd_src.keys())

    def suffix_after(s: str, token: str) -> Optional[str]:
        i = s.find(token)
        return s[i:] if i >= 0 else None

    remapped: Dict[str, object] = {}
    for k, v in sd_src.items():
        suf = (
            suffix_after(k, "ebc.embedding_bags.")
            or suffix_after(k, "query_proj.")
            or suffix_after(k, "candidate_proj.")
        )
        if suf is not None:
            remapped[suf] = v

    if len(remapped) == 0:
        # Help the user debug if mapping failed
        sample = "\n".join(list(sd_src.keys())[:50])
        raise RuntimeError(
            "Could not map any checkpoint keys onto the TwoTower model. "
            "Sample checkpoint keys:\n" + sample
        )

    model_is_quant = any(k.endswith("qweight") for k in model.state_dict().keys())
    if saved_is_quant and not model_is_quant:
        trec_infer_modules.quantize_embeddings(model, dtype=torch.qint8, inplace=True)
    if (not saved_is_quant) and model_is_quant:
        raise RuntimeError(
            "Checkpoint is FP weights but model is quantized; rebuild model without quantization."
        )

    # Try gather (for sharded states); if that fails, fall back to regular load.
    try:
        state_dict_gather(remapped, model.state_dict())
    except Exception:
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            print("[WARN] Missing keys when loading:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("[WARN] Unexpected keys when loading:", unexpected[:10], "..." if len(unexpected) > 10 else "")


def _kjt_for_users(users: List[str], user2ix: Dict[str, int], oov_idx: Optional[int]) -> KeyedJaggedTensor:
    u_idxs = [user2ix.get(u, oov_idx if oov_idx is not None else -1) for u in users]
    if oov_idx is None and any(i < 0 for i in u_idxs):
        missing = [u for u, i in zip(users, u_idxs) if i < 0]
        raise ValueError(f"OOV users without OOV row: {missing[:5]}{'...' if len(missing)>5 else ''}")

    B = len(users)
    # values contain ONLY user indices (sum(lengths) must equal len(values))
    values = torch.tensor(u_idxs, dtype=torch.long)
    # First B entries are user lengths (=1); next B are ISBN lengths (=0)
    lengths = torch.tensor([1] * B + [0] * B, dtype=torch.int32)

    return KeyedJaggedTensor.from_lengths_sync(
        keys=[UID_KEY, ITEMID_KEY],
        values=values,
        lengths=lengths,
    )


# ------------------------------- CLI --------------------------------

@click.command()
@click.option("--save-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="Folder with model.pt and faiss.index saved by training.")
@click.option("--data-path", type=click.Path(exists=True, file_okay=False), default="data", show_default=True)
@click.option("--id-maps-json", type=str, default="id_maps.json", show_default=True)
@click.option("--embedding-dim", type=int, default=64, show_default=True,
              help="Will be overridden by checkpoint dim if different.")
@click.option("--layer-sizes", type=str, default="128,64", show_default=True,
              help="Comma-separated sizes; must match training.")
@click.option("--metric", type=click.Choice(["ip", "l2"], case_sensitive=False), default="ip", show_default=True)
@click.option("--topk", type=int, default=20, show_default=True)
@click.option("--use-faiss-gpu/--no-faiss-gpu", default=False, show_default=True)
@click.option("--books-csv", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Optional Books.csv for titles in output.")
@click.option("--user", "users", multiple=True, required=True,
              help="One or more User-ID values (repeat flag).")
def main(
    save_dir: str,
    data_path: str,
    id_maps_json: str,
    embedding_dim: int,
    layer_sizes: str,
    metric: str,
    topk: int,
    use_faiss_gpu: bool,
    books_csv: Optional[str],
    users: List[str],
) -> None:
    # 0) PG init for ShardedTensor deserialization
    _init_pg_once()

    # 1) Load ID maps & titles
    id_maps = _load_id_maps(os.path.join(data_path, id_maps_json))
    user2ix = id_maps["user2ix"]
    isbn2ix = id_maps["isbn2ix"]
    ix2isbn = _invert_map(isbn2ix)
    titles = _load_titles(books_csv)

    # 2) Infer sizes from checkpoint (authoritative)
    ckpt_path = os.path.join(save_dir, "model.pt")
    ckpt_sizes = _infer_table_sizes_from_ckpt(ckpt_path)
    num_users = ckpt_sizes["users"]
    num_items = ckpt_sizes["items"]

    # OOV if training added a +1 row
    add_oov = (num_users > len(user2ix)) or (num_items > len(isbn2ix))
    oov_user = (num_users - 1) if add_oov else None

    # Keep embedding_dim in sync with checkpoint if it differs
    if ckpt_sizes["dim"] != embedding_dim:
        print(f"[INFO] embedding_dim={embedding_dim} overridden by checkpoint dim={ckpt_sizes['dim']}.")
        embedding_dim = ckpt_sizes["dim"]

    layer_sizes_list = [int(s) for s in layer_sizes.split(",") if s.strip()]

    # 3) Rebuild model and load checkpoint flexibly
    model = _build_two_tower_cpu(embedding_dim, num_users, num_items, layer_sizes_list)
    _load_state_flex(model, ckpt_path)

    # 4) Load FAISS index (CPU) and optionally move to GPU
    import faiss
    index = faiss.read_index(os.path.join(save_dir, "faiss.index"))
    if use_faiss_gpu and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"[WARN] Could not move FAISS to GPU: {e}. Using CPU index.")

    # 5) Build user KJT -> user vectors -> search
    kjt = _kjt_for_users(list(users), user2ix=user2ix, oov_idx=oov_user)
    with torch.no_grad():
        pooled = model.ebc(kjt)
        u_vec = model.query_proj(pooled[UID_KEY])
        u_vec = _norm_if_needed(u_vec, metric)
        q = u_vec.cpu().numpy().astype("float32", copy=False)

    D, I = index.search(q, topk)

    # 6) Pretty print
    for uid, dists, ids in zip(users, D, I):
        print(f"\nUser {uid}:")
        for rank, (iid, score) in enumerate(zip(ids, dists), 1):
            isbn = ix2isbn.get(int(iid), f"<missing:{int(iid)}>")
            t = titles.get(isbn, "")
            suffix = f" — {t}" if t else ""
            print(f"  {rank:2d}. {isbn}{suffix}  (score={float(score):.4f})")


if __name__ == "__main__":
    # Run from repo root:  python -m retrieval.two_tower_retrieval --help
    main()
