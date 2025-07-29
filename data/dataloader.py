# retrieval/data/dataloader.py
import hashlib
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchrec.datasets.utils import Batch, safe_cast
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .booksdataset import BookCrossingIterableDataset, DEFAULT_RATINGS_COLUMN_NAMES

# CSV keys coming from your row_mapper / DEFAULT_COLUMN_NAMES
# UID_KEY = "User-ID"
# ITEMID_KEY = "ISBN"
# RATING_KEY = "Book-Rating"

UID_KEY, ITEMID_KEY, RATING_KEY = DEFAULT_RATINGS_COLUMN_NAMES[:3]

def _hash_str(s: str, mod: int) -> int:
    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    return int(h, 16) % mod

def _collate_to_batch(rows: List[Dict[str, str]], num_embeddings: int) -> Batch:
    # IDs as strings → hash to [0, num_embeddings)
    u_vals = [_hash_str(r[UID_KEY], num_embeddings) for r in rows]
    i_vals = [_hash_str(r[ITEMID_KEY], num_embeddings) for r in rows]

    # Build KJT with keys expected by the two-tower example: ["userId","movieId"]
    values  = torch.tensor(u_vals + i_vals, dtype=torch.long)
    lengths = torch.ones(len(rows) * 2, dtype=torch.int32)  # 2 features per sample

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=[UID_KEY, ITEMID_KEY],
        values=values,
        lengths=lengths,
    )

    # Labels from rating (binarize)
    labels = torch.tensor(
        [1.0 if safe_cast(r[RATING_KEY], float, 0.0) >= 4.0 else 0.0 for r in rows],
        dtype=torch.float32,
    )
    return Batch(sparse_features=kjt, dense_features=None, labels=labels)

def get_dataloader(
    batch_size: int,
    num_embeddings: int,
    pin_memory: bool = False,
    num_workers: int = 0,
    *,
    data_path: str = "data",
    include_users_data: bool = True,
    include_books_data: bool = True,
    filter_to_catalog: bool = False
) -> DataLoader:
    """
    Single DataLoader (like the original example), now reading Book-Crossing.
    num_embeddings = hash bucket size used for BOTH userId and movieId.
    """
    dataset = BookCrossingIterableDataset(
        data_path,
        include_users_data=include_users_data,
        include_books_data=include_books_data,
        filter_to_catalog=filter_to_catalog
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda rows: _collate_to_batch(rows, num_embeddings),
        drop_last=False,
    )
    return loader
