# retrieval/data/dataloader.py
import re
import hashlib
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchrec.datasets.utils import Batch, safe_cast
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .booksdataset import (
    BookCrossingIterableDataset,
    DEFAULT_RATINGS_COLUMN_NAMES,
)
from .schema import (
    UID_KEY, ITEMID_KEY, RATING_KEY,
    USER_COUNTRY, USER_AGE, ITEM_AUTHOR, ITEM_PUB, ITEM_YEAR, ITEM_TITLE,
)

# ----------------- small helpers -----------------

def _hash_str(s: str, mod: int) -> int:
    return int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16) % mod

def _age_bucket(a: str, buckets: int = 10) -> int:
    try:
        v = int(a)
    except Exception:
        v = 0
    v = max(0, min(99, v))
    step = max(1, 100 // buckets)
    return min(v // step, buckets - 1)

def _year_bucket(y: str, buckets: int = 12, start: int = 1950, end: int = 2025) -> int:
    try:
        v = int(y)
    except Exception:
        v = 0
    if v < start or v > end:
        return 0
    span = end - start + 1
    step = max(1, span // (buckets - 1))
    return 1 + min((v - start) // step, buckets - 2)

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())

# ----------------- collate -> Batch -----------------

def _collate_to_batch(
    rows: List[Dict[str, str]],
    *,
    id_maps: Dict[str, Dict[str, int]],
    add_oov: bool = True,
    side_features: bool = False,
    side_vocabs: Optional[Dict[str, int]] = None,
) -> Batch:
    """
    Build a Batch with KJT.
    - Contiguous IDs for UID_KEY / ITEMID_KEY from id_maps (OOV to last row if enabled).
    - Optional side features (hashed buckets) when side_features=True.
    """
    assert id_maps is not None and "user2ix" in id_maps and "isbn2ix" in id_maps, \
        "id_maps must contain 'user2ix' and 'isbn2ix' for contiguous IDs."

    side_vocabs = side_vocabs or {
        USER_COUNTRY: 5000,
        USER_AGE: 10,
        ITEM_AUTHOR: 200_000,
        ITEM_PUB: 80_000,
        ITEM_YEAR: 12,
        ITEM_TITLE: 100_000
    }

    u2i = id_maps["user2ix"]
    i2i = id_maps["isbn2ix"]

    num_users = len(u2i) + (1 if add_oov else 0)
    num_items = len(i2i) + (1 if add_oov else 0)
    oov_user = num_users - 1 if add_oov else -1
    oov_item = num_items - 1 if add_oov else -1

    # -------- base features: contiguous IDs --------
    u_vals = [u2i.get(r[UID_KEY], oov_user) for r in rows]
    i_vals = [i2i.get(r[ITEMID_KEY], oov_item) for r in rows]

    if not add_oov:
        keep_idx = [j for j, (u, it) in enumerate(zip(u_vals, i_vals)) if u >= 0 and it >= 0]
        rows   = [rows[j] for j in keep_idx]
        u_vals = [u_vals[j] for j in keep_idx]
        i_vals = [i_vals[j] for j in keep_idx]

    keys: List[str] = [UID_KEY, ITEMID_KEY]
    values_flat: List[int] = []
    lengths_flat: List[int] = []

    def _append_single_feature(vals: List[int]) -> None:
        # one value per sample
        values_flat.extend(vals)
        lengths_flat.extend([1] * len(vals))

    def _append_multihot_feature(lists: List[List[int]]) -> None:
        # variable number per sample
        for lst in lists:
            values_flat.extend(lst)
            lengths_flat.append(len(lst))

    _append_single_feature(u_vals)
    _append_single_feature(i_vals)

    # -------- side features (hashed buckets) --------
    if side_features:
        # NOTE: field names (e.g., "country", USER_AGE, "Book-Author", "Publisher", "Year-Of-Publication", "Book-Title")
        # come from  row_mapper in booksdataset.py. 
        uc_vals = [_hash_str(r.get(USER_COUNTRY, ""),                 side_vocabs[USER_COUNTRY])   for r in rows]
        ua_vals = [_age_bucket(r.get(USER_AGE, "0"),                  side_vocabs[USER_AGE])       for r in rows]
        ia_vals = [_hash_str(r.get(ITEM_AUTHOR, ""),             side_vocabs[ITEM_AUTHOR])    for r in rows]
        ip_vals = [_hash_str(r.get(ITEM_PUB, ""),               side_vocabs[ITEM_PUB]) for r in rows]
        iy_vals = [_year_bucket(r.get(ITEM_YEAR, "0"), side_vocabs[ITEM_YEAR])      for r in rows]
        it_vals = [[_hash_str(t, side_vocabs[ITEM_TITLE]) for t in _tokenize(r.get(ITEM_TITLE,""))] for r in rows]

        keys += [USER_COUNTRY, USER_AGE, ITEM_AUTHOR, ITEM_PUB, ITEM_YEAR]
        _append_single_feature(uc_vals)
        _append_single_feature(ua_vals)
        _append_single_feature(ia_vals)
        _append_single_feature(ip_vals)
        _append_single_feature(iy_vals)
        # could use a text encoder instead
        _append_multihot_feature(ititle_lists)


    # Build KJT in the same order as 'keys'
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.tensor(values_flat, dtype=torch.long),
        lengths=torch.tensor(lengths_flat, dtype=torch.int32),
    )

    # Labels from rating (binarize)
    labels = torch.tensor(
        [1.0 if safe_cast(r[RATING_KEY], float, 0.0) >= 4.0 else 0.0 for r in rows],
        dtype=torch.float32,
    )
    return Batch(sparse_features=kjt, dense_features=None, labels=labels)

# ----------------- factory -----------------

def get_dataloader(
    batch_size: int,
    num_embeddings: int = None,  # obsolete with contiguous IDs
    pin_memory: bool = False,
    num_workers: int = 0,
    *,
    data_path: str = "data",
    include_users_data: bool = True,
    include_books_data: bool = True,
    filter_to_catalog: bool = False,
    id_maps: Optional[Dict[str, Dict[str, int]]] = None,
    add_oov: bool = True,
    side_features: bool = False,
    side_vocabs: Optional[Dict[str, int]] = None,
) -> DataLoader:
    """
    Returns ONE DataLoader yielding TorchRec Batch objects.
    - Uses contiguous ID maps for User-ID / ISBN (id_maps must be provided).
    - When side_features=True, emits extra KJT keys for user/book metadata.
    """
    dataset = BookCrossingIterableDataset(
        data_path,
        include_users_data=include_users_data,
        include_books_data=include_books_data,
        filter_to_catalog=filter_to_catalog,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda rows: _collate_to_batch(
            rows,
            id_maps=id_maps,
            add_oov=add_oov,
            side_features=side_features,
            side_vocabs=side_vocabs,
        ),
        drop_last=False,
    )
    return loader
