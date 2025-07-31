# retrieval/data/dataloader.py
import re, hashlib
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchrec.datasets.utils import Batch, safe_cast
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .booksdataset import BookCrossingIterableDataset, DEFAULT_RATINGS_COLUMN_NAMES
from .schema import UID_KEY, ITEMID_KEY, RATING_KEY, USER_COUNTRY, USER_AGE, ITEM_AUTHOR, ITEM_PUB, ITEM_YEAR, ITEM_TITLE
# CSV keys coming from your row_mapper / DEFAULT_COLUMN_NAMES
# UID_KEY = "User-ID"
# ITEMID_KEY = "ISBN"
# RATING_KEY = "Book-Rating"

#UID_KEY, ITEMID_KEY, RATING_KEY = DEFAULT_RATINGS_COLUMN_NAMES[:3]


def _collate_to_batch(rows, id_maps=None, add_oov=True, num_embeddings=None):
    # use contiguous ids
    u2i = id_maps["user2ix"]
    b2i = id_maps["isbn2ix"]

    # column names (first two of DEFAULT_RATINGS_COLUMN_NAMES)
    UID_KEY, ITEMID_KEY = DEFAULT_RATINGS_COLUMN_NAMES[:2]

    u_vals = [u2i.get(r[UID_KEY], -1) for r in rows]
    i_vals = [b2i.get(r[ITEMID_KEY], -1) for r in rows]

    if add_oov:
      u_oov = len(u2i); i_oov = len(b2i)
      u_vals = [v if v >= 0 else u_oov for v in u_vals]
      i_vals = [v if v >= 0 else i_oov for v in i_vals]
    else:
      # drop rows with unknown ids
      keep = [i for i,(u,iid) in enumerate(zip(u_vals,i_vals)) if u>=0 and iid>=0]
      rows   = [rows[i] for i in keep]
      u_vals = [u_vals[i] for i in keep]
      i_vals = [i_vals[i] for i in keep]

    keys = [UID_KEY, ITEMID_KEY]
    values_list = [u_vals, i_vals]
    lengths_list = [1]*len(rows) + [1]*len(rows)
    if side_features:
        # build integer indices for each side feature (hash or maps)
        keys += [USER_COUNTRY, USER_AGE, ITEM_AUTHOR, ITEM_PUB, ITEM_YEAR]  # + ITEM_TITLE if used
        values_list += [uc_vals, ua_vals, ia_vals, ip_vals, iy_vals]        # + ititle_vals
        lengths_list += [1]*len(rows)*5                                     # multi-hot: use actual per-row lengths

    values = torch.tensor([*values_list[0], *values_list[1], *...], dtype=torch.long)
    lengths = torch.tensor(lengths_list, dtype=torch.int32)

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
    num_embeddings: int = None,
    pin_memory: bool = False,
    num_workers: int = 0,
    *,
    data_path: str = "data",
    include_users_data: bool = True,
    include_books_data: bool = True,
    filter_to_catalog: bool = False,
    id_maps=None, 
    add_oov=True
) -> DataLoader:
    """
    Single DataLoader (like the original example), now reading Book-Crossing.
    num_embeddings = obsolete
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
        collate_fn=lambda rows: _collate_to_batch(rows, num_embeddings=num_embeddings, id_maps=id_maps, add_oov=add_oov),
        drop_last=False,
    )
    return loader
