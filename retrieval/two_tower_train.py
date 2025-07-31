#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import os
from typing import List, Optional, Literal
import json 
import click

import faiss  # @manual=//faiss/python:pyfaiss_gpu
import faiss.contrib.torch_utils  # @manual=//faiss/contrib:faiss_contrib_gpu
import torch
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import inference as trec_infer
#from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES
from retrieval.data.booksdataset import DEFAULT_RATINGS_COLUMN_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from retrieval.data.schema import UID_KEY, ITEMID_KEY, USER_COUNTRY, USER_AGE, ITEM_AUTHOR, ITEM_PUB, ITEM_YEAR, ITEM_TITLE

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/retrieval/data:dataloader
    from data.dataloader import get_dataloader

    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/retrieval:knn_index
    from knn_index import get_index

    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/retrieval/modules:two_tower
    from modules.two_tower import TwoTower, TwoTowerTrainTask
except ImportError:
    pass

# internal import
try:
    from .data.dataloader import get_dataloader  # noqa F811
    from .knn_index import get_index  # noqa F811
    from .modules.two_tower import TwoTower, TwoTowerTrainTask  # noqa F811
except ImportError:
    pass


@click.command()
@click.option(
    "--catalog-mode",
    type=click.Choice(["joined", "full"], case_sensitive=False),
    default="joined",
    show_default=True,
    help="joined = train only on rows that exist in Users/Books (inner-join). "
         "full = keep all ratings rows (left-join with defaults).",
)
@click.option("--data-path", type=click.Path(exists=True, file_okay=False), default="data",
              help="Folder containing Ratings.csv / Users.csv / Books.csv and id-maps JSON.")
@click.option("--id-maps-json", type=click.STRING, default="id_maps.json",
              help="Filename of the JSON with user2ix/isbn2ix in data_path.")
@click.option("--add-oov/--no-add-oov", default=True,
              help="Reserve an extra row per table for unseen IDs (OOV fallback).")
@click.option("--embedding-dim", type=int, default=64, show_default=True)
@click.option("--layer-sizes", type=str, default="128,64",
              help="Comma-separated MLP sizes; last size should match embedding-dim.")
@click.option("--embed-lr", type=float, default=0.02, show_default=True,
              help="Learning rate for embedding tables (RowWiseAdagrad).")
@click.option("--dense-lr", type=float, default=3e-4, show_default=True,
              help="Learning rate for dense params (Adam).")
@click.option("--batch-size", type=int, default=8192, show_default=True)
@click.option("--num-iterations", type=int, default=10000, show_default=True,
              help="Number of training steps (batches).")
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--faiss-num-centroids", type=int, default=1024, show_default=True)
@click.option("--faiss-num-subquantizers", type=int, default=16, show_default=True)
@click.option("--faiss-bits-per-code", type=int, default=8, show_default=True)
@click.option("--faiss-num-probe", type=int, default=8, show_default=True)
@click.option(
    "--side-features/--no-side-features",
    default=False,
    show_default=True,
    help="If enabled, train with Users.csv and Books.csv side features in both towers.",
)
@click.option(
    "--save-dir",
    type=click.STRING,
    default=None,
    help="Directory to save model and faiss index. If None, nothing is saved",
)
def main(
    catalog_mode: Literal["joined", "full"],
    data_path: str,
    id_maps_json: str,
    add_oov: bool,
    embedding_dim: int,
    layer_sizes: str,
    embed_lr: float,
    dense_lr: float,
    batch_size: int,
    num_iterations: int,
    num_workers: int,
    faiss_num_centroids: int,
    faiss_num_subquantizers: int,
    faiss_bits_per_code: int,
    faiss_num_probe: int,
    save_dir: Optional[str],
    side_features: bool
) -> None:
    # parse layer sizes
    layer_sizes_list = [int(x) for x in layer_sizes.split(",") if x.strip()]
    if not layer_sizes_list:
        layer_sizes_list = [128, 64]

    # load contiguous ID maps
    maps_path = os.path.join(data_path, id_maps_json)
    with open(maps_path, "r") as f:
        id_maps = json.load(f)
    user2ix = id_maps["user2ix"]
    isbn2ix = id_maps["isbn2ix"]

    # table sizes (+1 if OOV)
    num_embeddings_user = int(id_maps.get("num_embeddings_user", len(user2ix)))
    num_embeddings_item = int(id_maps.get("num_embeddings_item", len(isbn2ix)))
    if add_oov:
        num_embeddings_user += 1
        num_embeddings_item += 1

    train(
        data_path=data_path,
        id_maps={"user2ix": user2ix, "isbn2ix": isbn2ix},
        add_oov=add_oov,
        num_embeddings_user=num_embeddings_user,
        num_embeddings_item=num_embeddings_item,
        embedding_dim=embedding_dim,
        layer_sizes=layer_sizes_list,
        embed_lr=embed_lr,
        dense_lr=dense_lr,
        batch_size=batch_size,
        num_iterations=num_iterations,
        num_workers=num_workers,
        num_centroids=faiss_num_centroids,
        num_subquantizers=faiss_num_subquantizers,
        bits_per_code=faiss_bits_per_code,
        num_probe=faiss_num_probe,
        save_dir=save_dir,
        side_features=side_features
    )


def train(
    *,
    data_path: str,
    id_maps: dict,
    add_oov: bool,
    num_embeddings_user: int,
    num_embeddings_item: int,
    embedding_dim: int = 64,
    layer_sizes: Optional[List[int]] = None,
    embed_lr: float = 0.02,
    dense_lr: float = 3e-4,
    batch_size: int = 8192,
    num_iterations: int = 10000,
    num_workers: int = 0,
    num_centroids: int = 1024,
    num_subquantizers: int = 16,
    bits_per_code: int = 8,
    num_probe: int = 8,
    save_dir: Optional[str] = None,
    side_features: bool = False

) -> None:
    """
    Trains a simple Two Tower (UV) model, which is a simplified version of [A Dual Augmented Two-tower Model for Online Large-scale Recommendation](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf).
    Torchrec is used to shard the model, and is pipelined so that dataloading, data-parallel to model-parallel comms, and forward/backward are overlapped.
    It is trained on random data in the format of [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) dataset in SPMD fashion.
    The distributed model is gathered to CPU.
    The item (movie) towers embeddings are used to train a FAISS [IVFPQ](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint) index, which is serialized.
    The resulting `KNNIndex` can be queried with batched `torch.Tensor`, and will return the distances and indices for the approximate K nearest neighbors of the query embeddings. The model itself is also serialized.

    Args:
        num_embeddings (int): The number of embeddings the embedding table
        embedding_dim (int): embedding dimension of both embedding tables
        layer_sizes (List[int]): list representing layer sizes of the MLP. Last size is the final embedding size
        learning_rate (float): learning_rate
        batch_size (int): batch size to use for training
        num_iterations (int): number of train batches
        num_centroids (int): The number of centroids (Voronoi cells)
        num_subquantizers (int): The number of subquanitizers in Product Quantization (PQ) compression of subvectors
        bits_per_code (int): The number of bits for each subvector in Product Quantization (PQ)
        num_probe (int): The number of centroids (Voronoi cells) to probe. Must be <= num_centroids. Sweeping powers of 2 for nprobe and picking one of those based on recall statistics (e.g., 1, 2, 4, 8, ..,) is typically done.
        save_dir (Optional[str]): Directory to save model and faiss index. If None, nothing is saved
    """
    if layer_sizes is None:
        layer_sizes = [128, 64]

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)

    # config for encoding userID and itemID
    two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[feature_name],
        )
        for feature_name, num_embeddings in zip(two_tower_column_names, [num_embeddings_user, num_embeddings_item])
    ]
    if side_features:
        eb_configs += [
            EmbeddingBagConfig(name=USER_COUNTRY,  embedding_dim=D, num_embeddings=65_536,   feature_names=[USER_COUNTRY]),
            EmbeddingBagConfig(name=USER_AGE,      embedding_dim=D, num_embeddings=20,      feature_names=[USER_AGE]),
            EmbeddingBagConfig(name=ITEM_AUTHOR,   embedding_dim=D, num_embeddings=200_000, feature_names=[ITEM_AUTHOR]),
            EmbeddingBagConfig(name=ITEM_PUB      ,embedding_dim=D, num_embeddings=66_000,  feature_names=[ITEM_PUB]),
            EmbeddingBagConfig(name=ITEM_YEAR,     embedding_dim=D, num_embeddings=20,      feature_names=[ITEM_YEAR]),
            EmbeddingBagConfig(name=ITEM_TITLE,    embedding_dim=D, num_embeddings=200_000, feature_names=[ITEM_TITLE]),
        ]

    embedding_bag_collection = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_model = TwoTower(
        embedding_bag_collection=embedding_bag_collection,
        layer_sizes=layer_sizes,
        device=device,
    )
    two_tower_train_task = TwoTowerTrainTask(two_tower_model)
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        two_tower_train_task.two_tower.ebc.parameters(),
        {"lr": embed_lr},
    )
    model = DistributedModelParallel(
        module=two_tower_train_task,
        device=device,
    )

    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.Adam(params, lr=dense_lr),
    )

    catalog_mode: Literal["joined","full"]="joined"
    dataloader = get_dataloader(
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
        data_path=data_path,
        filter_to_catalog=(catalog_mode=="joined"),
        id_maps=id_maps,
        add_oov=add_oov,
        include_users_data=side_features,
        include_books_data=side_features
    )
    dl_iterator = iter(dataloader)
    train_pipeline = TrainPipelineSparseDist(
        model,
        optimizer,
        device,
    )

    # Train model
    for _ in range(num_iterations):
        try:
            train_pipeline.progress(dl_iterator)
        except StopIteration:
            break

    checkpoint_pg = dist.new_group(backend="gloo")
    # Copy sharded state_dict to CPU.
    cpu_state_dict = state_dict_to_device(
        model.state_dict(), pg=checkpoint_pg, device=torch.device("cpu")
    )

    ebc_cpu = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_cpu = TwoTower(
        embedding_bag_collection=ebc_cpu,
        layer_sizes=layer_sizes,
    )
    two_tower_train_cpu = TwoTowerTrainTask(two_tower_cpu)
    if rank == 0:
        two_tower_train_cpu = two_tower_train_cpu.to_empty(device="cpu")
    state_dict_gather(cpu_state_dict, two_tower_train_cpu.state_dict())
    dist.barrier()

    # Create and train FAISS index for the item (movie) tower on CPU
    if rank == 0:
        index = get_index(
            embedding_dim=embedding_dim,
            num_centroids=num_centroids,
            num_probe=num_probe,
            num_subquantizers=num_subquantizers,
            bits_per_code=bits_per_code,
            device=torch.device("cpu"),
        )

        num_items_no_oov = num_embeddings_item - (1 if add_oov else 0)
        values = torch.arange(num_items_no_oov, device=torch.device("cpu"))
        kjt = KeyedJaggedTensor(
            keys=two_tower_column_names,
            values=values,
            lengths=torch.tensor([0] * num_items_no_oov + [1] * num_items_no_oov, device=torch.device("cpu")),
        )

        # Get the embeddings of the item(movie) tower by querying model
        with torch.no_grad():
            lookups = two_tower_cpu.ebc(kjt)[two_tower_column_names[1]]
            item_embeddings = two_tower_cpu.candidate_proj(lookups)
        index.train(item_embeddings)
        index.add(item_embeddings)

        if save_dir is not None:
            save_dir = save_dir.rstrip("/")
            quant_model = trec_infer.modules.quantize_embeddings(
                model, dtype=torch.qint8, inplace=True
            )
            torch.save(quant_model.state_dict(), f"{save_dir}/model.pt")
            # pyre-ignore[16]
            faiss.write_index(index, f"{save_dir}/faiss.index")


if __name__ == "__main__":
    main()
