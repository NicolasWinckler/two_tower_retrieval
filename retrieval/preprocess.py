# Build contiguous maps: preprocessing required in full catalog mode (take into account non interacted user/item via additional features)
import pandas as pd
import json
import argparse
from pathlib import Path

def existing_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"'{p}' is not an existing directory")
    return p

def parse_args():
    parser = argparse.ArgumentParser(description="Example that takes a root directory.")
    parser.add_argument(
        "-i", "--root-dir",
        type=existing_dir,
        default=Path.cwd(),
        help="Root directory (must exist). Defaults to current working directory."
    )
    parser.add_argument(
        "-o", "--out-dir",
        default=Path.cwd(),
        help="Output directory. Defaults to current working directory."
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    print(f"Using root dir: {args.root_dir}")
    data_path = args.root_dir
    save_dir = args.out_dir

    users = pd.read_csv(f"{data_path}/Users.csv", usecols=["User-ID"])
    books = pd.read_csv(f"{data_path}/Books.csv", usecols=["ISBN"])

    user2ix = {u:i for i,u in enumerate(users["User-ID"].astype(str).unique())}
    isbn2ix = {b:i for i,b in enumerate(books["ISBN"].astype(str).unique())}
    
    num_embeddings_user = len(user2ix)
    num_embeddings_item = len(isbn2ix)
    output = {"user2ix": user2ix, "isbn2ix": isbn2ix, "num_embeddings_user": num_embeddings_user, "num_embeddings_item": num_embeddings_item}

    # Save for reuse (and serving)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/id_maps.json","w") as f:
        json.dump(output, f)
