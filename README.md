# TorchRec retrieval example for the Kaggle's Book dataset (work in progress)

This example is the Two Tower model for recommendation from the [TorchRec](https://github.com/pytorch/torchrec) example, adapted for the [book recommendation dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data).


## Installation

Edit the .env file and set your environment variable. Eventually, edit the docker-compose.yaml to set the volumes and ports. Then run the following commands:
```bash
docker compose build
docker compose up
```

## How to run?
### Prepare data and environment variables
Download and unzip the Book dataset. Set input and output via environment variables:
```bash
export DATA_PATH=/path/to/csv/fles
export SAVE_DIR=/path/to/output
```

Run the preprocessng script only once:
```bash
python3 retrieval/preprocess.py --root-dir ${DATA_PATH} --out-dir ${DATA_PATH}
```

### Training
Run the traning script:
```bash
./run_train_baseline.sh
```

### Inference
Run the inference script:
```bash
./run_infer_baseline.sh
```



## Added features
* Docker image with TorchRec and fbgemm-gpu and other dependencies
* Torch dataset and dataloader for the book recommendation dataset
* Incorporate user and item metadata (not yet tested in training/inference pipeline)
* handle partial data and metadata
* Adapt training and inference pipeline
* Add CLI to training and inference
* Currently work on single process, network on gpu and faiss on cpu

## TODO
* Add evaluation pipeline and metrics
* Add a ranking model on top of retrieved candidate
* Test pipeline with additional user/item metadata
* Test distributed training and sharding
* Support faiss on gpu
* Add text encoder for book title
* Retrieve book covers and add an image encoder for additional metada
* Clean and simplify workflow