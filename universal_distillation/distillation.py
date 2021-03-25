from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    random_split,
    RandomSampler,
    BatchSampler,
    DistributedSampler,
)
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import logging
import logging.config
import yaml
from os import cpu_count
from typing import Optional

from modules.base import BaseTransformer
from data.jit_data_module import JITDataModule

from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
)

with open("logging.yaml", "rt") as f:
    config = yaml.safe_load(f.read())
    f.close()

# logging.config.dictConfig(config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-10s - %(levelname)-5s - %(message)s",
)
logger = logging.getLogger(__name__)

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaseTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_module = JITDataModule(
        file_path="/cw/dtaijupiter/NoCsBack/dtai/pieterd/projects/fair-distillation/data/oscar_dutch/nl_dedup_tiny.txt",
        tokenizer="pdelobelle/robbert-v2-dutch-base",
    )

    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # dataset_train, dataset_val = random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)])

   
    # val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)
    # for batch in train_loader:
    #    print(batch)
    #    print(len(batch.input_ids))
    #    break
    # ------------
    # model
    # ------------
    model = BaseTransformer("pdelobelle/robbert-v2-dutch-base", **vars(args))

    # ------------
    # training
    # ------------
    logger = TensorBoardLogger("tb_logs", name="my_model")

    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data_module)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
