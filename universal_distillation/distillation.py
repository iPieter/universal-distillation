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

from jit_dataloader import JITTokenizedDataset
from modules.base import BaseTransformer

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


class LRPolicy(object):
    def __init__(self, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )


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
    dataset = JITTokenizedDataset(
        file_path="/cw/dtaijupiter/NoCsBack/dtai/pieterd/projects/fair-distillation/data/oscar_dutch/nl_dedup_tiny.txt",
        tokenizer="pdelobelle/robbert-v2-dutch-base",
    )

    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # dataset_train, dataset_val = random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)])

    sampler = RandomSampler(dataset)
    # sampler = DistributedSampler(sampler)

    # groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
    sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)

    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=dataset.batch_sequences,
        num_workers=args.num_workers,
    )
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
    trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
