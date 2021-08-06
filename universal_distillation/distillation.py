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
from pytorch_lightning.callbacks import Callback
from typing import Callable

import os

from universal_distillation.modules.base import BaseTransformer
from universal_distillation.data.jit import JITDataModule

from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
)
from pytorch_lightning.plugins import DDPPlugin

# logging.config.dictConfig(config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)-10s - %(levelname)-5s - %(message)s",
)
logger = logging.getLogger(__name__)

class CommitCallback(Callback):
    def __init__(self, path: str, save_checkpoint: Callable) -> None:
        super().__init__()
        self.path = path
        self.save_checkpoint = save_checkpoint


    def on_validation_end(self, trainer, pl_module):
        logger.info('Commit this data')
        os.system(f"cd {self.path}; git add tb_logs; git commit -m 'Logging of epoch {trainer.current_epoch} step {trainer.global_step}'; git push")
        self.save_checkpoint()        
        

def cli_main():
    """
    Run universal-distillation from the command line.
    """
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--load_counts", type=str, required=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaseTransformer.add_model_specific_args(parser)
    args = parser.parse_args()

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.teacher)

    logger.info(f"Initializing tokenizer {tokenizer}")

    data_module = JITDataModule(
        train_path=args.data,
        val_path=args.val_data,
        tokenizer=tokenizer,
    )

    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # dataset_train, dataset_val = random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)])

    #constraints = [[2016, 2002]]  # she  # he

    #model = BaseTransformer(args.teacher, constraints=constraints, **vars(args))
    model = BaseTransformer(args.teacher, **vars(args))

    # ------------
    # training
    # ------------
    tb_logger = TensorBoardLogger(args.save_dir, name="tb_logs", default_hp_metric=False)

    tokenizer.save_pretrained(args.save_dir)


    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        # accelerator="ddp",
        # plugins=[DDPPlugin(find_unused_parameters=False)],
        # profiler="simple",
        callbacks=[CommitCallback(args.save_dir, lambda model=model : model.student.save_pretrained(args.save_dir))],
        checkpoint_callback=False
    )
    trainer.fit(model, data_module)

    model.student.save_pretrained(args.save_dir)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
