from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, RandomSampler, BatchSampler

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups

from jit_dataloader import JITTokenizedDataset
import logging
import logging.config

from typing import Optional

import yaml

from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
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

class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=self.config)
        self.model.resize_token_embeddings(40000)
        

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        #print(batch)
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        return {'loss': val_loss}

    def validation_epoch_end(self, outputs):
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        return loss

    def setup(self, stage):
        if stage == 'fit':
            # Gegt dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.train_batch_size)) #* max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseTransformer")        
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
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
    #dataset_train, dataset_val = random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)])

    sampler = RandomSampler(dataset)

    # groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
    sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)

    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=dataset.batch_sequences,
    )
    # val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)
    #for batch in train_loader:
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
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
