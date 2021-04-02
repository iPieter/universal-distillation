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
from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
)



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
        accumulate_grad_batches: int = 1,
        max_epochs: int = 3,
        eval_splits: Optional[list] = None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        self.config.num_hidden_layers = 6
        self.student = AutoModelForMaskedLM.from_config(self.config)
        #self.student.resize_token_embeddings(40000)

        self.teacher = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        #self.teacher.resize_token_embeddings(40000)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")

        self.temperature = 2.0
        self.alpha_ce = 5.0
        self.alpha_mlm = 2.0
        self.alpha_cos = 1.0

    def forward(self, **inputs):
        return self.student(**inputs)

    def training_step(self, batch, batch_idx):
        # print(batch)
        loss_mlm, student_logits = self(**batch, return_dict=False)

        with torch.no_grad():
            teacher_logits = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            ).logits

        mask = batch["attention_mask"].bool().unsqueeze(-1).expand_as(student_logits)
        student_logits_slct = torch.masked_select(student_logits, mask)
        student_logits_slct = student_logits_slct.view(-1, student_logits.size(-1))
        teacher_logits_slct = torch.masked_select(teacher_logits, mask)
        teacher_logits_slct = teacher_logits_slct.view(-1, student_logits.size(-1))
        assert teacher_logits_slct.size() == student_logits_slct.size()

        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(student_logits_slct / self.temperature, dim=-1),
                F.softmax(teacher_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce + self.alpha_mlm * loss_mlm

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        return {"loss": val_loss}

    def validation_epoch_end(self, outputs):
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(
            self.metric.compute(predictions=preds, references=labels), prog_bar=True
        )
        return loss

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (
                    len(train_loader.dataset) // (self.hparams.train_batch_size)
                )  # * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.student
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = {
            "scheduler": LambdaLR(
                optimizer,
                lr_lambda=LRPolicy(self.hparams.warmup_steps, self.total_steps),
            ),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
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