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
from typing import Optional, List
from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
)


class LRPolicy:
    """
    Pickable learning rate policy.

    When using multiple GPU's or nodes, communication uses pickle and the default
    transformers learning rate policy with warmup uses a non-pickable lambda.
    """
    def __init__(self, num_warmup_steps, num_training_steps):
        """
        Initialize pickable learning rate policy.

        Args:
            num_warmup_steps: Number of training steps used as warmup steps
            num_training_steps: Total number of training steps
        """
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
    """
    Base distillation model.
    """
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
        temperature: float = 2.0,
        alpha_teacher_mlm: float = 5.0,
        alpha_mlm: float = 2.0,
        alpha_hiddden: float = 1.0,
        constraints = None,
        **kwargs
    ):
        """
        Constructor for a base distillation model.

        Args:
            model_name_or_path: name or path of the model, follows Transformers identifiers.
            learning_rate: Maximum learning rate.
            adam_epsilon: Epsilon hyperparameter for Adam.
            warmup_steps: Number of warmup batches.
            weight_decay: Weight decay hyperparameter
            train_batch_size: Training batch size per GPU.
            eval_batch_size: Evaluation batch size per GPU.
            accumulate_grad_batches: Gradient accumulation multiplier.
            max_epochs: Number of epochs.
            temperature: Knowledge distillation hyperparameter.
            alpha_teacher_mlm: Weight of the MLM distillation loss.
            alpha_mlm: Weight of the regular MLM loss, i.e. pretraining.
            alpha_hiddden: Weight of the hidden state distillation loss.
        """
        super().__init__()

        self.save_hyperparameters()

        self.config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        self.config.num_hidden_layers = 6
        self.student = AutoModelForMaskedLM.from_config(self.config)
        self.student.resize_token_embeddings(40000)

        self.teacher = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.teacher.resize_token_embeddings(40000)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        self.temperature = temperature
        self.alpha_teacher_mlm = alpha_teacher_mlm
        self.alpha_mlm = alpha_mlm
        self.alpha_hiddden = alpha_hiddden

        self.contraints = constraints

    def forward(self, **inputs):
        return self.student(**inputs)

    def training_step(self, batch, batch_idx):
        # print(batch)
        loss_mlm, student_logits, student_hidden_states = self(**batch, return_dict=False, output_hidden_states=True)

        with torch.no_grad():
            teacher_logits, teacher_hidden_states = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=False,
                output_hidden_states=True
            )
        
        for constraint in self.constraints:
            tmp = (teacher_logits[:, :, constraint[0]] + teacher_logits[:, :, constraint[1]])/2
            teacher_logits[:, :, constraint[0]] = tmp
            teacher_logits[:, :, constraint[1]] = tmp

        mask = batch["attention_mask"].bool().unsqueeze(-1).expand_as(student_logits)
        student_logits_slct = torch.masked_select(student_logits, mask)
        student_logits_slct = student_logits_slct.view(-1, student_logits.size(-1))
        teacher_logits_slct = torch.masked_select(teacher_logits, mask)
        teacher_logits_slct = teacher_logits_slct.view(-1, student_logits.size(-1))
        assert teacher_logits_slct.size() == student_logits_slct.size()

        loss_teacher_mlm = (
            self.ce_loss_fct(
                F.log_softmax(student_logits_slct / self.temperature, dim=-1),
                F.softmax(teacher_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_teacher_mlm * loss_teacher_mlm + self.alpha_mlm * loss_mlm

        if self.alpha_hiddden > 0.0:
            student_hidden_states = student_hidden_states[-1]  # (bs, seq_length, dim)
            teacher_hidden_states = teacher_hidden_states[-1]  # (bs, seq_length, dim)
            mask = batch["attention_mask"].bool().unsqueeze(-1).expand_as(student_hidden_states)  # (bs, seq_length, dim)
            assert student_hidden_states.size() == teacher_hidden_states.size()
            dim = student_hidden_states.size(-1)

            student_hidden_states_slct = torch.masked_select(student_hidden_states, mask)  # (bs * seq_length * dim)
            student_hidden_states_slct = student_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            teacher_hidden_states_slct = torch.masked_select(teacher_hidden_states, mask)  # (bs * seq_length * dim)
            teacher_hidden_states_slct = teacher_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = student_hidden_states_slct.new(student_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_hidden = self.cosine_loss_fct(student_hidden_states_slct, teacher_hidden_states_slct, target)
            loss += self.alpha_hiddden * loss_hidden

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