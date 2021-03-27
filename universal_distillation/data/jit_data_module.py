from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerBase

from .jit_dataloader import JITTokenizedDataset


class JITDataModule(LightningDataModule):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer

    def train_dataloader(self):
        train_split = JITTokenizedDataset(
            file_path=self.file_path,
            tokenizer=self.tokenizer,
        )
        return DataLoader(
            train_split,
            batch_size=6,
            collate_fn=train_split.batch_sequences,
            pin_memory=True,
            #num_workers=40
        )
