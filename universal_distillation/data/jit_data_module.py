from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .jit_dataloader import JITTokenizedDataset


class JITDataModule(LightningDataModule):
    def __init__(self, file_path, tokenizer):
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
            pin_memory=True,
            num_workers=40
        )
