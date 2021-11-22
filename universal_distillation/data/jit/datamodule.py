from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerBase
from typing import Optional
from .dataloader import JITTokenizedDataset


class JITDataModule(LightningDataModule):
    """Data module that uses the tokenizer directly on a file."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
    ):
        """Create a JITDataModule with a tokenizer and a file path."""
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tokenizer = tokenizer

    def train_dataloader(self):
        train_split = JITTokenizedDataset(
            file_path=self.train_path,
            tokenizer=self.tokenizer,
        )
        return DataLoader(
            train_split,
            batch_size=6,
            collate_fn=train_split.batch_sequences,
            pin_memory=True,
            # num_workers=40
        )

    def val_dataloader(self):
        val_split = JITTokenizedDataset(
            file_path=self.val_path,
            tokenizer=self.tokenizer,
        )
        return DataLoader(
            val_split,
            batch_size=1,
            pin_memory=True,
            collate_fn=val_split.prepare_ppll
            # num_workers=40
        )

    def test_dataloader(self):
        test_split = JITTokenizedDataset(
            file_path=self.test_path,
            tokenizer=self.tokenizer,
        )
        return DataLoader(
            test_split,
            batch_size=1,
            pin_memory=True,
            collate_fn=test_split.prepare_ppll
            # num_workers=40
        )