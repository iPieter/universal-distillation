from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerBase

from .jit_dataloader import JITTokenizedDataset


class JITDataModule(LightningDataModule):
    """Data module that uses the tokenizer directly on a file."""

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase):
        """Create a JITDataModule with a tokenizer and a file path.
    
        """
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
            #batch_size=6,
            collate_fn=train_split.batch_sequences,
            pin_memory=True,
            # num_workers=40
        )
