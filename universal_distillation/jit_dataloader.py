from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from transformers.tokenization_utils_base import BatchEncoding
logger = logging.getLogger("dataloader")


class JITTokenizedDataset(Dataset):
    """
    Pytorch Dataset that tokenizes a textual dataset just in time (JIT). With HuggingFace's fast tokenizers,
    this should not be an issue on a reasonably fast CPU.

    For Universal Distillation, multiple tokenizations are required and the results are aligned.
    """

    def __init__(self, file_path: str, tokenizer: str):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf8") as fp:
            self.data = fp.readlines()

        logger.info(f"Loaded {len(self.data)} lines")

        logger.info(f"Initializing tokenizer {tokenizer}")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def batch_sequences(self, batch):
        output: BatchEncoding = self.tokenizer.batch_encode_plus(
            batch, padding=True, truncation=True, return_tensors="pt"
        )

        output['lengths'] = torch.Tensor([len(x) for x in self.tokenizer.batch_encode_plus(batch).input_ids], , dtype=torch.long)

        return output
