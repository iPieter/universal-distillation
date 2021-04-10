from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from transformers.tokenization_utils_base import BatchEncoding
import math
from typing import Union, Collection, Optional
import itertools

logger = logging.getLogger("dataloader")


class JITTokenizedDataset(Dataset):
    """
    Pytorch Dataset that tokenizes a textual dataset just in time (JIT).

    With HuggingFace's fast tokenizers, this should not be an issue on a reasonably fast CPU.

    For Universal Distillation, multiple tokenizations are required and the results are aligned.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: Optional[Collection[PreTrainedTokenizerBase]] = None,
    ):
        """
        Create a Dataset with one or more tokenizers.

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

        self.tokenizer = tokenizer
        
        self.tokenizer.model_max_length = 512
        self.mlm_mask_prop = 0.15

        self.token_probs = torch.ones(self.tokenizer.vocab_size)
        self.token_probs[self.tokenizer.pad_token_id] = 0

        if teacher_tokenizers:
            logger.info(f"Found {len(teacher_tokenizers)} teacher tokenizers.")
            self.teacher_tokenizers = teacher_tokenizers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def batch_sequences(self, batch):
        """
        Create tokenized sequences from an array of text sequences.

        Args:
            batch: A collection of text sentences or a single sentence.
        """
        if type(batch) == str:
            batch = [batch]

        output: BatchEncoding = self.tokenizer.batch_encode_plus(
            batch, padding=True, truncation=True, return_tensors="pt"
        )

        # TODO more efficient implementation
        output["lengths"] = torch.tensor(
            [
                len(x)
                for x in self.tokenizer.batch_encode_plus(
                    batch, truncation=True
                ).input_ids
            ],
            dtype=torch.long,
        )

        return self._mlm_objective(output)

    
    def _align_tokens(sentence, target_tokenizer, tokenizer2):
        """
        Function that aligns the tokens of two tokenizers. One is considered the target tokenizer.
        """
        lower_caseing = target_tokenizer.do_lower_case or tokenizer2.do_lower_case
        if lower_caseing:
            print("At least one tokenizer is uncased, continuing with uncased alignment.")
        
        aligned_tokens = []
        
        target_tokens = iter(target_tokenizer.encode(sentence))
        source_tokens = iter(tokenizer2.encode(sentence))
        
        source_underscore = target_tokenizer.convert_tokens_to_ids("_")
        target_underscore = tokenizer2.convert_tokens_to_ids("_")
        
        for token1, token2 in list(itertools.zip_longest(target_tokens, source_tokens)):
            token1 = token1 if token1 else target_tokenizer.pad_token_id
            token2 = token2 if token2 else tokenizer2.pad_token_id
            
            t1, t2 = target_tokenizer.decode([source_underscore, token1]).replace("_","").strip(), tokenizer2.decode([target_underscore, token2]).replace("_","").strip()
            
            if t1.lower() == t2.lower() if lower_caseing else t1 == t2:
                # Tokens match, add them
                aligned_tokens.append([t1, t2])
            elif t1 in [target_tokenizer.special_tokens_map_extended[t] for t in target_tokenizer.special_tokens_map_extended]:
                aligned_tokens.append([t1, t2])
            else:
                # Tokens don't match, build sequences from left and right tokens until they do
                # starting with shortest sequence
                if len(t1) > len(t2):
                    t1 += next(target_tokens)
                else:
                    t2 += next(source_tokens)

                aligned_tokens.append([t1, "Not matched", t2])


                pass
        return aligned_tokens
        

    def _mlm_objective(self, batch):
        """
        Prepare the batch: from the token_ids and the lenghts, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch.input_ids, batch.lengths
        # token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        # x_prob = self.token_probs[token_ids.flatten().long()]

        x_prob = self.token_probs[token_ids.flatten().long()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.tokenizer.pad_token_id] = 0

        self.pred_probs = torch.tensor(
            [0.8000, 0.1000, 0.1000], device=token_ids.device
        )  # TODO parametrize

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.tokenizer.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.tokenizer.mask_token_id)
        probs = torch.multinomial(
            self.pred_probs, len(_token_ids_real), replacement=True
        )
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        token_ids = token_ids.long().masked_scatter(pred_mask.bool(), _token_ids.long())

        mlm_labels[
            ~pred_mask
        ] = (
            -100
        )  # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        # assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return {
            "input_ids": token_ids,
            "attention_mask": batch.attention_mask,
            "labels": mlm_labels,
        }
