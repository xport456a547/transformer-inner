from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer
from tokenizers import CharBPETokenizer, SentencePieceBPETokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import random
from random import randint
import torch
import re

class PreTrainDataset(object):

    def __init__(self, path, train_cfg, model_cfg):

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_cfg.tokenizer_prefix)
        except:
            print("Loading custom tokenizer")
            self.tokenizer = self.load_custom_tokenizer(
                model_cfg.tokenizer_prefix)

        model_cfg.vocab_size = len(self.tokenizer)
        self.mask_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)
        
        self.data = open(path, "r", encoding="utf-8").read()
        self.data = re.sub(r" ?\n ?", r"\n", self.data)
        self.data = self.data.split("\n\n")

        # Exclude titles
        #self.data = [d.strip() for d in self.data if len(d) > 0 and d.strip()[0] != "="]

        self.batch_size = train_cfg.batch_size
        self.dataset_size = len(self.data)

        self.max_len = model_cfg.max_len
        self.keep_prob = train_cfg.keep_prob

        self.max_masked_words = round(train_cfg.mask_prob * model_cfg.max_len)
        self.mask_masked_tokens_in_attn = train_cfg.mask_masked_tokens_in_attn

        self.is_pretokenized = model_cfg.is_pretokenized

        self.step = 0
        self.dataset_indexes = [i for i in range(self.dataset_size)]
        self.reset_epoch()

    def __iter__(self):

        while self.step < self.dataset_size - torch.cuda.device_count():

            data = self.data[self.step: self.step + self.batch_size]
            data = [d.split("\n") for d in data]
            data = [" ".join(d[randint(0, len(d) - 1):]).strip() for d in data]

            batch_size = len(data)
            if batch_size % torch.cuda.device_count() != 0:
                data = data[: -batch_size % torch.cuda.device_count()]

            batch_size = len(data)

            # See https://huggingface.co/transformers/preprocessing.html
            if self.is_pretokenized:
                data = [d.split() for d in data]

            data = self.tokenizer(data, max_length=self.max_len,
                                  padding='max_length', truncation=True, return_tensors="pt", is_pretokenized=self.is_pretokenized)

            label_mask = torch.zeros(batch_size, self.max_len).float()
            for i in range(batch_size):
                indices = torch.randperm(self.max_len)[:self.max_masked_words]
                label_mask[i, indices] += 1.

            label_mask *= data["attention_mask"]
            keep_mask = label_mask * \
                torch.bernoulli(torch.ones_like(label_mask) - self.keep_prob)

            input_ids = data["input_ids"] * \
                (1 - keep_mask) + keep_mask * self.mask_id
            label = data["input_ids"] * label_mask
            attn_mask = data["attention_mask"].float()

            if self.mask_masked_tokens_in_attn:
                attn_mask *= (1. - keep_mask)

            self.step += batch_size

            yield input_ids.long(), attn_mask.float(), label.long(), label_mask

    def reset_epoch(self):

        print("Shuffling Dataset")
        self.step = 0
        random.shuffle(self.dataset_indexes)
        self.data = [self.data[i] for i in self.dataset_indexes]

    def get_dataset_state(self):

        return self.step, self.dataset_indexes

    def set_dataset_state(self, step, dataset_indexes):

        self.step = step
        self.dataset_indexes = dataset_indexes
        self.data = [self.data[i] for i in dataset_indexes]

    def load_custom_tokenizer(self, path):
        tokenizer = ByteLevelBPETokenizer(
            path + "-vocab.json", path + "-merges.txt")
        # Add preprocessing tokens like Roberta
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        return PreTrainedTokenizerFast(tokenizer, pad_token="<pad>", mask_token="<mask>", unk_token="<unk>", bos_token="<s>", eos_token="</s>")


class GlueDataset(object):

    def __init__(self, data, labels, train_cfg, model_cfg):

        assert type(data) == list or type(
            data) == tuple, "Expect a list of sentences or a tuple of lists"
        assert type(labels) == list, "Expect a list of labels"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_cfg.tokenizer_prefix)
        except:
            print("Loading custom tokenizer")
            self.tokenizer = self.load_custom_tokenizer(
                model_cfg.tokenizer_prefix)

        self.data = data
        self.labels = labels

        model_cfg.vocab_size = len(self.tokenizer)
        self.batch_size = train_cfg.batch_size
        self.max_len = model_cfg.max_len
        self.reduced_max_len = model_cfg.reduced_max_len

        if isinstance(data, tuple):
            # For successive sentences tasks
            assert len(data) == 2, "Only 2 successive sentences possible"
            self.data = tuple(zip(*data))

        self.dataset_size = len(self.data)

        self.step = 0
        self.dataset_indexes = [i for i in range(self.dataset_size)]
        # self.reset_epoch()

    def __iter__(self):

        while self.step < self.dataset_size - torch.cuda.device_count():

            data = self.data[self.step: self.step + self.batch_size]
            label = torch.tensor(
                self.labels[self.step: self.step + self.batch_size])

            batch_size = len(data)
            if batch_size % torch.cuda.device_count() != 0:
                data = data[: -batch_size % torch.cuda.device_count()]
                label = label[: -batch_size % torch.cuda.device_count()]

            batch_size = len(data)

            if isinstance(data[0], tuple):
                data = list(zip(*data))
                data = self.tokenizer(data[0], data[1], max_length=self.reduced_max_len,
                                      padding='max_length', truncation=True, return_tensors="pt")
            else:
                data = self.tokenizer(
                    data, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

            input_ids = data["input_ids"]
            attn_mask = data["attention_mask"].float()

            self.step += batch_size
            yield input_ids.long(), attn_mask.float(), label.long()

    def reset_epoch(self):

        print("Shuffling Dataset")
        self.step = 0
        random.shuffle(self.dataset_indexes)
        self.data = [self.data[i] for i in self.dataset_indexes]
        self.labels = [self.labels[i] for i in self.dataset_indexes]

    def get_dataset_state(self):

        return self.step, self.dataset_indexes

    def set_dataset_state(self, step, dataset_indexes):

        self.step = step
        self.dataset_indexes = dataset_indexes
        self.data = [self.data[i] for i in dataset_indexes]
        self.labels = [self.labels[i] for i in self.dataset_indexes]

    def get_n_labels(self):
        return len(list(set(self.labels)))

    def load_custom_tokenizer(self, path):
        tokenizer = ByteLevelBPETokenizer(
            path + "-vocab.json", path + "-merges.txt")
        # Add preprocessing tokens like Roberta
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        return PreTrainedTokenizerFast(tokenizer, pad_token="<pad>", mask_token="<mask>", unk_token="<unk>", bos_token="<s>", eos_token="</s>")
