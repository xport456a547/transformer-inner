from transformers import AutoTokenizer
import random
from random import randint
import torch

class PreTrainDataset(object):
    
    def __init__(self, path, train_cfg, model_cfg):

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        self.data = open(path, "r", encoding="utf-8").read().split("\n\n")
        self.batch_size = train_cfg.batch_size

        self.mask_id = model_cfg.mask_id
        self.max_len = model_cfg.max_len

        self.max_masked_words = round(train_cfg.mask_prob * model_cfg.max_len)

        self.dataset_size = len(self.data)

        self.step = 0
        self.dataset_indexes = [i for i in range(self.max_len)]

    def __iter__(self):

        while self.step < self.dataset_size:

            data = self.data[self.step: self.step + self.batch_size]
            data = [d.split("\n") for d in data]
            data = [" ".join(d[randint(0, len(d) - 1):]) for d in data]
  
            
            data = self.tokenizer(data, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

            label_mask = torch.zeros(self.batch_size, self.max_len).float()
            for i in range(self.batch_size):
                indices = torch.randperm(self.max_len)[:self.max_masked_words]
                label_mask[i, indices] += 1.

            label_mask *= data["attention_mask"]
            input_ids = data["input_ids"] * (1 - label_mask) + label_mask * self.mask_id
            label = data["input_ids"] * label_mask

            self.step += self.batch_size

            yield input_ids.long(), data["attention_mask"].float(), label.long(), label_mask

    def reset_epoch(self):

        self.data = zip(self.data, self.dataset_indexes)
        random.shuffle(self.data)
        self.data, self.dataset_indexes = zip(*self.data)

    def get_dataset_state(self):

        return self.step, self.dataset_indexes

    def set_dataset_state(self, step, dataset_indexes):

        self.step = step
        self.dataset_indexes = dataset_indexes
        self.data = [self.data[i] for i in dataset_indexes]
    