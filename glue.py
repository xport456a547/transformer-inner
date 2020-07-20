import itertools
import csv
import fire

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import *
from optim import *
from dataset import *

from utils import *

class MRPC(object):
    """ Dataset class for MRPC """
    def __init__(self, file):
        
        f = open(file, 'r', encoding="utf-8")
        self.data = csv.reader(f, delimiter='\t', quotechar=None)
        self.data = [_ for _ in self.data]

    def get_dataset(self):
        labels, text_a, text_b = [], [], []
        for line in itertools.islice(self.data, 1, None):
            # Avoid a crash caused by few bad lines in the dataset
            if len(line) == 5:
                labels.append(int(line[0]))
                text_a.append(line[3])
                text_b.append(line[4])
        return (text_a, text_b), labels

class QNLI(object):
    """ Dataset class for QNLI """
    def __init__(self, file):
        
        f = open(file, 'r', encoding="utf-8")
        self.data = csv.reader(f, delimiter='\t', quotechar=None)
        self.data = [_ for _ in self.data]

    def get_dataset(self):
        labels, text_a, text_b = [], [], []
        for line in itertools.islice(self.data, 1, None):
            # Avoid a crash caused by few bad lines in the dataset
            if len(line) == 4:
                labels.append(1 if line[3] == "entailment" else 0)
                text_a.append(line[1])
                text_b.append(line[2])
        return (text_a, text_b), labels

class QQP(object):
    """ Dataset class for QQP """
    def __init__(self, file):
        
        f = open(file, 'r', encoding="utf-8")
        self.data = csv.reader(f, delimiter='\t', quotechar=None)
        self.data = [_ for _ in self.data]

    def get_dataset(self):
        labels, text_a, text_b = [], [], []
        for line in itertools.islice(self.data, 1, None):
            # Avoid a crash caused by few bad lines in the dataset
            if len(line) == 6:
                labels.append(int(line[5]))
                text_a.append(line[3])
                text_b.append(line[4])
        return (text_a, text_b), labels

class RTE(object):
    """ Dataset class for RTE """
    def __init__(self, file):
        
        f = open(file, 'r', encoding="utf-8")
        self.data = csv.reader(f, delimiter='\t', quotechar=None)
        self.data = [_ for _ in self.data]

    def get_dataset(self):
        labels, text_a, text_b = [], [], []
        for line in itertools.islice(self.data, 1, None):
            # Avoid a crash caused by few bad lines in the dataset
            if len(line) == 4:
                labels.append(1 if line[3] == "entailment" else 0)
                text_a.append(line[1])
                text_b.append(line[2])
        return (text_a, text_b), labels

class SST(object):
    """ Dataset class for SST """
    def __init__(self, file):
        
        f = open(file, 'r', encoding="utf-8")
        self.data = csv.reader(f, delimiter='\t', quotechar=None)
        self.data = [_ for _ in self.data]

    def get_dataset(self):
        labels, text_a = [], []
        for line in itertools.islice(self.data, 1, None):
            labels.append(int(line[1]))
            text_a.append(line[0])
            
        return text_a, labels

class WNLI(object):
    """ Dataset class for WNLI """
    def __init__(self, file):
        
        f = open(file, 'r', encoding="utf-8")
        self.data = csv.reader(f, delimiter='\t', quotechar=None)
        self.data = [_ for _ in self.data]

    def get_dataset(self):
        labels, text_a, text_b = [], [], []
        for line in itertools.islice(self.data, 1, None):
            # Avoid a crash caused by few bad lines in the dataset
            if len(line) == 4:
                labels.append(int(line[3]))
                text_a.append(line[1])
                text_b.append(line[2])
        return (text_a, text_b), labels

def get_data_and_optimizer_from_dataset(data_file, task):
    if task == "mrpc":
        return MRPC(data_file).get_dataset(), nn.CrossEntropyLoss()
    if task == "qnli":
        return QNLI(data_file).get_dataset(), nn.CrossEntropyLoss()
    if task == "qqp":
        return QQP(data_file).get_dataset(), nn.CrossEntropyLoss()
    if task == "rte":
        return RTE(data_file).get_dataset(), nn.CrossEntropyLoss()
    if task == "sst":
        return SST(data_file).get_dataset(), nn.CrossEntropyLoss()
    if task == "wnli":
        return WNLI(data_file).get_dataset(), nn.CrossEntropyLoss()
