import numpy as np
import torch
import torch.nn as nn
import argparse

from utils import *
from dataset import *
from models import *
from optim import *
from train import *

def main(args):

    train_cfg = config_from_json(args.train_cfg)
    model_cfg = config_from_json(args.model_cfg)

    set_seeds(train_cfg.seed)

    print("Loading dataset")
    loader = PreTrainDataset(args.data_file, train_cfg, model_cfg)
    model = BertInner(model_cfg)

    if train_cfg.optimizer == "lamb":
        optimizer = Lamb(model.parameters(), lr=train_cfg.lr)
    else:
        optimizer = optim4GPU(train_cfg, model)

    trainer = Trainer(loader, model, optimizer, args.save_dir, get_device(), train_cfg.parallele)

    if args.load_dir != "":
        print("Loading checkpoint")
        trainer.load_model(args.load_dir, args.load_dataset_state)

    trainer.train(train_cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ALBERT Language Model')
    parser.add_argument('--data_file', type=str, default='./data/sample.txt')
    parser.add_argument('--train_cfg', type=str, default='./config/train.json')
    parser.add_argument('--model_cfg', type=str, default='./config/model.json')

    parser.add_argument('--save_dir', type=str, default='./saved_models/bert_inner')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--load_dataset_state', action='store_true')

    parser.add_argument('--log_dir', type=str, default='./log')

    args = parser.parse_args()
    main(args=args)
