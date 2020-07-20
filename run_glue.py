import argparse
from utils import *
from dataset import *
from models import *
from optim import *
from train import *
from glue import *
import torch_optimizer
import torch.optim as optim
import warnings
import apex

def main(args):

    train_cfg = config_from_json(args.train_cfg)
    model_cfg = config_from_json(args.model_cfg)
    model_cfg.block_size = model_cfg.max_len // model_cfg.n_blocks
    set_seeds(train_cfg.seed)

    if model_cfg.projection not in ["dense", "cnn"]:
        if args.max_len == 0:
            model_cfg.reduced_max_len = model_cfg.max_len
        else:
            model_cfg.reduced_max_len = args.max_len
        if args.reduce_block_size:
            assert model_cfg.reduced_max_len % model_cfg.n_blocks == 0, "Reduced len cannot be divided by n_blocks"
            model_cfg.block_size = model_cfg.reduced_max_len // model_cfg.n_blocks
        else:
            assert model_cfg.reduced_max_len % model_cfg.block_size == 0, "Reduced len cannot be divided by initial block_size"
            model_cfg.n_blocks = model_cfg.reduced_max_len // model_cfg.block_size
        print("max_len:", model_cfg.reduced_max_len, "block_size:", model_cfg.block_size, "n_blocks:", model_cfg.n_blocks)
    else:
        if args.max_len != 0:
            warnings.warn("Projection is incompatible with a reduced max len, using default max_len")

    
    print("Loading dataset")
    #data_file = get_filename(args.data_file, train_cfg.task)
    (data, labels), criterion = get_data_and_optimizer_from_dataset(args.data_file, train_cfg.task)

    loader = GlueDataset(data, labels, train_cfg, model_cfg)
    model = BertInnerForSequenceClassification(model_cfg, loader.get_n_labels(), criterion)

    if train_cfg.optimizer == "lamb":
        if train_cfg.opt_level != "" and train_cfg.opt_level is not None:
            optimizer = apex.optimizers.FusedLAMB(model.parameters(), **train_cfg.optimizer_parameters)
        else:
            optimizer = torch_optimizer.Lamb(model.parameters(), **train_cfg.optimizer_parameters)

    elif train_cfg.optimizer == "radam":
        optimizer = torch_optimizer.RAdam(model.parameters(), **train_cfg.optimizer_parameters)
    elif train_cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), **train_cfg.optimizer_parameters)
    else:
        optimizer = optim4GPU(train_cfg, model)

    trainer = GlueTrainer(loader, model, optimizer, args.save_dir, get_device(), train_cfg.parallel)

    if args.load_model != "":
        print("Loading checkpoint")
        trainer.load_model(args.load_model, args.load_dataset_state)

    if not args.eval:
        trainer.train(train_cfg)
    else:
        trainer.eval(train_cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Model')
    parser.add_argument('--data_file', type=str, default='./glue_data/MRPC/dev.tsv')

    parser.add_argument('--train_cfg', type=str, default='./config/train_glue.json')
    parser.add_argument('--model_cfg', type=str, default='./config/model.json')

    parser.add_argument('--save_dir', type=str, default='./saved_models/bert_inner_eval')
    parser.add_argument('--load_model', type=str, default='./saved_models/bert_inner_steps_180000')
    parser.add_argument('--load_dataset_state', action='store_true')
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--reduce_block_size', action='store_true')

    parser.add_argument('--log_dir', type=str, default='./log')

    args = parser.parse_args()
    main(args=args)
