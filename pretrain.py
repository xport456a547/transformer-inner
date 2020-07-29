import argparse
from utils import *
from dataset import *
from models import *
from optim import *
from train import *
import torch_optimizer
import apex

def main(args):

    train_cfg = config_from_json(args.train_cfg)
    model_cfg = config_from_json(args.model_cfg)
    model_cfg.block_size = model_cfg.max_len // model_cfg.n_blocks

    eval_config = Namespace(**vars(train_cfg))
    eval_config.autoregressive = True
    eval_config.randomize_data = False

    set_seeds(train_cfg.seed)


    print('train',train_cfg)
    print('eval', eval_config)
    print('model', model_cfg)

    print("Loading datasets")
    loader_train = PreTrainDataset(args.data_train, train_cfg, model_cfg)
    loader_eval  = PreTrainDataset(args.data_valid, eval_config, model_cfg)

    model = BertInnerForMaskedLM(model_cfg)

    if train_cfg.optimizer == "lamb":
        if train_cfg.opt_level != "" and train_cfg.opt_level is not None:
            optimizer = apex.optimizers.FusedLAMB(model.parameters(), **train_cfg.optimizer_parameters)
        else:
            optimizer = torch_optimizer.Lamb(model.parameters(), **train_cfg.optimizer_parameters)

    elif train_cfg.optimizer == "radam":
        optimizer = torch_optimizer.RAdam(model.parameters(), **train_cfg.optimizer_parameters)
    else:
        optimizer = optim4GPU(train_cfg, model)

    trainer = Trainer(loader_train, model, optimizer, args.save_dir, get_device(), train_cfg.parallel, train_cfg.opt_level)

    if args.load_model != "":
        print("Loading checkpoint")
        trainer.load_model(args.load_model, args.load_dataset_state)

    trainer.train(train_cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Model')
    parser.add_argument('--data_train', type=str, default='./data/sample.txt')
    parser.add_argument('--data_valid', type=str, default='./data/sample.txt')

    parser.add_argument('--train_cfg', type=str, default='./config/train.json')
    parser.add_argument('--model_cfg', type=str, default='./config/model.json')

    parser.add_argument('--save_dir', type=str, default='./saved_models/bert_inner')
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--load_dataset_state', action='store_true')

    parser.add_argument('--log_dir', type=str, default='./log')

    args = parser.parse_args()


    '''
    args.data_train = '/data/xp/transformer_inner/test.data'
    args.data_train = '/data/xp/transformer_inner/test.data'
    args.data_valid = '/data/xp/transformer_inner/test.data'
    args.data_train = '/data/xp/transformer_inner/text8/text8.train.modif'
    args.data_valid = '/data/xp/transformer_inner/text8/text8.valid.modif.tiny'
    '''
    main(args=args)