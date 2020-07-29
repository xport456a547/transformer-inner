import argparse
from utils import *
from dataset import *
from models import *
from optim import *
from train import *
import torch_optimizer
import apex
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader


class PreTrainDatasetTFW(object):
    """
        Wrapper for Transformers DataLoader
    """

    def __init__(self, path, tokenizer, train_cfg, model_cfg, is_train):
        print("Loading dataset")
        print(path, "tok.max_len:", model_cfg.max_len)
        self.tokenizer = tokenizer
        self.train_cfg = train_cfg
        self._dataset = TextDataset(tokenizer=tokenizer, file_path=path, block_size=model_cfg.max_len)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=self.train_cfg.mask_prob)
        self.is_train = is_train
        if not self.is_train:
            self.sampler = RandomSampler(self._dataset)
            self.mask_keep_prob = self.train_cfg.keep_prob
        else:
            self.sampler = SequentialSampler(self._dataset)
            self.mask_keep_prob = 1 # no unmasking considered during validation & testing

        self.dataloader = DataLoader(
            self._dataset,
            batch_size=train_cfg.batch_size,
            sampler=self.sampler,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def get_batch_data(self, inputs_):
        """
        Build the datastructures used for training
        :param inputs_:
        :return: inputs, attn_mask, labels, labels_mask
        inputs[i] tokenized version of string i with masked token
        attn_mask[i] vector of boolean values with 1 when the corresponding token id should be taken into account
        labels[i] tokenized version of string i without masked token
        labels_mask[i] boolean vec: 1 when the corresponding label id counts for loss computation
        """

        attn_mask = torch.ones(inputs_['input_ids'].shape).float()
        attn_mask[inputs_['input_ids'] == self.tokenizer.pad_token_id] = 0.
        attn_mask[inputs_['input_ids'] == self.tokenizer.sep_token_id] = 0.

        inputs_['labels'][inputs_['input_ids'] != self.tokenizer.mask_token_id] = 0
        labels_mask = torch.zeros(inputs_['input_ids'].shape).float()
        labels_mask[inputs_['input_ids'] == self.tokenizer.mask_token_id] = 1

        keep_mask = labels_mask * torch.bernoulli(torch.ones_like(attn_mask) - self.mask_keep_prob)
        unmask = (labels_mask - keep_mask).long()

        '''
        print("inputs", inputs_['input_ids'][0])
        print("labels_mask", labels_mask[0])
        print("unmask", unmask[0])
        '''

        inputs_['input_ids'] = inputs_['input_ids'] * (1 - unmask) + inputs_['labels'] * unmask

        if self.train_cfg.mask_masked_tokens_in_attn:
            attn_mask *= (1. - keep_mask)

        '''
        print(self.tokenizer.convert_ids_to_tokens(inputs_['input_ids'][0]))
        print("inputs", inputs_['input_ids'][0])
        print("attn mask", attn_mask[0])
        print("labels", inputs_['labels'][0])
        print("label mask", labels_mask[0])
        exit()
        '''

        return inputs_['input_ids'].long(), attn_mask.float(), inputs_['labels'].long(), labels_mask

    def __len__(self):
        return int(len(self._dataset) / self.train_cfg.batch_size)

    def __iter__(self):
        for step, inputs_ in enumerate(self.dataloader):
            inputs, attn_mask, labels, labels_mask = self.get_batch_data(inputs_)
            yield inputs, attn_mask, labels, labels_mask

    def reset_epoch(self):
        return

    def get_dataset_state(self):
        return 0

    def set_dataset_state(self, step, dataset_indexes):
        print('[Warning] set_dataset_state unsupported with PreTrainDatasetTFW')
        return 0


def load_custom_tokenizer(path):
    tokenizer = ByteLevelBPETokenizer(path + "-vocab.json", path + "-merges.txt")
    # Add preprocessing tokens like Roberta
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    return PreTrainedTokenizerFast(tokenizer, pad_token="<pad>", mask_token="<mask>", unk_token="<unk>",
                                   bos_token="<s>", eos_token="</s>")


def main(args):
    train_cfg = config_from_json(args.train_cfg)
    model_cfg = config_from_json(args.model_cfg)
    model_cfg.block_size = model_cfg.max_len // model_cfg.n_blocks

    eval_config = Namespace(**vars(train_cfg))
    set_seeds(train_cfg.seed)

    print('train', train_cfg)
    print('eval', eval_config)
    print('model', model_cfg)

    print("Loading Tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_prefix)
    except:
        print("Loading custom tokenizer")
        tokenizer = load_custom_tokenizer(model_cfg.tokenizer_prefix)
    print(tokenizer)
    model_cfg.vocab_size = len(tokenizer)

    loader_train = PreTrainDatasetTFW(args.data_train, tokenizer, train_cfg, model_cfg, is_train=True)
    loader_eval  = PreTrainDatasetTFW(args.data_valid, tokenizer, train_cfg, model_cfg, is_train=False)

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

    trainer = Trainer(loader_train, loader_eval, model, optimizer, args.save_dir, get_device(), train_cfg.parallel,
                      train_cfg.opt_level)

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
    args.data_valid = '/data/xp/transformer_inner/test.data'
    args.data_train = '/data/xp/transformer_inner/test.data'
    args.data_valid = '/data/xp/transformer_inner/test.data'

    args.data_train = '/data/xp/transformer_inner/text8/text8.train.modif'
    args.data_valid = '/data/xp/transformer_inner/text8/text8.valid.modif.tiny'
    args.data_train = '/data/nlp/wikitext-103-raw/wiki.test.raw'
   '''

    main(args)
