import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from apex import amp

class Trainer(object):

    def __init__(self, loader, model, optimizer, save_dir, device, parallel, opt_level=""):

        self.loader = loader
        self.model = model.to(device)
        self.optimizer = optimizer

        self.criterion = nn.CrossEntropyLoss()

        self.save_dir = save_dir
        self.device = device
        self.global_step = 0

        self.opt_level = opt_level

        if opt_level != "" and opt_level is not None:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)
        
        if parallel:
            self.model = nn.DataParallel(self.model)

    def train(self, train_cfg):

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(train_cfg.n_epochs):

            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            acc_sum = 0.

            iter_bar = tqdm(self.loader, desc='Iter (loss=X.XXX)',
                            initial=self.global_step)

            for i, batch in enumerate(iter_bar):
                inputs, attn_mask, labels, labels_mask = batch
                inputs, attn_mask, labels, labels_mask = inputs.to(self.device), attn_mask.to(
                    self.device), labels.to(self.device), labels_mask.to(self.device)

                loss, outputs, labels = self.model(
                    inputs, attn_mask, labels, labels_mask)

                loss = loss.mean() / train_cfg.accumulation_steps

                if self.opt_level != "" and self.opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.global_step % train_cfg.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                acc = self.get_acc(outputs, labels)
                loss_sum += loss.item()*train_cfg.accumulation_steps
                acc_sum += acc

                self.global_step += 1

                iter_bar.set_description('Iter (loss=%5.3f accuracy=%5.3f)' % (loss.item()*train_cfg.accumulation_steps, acc))

                if self.global_step % train_cfg.save_steps == 0:
                    self.save_model()

                if train_cfg.total_steps and train_cfg.total_steps < self.global_step:
                    print('Epoch %d/%d : Average Loss %5.3f' %
                          (epoch+1, train_cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save_model()
                    return

            print('Epoch %d/%d : Average Loss: %5.3f Average Acc: %5.3f' %
                  (epoch+1, train_cfg.n_epochs, loss_sum/(i+1), acc_sum/(i+1)))
            self.loader.reset_epoch()

        self.save_model()

    def save_model(self):

        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dataset_state': self.loader.get_dataset_state(),
        }, self.save_dir + "_steps_" + str(self.global_step))

    def load_model(self, path, load_dataset_state=False):

        checkpoint = torch.load(path)
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])

        checkpoint["optimizer_state_dict"]["param_groups"] = self.optimizer.state_dict()["param_groups"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_dataset_state:
            self.loader.set_dataset_state(*checkpoint["dataset_state"])

    def get_acc(self, outputs, labels):
        return accuracy_score(labels.cpu().detach().numpy(), torch.argmax(outputs, dim=-1).cpu().detach().numpy())


class GlueTrainer(object):

    def __init__(self, loader, model, optimizer, save_dir, device, parallel):

        self.loader = loader
        self.model = model.to(device)
        self.optimizer = optimizer

        self.criterion = nn.CrossEntropyLoss()

        self.save_dir = save_dir
        self.device = device
        self.global_step = 0

        if parallel:
            self.model = nn.DataParallel(self.model)

    def train(self, train_cfg):
        
        self.train_cfg = train_cfg
        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(train_cfg.n_epochs):

            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            acc_sum = 0.

            iter_bar = tqdm(self.loader, desc='Iter (loss=X.XXX)',
                            initial=self.global_step)

            for i, batch in enumerate(iter_bar):
                inputs, attn_mask, labels = batch
                inputs, attn_mask, labels = inputs.to(self.device), attn_mask.to(
                    self.device), labels.to(self.device)

                loss, outputs, labels = self.model(
                    inputs, attn_mask, labels)

                loss = loss.mean() / train_cfg.accumulation_steps
                loss.backward()

                if self.global_step % train_cfg.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                acc = self.get_acc(outputs, labels)
                loss_sum += loss.item() * train_cfg.accumulation_steps
                acc_sum += acc

                self.global_step += 1

                iter_bar.set_description('Iter (loss=%5.3f accuracy=%5.3f)' % (loss.item() * train_cfg.accumulation_steps, acc))

                if self.global_step % train_cfg.save_steps == 0:
                    self.save_model()

                if train_cfg.total_steps and train_cfg.total_steps < self.global_step:
                    print('Epoch %d/%d : Average Loss %5.3f' %
                          (epoch+1, train_cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save_model()
                    return

            print('Epoch %d/%d : Average Loss: %5.3f Average Acc: %5.3f' %
                  (epoch+1, train_cfg.n_epochs, loss_sum/(i+1), acc_sum/(i+1)))
            self.save_model()
            self.loader.reset_epoch()

    def eval(self, train_cfg):
        
        self.train_cfg = train_cfg
        self.model.eval()

        for epoch in range(1):

            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            acc_sum = 0.

            iter_bar = tqdm(self.loader, desc='Iter (loss=X.XXX)',
                            initial=self.global_step)

            for i, batch in enumerate(iter_bar):
                inputs, attn_mask, labels = batch
                inputs, attn_mask, labels = inputs.to(self.device), attn_mask.to(
                    self.device), labels.to(self.device)

                loss, outputs, labels = self.model(
                    inputs, attn_mask, labels)

                acc = self.get_acc(outputs, labels)
                loss = loss.mean()
                loss_sum += loss.item()
                acc_sum += acc

                self.global_step += 1

                iter_bar.set_description('Iter (loss=%5.3f accuracy=%5.3f)' % (loss.item(), acc))

                if train_cfg.total_steps and train_cfg.total_steps < self.global_step:
                    print('Epoch %d/%d : Average Loss: %5.3f Average Acc: %5.3f' % (epoch+1, train_cfg.n_epochs, loss_sum/(i+1), acc_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    return

            print('Epoch %d/%d : Average Loss: %5.3f Average Acc: %5.3f' %
                  (epoch+1, train_cfg.n_epochs, loss_sum/(i+1), acc_sum/(i+1)))

    def save_model(self):

        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dataset_state': self.loader.get_dataset_state(),
        }, self.save_dir + "_" + str(self.train_cfg.task) + "_steps_" + str(self.global_step))

    def load_model(self, path, load_dataset_state=False):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    def get_acc(self, outputs, labels):
        return accuracy_score(labels.cpu().detach().numpy(), torch.argmax(outputs, dim=-1).cpu().detach().numpy())
