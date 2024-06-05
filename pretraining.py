import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime
import csv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.nt_xent import NTXentLoss, Weight_NTXentLoss
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
NO_WEIGHT_EPOCH_NUMS = 5
apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except Exception as e:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


def write_csv(path, data, write_type='a'):
    with open(path, write_type, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


class AMPK(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        self.dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', self.dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.weight_nt_xent_criterion = Weight_NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, epoch_counter, data_ori, data_hard_pos, data_soft_pos, data_soft_neg):
        # get the representations and the projections
        r_ori, z_ori = model(data_ori)
        r_hard_pos, z_hard_pos = model(data_hard_pos)
        r_soft_pos, z_soft_pos = model(data_soft_pos)
        r_soft_neg, z_soft_neg = model(data_soft_neg)

        # get the representations and the projections
        # normalize projection feature vectors
        z_ori = F.normalize(z_ori, dim=1)
        z_hard_pos = F.normalize(z_hard_pos, dim=1)
        z_soft_pos = F.normalize(z_soft_pos, dim=1)
        z_soft_neg = F.normalize(z_soft_neg, dim=1)

        if epoch_counter < NO_WEIGHT_EPOCH_NUMS:
            loss_ori_hp, loss_ori_sp, loss_hp_sp = self.nt_xent_criterion(z_ori, z_hard_pos, z_soft_pos, z_soft_neg)
        else:
            loss_ori_hp, loss_ori_sp, loss_hp_sp = self.weight_nt_xent_criterion(z_ori, z_hard_pos, z_soft_pos, z_soft_neg)

        return loss_ori_hp, loss_ori_sp, loss_hp_sp

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()
        if self.config['model_type'] == 'gin':
            from models.ginet_ampk import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_ampk import GCN
            model = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        else:
            raise ValueError('Undefined GNN model.')
        #print(model)
        print(self.config['model_type'])

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'],
            weight_decay=eval(self.config['weight_decay'])
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs'] - self.config['warm_up'],
            eta_min=0, last_epoch=-1
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (data_ori, data_hard_pos, data_soft_pos, data_soft_neg) in enumerate(train_loader):
                optimizer.zero_grad()
                data_ori = data_ori.to(self.device)
                data_hard_pos = data_hard_pos.to(self.device)
                data_soft_pos = data_soft_pos.to(self.device)
                data_soft_neg = data_soft_neg.to(self.device)

                # 模型训练，计算三种损失
                loss_ori_hp, loss_ori_sp, loss_hp_sp = self._step(model, epoch_counter, data_ori, data_hard_pos, data_soft_pos, data_soft_neg)
                loss = loss_ori_hp + loss_ori_sp + loss_hp_sp

                if bn % 10 == 0 or (bn <= 2000 and epoch_counter == 0):
                    print('| epoch: %d | step: %d  | loss: %.5f | loss_1: %.5f | loss_2: %.5f | loss_3: %.5f |' % (epoch_counter, bn, loss.item(), loss_ori_hp.item(), loss_ori_sp.item(), loss_hp_sp.item()))
                    write_csv(f'./ckpt/{self.dir_name}/train_Loss.csv', [epoch_counter, bn, loss.item(), loss_ori_hp.item(), loss_ori_sp.item(), loss_hp_sp.item()])

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)

                loss.backward()
                optimizer.step()
                n_iter += 1

            # if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))
            # 保存模型参数：ckpt/时间名/checkpoints/

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, epoch_counter, valid_loader)
                print(epoch_counter, valid_loss, '(validation)')
                write_csv(f'./ckpt/{self.dir_name}/valid_Loss.csv', [epoch_counter, valid_loss])

                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self, model, epoch_counter, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            # for (xis, xjs) in valid_loader:
            for (data_ori, data_hard_pos, data_soft_pos, data_soft_neg) in valid_loader:
                data_ori = data_ori.to(self.device)
                data_hard_pos = data_hard_pos.to(self.device)
                data_soft_pos = data_soft_pos.to(self.device)
                data_soft_neg = data_soft_neg.to(self.device)

                loss_ori_hp, loss_ori_sp, loss_hp_sp = self._step(model, epoch_counter, data_ori, data_hard_pos, data_soft_pos, data_soft_neg)
                loss = loss_ori_hp + loss_ori_sp + loss_hp_sp
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter

        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if config['aug'] == 'node':
        from dataset.dataset import MoleculeDatasetWrapper
    elif config['aug'] == 'subgraph':
        from dataset.dataset_subgraph import MoleculeDatasetWrapper
    elif config['aug'] == 'mix':
        from dataset.dataset_mix import MoleculeDatasetWrapper
    else:
        raise ValueError('Not defined molecule augmentation!')

    # 生成相应的数据dataloader
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    model = AMPK(dataset, config)
    model.train()


if __name__ == "__main__":
    main()
