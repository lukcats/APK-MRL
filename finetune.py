import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
import csv
import torch
from augmentation_module import *
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from datetime import datetime
from dataset.dataset_test import MolTestDatasetWrapper
torch.multiprocessing.set_sharing_strategy('file_system')
apex_support = False
LOAD_MODE_NAME = 'model_9.pth'
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except Exception:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


def write_csv(path, data, write_type='a'):
    with open(path, write_type, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config, task_name):
        self.config = config
        self.task_name = task_name
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['dataset']['target']
        log_dir = os.path.join('finetune', self.task_name, dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.pre_val = 0.0
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            self.pre_val = 9999.99
            if self.task_name in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        _, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None

        if self.task_name in ['qm7', 'qm9']:
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_lin' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_cls = self._validate(model, valid_loader, epoch_counter)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rgr = self._validate(model, valid_loader, epoch_counter)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            self._test(model, test_loader, epoch_counter, False)
        self._test(model, test_loader, None, True)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, LOAD_MODE_NAME), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, epoch_counter):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.task_name in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Epoch:', epoch_counter, 'Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Epoch:', epoch_counter, 'Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Epoch:', epoch_counter, 'Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader, epoch_counter, finnal_epoch):
        if finnal_epoch:
            model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.task_name in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                self.pre_val = self.mae if self.pre_val > self.mae else self.pre_val
                # print('Test loss:', test_loss, 'Test MAE:', self.mae)
                write_csv(f'./{self.writer.log_dir}/TEST_RESULTS.csv', [epoch_counter, test_loss, self.mae])
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                self.pre_val = self.rmse if self.pre_val > self.rmse else self.pre_val
                # print('Test loss:', test_loss, 'Test RMSE:', self.rmse)
                write_csv(f'./{self.writer.log_dir}/TEST_RESULTS.csv', [epoch_counter, test_loss, self.rmse])
        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            self.pre_val = self.roc_auc if self.pre_val < self.roc_auc else self.pre_val
            # print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
            write_csv(f'./{self.writer.log_dir}/TEST_RESULTS.csv', [epoch_counter, test_loss, self.roc_auc])


def main(config_input, task_name_input):
    dataset = MolTestDatasetWrapper(config_input['batch_size'], **config_input['dataset'])

    fine_tune = FineTune(dataset, config_input, task_name_input)
    fine_tune.train()

    if config_input['dataset']['task'] == 'classification':
        # return fine_tune.roc_auc
        return fine_tune.pre_val
    if config_input['dataset']['task'] == 'regression':
        if task_name_input in ['qm7', 'qm8', 'qm9']:
            # return fine_tune.mae
            return fine_tune.pre_val
        else:
            # return fine_tune.rmse
            return fine_tune.pre_val


if __name__ == "__main__":
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    task_names = config['task_name']

    for task_name in task_names:

        if task_name == 'BBBP':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
            target_list = ["p_np"]

        elif task_name == 'Tox21':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/tox21/tox21.csv'
            target_list = [
                "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
            ]

        elif task_name == 'ClinTox':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/clintox/clintox.csv'
            target_list = ['CT_TOX', 'FDA_APPROVED']

        elif task_name == 'HIV':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/hiv/HIV.csv'
            target_list = ["HIV_active"]

        elif task_name == 'BACE':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/bace/bace.csv'
            target_list = ["Class"]

        elif task_name == 'SIDER':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/sider/sider.csv'
            target_list = [
                "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
                "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
                "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
                "Reproductive system and breast disorders",
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                "General disorders and administration site conditions", "Endocrine disorders",
                "Surgical and medical procedures", "Vascular disorders",
                "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
                "Congenital, familial and genetic disorders", "Infections and infestations",
                "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
                "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
                "Ear and labyrinth disorders", "Cardiac disorders",
                "Nervous system disorders", "Injury, poisoning and procedural complications"
            ]

        elif task_name == 'MUV':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/muv/muv.csv'
            target_list = [
                'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
                'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
                'MUV-652', 'MUV-466', 'MUV-832'
            ]

        elif task_name == 'FreeSolv':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
            target_list = ["expt"]

        elif task_name == 'ESOL':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/esol/esol.csv'
            target_list = ["measured log solubility in mols per litre"]

        elif task_name == 'Lipo':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
            target_list = ["exp"]

        elif task_name == 'qm7':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/qm7/qm7.csv'
            target_list = ["u0_atom"]

        elif task_name == 'qm8':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/qm8/qm8.csv'
            target_list = [
                "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0",
                "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
            ]

        elif task_name == 'qm9':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/qm9/qm9.csv'
            target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

        else:
            raise ValueError('Undefined downstream task!')

        print(config)

        results_list = []
        for target in target_list:
            try:
                config['dataset']['target'] = target
                result = main(config, task_name)
                results_list.append([target, result])
            except Exception:
                continue

        os.makedirs('experiments', exist_ok=True)
        df = pd.DataFrame(results_list)
        df.to_csv(
            'experiments/{}_{}_finetune.csv'.format(config['fine_tune_from'], task_name),
            mode='a', index=False, header=False)
