import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import networkx as nx
from rdkit.Chem.Scaffolds import MurckoScaffold
import re
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import tqdm
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from collections import deque
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, RWMol
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path):
    smiles_data = []
    fp = open(data_path, 'r')
    for smiles in tqdm.tqdm(fp, ncols=120):
        smiles = smiles.strip()
        smiles_data.append(smiles)
    fp.close()
    return smiles_data


class My_MultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def _get_data(self):
        # Fetches data from `self._data_queue`.
        #
        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        try:
            if self._timeout > 0:
                success, data = self._try_get_data(self._timeout)
                if success:
                    return data
                else:
                    raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
            elif self._pin_memory:
                while self._pin_memory_thread.is_alive():
                    success, data = self._try_get_data()
                    if success:
                        return data
                else:
                    # while condition is false, i.e., pin_memory_thread died.
                    raise RuntimeError('Pin memory thread exited unexpectedly')
                # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
                # need to call `.task_done()` because we don't use `.join()`.
            else:
                while True:
                    success, data = self._try_get_data()
                    if success:
                        return data
        except:
            self._next_data()


class MyDataloader(DataLoader):

    def _get_iterator(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return My_MultiProcessingDataLoaderIter(self)


class MoleculeDataset(Dataset):

    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)
        self.qe = deque(maxlen=50)
        self.du = Chem.MolFromSmiles('*')
        self.patt = Chem.MolFromSmiles('[H]')
        self.qe.append(AllChem.ReplaceSubstructs(Chem.MolFromSmiles('[*]C'), self.du, self.patt, True)[0])

    def __getitem__(self, index):

        smi_copy = self.smiles_data[index]
        # print("SMI:" + '\t' + smi_copy )
        # try:
        hard_pos_mol, soft_pos_mol, soft_neg_mol = self.aug_data(smi_copy)
        # print("SMI:" + '\t' + 'GetMol' )
        ori_mol_copy = Chem.MolFromSmiles(smi_copy)
        ori_mol_copy_AddH = Chem.AddHs(ori_mol_copy)
        x_ori, edge_index_ori, edge_attr_ori = self.get_data_mol(ori_mol_copy_AddH)

        x_hard_pos, edge_index_hard_pos, edge_attr_hard_pos = self.get_data_mol(hard_pos_mol)
        x_soft_pos, edge_index_soft_pos, edge_attr_soft_pos = self.get_data_mol(soft_pos_mol)
        x_soft_neg, edge_index_soft_neg, edge_attr_soft_neg = self.get_data_mol(soft_neg_mol)
        # print("SMI:" + '\t' + 'GetData' )
        data_ori = Data(x=x_ori, edge_index=edge_index_ori, edge_attr=edge_attr_ori)
        data_hard_pos = Data(x=x_hard_pos, edge_index=edge_index_hard_pos, edge_attr=edge_attr_hard_pos)
        data_soft_pos = Data(x=x_soft_pos, edge_index=edge_index_soft_pos, edge_attr=edge_attr_soft_pos)
        data_soft_neg = Data(x=x_soft_neg, edge_index=edge_index_soft_neg, edge_attr=edge_attr_soft_neg)
        # time.sleep(0.5)
        # print("SMI:" + '\t' + 'ReturnData' )
        return (data_ori, data_hard_pos, data_soft_pos, data_soft_neg)
        # except:
        #    return None
        # return (data_ori, data_ori, data_ori, data_ori)

    def __len__(self):
        return len(self.smiles_data)

    def get_data_mol(self, mol):
        #########################
        # Get the molecule info #
        #########################
        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]  # row = [start1 ,                       end1,                       start2, end2 ...]
            col += [end, start]  # col = [end1 ,                         start1,                     end2, start2 ...]
            edge_feat.append([  # col = [[BOND_LIST1 , BONDDIR_LIST1], [BOND_LIST1, BONDDIR_LIST] ...]
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        return x, edge_index, edge_attr

    def aug_data(self, smi):
        # print('Aug_smi' + {smi})
        mol_noH = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol_noH)
        try:
            ############
            # hard_pos #
            ############
            hard_pos_mol_noH = deepcopy(mol_noH)
            atom_withHs_idx = [atom_i.GetIdx() for atom_i in hard_pos_mol_noH.GetAtoms() if
                               (atom_i.GetAtomicNum() == 6 and atom_i.GetImplicitValence() > 0)]
            replce_atom_num = random.choice(atom_withHs_idx)
            smi_core = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
            if smi_core == '' or smi_core == smi:
                raise Exception("No Scaffold")
            mol_core = Chem.MolFromSmiles(smi_core)
            Sidechains = Chem.ReplaceCore(hard_pos_mol_noH, mol_core)
            if Sidechains == None:
                raise Exception("No Sidechains")
            Sidechains_mols_init = Chem.GetMolFrags(Sidechains, asMols=True)
            if len(Sidechains_mols_init) == 0 or len(atom_withHs_idx) == 0:
                raise Exception("Scaffold == Mol")

            Sidechains_mol = random.choice(Sidechains_mols_init)
            Sidechains_mol_Nodummy = AllChem.ReplaceSubstructs(Sidechains_mol, self.du, self.patt, True)[0]
            Chem.SanitizeMol(Sidechains_mol_Nodummy)
            self.qe.append(Sidechains_mol_Nodummy)
            hard_pos_rs_mol = AllChem.ReplaceSubstructs(Sidechains_mol_Nodummy, self.patt, hard_pos_mol_noH,
                                                        replacementConnectionPoint=replce_atom_num)[0]
            hard_pos_rs_mol_AddHs = Chem.AddHs(hard_pos_rs_mol)
            Chem.SanitizeMol(hard_pos_rs_mol_AddHs)
            hard_pos_mols_oupt = deepcopy(hard_pos_rs_mol_AddHs)
        except:
            ############
            # hard_pos #
            ############
            hard_pos_mols_oupt = deepcopy(mol)
        # print('End Hard POS' + {Chem.MolToSmiles(hard_pos_mols_oupt)})
        try:
            ############
            # soft_pos #
            ############
            soft_pos_mol_noH = deepcopy(mol_noH)
            atom_withHs_idx = [atom_i.GetIdx() for atom_i in soft_pos_mol_noH.GetAtoms() if
                               (atom_i.GetAtomicNum() == 6 and atom_i.GetImplicitValence() > 0)]
            replce_atom_num = random.choice(atom_withHs_idx)
            soft_pos_rs_mol = AllChem.ReplaceSubstructs(random.choice(self.qe), self.patt, soft_pos_mol_noH,
                                                        replacementConnectionPoint=replce_atom_num)[0]

            soft_pos_rs_mol_AddHs = Chem.AddHs(soft_pos_rs_mol)
            Chem.SanitizeMol(soft_pos_rs_mol_AddHs)
            soft_pos_mols_oupt = deepcopy(soft_pos_rs_mol_AddHs)

        except:
            ############
            # soft_pos #
            ############
            try:
                soft_pos_mw = RWMol(soft_pos_mol_noH)
                dgree_zero_atom_index = [i.GetIdx() for i in soft_pos_mw.GetAtoms() if i.GetDegree() == 1]
                soft_pos_mw.RemoveAtom(random.choice(dgree_zero_atom_index))
                soft_pos_mw_mol = Chem.AddHs(soft_pos_mw.GetMol())
                Chem.SanitizeMol(soft_pos_mw_mol)
                soft_pos_mols_oupt = deepcopy(soft_pos_mw_mol)

            except:
                soft_pos_mols_oupt = deepcopy(mol)
        # print('End Soft POS' + {Chem.MolToSmiles(soft_pos_mols_oupt)})

        try:
            ############
            # soft_neg #
            ############
            soft_neg_mol_noH = deepcopy(mol_noH)
            soft_neg_mols_pt2 = []
            soft_neg_mw = RWMol(soft_neg_mol_noH)
            del_nums = len([i.GetIdx() for i in soft_neg_mw.GetAtoms() if i.GetDegree() == 1])
            if del_nums != 0:
                for _ in range(del_nums):
                    soft_neg_dgree_one_atom_index = [i.GetIdx() for i in soft_neg_mw.GetAtoms() if i.GetDegree() == 1]
                    soft_neg_mw.RemoveAtom(random.choice(soft_neg_dgree_one_atom_index))
                    try:
                        soft_neg_mw_mol = Chem.AddHs(soft_neg_mw.GetMol())
                        Chem.SanitizeMol(soft_neg_mw_mol)
                        soft_neg_mols_pt2.append(soft_neg_mw_mol)
                    except:
                        continue
                if len(soft_neg_mols_pt2) == 0:
                    raise Exception("No")
                else:
                    soft_neg_mols_oupt = random.choice(soft_neg_mols_pt2)
            else:
                soft_neg_mols_oupt = deepcopy(mol)

        except:
            ############
            # soft_neg #
            ############
            soft_neg_mols_oupt = deepcopy(mol)
        # print('End Soft Neg' + {Chem.MolToSmiles(soft_neg_mols_oupt)})

        ############
        # hard_neg #
        ############
        # return hard_pos_smis, soft_pos_smis, soft_neg_smis
        return hard_pos_mols_oupt, soft_pos_mols_oupt, soft_neg_mols_oupt


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, train_data_path, valid_data_path):
        super(object, self).__init__()
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        # train_dataset = MoleculeDataset(data_path=self.data_path)
        train_data = MoleculeDataset(self.train_data_path)
        valid_data = MoleculeDataset(self.valid_data_path)
        # train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_data, valid_data)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_data, valid_data):
        # =---------------------------------------------------
        # obtain training indices that will be used for validation
        # num_train = len(all_smis)
        # indices = list(range(num_train))
        # np.random.shuffle(indices)
        # train_dataset = []
        # valid_dataset = []
        #
        # split = int(np.floor(self.valid_size * num_train))
        # train_idx, valid_idx = indices[split:], indices[:split]
        # # train_idx, valid_idx = indices[:-split], indices[-split:]
        #
        # for idx in indices:
        #     if idx in valid_idx:
        #         train_dataset.append(all_smis[idx])
        #     else:
        #         valid_dataset.append(all_smis[idx])
        # train_dataset = np.array(all_smis)[train_idx].tolist()
        # valid_dataset = np.array(all_smis)[valid_idx].tolist()
        # define samplers for obtaining training and validation batches
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = MyDataloader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                    drop_last=True, timeout=30)
        valid_loader = MyDataloader(valid_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                    drop_last=True, timeout=30)
        # =---------------------------------------------------

        # train_dataset, test_dataset = random_split(
        #     dataset=train_dataset,
        #     lengths=[num_train - split, split]
        # )
        #
        # train_loader = DataLoader(
        #     train_dataset, batch_size=self.batch_size, shuffle=True,
        #     num_workers=self.num_workers, drop_last=True)
        #
        # valid_loader = DataLoader(
        #     train_dataset, batch_size=self.batch_size, shuffle=False,
        #     num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader



