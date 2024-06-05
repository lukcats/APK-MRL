import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
from rdkit.Chem.Scaffolds import MurckoScaffold
import re
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import tqdm
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
from collections import deque
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, RWMol
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
qe = deque(maxlen=512)
qe.append(Chem.MolFromSmarts('[H]C'))

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

def aug_data(smi):
    mol_input = Chem.MolFromSmiles(smi)
    mol_noH = Chem.RemoveHs(mol_input)
    mol = Chem.AddHs(mol_noH)
    patt = Chem.MolFromSmarts('[H]')
    try:
        ############
        # hard_pos #
        ############
        atom_withHs_idx = [atom_i.GetIdx() for atom_i in mol_noH.GetAtoms() if atom_i.GetAtomicNum() == 6 and (atom_i.GetTotalNumHs() >= 1)]
        smi_core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol_noH, includeChirality=False)
        if smi_core == '':
            raise Exception("No Scaffold")
        mol_core = Chem.MolFromSmiles(smi_core)
        Sidechains = Chem.ReplaceCore(mol_noH, mol_core)
        Sidechains_mols_init = Chem.GetMolFrags(Sidechains, asMols=True)
        if len(Sidechains_mols_init) or len(atom_withHs_idx) == 0:
            raise Exception("Scaffold == Mol")
        Sidechains_mols = []
        for Sidechains_mol_init in Sidechains_mols_init:
            sc_smi_w_sig = Chem.MolToSmiles(Sidechains_mol_init)
            Sidechains_smi = re.sub(r'\[\d*\*\]', '[H]', sc_smi_w_sig)
            Sidechains_mol = Chem.MolFromSmarts(Sidechains_smi)
            try:
                Sidechains_mol = Chem.RemoveHs(Sidechains_mol)
                Chem.SanitizeMol(Sidechains_mol)
                Sidechains_mols.append(Sidechains_mol)
            except:
                continue
        if len(Sidechains_mols) == 0:
            raise Exception("Sidechains_mol == Mol")
        mol_added = random.choice(Sidechains_mols)
        qe.append(mol_added)
        hard_pos_rs_mols = AllChem.ReplaceSubstructs(mol_added, patt, mol_noH, replacementConnectionPoint=random.choice(atom_withHs_idx))
        hard_pos_mols = []
        for hard_pos_rs_mol in hard_pos_rs_mols:
            try:
                hard_pos_rs_mol = Chem.MolFromSmiles(Chem.MolToSmiles(hard_pos_rs_mol))
                hard_pos_rs_mol = Chem.AddHs(Chem.RemoveHs(hard_pos_rs_mol))
                Chem.SanitizeMol(hard_pos_rs_mol)
                hard_pos_mols.append(hard_pos_rs_mol)
            except:
                continue
        if len(hard_pos_mols) == 0:
            raise Exception("No_pos_mols")
        hard_pos_mols_oupt = random.choice(hard_pos_mols)

    except:
        ############
        # hard_pos #
        ############
        hard_pos_mols_oupt = mol

    try:
        ############
        # soft_pos #
        ############
        soft_pos_mols = []
        # del_subs = [Chem.MolFromSmiles(Sidechains_smi) for Sidechains_smi in Sidechains_smis if len(Sidechains_smi) > 1]
        # if len(del_subs) == 0:
        #     raise Exception("Del error")
        # rs_mols = [AllChem.DeleteSubstructs(mol_noH, del_sub) for del_sub in del_subs]
        atom_withHs_idx = [atom_i.GetIdx() for atom_i in mol_noH.GetAtoms() if atom_i.GetAtomicNum() == 6 and (atom_i.GetTotalNumHs() >= 1)]
        soft_pos_rs_mols = AllChem.ReplaceSubstructs(random.choice(qe), patt, mol_noH, replacementConnectionPoint=random.choice(atom_withHs_idx))
        for soft_pos_rs_mol in soft_pos_rs_mols:
            try:
                soft_pos_rs_mol = Chem.MolFromSmiles(Chem.MolToSmiles(soft_pos_rs_mol))
                soft_pos_rs_mol = Chem.AddHs(Chem.RemoveHs(soft_pos_rs_mol))
                Chem.SanitizeMol(soft_pos_rs_mol)
                soft_pos_mols.append(soft_pos_rs_mol)
            except:
                continue
        if len(soft_pos_mols) == 0:
            raise Exception("No")
        soft_pos_mols_oupt = random.choice(soft_pos_mols)

    except:
        ############
        # soft_pos #
        ############
        try:
            soft_pos_mw = RWMol(mol_noH)
            dgree_zero_atom_index = [j for j, x in enumerate([i.GetDegree() for i in mol_noH.GetAtoms()]) if x == 1]
            # for i in range(len(dgree_zero_atom_index) // 2 + 1)
            soft_pos_mw.RemoveAtom(random.choice(dgree_zero_atom_index))
            soft_pos_mw_mol = Chem.AddHs(soft_pos_mw.GetMol())
            Chem.SanitizeMol(soft_pos_mw_mol)
            soft_pos_mols_oupt = soft_pos_mw_mol
        except:
            soft_pos_mols_oupt = mol

    try:
        ############
        # soft_neg #
        ############
        soft_neg_mols_pt2 = []
        soft_neg_mw = RWMol(mol_noH)
        soft_neg_dgree_zero_atom_index_length = [j for j, x in enumerate([i.GetDegree() for i in mol_noH.GetAtoms()]) if x == 1]
        del_nums = len(soft_neg_dgree_zero_atom_index_length)
        for index in range(del_nums):
            soft_neg_dgree_zero_atom_index = [j for j, x in enumerate([i.GetDegree() for i in soft_neg_mw.GetAtoms()]) if x == 1]
            soft_neg_mw.RemoveAtom(random.choice(soft_neg_dgree_zero_atom_index))
            try:
                soft_neg_mw_mol = Chem.AddHs(soft_neg_mw.GetMol())
                Chem.SanitizeMol(soft_neg_mw_mol)
                soft_neg_mols_pt2.append(soft_neg_mw_mol)
            except:
                continue
        if len(soft_neg_mols_pt2) == 0:
            raise Exception("No")
        soft_neg_mols_oupt = random.choice(soft_neg_mols_pt2)

    except:

        ############
        # soft_neg #
        ############
        soft_neg_mols_oupt = mol

    return hard_pos_mols_oupt, soft_pos_mols_oupt, soft_neg_mols_oupt


def read_smiles(data_path):
    smiles_data = []
    fp = open(data_path, 'r')
    for smiles in tqdm.tqdm(fp, ncols=120):
        smiles = smiles.strip()
        smiles_data.append(smiles)
    fp.close()
    return smiles_data

    #
    # smiles_data = []
    # fp = open(data_path, 'r')
    # for smiles in tqdm.tqdm(fp, ncols=120):
    #     smiles = smiles.strip()
    #     try:
    #         mol = Chem.MolFromSmiles(smiles)
    #         Chem.SanitizeMol(mol)
    #     except:
    #         print(smiles)
    #         continue
    #     smiles_data.append(smiles)
    # return smiles_data
    # with open(data_path) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     for i, row in tqdm.tqdm(enumerate(csv_reader)):
    #         smiles = row[-1]
    #         try:
    #             mol = Chem.MolFromSmiles(smiles)
    #             Chem.SanitizeMol(mol)
    #         except:
    #             print(smiles)
    #             continue
    #         smiles_data.append(smiles)
    # return smiles_data


def get_data_mol(mol):
    # mol_noH = Chem.MolFromSmiles(smi)
    # mol = Chem.AddHs(mol_noH)

    # N = mol.GetNumAtoms()
    # M = mol.GetNumBonds()
    # atoms = mol.GetAtoms()
    # bonds = mol.GetBonds()

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


class MoleculeDataset(Dataset):

    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)  # SMILES 列表

    def __getitem__(self, index):
        smi_copy = self.smiles_data[index]

        hard_pos_mol, soft_pos_mol, soft_neg_mol = aug_data(smi_copy)

        ori_mol_copy = Chem.MolFromSmiles(smi_copy)
        ori_mol_copy = Chem.AddHs(ori_mol_copy)
        x_ori, edge_index_ori, edge_attr_ori = get_data_mol(ori_mol_copy)

        x_hard_pos, edge_index_hard_pos, edge_attr_hard_pos = get_data_mol(hard_pos_mol)
        x_soft_pos, edge_index_soft_pos, edge_attr_soft_pos = get_data_mol(soft_pos_mol)
        x_soft_neg, edge_index_soft_neg, edge_attr_soft_neg = get_data_mol(soft_neg_mol)

        data_ori = Data(x=x_ori, edge_index=edge_index_ori, edge_attr=edge_attr_ori)
        data_hard_pos = Data(x=x_hard_pos, edge_index=edge_index_hard_pos, edge_attr=edge_attr_hard_pos)
        data_soft_pos = Data(x=x_soft_pos, edge_index=edge_index_soft_pos, edge_attr=edge_attr_soft_pos)
        data_soft_neg = Data(x=x_soft_neg, edge_index=edge_index_soft_neg, edge_attr=edge_attr_soft_neg)

        return (data_ori, data_hard_pos, data_soft_pos, data_soft_neg)

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # =---------------------------------------------------
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))  # 划分的index
        train_idx, valid_idx = indices[split:], indices[:split]  # 数据集划分
        # train_idx, valid_idx = indices[:-split], indices[-split:]  # 数据集划分

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)  # 随机采样操作
        valid_sampler = SubsetRandomSampler(valid_idx)  # 随机采样操作

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=True
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=True
        )
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
