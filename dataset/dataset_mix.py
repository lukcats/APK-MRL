import csv
import random
import numpy as np
import torch
import tqdm
from copy import deepcopy
from torch_geometric.data import Data, Dataset
from torch_geometric.loader.dataloader import DataLoader
from collections import deque
from rdkit import Chem
from augmentation_module import *
from rdkit.Chem.rdchem import HybridizationType, RWMol
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data.sampler import SubsetRandomSampler

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

QE_MOL = deque(maxlen=100)
DU_MOL = Chem.MolFromSmiles('*')
PATT_MOL = Chem.MolFromSmiles('[H]')
QE_MOL.append(AllChem.ReplaceSubstructs(Chem.MolFromSmiles('[*]C'), DU_MOL, PATT_MOL, True)[0])


def write_csv(path, data, write_type='a'):
    with open(path, write_type, newline='') as fp_csv_file:
        writer = csv.writer(fp_csv_file)
        writer.writerow(data)


def write_txt(path, data, write_type='a'):
    with open(path, write_type, newline='') as txt_file:
        txt_file.write(data + '\n')


def read_smiles(data_path):
    smiles_data = []
    fp = open(data_path, 'r')
    for smiles in tqdm.tqdm(fp, ncols=120):
        smiles = smiles.strip()
        smiles_data.append(smiles)
    fp.close()
    return smiles_data


# def cal_dis_info(mol):
#     try:
#         AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=2022, clearConfs=True, numThreads=0)
#         res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
#         eng_list = [i[1] if i[0] == 0 else 99999999 for i in res]
#         min_index = eng_list.index(min(eng_list))
#         xyz_index = mol.GetConformers()[min_index].GetPositions()
#         dis2zero = np.sqrt(np.sum(np.square(xyz_index), axis=1))
#         dis_opt = dis2zero / dis2zero.max()
#         dis_opt = np.where(dis_opt > 0, 1 / np.exp(dis_opt), 0.0)
#         dis_opt = dis_opt / dis_opt.max()
#         return torch.tensor(dis_opt, dtype=torch.float32).view(-1, 1)
#     except Exception:
#         return torch.ones((mol.GetNumAtoms())).view(-1, 1)


def get_data_mol(mol):
    #########################
    # Get the molecule info #
    #########################
    type_idx = []
    chirality_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    # x_dis = cal_dis_info(mol)
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


def aug_data(smi):
    try:  # ori_mol
        mol_noH = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol_noH)
        ori_mol_oupt = get_data_mol(mol)
    except Exception:
        return None, None, None, None

    try:
        ############
        # hard_pos #
        ############
        hard_pos_mol_noH = deepcopy(mol_noH)
        atom_withHs_idx = [atom_i.GetIdx() for atom_i in hard_pos_mol_noH.GetAtoms() if
                           (atom_i.GetAtomicNum() == 6 and atom_i.GetImplicitValence() > 0)]
        replace_atom_num = random.choice(atom_withHs_idx)
        smi_core = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
        if smi_core == '' or smi_core == smi:
            raise Exception("No Scaffold")
        mol_core = Chem.MolFromSmiles(smi_core)
        Sidechains = Chem.ReplaceCore(hard_pos_mol_noH, mol_core)
        if Sidechains is None:
            raise Exception("No sidecars")
        Sidechains_mols_init = Chem.GetMolFrags(Sidechains, asMols=True)
        if len(Sidechains_mols_init) == 0 or len(atom_withHs_idx) == 0:
            raise Exception("Scaffold == Mol")

        Sidechains_mol = random.choice(Sidechains_mols_init)
        Sidechains_mol_Nodummy = AllChem.ReplaceSubstructs(Sidechains_mol, DU_MOL, PATT_MOL, True)[0]
        Chem.SanitizeMol(Sidechains_mol_Nodummy)
        QE_MOL.append(Sidechains_mol_Nodummy)
        hard_pos_rs_mol = AllChem.ReplaceSubstructs(Sidechains_mol_Nodummy, PATT_MOL, hard_pos_mol_noH,
                                                    replacementConnectionPoint=replace_atom_num)[0]
        hard_pos_rs_mol_AddHs = Chem.AddHs(Chem.RemoveHs(hard_pos_rs_mol))
        Chem.SanitizeMol(hard_pos_rs_mol_AddHs)
        hard_pos_mols_oupt = get_data_mol(hard_pos_rs_mol_AddHs)
    except Exception:
        ############
        # hard_pos #
        ############
        hard_pos_mols_oupt = deepcopy(ori_mol_oupt)
    try:
        ############
        # soft_pos #
        ############
        soft_pos_mol_noH = deepcopy(mol_noH)
        atom_withHs_idx = [atom_i.GetIdx() for atom_i in soft_pos_mol_noH.GetAtoms() if
                           (atom_i.GetAtomicNum() == 6 and atom_i.GetImplicitValence() > 0)]
        replace_atom_num = random.choice(atom_withHs_idx)
        soft_pos_rs_mol = AllChem.ReplaceSubstructs(random.choice(QE_MOL), PATT_MOL, soft_pos_mol_noH,
                                                    replacementConnectionPoint=replace_atom_num)[0]

        soft_pos_rs_mol_AddHs = Chem.AddHs(Chem.RemoveHs(soft_pos_rs_mol))
        Chem.SanitizeMol(soft_pos_rs_mol_AddHs)
        soft_pos_mols_oupt = get_data_mol(soft_pos_rs_mol_AddHs)

    except Exception:
        ############
        # soft_pos #
        ############
        try:
            soft_pos_mw = RWMol(soft_pos_mol_noH)
            dgree_zero_atom_index = [i.GetIdx() for i in soft_pos_mw.GetAtoms() if i.GetDegree() == 1]
            soft_pos_mw.RemoveAtom(random.choice(dgree_zero_atom_index))
            soft_pos_mw_mol = Chem.AddHs(soft_pos_mw.GetMol())
            Chem.SanitizeMol(soft_pos_mw_mol)
            soft_pos_mols_oupt = get_data_mol(soft_pos_mw_mol)
        except:
            soft_pos_mols_oupt = deepcopy(ori_mol_oupt)

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
                soft_neg_mol_pt2 = random.choice(soft_neg_mols_pt2)
                soft_neg_mols_oupt = get_data_mol(soft_neg_mol_pt2)
        else:
            soft_neg_mols_oupt = deepcopy(ori_mol_oupt)

    except Exception:
        ############
        # soft_neg #
        ############
        soft_neg_mols_oupt = deepcopy(ori_mol_oupt)
    ############
    # hard_neg #
    ############
    return ori_mol_oupt, hard_pos_mols_oupt, soft_pos_mols_oupt, soft_neg_mols_oupt


class MoleculeDataset(Dataset):

    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)

    def __getitem__(self, index):

        smi_copy = self.smiles_data[index]
        # print('Preper SMILES' + '\t' + smi_copy)
        # write_csv('./smis.csv', [index, smi_copy])
        # write_txt('./smis.txt',str(os.getpid()) + '\t' + str(index) + '\t' + smi_copy )
        ori_mol, hard_pos_mol, soft_pos_mol, soft_neg_mol = aug_data(smi_copy)

        if ori_mol is None:
            idx = np.random.randint(0, len(self) - 1)
            data_ori_cache, data_hard_pos_cache, data_soft_pos_cache, data_soft_neg_cache = self[idx]
            return data_ori_cache, data_hard_pos_cache, data_soft_pos_cache, data_soft_neg_cache

        else:
            data_ori = Data(x=ori_mol[0], edge_index=ori_mol[1], edge_attr=ori_mol[2])
            data_hard_pos = Data(x=hard_pos_mol[0], edge_index=hard_pos_mol[1], edge_attr=hard_pos_mol[2])
            data_soft_pos = Data(x=soft_pos_mol[0], edge_index=soft_pos_mol[1], edge_attr=soft_pos_mol[2])
            data_soft_neg = Data(x=soft_neg_mol[0], edge_index=soft_neg_mol[1], edge_attr=soft_neg_mol[2])
            return data_ori, data_hard_pos, data_soft_pos, data_soft_neg

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
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=True
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, valid_loader
