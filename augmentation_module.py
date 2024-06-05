import numpy as np
from rdkit import Chem
from KGprocess import DataProcess
from utils import *
from typing import Union
import torch

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14
SMILES_TO_GRAPH = {}


class Molecular:
    def __init__(self, smiles):

        # load the kg data
        self.data = DataProcess()
        self.fg2emb = self.data.fg_emb
        self.ele2emb = self.data.ele_emb
        self.rel_emb = self.data.rel_emb
        self.eletype_list = [i for i in range(118)]
        self.hrc2emb = {}
        self.generate_hrc_emb()
        self.smart = []
        self.smart2name = dict()
        self.generate_smart()
        if type(smiles) == str:
            self.mol = MolGraph(str(smiles), self.smart, self.fg2emb, self.smart2name)
        else:
            mol = []
            for smile in smiles:
                g = MolGraph(smile, self.smart, self.fg2emb, self.smart2name)
                mol.append(g)
            self.mol = mol

    def generate_hrc_emb(self):
        for eletype in self.eletype_list:
            hrc_emb = np.random.rand(14)
            self.hrc2emb[eletype] = hrc_emb

    def generate_smart(self):
        with open(self.data.fg_path, "r") as f:
            funcgroups = f.read().strip().split('\n')
            name = [i.split()[0] for i in funcgroups]
            self.smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
            self.smart2name = dict(zip(self.smart, name))

    def get_features(self):
        return self.mol


# build the mol graph
class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.

    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.

    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str,  smart: List, fg2emb: dict, smart2name: dict) -> None:
        """
        Computes the graph structure and featurization of a molecule.
        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds

        self.f_atoms = []  # atom features
        self.f_bonds = []  # concat(in_atom, bond) features

        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []

        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        self.f_fgs = match_fg(mol, smart, fg2emb, smart2name)
        self.n_fgs = len(self.f_fgs)  # 13

        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for _, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])  #

        # Get bond features  ->  Upper triangular matrix
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                # There judge bond exist
                if bond is None:
                    continue
                f_bond = bond_features(bond=bond)
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1

                # a2b=[[],[],[]....]
                self.a2b[a2].append(b1)  # b1 = a1 --> a2   a1 is match index of node  == atom -> bond :index
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)

                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
                self.bonds.append(np.array([a1, a2]))


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.
    """
    return ATOM_FDIM


def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """

    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def match_fg(mol, smart, fg2emb, smart2name):
    """
    Match the functional groups of molecular
    :param mol: The molecular
    smart: mol
    fg2emb: {: embeding,.....}
    smart2name: {mol: ，.....}
    :return:
    """
    fg_emb = [[1] * 133]
    pad_fg = [[0] * 133]
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            fg_emb.append(fg2emb[smart2name[sm]].tolist())

    result = []
    for row in fg_emb:
        result.append(row[:132])

    result_np = np.array(result)
    result = np.sum(result_np, axis=0)
    result = result.reshape(1, -1)
    result=torch.from_numpy(result).float()
    return result
