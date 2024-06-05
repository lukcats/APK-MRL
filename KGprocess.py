import pickle
import numpy as np
from sklearn.decomposition import PCA
from rdflib import Graph
from pathlib import Path
from typing import Tuple, List
from gensim.models import KeyedVectors
from utils import *

class DataProcess:
    """
    Generate the base data embedding
    """
    def __init__(self):
        # VSCode and PyCharm Path have different.
        # VSCode can use cwd, The PyCharm use parent
        cur_path = Path.cwd()
        parent_path = cur_path.parent
        # The current data path
        print(parent_path)
        self.path = parent_path.joinpath("KG_augmentation\KG_graph")
        # check the embedding, True if Exist,  else False
        fg_flag, ele_flag, rel_flag = self.check_pkl()
        self.onto_path = self.path.joinpath("elementkgontology.embeddings.txt")
        self.fg_path = self.path.joinpath("funcgroup.txt")
        fg_filename = self.path.joinpath("fg2emb.pkl")
        self.onto_emb = self.get_onto_emb()
        self.fg_name = self.get_fg_name()
        if fg_flag:
            # Load the fg
            self.fg_emb = pickle.load(open(fg_filename, 'rb'))
        else:
            self.fg_emb = self.get_emb_dict(self.onto_emb, self.fg_name, fg_filename, True)

        self.element_symbols = [get_atom_symbol(i) for i in range(1, 109)]
        ele_filename = self.path.joinpath("ele2emb.pkl")
        if ele_flag:
            self.ele_emb = pickle.load(open(ele_filename, 'rb'))
        else:
            self.ele_emb = self.get_emb_dict(self.onto_emb, self.element_symbols, ele_filename, True)

        # getting the rel2emb dict
        rel_filename = self.path.joinpath("rel2emb.pkl")
        if rel_flag:
            self.rel_emb = pickle.load(open(rel_filename, 'rb'))
        else:
            self.rel_emb = self.get_relation_emb(self.onto_emb, self.element_symbols, rel_filename, True)

    def check_pkl(self) -> Tuple[bool, bool, bool]:
        """
        This function check whether the embedding is generated.
        :return: the functional group embedding, element embedding, relationship embedding flag.
        """
        file = "fg2emb.pkl"
        fg_emb_flag = Path(self.path.joinpath(file)).exists()
        file = "ele2emb.pkl"
        ele_emb_flag = Path(self.path.joinpath(file)).exists()
        file = "rel2emb.pkl"
        rel_emb_flag = Path(self.path.joinpath(file)).exists()
        return fg_emb_flag, ele_emb_flag, rel_emb_flag

    def get_onto_emb(self):
        """
        get the all embedding of element and functional groups
        """
        return KeyedVectors.load_word2vec_format(self.onto_path, binary=False)
        # 加载 Word2Vec 向量文件，{key:value}

    def get_fg_name(self) -> List:
        """
        get all functional groups name
        """
        with open(self.fg_path, 'r') as f:
            func_groups = f.read().strip().split('\n')
            name = [i.split()[0] for i in func_groups]
        return name

    def get_emb_dict(self, emb_data, element, emb_files, flag):
        """
        get embedding dict
        """
        emb_dict = {}
        for item in element:
            fg_name = "http://www.semanticweb.org/ElementKG#" + item
            ele_emb = emb_data[fg_name]
            emb_dict[item] = ele_emb

        if flag:
            #  functional groups embedding
            pickle.dump(emb_dict, open(emb_files, 'wb'))
        return emb_dict
    def get_relation_emb(self, onto_emb, ele_symbols, save_file, save_flag=False) -> dict:

        obj_pro_path = self.path.joinpath('objectproperty.txt')
        self.obj_property_matrix = get_property(obj_pro_path)
        v_property, pro_emb = get_property_emb(onto_emb, self.obj_property_matrix)
        pro_emb_matrix = np.concatenate(pro_emb, axis=0)
        pca = PCA(n_components=14)
        pro_emb = pca.fit_transform(pro_emb)

        property_emb_dict = get_property_utlimate_emb_dict(pro_emb, v_property)

        g_files = str(self.path.joinpath('elementkg.owl'))
        g = Graph()
        g.parse(g_files, format="xml")

        rel_emb = {}
        for i in range(len(ele_symbols) - 1):
            for j in range(1, len(ele_symbols)):
                #  SparQL
                #  query the relation
                qr = "select ?relation where { \
                     <http://www.semanticweb.org/ElementKG#" + ele_symbols[i] + "> \
                      ?relation <http://www.semanticweb.org/ElementKG#" + ele_symbols[j] + ">}"
                relations = g.query(qr)
                relations = list(relations)
                relations = [property_emb_dict[rel[0]] for rel in relations]

                if relations:
                    relation = np.mean(relations, axis=0)
                    rel_emb[(i, j)] = relation
                    rel_emb[(j, i)] = relation

        if save_flag:
            pickle.dump(rel_emb, open(save_file, 'wb'))

        return rel_emb


