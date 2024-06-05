
from rdkit import Chem
from typing import Tuple, List
from rdflib import URIRef

# get property -> embedding matrix
def get_property(filename) -> list:
    """
    # get property -> embedding matrix
    """
    obj_property_list = []
    with(open(filename, 'r')) as f:
        for line in f.readlines():
            temp = line.split()
            obj_property_list.extend(temp)

    return obj_property_list


def get_atom_symbol(atomic_number):
    '''
    get all element symbol
    '''
    return Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), atomic_number)


def get_property_emb(emb_data, property) -> Tuple[list, list]:
    """
    get valid property and embedding
    """
    valid_pro = []
    pro_emb = []
    for p in property:
        p_name = "http://www.semanticweb.org/ElementKG#" + p
        if p_name in emb_data:
            valid_pro.append(p)
            pro_emb.append(emb_data[p_name])

    return (valid_pro, pro_emb)

# get property -> embedding dict
def get_property_utlimate_emb_dict(org_emb_data, pro_list) -> dict:
    """
    get new propery embedding dict
    """
    property_emb = {}
    for p in range(len(pro_list)):
        p_name = URIRef(f"http://www.semanticweb.org/ElementKG#" + pro_list[p])
        property_emb[p_name] = org_emb_data[p]

    return property_emb
