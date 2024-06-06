import torch
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        print("Error: Invalid selfies string")
        return None

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

def is_valid_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    else:
        return True