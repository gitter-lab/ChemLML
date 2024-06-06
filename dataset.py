import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import networkx as nx
from rdkit import Chem
# Assume SmilesTokenizer is implemented or imported as mentioned in previous responses

class MoleculeDataset(Dataset):
    def __init__(self, data):
        self.descriptions, self.selfies, self.smiles = zip(*data)
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, index):
        return {
            "description": self.descriptions[index],
            "selfies": self.selfies[index],
            "smiles": self.smiles[index]
        }

class DrugTargetDataset(Dataset):
    def __init__(self, data):
        self.seq, self.selfies, self.smiles = zip(*data)
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, index):
        text = self.seq[index]
        smiles = self.smiles[index]
        selfies = self.selfies[index]
        return {
            "description": text,
            "smiles": smiles,
            "selfies": selfies
        }

import h5py
class PocketMolDataset(Dataset):
    def __init__(self, file):
        hdf = h5py.File(file, 'r')
        self.array_group = hdf['arrays']
        self.string_group = hdf['strings']
    def __len__(self):
        return len(self.array_group)

    def __getitem__(self, index):
        return {
            "smiles": self.string_group[f'strings_{index}'][...][0],
            "selfies": self.string_group[f'strings_{index}'][...][1],
            "pocket_seq": self.string_group[f'strings_{index}'][...][2],
            "pocket_coord": self.array_group[f'array_{index}'][...],
        }

import numpy as np
class PocketMolDataLoader():
    def __init__(self, dataset, batch_tokens=1000, shuffle=True, collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset.string_group[f'strings_{i}'][...][2]) for i in range(self.size)]
        self.batch_tokens = batch_tokens
        sorted_ix = np.argsort(self.lengths)
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_tokens:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters
        if drop_last == True:
            if len(self.clusters[-1]) == 1:
                del self.clusters[-1]

    def __len(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch

class MoleculeGraphDataset(Dataset):
    def __init__(self, data):
        # Data is a list of tuples containing descriptions and SMILES strings
        self.descriptions, self.smiles = zip(*data)
        self.molecule_graph = []
        for smile in self.smiles:
            mol = Chem.MolFromSmiles(smile)

            # Initialize an empty graph
            G = nx.Graph()

            # Add atoms to graph
            for atom in mol.GetAtoms():
                G.add_node(atom.GetIdx(),
                        atomic_num=atom.GetAtomicNum(),
                        formal_charge=atom.GetFormalCharge(),
                        chiral_tag=atom.GetChiralTag(),
                        hybridization=atom.GetHybridization(),
                        num_explicit_hs=atom.GetNumExplicitHs(),
                        is_aromatic=atom.GetIsAromatic())

            # Add bonds to graph
            for bond in mol.GetBonds():
                G.add_edge(bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        bond_type=bond.GetBondType())
            self.molecule_graph.append(G)
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, index):
        text = self.descriptions[index]
        molecule = self.molecule_graph[index]
        return {
            "description": text,
            "molecule": molecule
        }



    

