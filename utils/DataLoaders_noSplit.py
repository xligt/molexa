import torch
import os, glob
import random
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from fastai.data.core import TfmdLists
from fastai.data.transforms import Transform, RandomSplitter

class TransformPd_multInput(Transform):
    def __init__(self):
        self.atom_ind_dict = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si': 14, 'P':15, 'S': 16, 'Cl':17, 'Br':35}
        self.atom_mass_dict = {'H':1.00797, 'C':12.011, 'N':14.007, 'O':15.999, 'F':18.99840316, 'Si': 28.085, 'P': 30.97376200, 'S': 32.07, 'Cl':35.45, 'Br':79.90}
        self.ind_atom_dict = {v:k for k,v in self.atom_ind_dict.items()}
        
    def encodes(self, pkl_idx):
        pkl, mol_tp, pkl_ind = pkl_idx.split(',')
        num_atoms = int((mol_tp.split('_'))[0])
        df = pd.read_pickle(pkl)
        row = df.iloc[int(pkl_ind)]
        anchor_indexes = [row.ptc1,row.ptc2]
        indexes = anchor_indexes+[i for i in range(1,num_atoms+1) if i not in anchor_indexes]
        
        columns = ['atom_'+str(index) for index in indexes]
        atoms = (row[columns].apply(lambda x: self.atom_ind_dict[x])).tolist()
        
        columns = ['q_'+str(index) for index in indexes]
        charges = (row[columns]).tolist()
        
        columns = [dim+'_'+str(index) for index in indexes for dim in ['x','y','z']]
        xyz = (row[columns]).tolist()
        
        columns = [dim+'_'+str(index) for index in indexes for dim in ['px','py','pz']]
        pxpypz = (row[columns]).tolist()

        columns = [dim+'_'+str(index) for index in indexes for dim in ['errx','erry','errz']]
        errxyz = (row[columns]).tolist()        

        pos_raw = torch.tensor(xyz, dtype=torch.float32).view(-1,3)
        pos_raw[torch.isnan(pos_raw)] = 0

        return Data(mol_tp=mol_tp,mol_name=row['mol_name'],count=row['count'],z = torch.tensor(atoms, dtype=torch.long),q = torch.tensor(charges, dtype=torch.long),
                     y = pos_raw, pos = torch.tensor(pxpypz, dtype=torch.float32).view(-1,3), errxyz = torch.tensor(errxyz, dtype=torch.float32).view(-1,3), natoms = torch.tensor([num_atoms], dtype=torch.long))
#        return Data(z = torch.tensor(atoms, dtype=torch.long),q = torch.tensor(charges, dtype=torch.long),
#                     y = pos_raw, pos = torch.tensor(pxpypz, dtype=torch.float32).view(-1,3), natoms = torch.tensor([num_atoms], dtype=torch.long))

class GeomDataLoaders():
    def __init__(self,dataset, batch_size, sampler=None, vshuffle=False):
        self.samples = DataLoader(dataset, batch_size=batch_size, sampler=sampler) if sampler is not None else DataLoader(dataset, batch_size=batch_size, shuffle=True)


def Get_Dataset_noSplit(path='../download/dataset/dataset_2_7/test/', rank=0):

    pkls = glob.glob(os.path.join(path, '**', '*.pkl'), recursive=True)
    pkl_idxs = []
    for pkl in pkls:
        df = pd.read_pickle(pkl)
        mol_tp = pkl.split('/')[-1].split('.')[0]
        mol_tps = [mol_tp]*df.shape[0]
        pkl_repeat = [pkl]*df.shape[0]
        pkl_idxs += [f"{pklf},{mt},{idx}" for pklf, mt, idx in zip(pkl_repeat, mol_tps, df.index)]
    tfm = TransformPd_multInput() 
    tls = TfmdLists(pkl_idxs, tfm, splits=None) 
        
    if rank==0: 
        print('# of samples:', len(tls.train))
        
    return tls

def Create_DataLoaders_noSplit(tls, batch_size=128, sampler=None, vshuffle=False):
    
    dls = GeomDataLoaders(tls, batch_size, sampler=sampler, vshuffle=vshuffle)
    
    return dls
