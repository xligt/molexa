import torch
import os, glob
import random
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from fastai.data.core import TfmdLists
from fastai.data.transforms import Transform, RandomSplitter



def random_unit_rotation_matrix():
    """
    Generates a random 3D rotation matrix uniformly distributed over SO(3).
    """
    # Generate random numbers uniformly distributed in [0, 1]
    u1 = torch.rand(1)  # Keep as tensor
    u2 = torch.rand(1)
    u3 = torch.rand(1)

    # Compute quaternion components using the method by Ken Shoemake
    sqrt_u1 = torch.sqrt(u1)
    sqrt_one_minus_u1 = torch.sqrt(1 - u1)
    two_pi_u2 = 2 * torch.pi * u2
    two_pi_u3 = 2 * torch.pi * u3

    q0 = sqrt_one_minus_u1 * torch.sin(two_pi_u2)
    q1 = sqrt_one_minus_u1 * torch.cos(two_pi_u2)
    q2 = sqrt_u1 * torch.sin(two_pi_u3)
    q3 = sqrt_u1 * torch.cos(two_pi_u3)

    # Form the quaternion
    q = torch.stack([q0, q1, q2, q3], dim=0).squeeze()

    # Normalize the quaternion (optional, but ensures numerical stability)
    q = q / q.norm()

    # Extract quaternion components
    w, x, y, z = q

    # Compute the rotation matrix from the quaternion
    rotation_matrix = torch.tensor([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

    return rotation_matrix


class TransformPd_multInput(Transform):
    def __init__(self, rotation=False):
        self.atom_ind_dict = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si': 14, 'P':15, 'S': 16, 'Cl':17, 'Br':35}
        self.atom_mass_dict = {'H':1.00797, 'C':12.011, 'N':14.007, 'O':15.999, 'F':18.99840316, 'Si': 28.085, 'P': 30.97376200, 'S': 32.07, 'Cl':35.45, 'Br':79.90}
        self.ind_atom_dict = {v:k for k,v in self.atom_ind_dict.items()}
        self.rotation = rotation
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

        pos_raw = torch.tensor(xyz, dtype=torch.float32).view(-1,3)
        pxpypz_raw = torch.tensor(pxpypz, dtype=torch.float32).view(-1,3)
        if self.rotation:# and torch.rand(1).item()>0.5:
            rot_mat = random_unit_rotation_matrix()
            pos_raw = torch.matmul(rot_mat, pos_raw.T).T
            pxpypz_raw = torch.matmul(rot_mat, pxpypz_raw.T).T
#        columns = [dim+'_'+str(index) for index in indexes for dim in ['errx','erry','errz']]
#        errxyz = (row[columns]).tolist()        

        
        pos_raw[torch.isnan(pos_raw)] = 0

#        return Data(mol_tp=mol_tp,mol_name=row['mol_name'],count=row['count'],z = torch.tensor(atoms, dtype=torch.long),q = torch.tensor(charges, dtype=torch.long),
#                     y = pos_raw, pos = torch.tensor(pxpypz, dtype=torch.float32).view(-1,3), errxyz = torch.tensor(errxyz, dtype=torch.float32).view(-1,3), natoms = torch.tensor([num_atoms], dtype=torch.long))
        return Data(z = torch.tensor(atoms, dtype=torch.long),q = torch.tensor(charges, dtype=torch.long),
                     y = pos_raw, pos = pxpypz_raw, natoms = torch.tensor([num_atoms], dtype=torch.long))

class GeomDataLoaders():
    def __init__(self,dataset_train, dataset_valid, batch_size, sampler=None, vshuffle=False):
        self.train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler) if sampler is not None else DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        self.valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=vshuffle)


def Get_Dataset(path='../download/dataset/dataset_2_7/', rank=0):

    dtps = ['train', 'valid']
    tls_dict = {}
    for dtp in dtps:
        pkls = glob.glob(os.path.join(path+dtp, '**', '*.pkl'), recursive=True)
        pkl_idxs = []
        for pkl in pkls:
            df = pd.read_pickle(pkl)
            mol_tp = pkl.split('/')[-1].split('.')[0]
            mol_tps = [mol_tp]*df.shape[0]
            pkl_repeat = [pkl]*df.shape[0]
            pkl_idxs += [f"{pklf},{mt},{idx}" for pklf, mt, idx in zip(pkl_repeat, mol_tps, df.index)]
        tfm = TransformPd_multInput(rotation=False) if dtp=='valid' else TransformPd_multInput(rotation=True) 
        tls_dict[dtp] = TfmdLists(pkl_idxs, tfm, splits=None) 
        
    if rank==0: 
        print('# of valid:', len(tls_dict['valid'].train), '# of train:', len(tls_dict['train'].train))
        
    return tls_dict['train'], tls_dict['valid']

def Create_DataLoaders(tls_train, tls_valid, batch_size=128, sampler=None, vshuffle=False):
    
    dls = GeomDataLoaders(tls_train, tls_valid, batch_size, sampler=sampler, vshuffle=vshuffle)
    
    return dls
