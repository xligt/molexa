import math, shutil, pickle, re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

Z_dict = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si': 14, 'P':15, 'S': 16, 'Cl':17, 'Br':35, 'I':53}
M_dict = {'H':1.00797, 'C':12.011, 'N':14.007, 'O':15.999, 'F':18.99840316, 'Si': 28.085, 'P': 30.97376200, 'S': 32.07, 'Cl':35.45, 'Br':79.90, 'I':126.90447}


def get_u1(v1,ax=0):
    """get the norm and unit vector of v1
    Parameters
    ----------
    v1: 1d or 2d array
    ax: axis along which this operation is to be applied
    Returns
    -------
    v1 norm, float or 1d array
    unit vector u1, 1d or 2d array
    """
    norm1 = np.linalg.norm(v1,axis=ax)
    return norm1,v1/norm1

def get_u2(u1,v2,ax):
    """find the unit vector u2 which is perpendicular to u1 and make a plane with u1 containing v2.
    Parameters
    ----------
    u1: unit vector u1
    v2: 1d or 2d array    
    ax: axis along which this operation is to be applied    
    Returns
    -------
    absolute value of the projection of v2 on u2, float or 1d array
    unit vector u2, 1d or 2d array
    """    
    m_v2_u1 = np.sum(v2*u1,axis=0)
    
    v2_u2 = v2 - u1*m_v2_u1
    norm2 = np.linalg.norm(v2_u2,axis=ax)
    return norm2,v2_u2/norm2

def get_u3(u1,u2):
    """find the unit vector u3 which is perpendicular to both u1 and u2.
    Parameters
    ----------
    u1: unit vector u1
    u2: unit vector u2
    Returns
    -------
    unit vector u3
    """        
    u3 = np.cross(u1.T,u2.T)
    return u3.T


def get_p1_p2_p3(u1,u2,u3,p,nm):
    """find the projections of p on unit vectors u1, u2 and u3
    Parameters
    ----------
    u1: unit vector u1
    u2: unit vector u2
    u3: unit vector u3
    Returns
    -------
    the projections of p on unit vectors u1, u2 and u3
    """            
    p1 = np.sum(p*u1,axis=0)    
    p2 = np.sum(p*u2,axis=0)
    p3 = np.sum(p*u3,axis=0)

    return p1/nm,p2/nm,p3/nm

def MF_Momen(momen_dict,ptc1,ptc2, correct_u1=False, recenter=np.array([1, 0, 0])):

    Mdict={}
    nm1,u1 = get_u1(momen_dict[ptc1],ax=0)

    if len(momen_dict.keys())==2:
        for k in momen_dict.keys():
            Mdict[k] = np.array([np.sum(momen_dict[k]*u1), 0, 0])
        return Mdict

    if nm1==0 and correct_u1:# this is for molecules like O2Si, where the heavist atom Si used to defined u1 is located at [0,0,0]
        for k in momen_dict.keys():
            momen_dict[k] += recenter

        _,u1 = get_u1(momen_dict[ptc1],ax=0)
    
    if isinstance(ptc2,list):
        v23 = momen_dict[ptc2[0]]+momen_dict[ptc2[1]]
    else:
        v23 = momen_dict[ptc2]
    
    norm2,u2 = get_u2(u1,v23,ax=0)  
    u3 = get_u3(u1,u2)    
    
    for k in momen_dict.keys():
        Mdict[k] = np.array(get_p1_p2_p3(u1,u2,u3,momen_dict[k],nm=1))
        if nm1==0 and correct_u1: Mdict[k] -= recenter
            
    return Mdict

def sort_atoms(z_dict):
    sorted_atom_lst = sorted(z_dict, key=z_dict.get, reverse=True)
    return sorted_atom_lst

def calc_orthgonality(v1,v2):
    return 1 - np.abs(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    
def momentum_transform(momen_dict, sorted_atom_lst):

    ptc1 = sorted_atom_lst[0]

    ###### start with the highest-Z atom (tag=1), use its momentum direction as positive X, and find the atom whose momentum is most orthogonal to that of X, these two atoms will be used as ptc1 and ptc2 respectively.
    momen_ptc1 = momen_dict[ptc1]
    orth = -np.inf
    ptc2=2
    for ptc in momen_dict.keys():
        if ptc==ptc1:continue 
        orth_temp = calc_orthgonality(momen_ptc1,momen_dict[ptc])
        if orth_temp > orth:
            ptc2 = ptc
            orth = orth_temp
                
    
    momen_dict_td = MF_Momen(momen_dict, ptc1=ptc1, ptc2=ptc2, correct_u1=True, recenter=np.array([1, 0, 0]))
    return momen_dict_td, ptc1, ptc2


def geom_transform(geom_dict, ptc1, ptc2):
    for ik, k in enumerate(geom_dict.keys()):
        if ik==0:
            geom_arr = geom_dict[k]
        else:
            geom_arr = np.vstack([geom_arr, geom_dict[k]])
            
    geom_arr_mean = np.mean(geom_arr,axis=0)
    geom_dict_td = {}
    for ik, k in enumerate(geom_dict.keys()):
        geom_dict_td[k] = geom_dict[k] - geom_arr_mean
        
    geom_dict_td = MF_Momen(geom_dict, ptc1=ptc1, ptc2=ptc2, correct_u1=True, recenter=np.array([1, 0, 0]))
    
    return geom_dict_td


def get_batch(mol_name, variation, atom_lst, z_dict, q_dict, geom_dict_td, momen_dict_td):
    inds = []
    atoms = []
    charges = []
    xyz = []
    pxpypz = []
    num_atoms = len(atom_lst)
    for i, atom in enumerate(atom_lst):
        inds.append(0)
        atoms.append(z_dict[atom])
        charges.append(q_dict[atom])
        if i == 0:
            xyz = geom_dict_td[atom][np.newaxis, :]
            pxpypz = momen_dict_td[atom][np.newaxis, :]
        else:
            xyz = np.concatenate([xyz, geom_dict_td[atom][np.newaxis, :]], axis = 0)
            pxpypz = np.concatenate([pxpypz, momen_dict_td[atom][np.newaxis, :]], axis = 0)
    
    batch = Data(mol_name = mol_name, variation = variation, z = torch.tensor(atoms, dtype=torch.long),q = torch.tensor(charges, dtype=torch.long), batch = torch.tensor(inds, dtype=torch.long),
                 y = torch.from_numpy(xyz).to(torch.float32), pos = torch.from_numpy(pxpypz).to(torch.float32), natoms = torch.tensor([num_atoms], dtype=torch.long))
    return batch
