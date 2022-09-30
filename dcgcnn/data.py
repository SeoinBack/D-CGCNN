from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import math as m
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

# Function added at D-CGCNN 

def compass(sph_coor, theta):
    """
    Arg:
    sph_coor : np.array(radius, azimuth (degree) , elevation (degree))
    theta : float

    Return:
    key :str longitude position + latitude position
    """
    
    lon_pos = m.floor(((sph_coor[1]+theta/2)%360)/theta)
    lat_pos = m.floor((sph_coor[2]+theta/2)/theta)
    
    
    #lat_pos = m.floor((sph_coor[2])/theta)
    #if lat_pos == 180/theta:
    #    lat_pos -= 1
        
    if (lat_pos == 0/theta) | (lat_pos == 180/theta):
        lon_pos = 0
    return str(lon_pos)+'-'+str(lat_pos)

def cart2sph(cart_coord, radian = False):
    """
    Arg:
    cart_coord : np.array(x, y, z)
    raidan : Return as radian else degree
    
    Return : 
    sph_coord : np.array(r, lambda, phi)
    """
    x = cart_coord[0]
    y = cart_coord[1]
    z = cart_coord[2]
    
    XYsq = x**2 + y**2
    r = m.sqrt(XYsq + z**2)
    elev = m.atan2(z, m.sqrt(XYsq))
    az = m.atan2(y,x)
    
    if radian:
        sph_coord = np.array([r,az,elev])
    else:
        lamb = (m.degrees(az) + 360) % 360
        phi = abs(m.degrees(elev) - 90)
        sph_coord = np.array([r,lamb,phi])

    return sph_coord
    
# CGCNN Function
    
def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):

    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):

    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance(object):

    def __init__(self, dmin, dmax, step, var=None):
    
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin-step, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
    
        return np.exp(-(distances - self.filter)**2 /
                      self.var**2)
        

class AtomInitializer(object):

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr=15, radius=8, dmin=0, step=0.4, theta = 45,
                 random_seed=123):
        assert 180 % theta == 0  
        self.root_dir = root_dir
        self.theta = theta
        self.keys = []
        
        #### Key generator for 180%theta != 0
        for lon in range(int(360/self.theta)):
            for lat in range(int((180+self.theta)/self.theta)):
                self.keys.append(str(lon)+'-'+str(lat))
        ####
        
        #### Key generator for 180%theta == 0    
        #for lon in range(int(360/self.theta)):
        #    for lat in range(int((180)/self.theta)):
        #        self.keys.append(str(lon)+'-'+str(lat))
        ####
        
        self.max_num_nbr, self.radius = max_num_nbr, radius

        
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        

        for index,nbr in enumerate(all_nbrs):
            
            drt_dict ={}
            checksum = {}
            for key in self.keys:
                drt_dict[key] = [0,np.zeros(len(self.gdf.filter)*3),0]
                checksum[key] = 0
            a = crystal[index].coords
            
            for nb in nbr:
                b = nb[0].coords
                nbr_idx = nb[2]
                cart_coords = np.array([j-i for i,j in zip(a,b)])
                sph_coords = cart2sph(cart_coords)
                key = compass(sph_coords,self.theta)
                
                gdf_vector = np.concatenate([self.gdf.expand(sph_coords[0]),self.gdf.expand(sph_coords[1]*self.radius/360), self.gdf.expand(sph_coords[2]*self.radius/180)])

                if checksum[key] == 0:
                    drt_dict[key] = [nbr_idx,gdf_vector]
                    checksum[key] = 1
                else:
                    pass
                
            drt_idx = []
            drt = []
            for item in drt_dict.items():
                drt_idx.append(item[1][0])
                drt.append(item[1][1])
            nbr_fea_idx.append(drt_idx)
            nbr_fea.append(drt)


        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
