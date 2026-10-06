import torch.utils.data as Data
import h5py
import numpy as np
import torch

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path, augment=False):
        self.h5file_path = h5file_path
        self.augment = augment
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        VIS, IR = torch.Tensor(VIS), torch.Tensor(IR)
        if self.augment:
            # Same random rot90/flip for both modalities so the pair stays aligned; patches are square.
            k = int(torch.randint(4, (1,)))
            VIS, IR = torch.rot90(VIS, k, dims=(1, 2)), torch.rot90(IR, k, dims=(1, 2))
            if torch.rand(1).item() < 0.5:
                VIS, IR = torch.flip(VIS, dims=(2,)), torch.flip(IR, dims=(2,))
        return VIS, IR
