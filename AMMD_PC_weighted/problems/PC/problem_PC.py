
from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.PC.state_PC import StatePC


class PC(object):
    NAME = 'PC'

    @staticmethod
    def get_total_dis(dataset, pi):
        d_set = dataset['demand']
        f_set = dataset['facility']
        weight = dataset['weight']
        batch_size, n_d, _ = d_set.size()
        _, n_f, _ = f_set.size()
        _, p = pi.size()

        dist = (f_set[:, :, None, :] - d_set[:, None, :, :]).norm(p=2, dim=-1)
        ndist = torch.mul(dist, weight.unsqueeze(1).expand(batch_size, n_f, n_d))
        facility_tensor = pi.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_d))
        dist_p = ndist.gather(1, facility_tensor)
        length = torch.min(dist_p, 1)
        lengths = length[0].max(-1)[0]

        return lengths

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PCDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePC.initialize(*args, **kwargs)


class PCDataset(Dataset):
    def __init__(self, size_d=20, size_f=20, p=4, num_samples=5000, filename=None, offset=0,  distribution=None,seed=0):
        super(PCDataset, self).__init__()
        if seed==1:
            torch.manual_seed(1234)
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                # self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
                self.data = data
        else:
            # Sample nodes randomly in [0, 1] square
            if size_d == size_f:
                self.data = [{
                    'demand': torch.FloatTensor(size_d, 2).uniform_(0, 1),
                    'weight': torch.FloatTensor(size_d).uniform_(0, 1),
                    'p': p,

                }
                    for i in range(num_samples)
                ]

                for j in range(num_samples):
                    self.data[j].update({'facility': self.data[j]['demand']})

            else:
                self.data = [{
                    'demand': torch.FloatTensor(size_d, 2).uniform_(0, 1),
                    'weight': torch.FloatTensor(size_d).uniform_(0, 1),
                    'facility': torch.FloatTensor(size_f, 2).uniform_(0, 1),
                    'p': p
                }
                    for k in range(num_samples)
                ]

        self.size = len(self.data)
        self.p = p

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
