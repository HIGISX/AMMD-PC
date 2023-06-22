from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.MCLP.state_MCLP import StateMCLP


class MCLP(object):
    NAME = 'MCLP'

    @staticmethod
    def get_total_num(dataset, pi):

        d_set = dataset['demand']
        f_set = dataset['facility']
        weight =dataset['weight']
        batch_size, n_d, _ = d_set.size()
        _, n_f, _ = f_set.size()
        _, p = pi.size()
        radius = dataset['radius'][0]
        dist = (f_set[:, :, None, :] - d_set[:, None, :, :]).norm(p=2, dim=-1)
        facility_tensor = pi.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_d))
        f_u_dist_tensor = dist.gather(1, facility_tensor)
        mask = f_u_dist_tensor < radius
        mask = torch.sum(mask, dim=1)
        cover=(mask>=1)+0
        cover_num = torch.sum(cover*(weight.squeeze(-1)),dim=1)
        # cover_num = torch.count_nonzero(mask, dim=-1)
        # cover_num = torch.as_tensor(cover_num, dtype=torch.float32)
        #
        return cover_num

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MCLPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMCLP.initialize(*args, **kwargs)


class MCLPDataset(Dataset):
    def __init__(self, size_d=20, size_f=20, p=4, r=0.4,num_samples=5000, filename=None, offset=0, distribution=None):
        super(MCLPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                # self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
                self.data = data
        else:
            # Sample points randomly in [0, 1] square
            if size_d == size_f:
                self.data = [{
                    'demand': torch.FloatTensor(size_d, 2).uniform_(0, 1),
                    'p': p,
                    'radius':r,
                    'weight': torch.FloatTensor(size_d).uniform_(0, 5)
                }
                    for i in range(num_samples)
                ]

                for j in range(num_samples):
                    self.data[j].update({'facility': self.data[j]['demand']})
            else:
                self.data = [{
                    'demand': torch.FloatTensor(size_d, 2).uniform_(0, 1),
                    'weight': torch.FloatTensor(size_d, 1).uniform_(0, 5),
                    'facility': torch.FloatTensor(size_f, 2).uniform_(0, 1),
                    'p': p,
                    'radius': r
                }
                    for k in range(num_samples)
                ]

        self.size = len(self.data)
        self.p = p
        self.radius = r

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]