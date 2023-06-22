from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle

from problems.PM.state_PM import StatePM
# from problems.csp.state_csp_varibleNC import StateCSP

from utils.beam_search import beam_search


class PM(object):
    NAME = 'PM'

    @staticmethod
    def get_total_dis(dataset, pi):
        d_set = dataset['demand']
        f_set = dataset['facility']
        batch_size, n_d, _ = d_set.size()
        _, n_f, _ = f_set.size()
        _, p = pi.size()

        dist = (f_set[:, :, None, :] - d_set[:, None, :, :]).norm(p=2, dim=-1)
        facility_tensor = pi.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_d))
        dist_p = dist.gather(1, facility_tensor)
        length = torch.min(dist_p, 1)
        lengths = length[0].sum(-1)
        return lengths

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PMDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePM.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = PM.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )

        return beam_search(state, beam_size, propose_expansions)


def cal_dist_matrix(loc):
    loc1 = torch.unsqueeze(loc, dim=1)
    loc2 = loc[None, ...]
    return torch.sum((loc1-loc2)**2, dim=-1) ** 0.5




class PMDataset(Dataset):
    def __init__(self, size_d=20, size_f=20, p=4, num_samples=5000, filename=None, offset=0, distribution=None):
        super(PMDataset, self).__init__()
        # torch.manual_seed(1234)
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
                    'p': p
                }
                    for i in range(num_samples)
                ]

                for j in range(num_samples):
                    self.data[j].update({'facility': self.data[j]['demand']})

            else:
                self.data = [{
                    'demand': torch.FloatTensor(size_d, 2).uniform_(0, 1),
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
