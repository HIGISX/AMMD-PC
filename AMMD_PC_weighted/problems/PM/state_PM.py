import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StatePM(NamedTuple):
    # Fixed input
    fac: torch.Tensor
    dem: torch.Tensor
    p: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    visited_: torch.Tensor  # Keeps track of nodes that have been visited

    # State
    facility: torch.Tensor  # B x p
    length: torch.Tensor  # obj val of current solution
    first_a: torch.Tensor
    prev_a: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(data, visited_dtype=torch.uint8):
        set_d = data['demand']
        set_f = data['facility']
        p = data['p'][0]
        batch_size, n_d, _ = set_d.size()
        _, n_f, _ = set_f.size()

        dist = (set_f[:, :, None, :] - set_d[:, None, :, :]).norm(p=2, dim=-1)
        facility_list = [[] for i in range(batch_size)]
        facility = torch.tensor(facility_list, device=set_f.device)

        length = torch.zeros(batch_size, 1, dtype=torch.long, device=set_f.device)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=set_f.device)
        return StatePM(
            fac=set_f,
            dem=set_d,
            p=p,
            dist=dist,
            first_a=prev_a,
            prev_a=prev_a,
            ids=torch.arange(batch_size, dtype=torch.int64, device=set_f.device)[:, None],  # Add steps dimension
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_f,
                    dtype=torch.uint8, device=set_f.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_f + 63) // 64, dtype=torch.int64, device=set_f.device)  # Ceil
            ),
            facility=facility,
            length=length,
            i=torch.zeros(1, dtype=torch.int64, device=set_f.device)  # Vector with length num_steps
        )




    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.length

    def get_length(self, facility):
        """
        :param facility: list, a list of facility index list,  if None, generate randomly
        :return: obj val of given facility_list
        """

        batch_size, n_d, _ = self.dem.size()
        _, p = facility.size()
        facility_tensor = facility.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_d))
        f_u_dist_tensor = self.dist.gather(1, facility_tensor)
        lengths = torch.sum(torch.min(f_u_dist_tensor, dim=1)[0], dim=-1)
        return lengths


    def update(self, selected):
        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        facility_list = self.facility.tolist()
        selected_list = selected.tolist()
        [facility_list[i].append(selected_list[i]) for i in range(len(selected))]

        new_facility = torch.tensor(facility_list, device=self.fac.device)
        new_length = self.get_length(new_facility)

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(visited_=visited_, facility=new_facility, prev_a=prev_a, length=new_length, i=self.i + 1)

    def all_finished(self):
        return self.i == self.p

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_ > 0  # Hacky way to return bool or uint8 depending on pytorch version