import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


def construct_solutions(actions):
    return actions


class StateMCLP(NamedTuple):
    # Fixed input
    fac: torch.Tensor
    dem: torch.Tensor
    p: torch.Tensor
    radius: torch.Tensor
    # weight:torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to   index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    facility: torch.Tensor  # B x p
    # mask_cover
    mask_cover: torch.Tensor
    cover_num: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(data, visited_dtype=torch.bool):
        set_d = data['demand']
        set_f = data['facility']
        p = data['p'][0]
        radius = data['radius'][0]
        batch_size, n_d, _ = set_d.size()
        _, n_f, _ = set_f.size()
        dist = (set_f[:, :, None, :] - set_d[:, None, :, :]).norm(p=2, dim=-1)

        facility_list = [[] for i in range(batch_size)]
        facility = torch.tensor(facility_list, device=set_f.device)

        cover_num = torch.zeros(batch_size, 1, dtype=torch.long, device=set_f.device)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=set_f.device)

        return StateMCLP(
            fac=set_f,
            dem=set_d,
            p=p,
            radius=radius,
            dist=dist,
            ids=torch.arange(batch_size, dtype=torch.int64, device=set_f.device)[:, None],  # Add steps dimension
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_f,
                    dtype=torch.bool, device=set_f.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_f + 63) // 64, dtype=torch.int64, device=set_f.device)  # Ceil
            ),
            mask_cover=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_f,
                    dtype=torch.bool, device=set_f.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_f + 63) // 64, dtype=torch.int64, device=set_f.device)  # Ceil
            ),
            facility=facility,
            cover_num=cover_num,
            prev_a=prev_a,
            i=torch.zeros(1, dtype=torch.int64, device=set_f.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.cover_num

    def get_cover_num(self, facility):
        """
        :param facility: list, a list of facility index list,  if None, generate randomly
        :return: obj val of given facility_list
        """

        batch_size, n_d, _ = self.dem.size()
        _, p = facility.size()
        radius = self.radius
        facility_tensor = facility.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_d))
        f_u_dist_tensor = self.dist.gather(1, facility_tensor)
        mask = f_u_dist_tensor < radius
        mask = torch.sum(mask, dim=1)
        cover_num = torch.count_nonzero(mask, dim=-1)

        return cover_num

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        cur_coord = self.fac[self.ids, prev_a]
        facility_list = self.facility.tolist()
        slected_list = selected.tolist()
        [facility_list[i].append(slected_list[i]) for i in range(len(selected))]

        new_facility = torch.tensor(facility_list, device=self.fac.device)
        new_cover_num = self.get_cover_num(new_facility)

        if self.visited_.dtype != torch.bool:
            visited_ = mask_long_scatter(self.visited_, prev_a)
        else:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        # mask covered cities
        mask_cover = visited_

        return self._replace(visited_=visited_, mask_cover=mask_cover, prev_a=prev_a,
                             facility=new_facility, cover_num=new_cover_num, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i == self.p

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.mask_cover
