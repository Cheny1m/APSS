import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as mnp

from apss.utils.boolmask import mask_long2bool, mask_long_scatter

def initialize_pp_state(loc, visited_dtype=ms.uint8):
    batch_size, n_loc, _ = loc.shape
    prev_a = ops.fill(ms.int64, (batch_size, 1), -1)
    return StatePP(
        loc=loc,
        ids=mnp.arange(0, batch_size, dtype=ms.int64).reshape(batch_size, 1),
        prev_a=prev_a,
        visited_=(
            ops.zeros((batch_size, 1, n_loc), ms.uint8)
            if visited_dtype == ms.uint8
            else ops.zeros((batch_size, 1, (n_loc + 63) // 64), ms.int64)
        ),
        i = ops.zeros(1,ms.int64)
    )

# @ms.jit
class StatePP:
    def __init__(self, 
                 loc: ms.Tensor, 
                 ids: ms.Tensor,
                 prev_a: ms.Tensor,
                 visited_: ms.Tensor,
                 i: ms.Tensor):
        self.loc = loc
        self.ids = ids
        self.prev_a = prev_a
        self.visited_ = visited_
        self.i = i

    @property
    def visited(self):
        if self.visited_.dtype == ms.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.shape[-2])
    
    def __getitem__(self, key):
        assert isinstance(key, ms.Tensor) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            # first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            # lengths=self.lengths[key],
            # cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    # @staticmethod
    # def initialize(loc, visited_dtype=ms.uint8):
    #     # print("初始化并返回state_pp对象")
    #     batch_size, n_loc, _ = loc.shape
    #     # prev_a = ops.zeros((batch_size, 1), ms.int64) - 1
    #     prev_a = ops.fill(ms.int64, (batch_size, 1), -1)
    #     return StatePP(
    #         loc=loc,
    #         # dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
    #         ids=mnp.arange(0, batch_size, dtype=ms.int64).reshape(batch_size, 1),  # Add steps dimension
    #         # first_a=prev_a,
    #         prev_a=prev_a,
    #         # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
    #         visited_=(  # Visited as mask is easier to understand, as long more memory efficient
    #             ops.zeros((batch_size, 1, n_loc),ms.uint8)
    #             if visited_dtype == ms.uint8
    #             else ops.zeros((batch_size, 1, (n_loc + 63) // 64), ms.int64)  # Ceil
    #         ),
    #         # lengths=ms.zeros((batch_size, 1)),
    #         # cur_coord=None,
    #         i=ops.zeros(1, ms.int64)  # Vector with length num_steps
    #     )
    
    @staticmethod
    def initialize(loc, visited_dtype=ms.uint8):
        batch_size, n_loc, _ = loc.shape
        prev_a = ops.fill(ms.int64, (batch_size, 1), -1)
        return StatePP(
            loc=loc,
            ids=ops.arange(0, batch_size, dtype=ms.int64).view(batch_size, 1),
            prev_a=prev_a,
            visited_=(
                ops.zeros((batch_size, 1, n_loc), ms.uint8)
                if visited_dtype == ms.uint8
                else ops.zeros((batch_size, 1, (n_loc + 63) // 64), ms.int64)
            ),
            i=ops.zeros(1, ms.int64)
        )
    

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, axis=-1)

    # def update(self, selected):

    #     # Update the state
    #     prev_a = selected.reshape(selected.shape[0], 1)  # Add dimension for step
    #     if self.visited_.dtype == ms.uint8:
    #         # Add one dimension since we write a single value
    #         visited_ = ms.scatter_add(self.visited_, -1, prev_a.reshape(prev_a.shape[0], prev_a.shape[1], 1), 1)
    #     else:
    #         visited_ = mask_long_scatter(self.visited_, prev_a)

    #     return self._replace(prev_a=prev_a, visited_=visited_, i=self.i + 1)
    
    def update(self, selected):
        prev_a = selected[:, None]

        if self.visited_.dtype == ms.uint8:
            # Mindspore1.10.1实现
            updates = mnp.ones(prev_a.shape,ms.uint8)
            visited_ = ops.tensor_scatter_elements(self.visited_, prev_a[:, :, None] ,updates[:,:,None],-1)
            # visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)  # You will need to implement mask_long_scatter for MindSpore

        return self._replace(prev_a=prev_a, visited_=visited_, i=self.i + 1)  # Assuming _replace is a method in your class


    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.shape[-2]

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.shape[-2] - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, axis=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.shape[-2]
        k = min(k, self.loc.shape[-2] - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, axis=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions

    def _replace(self, **kwargs):
        new_state = StatePP(
            loc=kwargs.get('loc', self.loc),
            ids=kwargs.get('ids', self.ids),
            prev_a=kwargs.get('prev_a', self.prev_a),
            visited_=kwargs.get('visited_', self.visited_),
            i=kwargs.get('i', self.i)
        )
        return new_state

    def __repr__(self):
        return f"StatePP(loc={self.loc}, ids={self.ids}, prev_a={self.prev_a}, visited_={self.visited_}, i={self.i})"
