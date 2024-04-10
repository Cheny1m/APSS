from typing import NamedTuple

import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor

from .lexsort import torch_lexsort

def beam_search(*args, **kwargs):
    beams, final_state = _beam_search(*args, **kwargs)
    return get_beam_search_results(beams, final_state)


def get_beam_search_results(beams, final_state):
    beam = beams[-1]  # Final beam
    if final_state is None:
        return None, None, None, None, beam.batch_size

    # First state has no actions/parents and should be omitted when backtracking
    actions = [beam.action for beam in beams[1:]]
    parents = [beam.parent for beam in beams[1:]]

    solutions = final_state.construct_solutions(backtrack(parents, actions))
    return beam.score, solutions, final_state.get_final_cost()[:, 0], final_state.ids.view(-1), beam.batch_size


def _beam_search(state, beam_size, propose_expansions=None,
                keep_states=False):

    beam = BatchBeam.initialize(state)

    # Initial state
    beams = [beam if keep_states else beam.clear_state()]

    # Perform decoding steps
    while not beam.all_finished():

        # Use the model to propose and score expansions
        parent, action, score = beam.propose_expansions() if propose_expansions is None else propose_expansions(beam)
        if parent is None:
            return beams, None

        # Expand and update the state according to the selected actions
        beam = beam.expand(parent, action, score=score)

        # Get topk
        beam = beam.topk(beam_size)

        # Collect output of step
        beams.append(beam if keep_states else beam.clear_state())

    # Return the final state separately since beams may not keep state
    return beams, beam.state


class BatchBeam(NamedTuple):
   
    score: Tensor
    state: None  
    parent: Tensor 
    action: Tensor
    batch_size: int  
    device: None

    @property
    def ids(self):
        return self.state.ids.view(-1)

    def __getitem__(self, key):
        assert isinstance(key, Tensor) or isinstance(key, slice) 
        return self._replace(
            score=self.score[key] if self.score is not None else None,
            state=self.state[key],
            parent=self.parent[key] if self.parent is not None else None,
            action=self.action[key] if self.action is not None else None
        )

    @staticmethod
    def initialize(state):
        batch_size = len(state.ids)
        device = state.ids.device
        return BatchBeam(
            score = Tensor(batch_size, dtype=ms.float32, device=device), 
            state = state,
            parent = None,
            action = None,
            batch_size = batch_size,
            device = device
        )

    def propose_expansions(self):
        mask = self.state.get_mask()

        # Mask always contains a feasible action
        indices = ms.ops.nonzero(mask[:, 0, :] == 0)  
        parent = indices[:, 0]
        action = indices[:, 1]

        return parent, action, None

    def expand(self, parent, action, score=None):
        return self._replace(
            score=score,  # The score is cleared upon expanding as it is no longer valid, or it must be provided
            state=self.state[parent].update(action),  # Pass ids since we replicated state
            parent=parent,
            action=action
        )

    def topk(self, k):
        idx_topk = segment_topk_idx(self.score, k, self.ids)
        return self[idx_topk]

    def all_finished(self):
        return self.state.all_finished()
    
    def cpu(self):
        return self.to_device(ms.cpu()) 

    def to(self, device):
        if device == self.device:
            return self
        return self._replace(
            score=self.score.to(device) if self.score is not None else None,
            state=self.state.to(device),
            parent=self.parent.to(device) if self.parent is not None else None,
            action=self.action.to(device) if self.action is not None else None
        )

    def clear_state(self):
        return self._replace(state=None)

    def size(self):
        return self.state.ids.size(0)

def segment_topk_idx(x, k, ids):

    assert x.ndim == 1
    assert ids.ndim == 1

    splits = ops.nonzero(ids[1:] - ids[:-1]) + 1
    splits = ms.ops.concat((Tensor([0], dtype=ms.int32), splits))

    group_offsets = Tensor(shape=[splits.max() + 1], dtype=ms.int32)
    group_offsets[ids[splits]] = splits
    offsets = group_offsets[ids]

    # We want topk so need to sort x descending so sort -x (be careful with unsigned data type!)
    idx_sorted = torch_lexsort((-(x if x.dtype != ms.uint8 else x.int()).detach(), ids))

    # This will filter first k per group (example k = 2)
    # ids     = [0, 0, 0, 1, 1, 1, 1, 2]
    # splits  = [0, 3, 7]
    # offsets = [0, 0, 0, 3, 3, 3, 3, 7]
    # offs+2  = [2, 2, 2, 5, 5, 5, 5, 9]
    # arange  = [0, 1, 2, 3, 4, 5, 6, 7]
    # filter  = [1, 1, 0, 1, 1, 0, 0, 1]
    # Use filter to get only topk of sorting idx
    
    # before
    # return idx_sorted[torch.arange(ids.size(0), out=ids.new()) < offsets + k]

    # after
    return idx_sorted[ms.Tensor(ms.arange(ids.shape[0]), dtype=ms.int32) < offsets + k]
    

def backtrack(parents, actions):

    cur_parent = parents[-1]
    reversed_aligned_sequences = [actions[-1]]

    for parent, sequence in reversed(list(zip(parents[:-1], actions[:-1]))):
        cur_parent = ops.gather(parent, cur_parent, -1)
        reversed_aligned_sequences.append(ops.gather(sequence, cur_parent, -1))

    return ops.stack(list(reversed(reversed_aligned_sequences)), -1) 

class CachedLookup:
    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), "CachedLookup does not support slicing, you can slice the result of an index operation instead"
        assert isinstance(key, ops.Tensor),  "Key must be a Tensor"
        if self.key is None:
            self.key = key
            self.current = self.orig[key]
        elif len(key) != len(self.key) or (key != self.key).any():
            self.key = key
            self.current = self.orig[key]
        return self.current