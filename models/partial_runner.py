import torch

from typing import Optional, List


class PartialRunner:
    def __init__(self):
        self.L = None

    def validate_args(
        self,
        input_ids: Optional[torch.Tensor],
        hidden_states: Optional[torch.Tensor],
        layer_indices: Optional[List[int]],
    ):
        layer_indices = list(layer_indices)
        # The layer_indices must be in sorted order
        assert layer_indices == sorted(layer_indices)
        # No duplicates
        assert len(layer_indices) == len(set(layer_indices))
        # The indices never go out of range
        assert min(layer_indices) >= 0
        assert max(layer_indices) < self.L
        # input_ids xor hidden_states are being used as input
        assert (input_ids is None) != (hidden_states is None)

        # If input_ids are specified, then we start at the first layer
        # Otherwise, if hidden_states are specified, we start at some other layer
        # if input_ids is not None:
        #     assert layer_indices[0] == 0
        # elif hidden_states is not None:
        #     assert layer_indices[0] != 0
