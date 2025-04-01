import torch
from botorch.models.deterministic import DeterministicModel
from torch import Tensor

class WeightedFidelityCostModel(DeterministicModel):
    r"""Deterministic, cost model operating on fidelity parameters.

    For each (q-batch) element of a candidate set `X`, this module computes a
    cost of the form

        cost = fixed_cost + weights[X[fidelity_dim]]

    """

    def __init__(
        self,
        fidelity_weights: list[float],
        fixed_cost: float = 0.01,
    ) -> None:
        r"""
        Args:
            fidelity_weights: A list of costs associated with fidelities, cost of
            i-th fidelity is costs[i]
            fixed_cost: The fixed cost of running a single candidate point (i.e.
                an element of a q-batch).
        """
        super().__init__()
        self.fixed_cost = fixed_cost
        weights = torch.tensor(fidelity_weights)
        self.register_buffer("weights", weights)
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the cost on a candidate set X.

        Computes a cost of the form

            cost = fixed_cost + weights[X[fidelity_dim]]

        for each element of the q-batch

        Args:
            X: A `batch_shape x q x d'`-dim tensor of candidate points.

        Returns:
            A `batch_shape x q x 1`-dim tensor of costs.
        """
        lin_cost = self.weights[X[...,-1].int()]

        return self.fixed_cost + lin_cost.unsqueeze(-1)
