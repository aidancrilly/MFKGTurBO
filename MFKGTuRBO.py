"""

Adapted from https://botorch.org/tutorials/turbo_1
and https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/

"""

import math
import numpy as np
from dataclasses import dataclass

import torch
from botorch.optim import optimize_acqf

from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# GP and BO hyperparameters
DEFAULT_NOISE_INTERVAL = [1e-8,1e-5]
DEFAULT_LENGTHSCALE_INTERVAL = [0.005,4.0]
DEFAULT_NUM_RESTARTS = 10
DEFAULT_RAW_SAMPLES  = 512

# Multi-Fidelity hyperparameters
DEFAULT_NUM_FANTASIES = 128

# TuRBO hyperparameters
DEFAULT_LENGTH = 0.8
DEFAULT_LENGTH_MIN = 0.5**7
DEFAULT_LENGTH_MAX = 1.6

@dataclass
class Domain:
    dim: int
    b_low : np.ndarray
    b_up : np.ndarray
    types : list
    steps : np.ndarray
    l_MultiFidelity : bool
    costs : list[float] | None = None

    def __post_init__(self):
        self.discrete_indices = [i for i in range(self.dim) if self.types[i] == 'discrete']
        self.discrete_dim = len(self.discrete_indices)
        self.discrete_bound = [int((self.b_up[i]-self.b_low[i])/self.steps[i]) for i in self.discrete_indices]
        if(self.l_MultiFidelity):
            self.minimal_fidelity  = 0
            self.target_fidelity   = len(self.costs)
            self.fidelities        = [i for i in range(self.target_fidelity)]
            self.fidelity_weights  = {self.dim: 1.0}
            self.target_fidelities = {self.dim: self.target_fidelity}
            self.fidelity_features = [{self.dim: f} for f in self.fidelities]

    def transform(self,X):
        X_scaled = X.copy()
        # Transform to [0,1]^d
        for n in range(self.dim):
            X_scaled[:,n] = (X_scaled[:,n]-self.b_low[n])/(self.b_up[n]-self.b_low[n])
        # Catch floating point errors
        X_scaled[np.isclose(X_scaled,0.0)] = 0.0
        X_scaled[np.isclose(X_scaled,1.0)] = 1.0
        return X_scaled

    def inverse_transform(self,X):
        X_scaled = X.copy()
        # Scale back to parameter space
        for n in range(self.dim):
            X_scaled[:,n] = (self.b_up[n]-self.b_low[n])*X_scaled[:,n]+self.b_low[n]
            # For discrete spaces, find nearest point
            if(self.types[n] == 'discrete'):
                X_scaled[:,n] = np.rint(X_scaled[:,n]/self.steps[n])*self.steps[n]
        return X_scaled
    
    def fidelity_project(self,X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)
    
    def duplicate_detection(self,X,X_next):
        if self.discrete_dim > 0:
            X_scaled        = self.transform(X)
            X_discrete      = np.rint(X_scaled[:,self.discrete_indices]/self.steps[self.discrete_indices]).astype(np.int32)
            X_next          = self.transform(X_next)
            X_next_discrete = np.rint(X_next[:,self.discrete_indices]/self.steps[self.discrete_indices]).astype(np.int32)
            N_candidates = X_next.shape[0]
            for i in range(N_candidates):
                X_set_discrete = np.concatenate((X_discrete,X_next_discrete[:i,:]),axis=0)
                while self.is_duplicate(X_next_discrete[i,:],X_set_discrete):
                    X_next_discrete[i,:] = self.random_step(X_next_discrete[i,:])
            
            X_next[:,self.discrete_indices] = X_next_discrete*self.steps[self.discrete_indices]
            X_next = self.inverse_transform(X_next)

        return X_next

    def is_duplicate(self,X_cand,X):
        dup = False
        for i in range(X.shape[0]):
            if all(X[i,:] == X_cand):
                dup = True
        return dup
    
    def random_step(self,X_cand):
        step = np.zeros(self.discrete_dim)
        while all(step == 0):
            step = np.random.randint(low=-1,high=2,size=self.discrete_dim)
        X_jump = X_cand+step
        print(X_jump,X_cand,step)
        X_jump[X_jump < 0] = 0
        for i in range(self.discrete_dim):
            if(X_jump[i] >= self.discrete_bound[i]):
                X_jump[i] = self.discrete_bound[i]-1
        return X_jump

@dataclass
class TurboState:
    """
    
    These settings assume that the domain has been scaled to [0,1]^d and that the same batch size is used for each iteration.
    
    """
    dim: int
    batch_size: int
    length: float = DEFAULT_LENGTH
    length_min: float = DEFAULT_LENGTH_MIN
    length_max: float = DEFAULT_LENGTH_MAX
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __init__(self,dim,batch_size,Ys):
        self.dim = dim
        self.batch_size = batch_size
        if(Ys is not None):
            self.best_value=Ys.max()

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

    def __call__(self,Y_next):
        if Y_next.max() > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, Y_next.max())
        if self.length < self.length_min:
            self.restart_triggered = True

def generate_MFKG_acqf(domain,cost_model,GP_model,bounds,num_restarts,raw_samples,num_fantasies):
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(GP_model),
        d=domain.dim+1,
        columns=[domain.dim],
        values=[domain.target_fidelity],
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 10, "maxiter": 200},
    )

    mfkg = qMultiFidelityKnowledgeGradient(
        model=GP_model,
        num_fantasies=num_fantasies,
        current_value=current_value,
        cost_aware_utility=cost_model,
        project=domain.fidelity_project,
    )

    return mfkg