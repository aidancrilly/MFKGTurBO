from MFKGTuRBO import *

from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.optim import optimize_acqf,optimize_acqf_mixed
from botorch.acquisition.cost_aware import InverseCostWeightedUtility

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from MFKernels import MF_Matern_SingleTaskGP
from CostModels import WeightedFidelityCostModel

import time

def generate_multifidelity_cost_model(domain,fixed_cost=0.25):
    cost_model = WeightedFidelityCostModel(fidelity_weights=domain.costs, fixed_cost=fixed_cost)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    return cost_aware_utility

def generate_initial_sample(domain,sampler,initial_samples,batch_size=1):
    # Make sure divisible by Nbatch and greater than initial_samples
    initial_samples = int(np.ceil(initial_samples/batch_size))*batch_size
    # Get sample
    X = sampler.random(initial_samples)
    X = domain.inverse_transform(X)
    return X,initial_samples

def generate_batch(
    domain,
    state,
    GP_model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d + fidelities if multi-fidelity
    Y,  # Function values
    batch_size,
    num_restarts,
    raw_samples
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Scale the trust region to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = GP_model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    tr_bounds = torch.stack([tr_lb, tr_ub])

    # Expected Improvement
    ei = qLogExpectedImprovement(GP_model, Y.amax())
    X_next, _ = optimize_acqf(
        ei,
        bounds=tr_bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return X_next

def generate_multifidelity_batch(
    domain,
    state,
    cost_model, # Cost model for multi-fidelity
    GP_model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d + fidelities
    Y,  # Function values
    batch_size,
    num_restarts,
    raw_samples,
    num_fantasies
):
    
    Xs =  X[:,:-1]
    assert Xs.min() >= 0.0 and Xs.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Scale the trust region to be proportional to the lengthscales
    x_center = Xs[Y.argmax(), :].clone()
    # Select lengthscale from first GP in kernel
    # weights = GP_model.covar_module.kernels[0].base_kernel.lengthscale.squeeze().detach()
    weights = GP_model.covar_module.base_kernel.kernels[0].covar_module_unbiased.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    tr_bounds = torch.stack([tr_lb, tr_ub])
    print(tr_bounds)
    mf_bounds = torch.concat([tr_bounds,torch.Tensor([domain.minimal_fidelity,domain.target_fidelity]).unsqueeze(-1)],dim=1)

    # Generate multi-fidelity acquisition function
    mfkg_acqf = generate_MFKG_acqf(
        domain,
        cost_model,
        GP_model,
        mf_bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        num_fantasies=num_fantasies
        )
    
    # generate new candidates
    start = time.time()
    X_next, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=mf_bounds,
        fixed_features_list=domain.fidelity_features,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    print(f'Generating candidates, elasped time: {time.time()-start}')

    return X_next

def suggest_next_locations(model_trainer,domain,state,Xs,Ys,num_restarts=DEFAULT_NUM_RESTARTS,raw_samples=DEFAULT_RAW_SAMPLES):
    # Train the model
    X_torch,Y_torch,train_Y,likelihood,model = model_trainer(domain,state,Xs,Ys)

    # Create a batch
    X_next = generate_batch(
        domain=domain,
        state=state,
        GP_model=model,
        X=X_torch,
        Y=train_Y,
        batch_size=state.batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    # Get to CPU and remove any AD info...
    X_next = X_next.detach().cpu().numpy()

    # Transform to real domain
    X_next = domain.inverse_transform(X_next)

    # Duplicate detection
    #X_next = domain.duplicate_detection(Xs,X_next)

    return X_next

def suggest_next_multifidelity_locations(model_trainer,cost_model,domain,state,Xs,Ys,Ss,num_restarts=DEFAULT_NUM_RESTARTS,raw_samples=DEFAULT_RAW_SAMPLES,num_fantasies=DEFAULT_NUM_FANTASIES):
    # Train the model
    start = time.time()
    X_torch,Y_torch,train_Y,likelihood,model = model_trainer(domain,state,Xs,Ys,Ss)
    print(f'Training, elasped time: {time.time()-start}')

    # Create a batch
    X_next = generate_multifidelity_batch(
        domain=domain,
        state=state,
        cost_model=cost_model,
        GP_model=model,
        X=X_torch,
        Y=train_Y,
        batch_size=state.batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        num_fantasies=num_fantasies
    )

    # Get to CPU and remove any AD info...
    X_next = X_next.detach().cpu().numpy()

    X_next,S_next = X_next[:,:-1],X_next[:,-1:]
    # Transform to real domain
    X_next = domain.inverse_transform(X_next)

    # Duplicate detection
    #X_next = domain.duplicate_detection(Xs,X_next)

    return X_next,S_next

def train_singlefidelity_model(domain,state,Xs,Ys,return_torch=True,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL):
    # Transform to [0,1]^d
    Xs_unit = domain.transform(Xs)

    # Convert to torch tensors
    X_torch = torch.tensor(Xs_unit, dtype=dtype, device=device)
    Y_torch = torch.tensor(Ys, dtype=dtype, device=device)

    # Fit a GP model
    Y_std = 1.0
    if(Ys.shape[0] > 1):
        Y_std = Y_torch.std()

    train_Y = (Y_torch - Y_torch.mean()) / Y_std
    likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(
            nu=2.5, ard_num_dims=state.dim, lengthscale_constraint=Interval(*lengthscale_interval)
        )
    )
    model = SingleTaskGP(
            X_torch, train_Y, covar_module=covar_module, likelihood=likelihood
        )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Fit the model
    fit_gpytorch_mll(mll)

    if(return_torch): # BO mode
        return X_torch,Y_torch,train_Y,likelihood,model
    else: # Switch to evaluation mode
        model.eval()
        likelihood.eval()
        return likelihood,model
    
# def train_multifidelity_model(domain,state,Xs,Ys,Ss,return_torch=True,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL):
#     # Transform to [0,1]^d
#     transformed_Xs = domain.transform(Xs)
#     Xs_unit = np.c_[transformed_Xs,Ss]

#     # Convert to torch tensors
#     X_torch = torch.tensor(Xs_unit, dtype=dtype, device=device)
#     Y_torch = torch.tensor(Ys, dtype=dtype, device=device)

#     # Fit a GP model
#     Y_std = 1.0
#     if(Ys.shape[0] > 1):
#         Y_std = Y_torch.std()

#     train_Y = (Y_torch - Y_torch.mean()) / Y_std
#     likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))

#     model = MF_Matern_SingleTaskGP(
#             X_torch, train_Y, nu=2.5, lengthscale_interval=lengthscale_interval, likelihood=likelihood
#         )
    
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)

#     # Fit the model
#     fit_gpytorch_mll(mll)
    
#     if(return_torch): # BO mode
#         return X_torch,Y_torch,train_Y,likelihood,model
#     else: # Switch to evaluation mode
#         model.eval()
#         likelihood.eval()
#         return likelihood,model
    
def train_multifidelity_model(domain,state,Xs,Ys,Ss,return_torch=True,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL):
    # Transform to [0,1]^d
    transformed_Xs = domain.transform(Xs)
    Xs_unit = np.c_[transformed_Xs,Ss]

    # Convert to torch tensors
    X_torch = torch.tensor(Xs_unit, dtype=dtype, device=device)
    Y_torch = torch.tensor(Ys, dtype=dtype, device=device)

    # Fit a GP model
    Y_std = 1.0
    if(Ys.shape[0] > 1):
        Y_std = Y_torch.std()

    train_Y = (Y_torch - Y_torch.mean()) / Y_std
    likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))

    model = SingleTaskMultiFidelityGP(
            X_torch, train_Y, data_fidelities = [domain.dim], likelihood=likelihood
        )
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Fit the model
    fit_gpytorch_mll(mll)
    
    if(return_torch): # BO mode
        return X_torch,Y_torch,train_Y,likelihood,model
    else: # Switch to evaluation mode
        model.eval()
        likelihood.eval()
        return likelihood,model