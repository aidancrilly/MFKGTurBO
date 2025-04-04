from MFKGTuRBO_utils import *
import matplotlib.pyplot as plt
from scipy.stats import qmc

def extract_2d_SF_posterior(domain, model, likelihood, N=100):
    """
    Extracts a 2D posterior distribution from an arbitrary-dimensional model.

    Args:
        domain (Domain): The problem domain containing parameter bounds.
        model (GPModel): The trained GP model.
        likelihood (Likelihood): The GP likelihood.
        N (int): Number of evaluation points per dimension.

    Returns:
        x1_grid, x2_grid (np.array): 2D mesh grid of selected parameters.
        mean_2d (np.array): Mean posterior values.
        var_2d (np.array): Variance of the posterior.
    """

    # Create a grid for the two selected parameters
    x1_grid = np.linspace(domain.b_low[0], domain.b_up[0], N)
    x2_grid = np.linspace(domain.b_low[1], domain.b_up[1], N)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)

    # Initialize test points
    test_points = np.ones((N * N, 2))
    test_points[:, 0] = x1_mesh.ravel()
    test_points[:, 1] = x2_mesh.ravel()

    # Transform to unit space
    test_points_unit = domain.transform(test_points)
    test_points_torch = torch.tensor(test_points_unit, dtype=dtype, device=device)

    # Predict posterior
    with torch.no_grad():
        observed_pred = likelihood(model(test_points_torch))
        mean_2d = observed_pred.mean.cpu().numpy()
        var_2d = observed_pred.variance.cpu().numpy()

    # Reshape to 2D grid
    mean_2d = mean_2d.reshape(N, N)
    var_2d = var_2d.reshape(N, N)

    return x1_mesh, x2_mesh, mean_2d, var_2d

def extract_2d_MF_posterior(domain, model, likelihood, N=100):
    """
    Extracts a 2D posterior distribution from an arbitrary-dimensional model.

    Args:
        domain (Domain): The problem domain containing parameter bounds.
        model (GPModel): The trained GP model.
        likelihood (Likelihood): The GP likelihood.
        N (int): Number of evaluation points per dimension.

    Returns:
        x1_grid, x2_grid (np.array): 2D mesh grid of selected parameters.
        mean_2d (np.array): Mean posterior values.
        var_2d (np.array): Variance of the posterior.
    """

    # Create a grid for the two selected parameters
    x1_grid = np.linspace(domain.b_low[0], domain.b_up[0], N)
    x2_grid = np.linspace(domain.b_low[1], domain.b_up[1], N)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)

    # Initialize test points
    test_points = np.ones((N * N, 2))
    test_points[:, 0] = x1_mesh.ravel()
    test_points[:, 1] = x2_mesh.ravel()

    # Transform to unit space
    test_points_unit = domain.transform(test_points)
    test_points_unit_S = np.c_[test_points_unit,np.ones(N*N)]
    test_points_torch = torch.tensor(test_points_unit_S, dtype=dtype, device=device)

    # Predict posterior
    with torch.no_grad():
        observed_pred = likelihood(model(test_points_torch))
        mean_2d = observed_pred.mean.cpu().numpy()
        var_2d = observed_pred.variance.cpu().numpy()

    # Reshape to 2D grid
    mean_hi_2d = mean_2d.reshape(N, N)
    var_hi_2d = var_2d.reshape(N, N)

    test_points_unit_S = np.c_[test_points_unit,np.zeros(N*N)]
    test_points_torch = torch.tensor(test_points_unit_S, dtype=dtype, device=device)

    # Predict posterior
    with torch.no_grad():
        observed_pred = likelihood(model(test_points_torch))
        mean_2d = observed_pred.mean.cpu().numpy()
        var_2d = observed_pred.variance.cpu().numpy()

    # Reshape to 2D grid
    mean_lo_2d = mean_2d.reshape(N, N)
    var_lo_2d = var_2d.reshape(N, N)

    return x1_mesh, x2_mesh, mean_hi_2d, mean_lo_2d, var_hi_2d, var_lo_2d

def f(x,mu,corr):
    d = x - mu
    inv_corr = np.linalg.inv(corr)
    arg = np.einsum('ni,ij,nj->n', d, inv_corr, d)
    return np.exp(-0.5*arg)

def func_for_optim(x,s):
    mu1  = np.array([0.4,0.6])
    sig1 = 0.8
    corr1 = sig1**2*np.array([[1.0,0.5],[0.5,1.0]])
    f1 = 2.0*f(x,mu1,corr1)

    mu2  = np.array([0.7,0.3])
    sig2 = 0.3
    corr2 = sig2**2*np.array([[1.0,-0.8],[-0.8,1.0]])
    f2 = (1-0.2*s[:,0])*f(x,mu2,corr2)

    mu3  = np.array([0.4,0.6])
    sig3 = 0.1
    corr3 = sig3**2*np.array([[1.0,0.2],[0.2,1.0]])
    f3 = (1-0.5*s[:,0])*f(x,mu3,corr3)

    mu4  = np.array([0.77,0.33])
    sig4 = 0.05
    corr4 = sig4**2*np.array([[1.0,0.2],[0.2,1.0]])
    f4 = 1.5*(1-0.95*s[:,0])*f(x,mu4,corr4)

    ftotal = f1 + f2 + f3 + f4
    return ftotal.reshape(-1,1)

dim = 2
Nbatch = 1
Nsample = 200
hi_fi_frac = 0.05
max_iter = 16
seed = 420

np.random.seed(seed)

sampler = qmc.LatinHypercube(d=2,seed=seed)
X = sampler.random(n=Nsample)
# X = np.random.uniform(low=0,high=1,size=(Nsample,2))

S = np.zeros((Nsample,1))
S[np.random.binomial(n=1,p=hi_fi_frac,size=Nsample).astype(int),0] = 1.0
Y = func_for_optim(X,S)

S_ind = S.flatten()
SF_X = X[S_ind==1]
SF_Y = np.atleast_2d(Y[S_ind==1])

# Multi-fidelity
MF_domain = Domain(
    dim=dim,
    b_low=np.zeros(dim),
    b_up=np.ones(dim),
    types=['continuous','continuous'],
    steps=np.ones(dim),
    l_MultiFidelity = True,
    costs = [0.4,0.6]
)

# Single fidelity
SF_domain = Domain(
    dim=dim,
    b_low=np.zeros(dim),
    b_up=np.ones(dim),
    types=['continuous','continuous'],
    steps=np.ones(dim),
    l_MultiFidelity = False
)

fig = plt.figure(dpi=200,figsize=(7,7))
ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6 = fig.add_subplot(336)
ax7 = fig.add_subplot(313)

SF_opt_state  = TurboState(dim=SF_domain.dim,batch_size=Nbatch,Ys=SF_Y)
MF_opt_state  = TurboState(dim=MF_domain.dim,batch_size=Nbatch,Ys=Y)
cost_model = generate_multifidelity_cost_model(MF_domain,fixed_cost=0.0)

ax7.set_xlabel('Cumulative Cost')
ax7.set_ylabel('Best Value')
ax7.set_ylim(1.0,4.0)

for current_iter in range(max_iter):
    print(current_iter)
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()

    # Multi-fidelity
    likelihood,model = train_multifidelity_model(MF_domain,MF_opt_state,X,Y,S,return_torch=False)
    x1_mesh, x2_mesh, mean_hi_2d, mean_lo_2d, var_hi_2d, var_lo_2d = extract_2d_MF_posterior(MF_domain, model, likelihood)

    Y_std = np.std(Y)
    Y_mean = np.mean(Y)
    mean_lo_2d = Y_std*mean_lo_2d+Y_mean
    mean_hi_2d = Y_std*mean_hi_2d+Y_mean
    ax1.pcolormesh(x1_mesh,x2_mesh,mean_lo_2d)
    ax2.pcolormesh(x1_mesh,x2_mesh,mean_hi_2d)

    ax4.pcolormesh(x1_mesh,x2_mesh,var_lo_2d,cmap='plasma')
    ax5.pcolormesh(x1_mesh,x2_mesh,var_hi_2d,cmap='plasma')

    S_ind = S.flatten()
    ax1.scatter(X[S_ind==0,0],X[S_ind==0,1],c=Y[S_ind==0],ec='k',vmin=np.amin(mean_lo_2d),vmax=np.amax(mean_lo_2d))
    ax2.scatter(X[S_ind==1,0],X[S_ind==1,1],c=Y[S_ind==1],ec='k',vmin=np.amin(mean_hi_2d),vmax=np.amax(mean_hi_2d))

    ax1.set_title('Low Fidelity')
    ax2.set_title('High Fidelity')

    X_next,S_next = suggest_next_multifidelity_locations(train_multifidelity_model,cost_model,MF_domain,MF_opt_state,X,Y,S)
    print(X_next,S_next)

    Y_next = func_for_optim(X_next,S_next)

    X = np.append(X,X_next.reshape(Nbatch,-1),axis=0)
    S = np.append(S,S_next.reshape(Nbatch,-1),axis=0)
    Y = np.append(Y,Y_next.reshape(Nbatch,-1),axis=0)

    if(S_next.flatten()[0] == 0):
        ax1.scatter(X_next[:,0],X_next[:,1],c=Y_next[:,0],ec='r',vmin=np.amin(mean_lo_2d),vmax=np.amax(mean_lo_2d))
    else:
        ax2.scatter(X_next[:,0],X_next[:,1],c=Y_next[:,0],ec='r',vmin=np.amin(mean_hi_2d),vmax=np.amax(mean_hi_2d))

    ax1.plot(x1_mesh.flatten()[np.argmax(mean_lo_2d)],x2_mesh.flatten()[np.argmax(mean_lo_2d)],'rx')
    ax2.plot(x1_mesh.flatten()[np.argmax(mean_hi_2d)],x2_mesh.flatten()[np.argmax(mean_hi_2d)],'rx')

    MF_opt_state(Y)

    # Single fidelity
    ax3.set_title('Single Fidelity')

    likelihood,model = train_singlefidelity_model(SF_domain,SF_opt_state,SF_X,SF_Y,return_torch=False)
    x1_mesh, x2_mesh, mean_2d, var_2d = extract_2d_SF_posterior(SF_domain, model, likelihood)

    Y_std = np.std(SF_Y)
    Y_mean = np.mean(SF_Y)
    mean_2d = Y_std*mean_2d+Y_mean
    ax3.pcolormesh(x1_mesh,x2_mesh,mean_2d)

    ax6.pcolormesh(x1_mesh,x2_mesh,var_2d,cmap='plasma')

    ax3.scatter(SF_X[:,0],SF_X[:,1],c=SF_Y,ec='k',vmin=np.amin(mean_2d),vmax=np.amax(mean_2d))

    X_next = suggest_next_locations(train_singlefidelity_model,SF_domain,SF_opt_state,SF_X,SF_Y)

    Y_next = func_for_optim(X_next,np.ones((X_next.shape[0],1)))

    SF_X = np.append(SF_X,X_next.reshape(Nbatch,-1),axis=0)
    SF_Y = np.append(SF_Y,Y_next.reshape(Nbatch,-1),axis=0)

    ax3.scatter(X_next[:,0],X_next[:,1],c=Y_next[:,0],ec='r',vmin=np.amin(mean_2d),vmax=np.amax(mean_2d))

    ax3.plot(x1_mesh.flatten()[np.argmax(mean_2d)],x2_mesh.flatten()[np.argmax(mean_2d)],'rx')

    SF_opt_state(SF_Y)

    MF_costs = 0.01*S[S==0].shape[0] + S[S==1].shape[0]
    SF_costs = SF_Y.shape[0]
    MF_best_f = np.amax(Y[S==1])
    SF_best_f = np.amax(SF_Y)

    ax7.plot(MF_costs,MF_best_f,'ro',label='Multi-fidelity')
    ax7.plot(SF_costs,SF_best_f,'bo',label='Single fidelity')

    if(current_iter == 0):
        ax7.legend(frameon=False)
    
    fig.tight_layout()
    fig.savefig(f'./plots/optim_{str(current_iter).zfill(3)}.png')