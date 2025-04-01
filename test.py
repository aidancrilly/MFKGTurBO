from MFKGTuRBO_utils import *
import matplotlib.pyplot as plt

def func_for_optim(x,s):
    return -np.sum((x-(0.5+0.1*s))**2,axis=-1,keepdims=True)

dim = 2
Nbatch = 1
Nsample = 10
max_iter = 10

# Single fidelity
domain = Domain(
    dim=dim,
    b_low=np.zeros(dim),
    b_up=np.ones(dim),
    types=['continuous','continuous'],
    steps=np.ones(dim),
    l_MultiFidelity = False
)
X = np.random.uniform(low=0,high=1,size=(Nsample,2))
Y = func_for_optim(X,np.ones((Nsample,1)))

opt_state = TurboState(dim=domain.dim,batch_size=Nbatch,Ys=Y)

# for current_iter in range(max_iter):
#     print(current_iter)

#     X_next = suggest_next_locations(train_singlefidelity_model,domain,opt_state,X,Y)

#     Y_next = func_for_optim(X_next,np.ones_like(X_next))

#     X = np.append(X,X_next.reshape(Nbatch,-1),axis=0)
#     Y = np.append(Y,Y_next.reshape(Nbatch,-1),axis=0)

#     opt_state(Y)
#     if(opt_state.restart_triggered):
#         print('TuRBO restart triggered, exiting...')
#         from sys import exit
#         exit()

# Multi-fidelity
MF_domain = Domain(
    dim=dim,
    b_low=np.zeros(dim),
    b_up=np.ones(dim),
    types=['continuous','continuous'],
    steps=np.ones(dim),
    l_MultiFidelity = True,
    costs = [0.1,1.0]
)
X = np.random.uniform(low=0,high=1,size=(Nsample,2))
S = np.random.randint(low=0,high=2,size=(Nsample,1))

Y = func_for_optim(X,S)
print(Y.shape)

opt_state  = TurboState(dim=MF_domain.dim,batch_size=Nbatch,Ys=Y)
cost_model = generate_multifidelity_cost_model(MF_domain,fixed_cost=0.0)

for current_iter in range(max_iter):
    print(current_iter)

    X_next,S_next = suggest_next_multifidelity_locations(train_multifidelity_model,cost_model,MF_domain,opt_state,X,Y,S)
    print(X_next,S_next)

    Y_next = func_for_optim(X_next,S_next)

    X = np.append(X,X_next.reshape(Nbatch,-1),axis=0)
    S = np.append(S,S_next.reshape(Nbatch,-1),axis=0)
    Y = np.append(Y,Y_next.reshape(Nbatch,-1),axis=0)

    opt_state(Y)
    if(opt_state.restart_triggered):
        print('TuRBO restart triggered, exiting...')
        from sys import exit
        exit()