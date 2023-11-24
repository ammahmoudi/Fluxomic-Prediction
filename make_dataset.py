import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import T2FProblem
from BioDataParser import Stoichiometry
#logging
from loguru import logger
import sys        # <!- add this line
# logger.remove(0)             # <- add this line
logger.add(sys.stdout, level="TRACE")   # <- add this line

data=Stoichiometry("./Data/A.txt","./Data/b.txt","./Data/lb.txt","./Data/ub.txt","./Data/projector.npy")

A=data.get_a_matrix()
X=data.get_b_vector()
X=X.reshape((1,X.shape[0]))
Y_min=data.get_lb()
Y_min=Y_min.reshape((1,Y_min.shape[0])) 
Y_max=data.get_ub()
Y_max=Y_max.reshape((1,Y_max.shape[0]))

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)


num_var = Y_max.shape[1]
num_ineq = 2*A.shape[0]
num_eq = A.shape[0]
num_examples = X.shape[0]


# num_ineq = 50
# for num_eq in [10, 30, 50, 70, 90]:
#     print(num_ineq, num_eq)
#     np.random.seed(17)
#     Q = np.diag(np.random.random(num_var))
#     p = np.random.random(num_var)
#     A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
#     X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
#     G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
#     h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

problem = T2FProblem(A,Y_min,Y_max,X)
problem.calc_Y()
print(len(problem.Y))

with open("./Data/recon2.2_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(problem, f)
logger.success("Data has been seccussfully generated")
