import numpy as np
import pickle
import torch
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import T2FProblem
from utils import SimpleProblem2
from BioDataParser import Stoichiometry


import matrix_tools as mt

#logging
import datetime

from loguru import logger
import sys        # <!- add this line
logger.remove()             # <- add this line
logger.add(sys.stdout, level="TRACE")   # <- add this line
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
log_path=".\logs\log-"+str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","-")+".log"
logger.add(log_path, level="TRACE", format=log_format, colorize=False, backtrace=True, diagnose=True)


data=Stoichiometry("./Data/A.txt","./Data/b.txt","./Data/lb.txt","./Data/ub.txt","./Data/projector.npy")

A=data.get_a_matrix()
# num_m,num_r=A.shape
good_rows="good-row-5733-2023-12-22-00-59-59-904165"
good_columns="good-column-5733-2023-12-22-01-05-10-456617"
A=mt.make_chosen_matrix(A,good_rows,good_columns)

# A=A[good_matrix.good_rows,:]
# A=A.T[good_matrix.good_columns, :]
# A=A.T

# print(np.linalg.matrix_rank(A))
# print(A.shape)
# A=mt.full_rank_maker_5(A,threshold=1e-6)
# print(A.shape)
# A=mt.full_rank_maker_5(A,threshold=1e-6,mode="column")
# print(A.shape)
mt.random_invertible_finder(A)
num_m,num_r=A.shape


X=data.get_b_vector()
X=X.reshape((X.shape[0], -1))
X=mt.make_chosen_matrix(X,good_rows,None, "row")
X=X.reshape((1,X.shape[0]))

Y_min=data.get_lb()
Y_min=Y_min.reshape((Y_min.shape[0], -1)) 
Y_min=mt.make_chosen_matrix(Y_min,good_columns,None, "row").flatten()

Y_max=data.get_ub()
Y_max=Y_max.reshape((Y_max.shape[0], -1)) 
Y_max=mt.make_chosen_matrix(Y_max,good_columns,None, "row").flatten()

# print(X.shape)


# define G. a (2r,r) shape matrix which upper part prepare y for upper bound and second part for lower bound 
G_up=np.eye(num_r)
G_down=-np.eye(num_r)
G=np.vstack((G_up,G_down))

# print(G.shape)
#define h by stack concatenating upper and negative of lower bounds
h=np.vstack((Y_max,-Y_min))

#define Q full zero in (r,r) shape
Q=np.zeros((num_r,num_r))

#define p (need to review)
p=np.ones((num_r,1))
print()

num_var = Y_max.shape[0]
num_ineq = G.shape[0]
num_eq = A.shape[0]
num_examples = X.shape[0]

# print(Q)
# a=torch.linalg.cholesky(torch.tensor(Q))
# print(a)




####old codes
# G=np.array([[[1],[-1]]])
# Q=np.array([0]*Y_max.shape[0])
# Q=Q.reshape((1,Y_min.shape[0])) 

# num_var = Y_max.shape[1]
# num_ineq = 2*A.shape[0]
# num_eq = A.shape[0]
# num_examples = X.shape[0]
# p=[1]*num_var
# print(len(p))



# problem = T2FProblem(A,Y_min,Y_max,X)
# problem.calc_Y()
# print(len(problem.Y))

problem2=SimpleProblem2(Q,p,A,G,h,X)
problem2.calc_Y()
# print(len(problem2.Y))
# print(problem2.Y)

with open("./datasets/T2F/recon2.2_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(problem2, f)
logger.success("Data has been seccussfully generated")
