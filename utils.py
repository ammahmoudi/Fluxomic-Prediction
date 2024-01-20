import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import scipy.linalg as la
import numpy as np
import pyomo.environ as pyo
# Import the solver
from pyomo.opt import SolverFactory

import osqp
from qpth.qp import QPFunction

from scipy.linalg import svd
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

# from pypower.api import case57
# from pypower.api import opf, makeYbus
# from pypower import idx_bus, idx_gen, ppoption

#logging
from loguru import logger
import pickle
import datetime

# import sys        # <!- add this line
# logger.remove()             # <- add this line
# logger.add(sys.stdout, level="TRACE")   # <- add this line
# log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
# log_path=".\logs\log-"+str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","-")+".log"
# logger.add(log_path, level="TRACE", format=log_format, colorize=False, backtrace=True, diagnose=True)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()



###################################################################
# SIMPLE PROBLEM
###################################################################

class T2FProblem:
    """ 
     minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h

        Where:
            num_r= Number of reactions
            num_m= number of metabolites
            num_ineq= 2*num_r
            num_m=num_eq 
            X=(num_examples,num_m)
            Y,Y_min,Y_max=(num_examples,num_r)  
            A=(num_m,num_r)
            G=(2*num_r,num_r)
            h=(num_examples, 2*num_r)
           
            Q=(num_r,num_r)
            p=(num_examples, num_r)



        More info:
            y=v
            p=-c
            Q=0        
            h=hstack[y_up,-y_min]
            G=hstack(I,-I)
            A=S
            Gy<=h

    """
    def __init__(self, Q, p, A, G, h, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        det = 0
        i = 0
        # print(self._ydim)
        # print(self._neq)
        good_columns="good-column-5733-2023-12-22-01-05-10-456617"
        with open(good_columns, "rb") as fp:   #Pickling
            columns = pickle.load(fp)
        while abs(det) < 0.0001 and i < 1000:
            # self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            # self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            # print(good_columns)
            # good_columns=np.nd
            self._other_vars=columns
            self._partial_vars = np.setdiff1d( np.arange(self._ydim), self._other_vars)

            det = torch.det(self._A[:, self._other_vars])
            logger.info("det of iteration "+str(i)+" is: "+str(det))
            i += 1
        if i == 1000:
            raise Exception
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'T2FProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def h(self):
        return self._h

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac
    @property
    def trainH(self):
        return self.h[:int(self.num*self.train_frac)]

    @property
    def validH(self):
        return self.h[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testH(self):
        return self.h[int(self.num*(self.train_frac + self.valid_frac)):]
    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]
    
    @property
    def device(self):
        return self._device
    
    def obj_fn(self, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T
    
    def get_h_by_indices(self, indices):
        return self.h[indices]

    def ineq_resid(self, X, Y,mode="full"):
        if mode=="test":h=self.testH
        elif mode=='valid':h=self.validH
        elif mode=='train':h=self.trainH
        else: h=self.h
        return Y@self.G.T - h

    def ineq_dist(self, X, Y,mode="full"):
        resids = self.ineq_resid(X, Y,mode=mode)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y, mode="full"):
        ineq_dist = self.ineq_dist(X, Y, mode=mode)
        return 2*ineq_dist@self.G

    def ineq_partial_grad(self, X, Y, mode="full"):
        if mode=="test":h=self.testH
        elif mode=='valid':h=self.validH
        elif mode=='train':h=self.trainH
        else: h=self.h
        logger.trace("calculating G effective with G_partial="+str(self.G[:, self.partial_vars].shape)+", G_others="+str(self.G[:, self.other_vars].shape)+", A_other_inv="+str(self._A_other_inv.shape)+", A_partial="+str(self._A_partial.shape))
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        logger.trace('G_effective with shape='+str(G_effective.shape))
        # print(G_effective)
        logger.trace("calculating h effective with h="+str(h.shape)+", X="+str(X.shape)+", A_other_inv="+str(self._A_other_inv.shape)+", G_others="+str(self.G[:, self.other_vars].shape))
        hx=(X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        logger.trace('hx with shape='+str(hx.shape))
        h_effective = h - hx
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        logger.trace('caluclating complete partial with Y='+str(Y.shape)+", Z="+str(Z.shape)+", A_partial="+str(self._A_partial.shape)+"A_other_inv="+str(self._A_other_inv.shape))
        logger.trace('partial_vars='+str(len(self.partial_vars))+", other_vars="+str(len(self.other_vars)))
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        # import pandas as pd
        # print("partial")
        # print(pd.DataFrame(Y.detach().cpu().numpy()).describe())
        return Y

    def opt_solve(self, X, solver_type='osqp', tol=1e-4,mode="full"):

        if solver_type == 'qpth':
            logger.trace('running qpth')
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False,check_Q_spd=False)(self.Q, self.p, self.G, self.h, self.A, X)
            end_time = time.time()
            # print(self.Q.count_nonzero())
            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
        
        elif solver_type == 'osqp':
            logger.trace('running osqp')
            Q, p, A, G = \
                self.Q_np, self.p_np, self.A_np, self.G_np
            if mode=="test":h=self.testH
            elif mode=='valid':h=self.validH
            elif mode=='train':h=self.trainH
            else: h=self.h
            h=h.detach().cpu().numpy()
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            # print("X ",str(X_np.shape))
            # print("h ",str(h.shape))
            # print("G ",G.shape)
            G1,G2=np.split(G,2)
            my_A = np.vstack([A, G1])
            y_max,y_min=np.split(h,indices_or_sections=2,axis=1)
            y_min=-y_min
            for X_id in range(len(X_np)):
                Xi=X_np[X_id]
                solver = osqp.OSQP()

                # my_X=X_np.reshape((X.shape[1], -1))
                # print("my_A",my_A.shape)
                # print("y_max:",y_max.shape)
                
                my_l = np.hstack([Xi, y_min[X_id].ravel()])
                my_u = np.hstack([Xi, y_max[X_id].ravel()])
                # print(my_l.shape)
                # print(my_u.shape)
                # print(my_l[5733],my_u[5733])
                solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()
                

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    Y.append(results.x)
                    logger.trace("Problem solved for sample "+str(X_id)+" in time "+str(end_time-start_time))
                else:
                    Y.append(np.ones(self.ydim) * np.nan)
                    logger.warning("Problem not solved for sample "+str(X_id)+" in time "+str(end_time-start_time))


            sols = np.array(Y)
            parallel_time = total_time/len(X_np)
            logger.success("Problem has been solved in time "+str(total_time)+",and parallel time="+str(parallel_time)+", using OSQP with Y in size:"+str(sols.shape))

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X,solver_type="osqp")[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)  
        self._num = feas_mask.sum()
        
        logger.info("Number of Feasible Samples="+str(self._num))
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        self._h = torch.tensor(self._h[feas_mask])
        self._Y=torch.tensor(Y)

        return Y

