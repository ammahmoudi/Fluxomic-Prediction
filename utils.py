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




class T2FProblem:
    """ 
        minimize y
        s.t.        AY=X
                    Y_min < Y < Y_max   

        num_r= Number of reactions
        num_m= numer of metabloites
        num_ineq= 2*num_r
        num_m=num_eq 
        X=(num_examples,num_m)
        A=(num_m,num_r)
        Y,Y_min,Y_max=(num_examples, num_r)        
    """
    def __init__(self, A, Y_min, Y_max, X, valid_frac=0.0833, test_frac=0.0833):
        self._A = torch.tensor(A)
        self._Y_min = torch.tensor(Y_min)
        self._Y_max = torch.tensor(Y_max)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = self._num_m = X.shape[1]
        self._ydim = self.num_r = A.shape[1]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = 2*A.shape[1]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self._partial_vars=[]
        self._other_vars=[]

        
        #trying to find a non-zero determinant submatrix of A to solve in a unique way.
        # self.find_square_submatrix(self.A)
        self.random_invertible_finder(self._A,self._A.shape[0])
        with open('Data\A_other_invers.txt', 'w') as testfile:
             for row in self._A_other_inv:
                 testfile.write(' '.join([str(a) for a in row]) + '\n')
        with open('Data\A_other_indexes.txt', 'w') as testfile:
             for row in self._other_vars:
                 testfile.write(' '.join([str(a) for a in row]) + '\n')
        # print(self._partial_vars)
        # print(self._A_other_inv)
        

        ### For Pytorch
        self._device = None

    def __str__(self):
        return 'T2F-R{}-INEQ{}-EQ{}-E{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    

    @property
    def A(self):
        return self._A


    @property
    def Y_min(self):
        return self._Y_min

    @property
    def Y_max(self):
        return self._Y_max
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
    def Y_max_np(self):
        return self.Y_max.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def Y_min_np(self):
        return self.Y_min.detach().cpu().numpy()

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
        return torch.norm(Y)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        resids=torch.cat([
            Y - self.Y_max,
            self.Y_min - Y
                          ], dim=0)
        return resids
    
    #have doubt about its correctness
    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids,min= 0)
    
    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A
    
    #have doubt about its correctness
    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*torch.tensor([[1,0],[0,-1]])@ineq_dist
    
    #have doubt about its correctness
    def ineq_partial_grad(self, X, Y):


        Y_max_effective = self.Y_max - (X @ self._A_other_inv.T)
        Y_min_effective =  (X @ self._A_other_inv.T)-self.Y_min
       
        grad_partial_max = 2 * torch.clamp(Y[:, self._partial_vars] - Y_max_effective, 0)
        grad_partial_min = 2 * torch.clamp(Y_min_effective - Y[:, self._partial_vars] , 0)

        Y = torch.zeros(2*X.shape[0], self.ydim, device=self.device)
        grad_partial= torch.cat([grad_partial_max,grad_other_min])
        grad_other_min= - (grad_partial_min @ self._A_partial.T) @ self._A_other_inv.T
        grad_other_max= - (grad_partial_max @ self._A_partial.T) @ self._A_other_inv.T
        grad_other= torch.cat([grad_other_max,grad_other_min])
        Y[:, self._partial_vars] = grad_partial
        Y[:, self._other_vars] = grad_other
       
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self._partial_vars] = Z
        Y[:, self._other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y
    

    # #ineq constraint
    # def ineq_constraint(model):
    #     return (model.Y_min,model.Y,model.Y_max)
    
    # #eq constraint
    # def eq_constraint(model):
    #     return model.Y.T@A==model.X
    
    #need to be filled  suitable optimization solvers
    def opt_solve(self, X, solver_type='pyomo', tol=1e-4):
        if solver_type == 'pyomo':
            
            # Create an abstract model
            model = pyo.AbstractModel()

            # Declare the sets
            model.R = pyo.RangeSet(self._num_r) # set of reactions
            model.M = pyo.RangeSet(self._num_m) # set of metabolites

            # Declare the parameters
            model.A = pyo.Param(model.M, model.R) # matrix A
            model.X = pyo.Param(model.M) # matrix X
            model.Y_min = pyo.Param(model.R) # matrix Y_min
            model.Y_max = pyo.Param(model.R) # matrix Y_max

            # Declare the variables
            model.Y = pyo.Var(model.R, domain=pyo.Reals) # matrix Y

            # Declare the objective function
            def obj_rule(model):
                return sum((model.Y[r] - 0)**2 for r in model.R) # minimize the second norm of Y
            model.obj = pyo.Objective(rule=obj_rule)

            # Declare the constraints
            def balance_rule(model, m):
                return sum(model.A[m, r] * model.Y[r] for r in model.R) == model.X[m] # Y @ A.T = X
            model.balance =pyo.Constraint(model.M, rule=balance_rule)

            def bound_rule(model, r):
                return pyo.inequality(model.Y_min[r], model.Y[r], model.Y_max[r]) # Y_min < Y < Y_max
            model.bound = pyo.Constraint(model.R, rule=bound_rule)

            logger.trace("Strarting glpk solver...")
            solver = SolverFactory('glpk')

            # Create a data dictionary
            data = {
                None: {
                    'A': {None: self.A_np},
                    'X': {None: self.X_np},
                    'Y_min': {None: self.Y_min_np},
                    'Y_max': {None: self.Y_max_np}
                }
            }

            # Create an instance of the model with the data
            instance = model.create_instance(data)

            # Solve the instance

            start_time = time.time()
            results = solver.solve(instance)
            end_time = time.time()
            logger.success("pyomo solver finished.")
            # sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
            return results, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)  
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

###################################################################
# SIMPLE PROBLEM
###################################################################

class SimpleProblem2:
    """ 
     minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h

        where:
            num_r= Number of reactions
            num_m= numer of metabloites
            num_ineq= 2*num_r
            num_m=num_eq 
            X=(num_m, num_examples)
            A=(num_m,num_r)
            Y,Y_min,Y_max=(num_r,num_examples)  
            p=(num_r,num_examples)
            Q=(num_r,num_r)
            G=(2*num_r,num_r)
            h=(2*num_r,num_examples)


        more info:
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
        while abs(det) < 0.0001 and i < 1000:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
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
        return 'SimpleProblem2-{}-{}-{}-{}'.format(
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

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_dist@self.G

    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
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
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    def opt_solve(self, X, solver_type='osqp', tol=1e-4):

        if solver_type == 'qpth':
            print('running qpth')
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False,check_Q_spd=False)(self.Q, self.p, self.G, self.h, self.A, X)
            end_time = time.time()
            print(self.Q.count_nonzero())
            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
        
        elif solver_type == 'osqp':
            print('running osqp')
            Q, p, A, G, h = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            
            solver = osqp.OSQP()
            my_A = np.vstack([A, np.eye(self.ydim,self.ydim)])
            y_max,y_min=np.split(h,2)
            y_min=-y_min
            my_l = np.vstack([X_np, y_min])
            my_u = np.vstack([X_np, y_max])
            print(my_l.shape)
            print(my_u.shape)
            print(my_l[5733],my_u[5733])
            solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
            start_time = time.time()
            results = solver.solve()
            end_time = time.time()

            total_time += (end_time - start_time)
            if results.info.status == 'solved':
                Y.append(results.x)
            else:
                Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time/len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X,solver_type="osqp")[0]
        # feas_mask =  ~np.isnan(Y).all(axis=0)  
        # self._num = feas_mask.sum()
        # self._X = self._X[feas_mask]
        # self._Y = torch.tensor(Y[feas_mask])
        self._Y=torch.tensor(Y)
        return Y

