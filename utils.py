import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)
import scipy.linalg as la
import numpy as np
import pyomo.environ as pyo
# Import the solver
from pyomo.opt import SolverFactory

# import osqp
# from qpth.qp import QPFunction

# from scipy.linalg import svd
# from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
# import scipy.io as spio
import time

# from pypower.api import case57
# from pypower.api import opf, makeYbus
# from pypower import idx_bus, idx_gen, ppoption

#logging
from loguru import logger
import sys        # <!- add this line
# logger.remove(0)             # <- add this line
logger.add(sys.stdout, level="TRACE")   # <- add this line

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
    
    def random_invertible_finder(self, matrix):
        det = 0
        i = 0
        max_i=1000
        
        
        while abs(det) < 0.0001 :
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars])
            i += 1
            logger.trace("i= "+str(i)+" | det(A_thers) = "+str(det))
        # if i == max_i:
            # raise Exception
        # else:

            if abs(det)>0.0001:
                self._A_partial = self._A[:, self._partial_vars]
                self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
                self._A_other=self._A[:,self._other_vars]
                logger.success("A_partial and A_others constructed successfully at i= "+str(i)+" | det(A_thers) = "+str(det))
                print(self._partial_vars)
                break

        else:
            logger.exception("i reached the maximum bound but the desired submatrix is not achieved.")
        
        


            
    
    def is_independent (self, col, selected, matrix):
    # This function checks if a column is linearly independent of a list of selected columns
    # If the selected list is empty, then the column is independent by default
        if not selected:
            logger.trace("selected matrix is empty and independent by default.")
            return True
        # Otherwise, we form a matrix by appending the column to the selected columns
        submatrix = torch.cat ([matrix [:, selected], col.unsqueeze (1)], dim=1)
        # We compute the rank of the submatrix using PyTorch's matrix_rank function
        rank = torch.linalg.matrix_rank (submatrix)
        # If the rank is equal to the number of selected columns plus one, then the column is independent
        # Otherwise, it is dependent
        result=rank == len (selected) + 1
        if result:
             logger.trace("selected matrix is independent")
        else:
            logger.trace("selected matrix is  not independent")

        
        return result


    def find_square_submatrix (self, matrix):
        # Get the number of rows and columns of the matrix
        rows, cols = matrix.size()
        # Initialize an empty list to store the indices of the selected columns
        selected = []
        self._partial_vars=[]
        self._other_vars=[]
        # self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
        # self._other_vars = np.setdiff1d( np.arange(self._ydim), self._partial_vars)
        
        # Loop through the columns of the matrix
        for i in range (cols):
            # Get the current column
            col = matrix [:, i]
            # Check if the current column is linearly independent of the selected columns
            if self.is_independent(col, selected, matrix):
                
                # If yes, add the index of the current column to the selected list
                selected.append (i)
                logger.info("A new column added to selected matrix. |col = "+str(i)+" | new size = "+str(len(selected)))
            # Check if we have found enough columns
            if len (selected) == rows:
                # If yes, return the submatrix formed by the selected columns
                logger.success("An invertible square submatrix has been found!")
                self._other_vars=selected
                self._partial_vars = np.setdiff1d( np.arange(self._ydim), self._other_vars)
                self._A_partial = self._A[:, self._partial_vars]
                self,_A_other = self=self._A[:,self._other_vars]
                self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
                logger.success("A_partial and A_others constructed successfully at i= "+str(i)+" | det(A_thers) = "+str(torch.det(self.A_other)))
                return matrix [:, selected]
        # If we reach here, it means we did not find any square submatrix that is invertible
        # Return an empty matrix
        logger.exception("No invertible square matrix found.Output is an empty tensor.")
        return torch.tensor ([])
    
    def column_subset_selection (self, matrix,k):

        self._partial_vars=[]
        self._other_vars=[]
        # Get the number of rows and columns of the matrix
        rows, cols = matrix.size()



        # Find a subset of k columns of Q that are linearly independent
        # This can be done by finding the pivot columns of R
        pivots=[]
        k_prime=0
        while(k_prime!=k):
            # Randomly sample k columns of A and form C
            self._other_vars = np.random.choice(cols, k, replace=False)
            self._partial_vars = np.setdiff1d( np.arange(self._ydim), self._other_vars)

            C = matrix[:, self._other_vars]

            # Compute the QR factorization of C
            Q, R = np.linalg.qr(C)
            # logger.trace("QR factorization of C has been completed.")
                
            pivots = np.abs(np.diag(R)) > 1e-10
            k_prime=np.count_nonzero(pivots)
            logger.trace("k prime = "+str(k_prime))


        Q_prime = Q[:, pivots]

        # Return the corresponding columns of A as A_prime
        A_prime = matrix[:, self._other_vars[pivots]]

        # Transpose A_prime and compute its QR factorization
        Q, R = np.linalg.qr(A_prime.T)

        # Return R as the square and invertible matrix of size k x k
        R = R.astype(np.float64)
        # print(R.shape) # (5000, 5000)
        # print(np.linalg.det(R)) # non-zero

        self._A_partial = self._A[:, self._partial_vars]
        self,_A_other = self=self._A[:,self._other_vars]
        logger.info("submatrix with shape "+R.shape+" and determinent "+torch.det(R)+" has been found!")
        return torch.tensor(R)
    




    def column_subset_selection_2 (self, matrix,k):

        # self._partial_vars=[]
        # self._other_vars=[]
        # Get the number of rows and columns of the matrix
        rows, cols = matrix.size()

        # A is the sparse matrix of size m x n
        # k is the desired number of columns to select, where k <= m
        # returns a square, invertible matrix of size k x k
        
        # compute the squared Euclidean norms of the columns of A
        norms = torch.sum(matrix**2, axis=0)
        
        # normalize the norms to obtain a probability distribution
        p = norms / torch.sum(norms)
        k_prime=0
        while(k_prime!=k):
            # Randomly sample k columns of A and form C
            # sample k columns of A according to p
            self._other_vars = np.random.choice(cols, k, replace=False, p=p)
            self._partial_vars = np.setdiff1d( np.arange(self._ydim), self._other_vars)

            C = matrix[:, self._other_vars]
            
            # compute the QR decomposition of C
            Q, R = la.qr(C, mode='economic')
            logger.trace("Q.R ro C = "+str(C.shape)+" :"+"Q= "+str(Q.shape)+", R="+str(R.shape))
            
            # find a set of k linearly independent columns of Q
            P = la.lu(R)[1] # permutation matrix that puts R in echelon form

            Q = Q @ P # permute the columns of Q accordingly
            logger.trace("Q.P ro P :"+"Q= "+str(Q.shape)+", P="+str(P.shape))
            P=torch.tensor(P)
            R=torch.tensor(R)
            indices = torch.where(torch.diag(R @ P) != 0)[0] # indices of nonzero diagonal entries
            
            k_prime=len(indices)
            logger.trace("k prime = "+str(k_prime))
        
        
        Q = Q[:, indices] # select the corresponding columns of Q
        self._other_vars=indices
        self._partial_vars = np.setdiff1d( np.arange(self._ydim), self._other_vars)

        # return the selected columns of A
        self._A_partial = self._A[:, self._partial_vars]
        self._A_other = self._A[:,self._other_vars]

        logger.info("A_others has been found with det= "+str(torch.det(self._A_other)))

        return matrix[:, self._other_vars]


   



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


