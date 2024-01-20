import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as svds
import sympy
import torch
import scipy.linalg as la
from timeit import default_timer as timer
import pickle



#logging
from loguru import logger
import datetime
import sys        # <!- add this line
logger.remove()             # <- add this line
logger.add(sys.stdout, level="TRACE")   # <- add this line
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
log_path=".\logs\log-"+str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","-")+".log"
logger.add(log_path, level="TRACE", format=log_format, colorize=False, backtrace=True, diagnose=True)



def full_rank_maker_1(A):
    """
        This method tries to makes input matrix A full-ranked.
        used SVD
        
        :return: full-ranked A matrix
    """
    print(A.shape)
    print(np.linalg.matrix_rank(A))
    # Compute the SVD of A
    U, S, V = np.linalg.svd (A)

    # Set the tolerance threshold
    tol = S.max () * max (A.shape) * np.finfo (A.dtype).eps

    # Find the indices of the singular values that are greater than the tolerance
    r = np.sum (S > tol)

    # Delete the rows of U that correspond to the smaller singular values
    U = np.delete (U, np.s_ [r:], axis=1)

    # Reconstruct the matrix A with full rank
    A = U @ np.diag (S [:r]) @ V [:r, :]

    # Print the result
    # print (A)
    # print (np.linalg.matrix_rank (A))
    print(A.shape)
    print(np.linalg.matrix_rank(A))
    return(A)



def full_rank_maker_2(A):
    """
        This method tries to makes input matrix A full-ranked.
        used SVD
        
        :return: full-ranked A matrix
    """
    # print(A.shape)
    print(np.linalg.matrix_rank(A))

    # Perform the SVD of A
    U, S, Vt = svds(A, k=min(A.shape)-1) # k is the number of singular values to compute

    # Set the tolerance for the singular values
    tol = 1e-10

    # Find the rank of A
    rank = np.sum(S > tol)
    print("rank ",str(rank))

    # Select the linearly independent rows or columns of A
    U_ind = U[:, :rank] # rows of A
    V_ind = Vt[:rank, :] # columns of A

    # Delete the linearly dependent rows or columns of A
    A_full = A[U_ind.nonzero()[0], :][:, V_ind.nonzero()[1]] # full rank matrix
    print(A_full.shape)
    print(np.linalg.matrix_rank(A_full))
    return A_full



def full_rank_maker_3(A):
    """
        This method tries to makes input matrix A full-ranked.
        used QR
        
        :return: full-ranked A matrix
    """
    _, inds = sympy.Matrix(A).T.rref()
    print(len(inds))
    print(inds)
    # A is a numpy array
    # Get the rank of A
    rank = np.linalg.matrix_rank(A)
    # Get the QR decomposition of A transpose
    Q, R, P = np.linalg.qr(A.T, mode='full', pivoting=True)
    # Get the indices of the linearly independent columns of A
    ind = P[:rank]
    # Return the submatrix of A with only those columns
    return A[:, ind]



def full_rank_maker_4(A):
    # Find the rank of A
    r = np.linalg.matrix_rank(A)

    # Initialize an empty list to store the indices of independent rows
    ind_rows = []

    # Loop over the rows of A
    for i in range(A.shape[0]):
        # Select the current row and the previously selected rows
        B = A[ind_rows + [i], :]
        # Check if the rank of B is equal to the number of rows in B
        if np.linalg.matrix_rank(B) == B.shape[0]:
            # If yes, then the current row is independent and can be added to the list
            ind_rows.append(i)

    # Select only the independent rows from A
    A_new = A[ind_rows, :]

    # Print the new matrix
    return A_new

def full_rank_maker_5(matrix,threshold=1e-6,mode="row"):
    start = timer()
    #if mode is col transpose matrix
    if (mode=="column"):matrix=matrix.T
    # Define your sparse matrix A
    A = sp.csr_matrix(matrix)

    # Initialize an empty list to store the indices of independent rows
    ind_rows = []

    # Initialize an empty list to store the orthonormal vectors
    Q = []

    # Loop over the rows of A
    d=0
    for i in range(A.shape[0]):
        # Select the current row and convert it to a dense array
        v = A[i, :].toarray()[0]
        # Initialize the orthogonal vector as the current row
        u = v.copy()
        # Loop over the previously computed orthonormal vectors
        for q in Q:
            # Subtract the projection of the current row onto each orthonormal vector
            u = u - np.dot(v, q) * q
        # Check if the orthogonal vector is zero or close to zero
        if np.linalg.norm(u) < threshold:
            d=d+1
            logger.trace("dependent row found: "+str(d))
            # If yes, then the current row is dependent and can be skipped
            continue
        else:
            # If no, then the current row is independent and can be added to the list
            ind_rows.append(i)
            logger.trace("independents: "+str(len(ind_rows)))
            # Normalize the orthogonal vector and add it to the orthonormal list
            Q.append(u / np.linalg.norm(u))

    # Convert the orthonormal list to a numpy array
    Q = np.array(Q)

    # Print the orthonormal matrix
    # print(Q)
    # print(ind_rows)
    ind_file_name="good-"+mode+"-"+str(len(ind_rows))+"-"+str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","-")
    with open(ind_file_name, "wb") as fp:   #Pickling
        pickle.dump(ind_rows, fp)

    # Select only the independent rows from A
    A_new = A[ind_rows, :]
    if (mode=="column"):A_new=A_new.T
    end = timer()
    logger.success("full rank matrix with shape "+str(A_new.shape)+" found within time: "+str(end-start))
    return A_new



def make_chosen_matrix(A,rows_path,columns_path,mode="mixed"):
    if(mode=="row" or mode=="mixed"):
        with open(rows_path, "rb") as fp:   #Pickling
            rows = pickle.load(fp)
        # print(rows)
        # print(A.shape)
        # print(len(rows))
        A=A[rows,:]
    if(mode=="column" or mode=="mixed"):
        with open(columns_path, "rb") as fp:   #Pickling
            columns = pickle.load(fp)
        A=A.T[columns, :]
        A=A.T
    
    return A



def random_invertible_finder(matrix):
    """
        This method tries to makes input matrix invertible by removing columns.
        by selecting random columns
        
        :return: partial_vars, other_vars, A_other, A_partial, A_other_inv
    """
    det = 0
    i = 0
    max_i=1000
    
    while abs(det) < 1e-6 and i<max_i :
        _partial_vars = np.random.choice(matrix.shape[1], matrix.shape[1]-matrix.shape[0], replace=False)
        _other_vars = np.setdiff1d( np.arange(matrix.shape[1]), _partial_vars)
        det = np.linalg.det(matrix[:, _other_vars])
        i += 1
        logger.trace("i= "+str(i)+" | det(A_others) = "+str(det))
    if i == max_i:
        logger.exception("i reached the maximum bound but the desired submatrix is not achieved.")
        raise Exception

    else:
            
            _A_partial = matrix[:, _partial_vars]
            
            _A_other_inv = np.linalg.inv(matrix[:, _other_vars])
            _A_other=matrix[:,_other_vars]
            logger.success("A_partial and A_others constructed successfully at i= "+str(i)+" | det(A_thers) = "+str(det))
            return _partial_vars,_other_vars,_A_other,_A_partial,_A_other_inv
            # print(_partial_vars)
    
def det_sparse(m):
    logger.trace("starting calculation of determininet of a sparse matrix of shape:"+str(m.shape))
    lu = svds.splu(m)
    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()
    d = diagL.prod()*diagU.prod()
    diagL = diagL.astype(np.complex128)
    diagU = diagU.astype(np.complex128)
    logdet = np.log(diagL).sum() + np.log(diagU).sum()
    det = np.exp(logdet) # usually underflows/overflows for large matrices
    return det

def random_invertible_finder_sparse(matrix):
    """
        This method tries to makes input matrix invertible by removing columns.
        by selecting random columns
        
        :return: partial_vars, other_vars, A_other, A_partial, A_other_inv
    """
    det = 0
    i = 0
    max_i = 1000
    matrix = sp.csr_matrix(matrix)
    
    while abs(det) < 0.0001 and i<max_i :
        _partial_vars = np.random.choice(matrix.shape[1], matrix.shape[1]-matrix.shape[0], replace=False)
        _other_vars = np.setdiff1d( np.arange(matrix.shape[1]), _partial_vars)
        
        
        # det = sp.linalg(matrix[:, _other_vars])
        det = det_sparse(matrix[:, _other_vars])
        i += 1
        logger.trace("i= "+str(i)+" | det(A_others) = "+str(det))
    if i == max_i:
        logger.exception("i reached the maximum bound but the desired submatrix is not achieved.")
        raise Exception

    else:
            
            _A_partial = matrix[:, _partial_vars]
            _A_other_inv = np.linalg.inverse(matrix[:, _other_vars])
            _A_other=matrix[:,_other_vars]
            logger.success("A_partial and A_others constructed successfully at i= "+str(i)+" | det(A_thers) = "+str(det))
            return _partial_vars,_other_vars,_A_other,_A_partial,_A_other_inv
            # print(_partial_vars)
            

        
def is_independent ( col, selected, matrix):
    """
        This function checks if a column is linearly independent of a list of selected columns
        If the selected list is empty, then the column is independent by default
        
        :return: result Boolean
    """
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

def find_square_submatrix (matrix):
    """
        This method tries to makes input matrix invertible by removing columns.
        by greedy algorithm
        
        :return: result matrix
    """
    
    # Get the number of rows and columns of the matrix
    rows, cols = matrix.size()
    # Initialize an empty list to store the indices of the selected columns
    selected = []
    _partial_vars=[]
    _other_vars=[]
    # _partial_vars = np.random.choice(matrix.shape[1], matrix.shape[1] - _neq, replace=False)
    # _other_vars = np.setdiff1d( np.arange(matrix.shape[1]), _partial_vars)
    
    # Loop through the columns of the matrix
    for i in range (cols):
        # Get the current column
        col = matrix [:, i]
        # Check if the current column is linearly independent of the selected columns
        if is_independent(col, selected, matrix):
            
            # If yes, add the index of the current column to the selected list
            selected.append (i)
            logger.info("A new column added to selected matrix. |col = "+str(i)+" | new size = "+str(len(selected)))
        # Check if we have found enough columns
        if len (selected) == rows:
            # If yes, return the submatrix formed by the selected columns
            logger.success("An invertible square submatrix has been found!")
            _other_vars=selected
            _partial_vars = np.setdiff1d( np.arange(matrix.shape[1]), _other_vars)
            _A_partial = matrix[:, _partial_vars]
            _A_other = self=matrix[:,_other_vars]
            _A_other_inv = torch.inverse(matrix[:, _other_vars])
            logger.success("A_partial and A_others constructed successfully at i= "+str(i)+" | det(A_thers) = "+str(torch.det(_A_other)))
            return matrix [:, selected]
    # If we reach here, it means we did not find any square submatrix that is invertible
    # Return an empty matrix
    logger.exception("No invertible square matrix found.Output is an empty tensor.")
    return torch.tensor ([])

def column_subset_selection (matrix,k):
    """
        This method tries to makes input matrix invertible by removing columns.
        Find a subset of k columns of Q that are linearly independent
        This can be done by finding the pivot columns of R
        
        :return: partial_vars, other_vars, A_other, A_partial, A_other_inv
    """
    # logger.info("rank of the A is : "+str(torch.linalg.matrix_rank(_A)))

    _partial_vars=[]
    _other_vars=[]
    # Get the number of rows and columns of the matrix
    rows, cols = matrix.size()




    while(True):
        pivots=[]
        k_prime=0
        while(k_prime!=k):
            # Randomly sample k columns of A and form C
            _other_vars = np.random.choice(cols, k, replace=False)
            _partial_vars = np.setdiff1d( np.arange(matrix.shape[1]), _other_vars)

            C = matrix[:, _other_vars]

            # Compute the QR factorization of C
        
            Q, R = torch.linalg.qr(C)
            logger.trace("QR factorization of C has been completed.")
            epsilon=0
            diag_R=torch.diag(R)
            logger.trace("R diag shape: "+str(diag_R.shape))
            pivots = torch.abs(diag_R) > epsilon

            k_prime=np.count_nonzero(pivots)
            logger.trace("k prime = "+str(k_prime))


        Q_prime = Q[:, pivots]

        # Return the corresponding columns of A as A_prime
        A_prime = matrix[:, _other_vars[pivots]]

        # Transpose A_prime and compute its QR factorization
        Q, R = torch.linalg.qr(A_prime.T)

        # Return R as the square and invertible matrix of size k x k
        # R = R.astype(np.float64)
        # print(R.shape) # (5000, 5000)
        # print(np.linalg.det(R)) # non-zero

        _A_partial = matrix[:, _partial_vars]
        _A_other=matrix[:,_other_vars]
        logger.info("rank of the A_other is : "+str(torch.linalg.matrix_rank(_A_other)))
        logger.info("submatrix with shape "+str(_A_other.shape)+" and determinent "+str(torch.det(_A_other))+" has been found!")
        try:
            _A_other_inv = torch.inverse(matrix[:, _other_vars])
        except Exception as error:
            logger.exception(error)
            continue
        break

    return _A_other

def column_subset_selection_2 (matrix, k):
    """
        This method tries to makes input matrix invertible by removing columns.
        Find a subset of k columns of Q that are linearly independent
        This can be done by finding the pivot columns of R
        
        :return: partial_vars, other_vars, A_other, A_partial, A_other_inv
    """
    # _partial_vars=[]
    # _other_vars=[]
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
    det=0
    while(abs(det)<0.0001):
        # Randomly sample k columns of A and form C
        # sample k columns of A according to p
        _other_vars = np.random.choice(cols, k, replace=False, p=p)
        _partial_vars = np.setdiff1d( np.arange(matrix.shape[1]), _other_vars)

        C = matrix[:, _other_vars]
        
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
        _other_vars=indices
        _partial_vars = np.setdiff1d( np.arange(matrix.shape[1]), _other_vars)

        # return the selected columns of A
        _A_partial = matrix[:, _partial_vars]
        _A_other = matrix[:,_other_vars]
        det=torch.det(_A_other)
        logger.info("A_others has been found with det= "+str(det))

    return matrix[:, _other_vars]
