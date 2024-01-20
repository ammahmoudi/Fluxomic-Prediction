import numpy as np
import pickle
import torch
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import T2FProblem
from BioDataParser import Stoichiometry
from Preprocessor import Preprocessor


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


preprocessor_path="./Data/Preproccesor_shape(741, 1713)"
with open(preprocessor_path, "rb") as fp:   #Pickling
            preprocessor = pickle.load(fp)

data=Stoichiometry("./Data/A.txt","./Data/b.txt","./Data/lb.txt","./Data/ub.txt","./Data/projector.npy")

A=data.get_a_matrix()
good_rows="good-row-5733-2023-12-22-00-59-59-904165"
# good_columns="good-column-5733-2023-12-22-01-05-10-456617"
A=mt.make_chosen_matrix(A,good_rows,None,mode='row')

# mt.random_invertible_finder(A)
num_m,num_r=A.shape


X=data.get_b_vector()
X=X.reshape((X.shape[0], -1))
logger.trace("X matrix reshpaed to  shape="+str(X.shape))
X=X@np.ones((1,preprocessor.number_of_samples))
logger.trace("X matrix expanded to shape="+str(X.shape))
X=mt.make_chosen_matrix(X,good_rows,None, "row")
logger.trace("X matrix filtered by good rows (metabolites) with shape="+str(X.shape))


Y_min=data.get_lb()
Y_min=Y_min.reshape((Y_min.shape[0], -1)) 
logger.trace("Y_min matrix reshpaed to shape="+str(Y_min.shape))
Y_max=data.get_ub()
Y_max=Y_max.reshape((Y_max.shape[0], -1)) 

logger.trace("Y_max matrix reshpaed to shape="+str(Y_max.shape))

Y_min=Y_min@np.ones((1,preprocessor.number_of_samples))
logger.trace("Y_min matrix expanded to shape="+str(Y_min.shape))
Y_max=Y_max@np.ones((1,preprocessor.number_of_samples))
logger.trace("Y_max matrix expanded to shape="+str(Y_max.shape))

Y_boundaries={}
import copy
for mode,active_reactions in preprocessor.active_reactions.items():
    logger.trace("Applying GPR data to Y boundaries for mode= "+mode+":")
    active_reactions_boolean=active_reactions>0
    Y_boundaries[mode]={}
    Y_boundaries[mode]['min']=copy.deepcopy(Y_min)
    Y_boundaries[mode]['max']=copy.deepcopy(Y_max)

    for sample_id in range(preprocessor.number_of_samples):
            for reaction_id in range(preprocessor.gpr_info.get_num_all_reactions()):
                if not active_reactions_boolean[sample_id][reaction_id]:
                    Y_boundaries[mode]['min'][reaction_id][sample_id]=0
                    Y_boundaries[mode]['max'][reaction_id][sample_id]=0

    # Y_boundaries[mode]['min']=mt.make_chosen_matrix(Y_boundaries[mode]['min'],good_columns,None, "row")

    # logger.trace("Y_min matrix filtered by good rows (reactions) with shape="+str(Y_boundaries[mode]['min'].shape))

    # Y_boundaries[mode]['max']=mt.make_chosen_matrix(Y_boundaries[mode]['max'],good_columns,None, "row")
    # logger.trace("Y_max matrix filtered by good rows (reactions) with shape="+str(Y_boundaries[mode]['max'].shape))


logger.success("Sample reaction Boundaries changed due to active reactions data.")
                   

# define G. a (2r,r) shape matrix in which the upper part prepares y for the upper bound and the second part for the lower bound 
G_up=np.eye(num_r)
G_down=-np.eye(num_r)
G=np.vstack((G_up,G_down))

#define Q complete zero in (r,r) shape
Q=np.zeros((num_r,num_r))

num_examples = X.shape[1]
#define p (need to review)
p=np.ones(num_r)

num_var = Y_boundaries['global']['max'].shape[0]
num_ineq = G.shape[0]
num_eq = A.shape[0]

for mode,active_reactions in preprocessor.active_reactions.items():
    # if(mode=="local_3state"):
          
        h=np.vstack((Y_boundaries[mode]['max'], -Y_boundaries[mode]["min"]))
        problem=T2FProblem(Q,p,A,G,h.T,X.T)
        problem.calc_Y()


        with open("./datasets/T2F/recon2.2_{}_dataset_var(R){}_ineq(2R){}_eq(M){}_ex(Samples){}".format(mode,num_var, num_ineq, num_eq, problem._num), 'wb') as f:
            pickle.dump(problem, f)
        logger.success("Data has been seccussfully generated and saver for "+"recon2.2_{}_dataset_var(R){}_ineq(2R){}_eq(M){}_ex(Samples){}".format(mode,num_var, num_ineq, num_eq, num_examples))
