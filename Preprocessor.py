
from BioDataParser import GPRMapParser
from BioNNDatasets import CustomTranscriptomicsDataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset,Dataset
import pickle

#logging
import datetime
import pickle
from loguru import logger
import sys        # <!- add this line
logger.remove()             # <- add this line
logger.add(sys.stdout, level="TRACE")   # <- add this line
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
log_path=".\logs\log-"+str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","-")+".log"
logger.add(log_path, level="TRACE", format=log_format, colorize=False, backtrace=True, diagnose=True)


class Preprocessor:

    def __init__(self, gpr_info:GPRMapParser, data:Dataset,number_of_features:int) -> None:
        self.gpr_info=gpr_info
        self.data=data
        self.number_of_features=number_of_features
        self.df=self.torch_dataset_to_pandas_dataframe(self.data,1)
        self.df=self.df.iloc[:,:-1]
        self.y=self.df.iloc[:,-1]
        self.data_statistics=self.df.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_numpy()

        self.active_genes_local_3state=self.apply_tresholding("local",3,True)
        self.active_genes_local_2state=self.apply_tresholding("local",2,True)
        self.active_genes_global_2state=self.apply_tresholding("global",2,True)
        self.active_genes_local=self.apply_tresholding("local",2,False)
        self.active_genes_global=self.apply_tresholding("global",2,False)

    def torch_dataset_to_pandas_dataframe(self, data:Dataset,fraction:float)-> pd.DataFrame:
        """
        This method, convert torch custom datset to pandas dataframe
        :param data: A Dataset object
        :return: pandas DataFrame object
        """

        column_names=list(np.arange(self.number_of_features))
        column_names.append("target")
        df=pd.DataFrame(np.column_stack(list(zip(*iter(data)))),columns=column_names)
        df=pd.concat([df.iloc[:,:-1].astype("float64"),df.iloc[:,-1].astype("category")],axis=1)
        return df[:int(fraction*len(df))]
    
    def calculate_tresholds(self, approach="global",use_boundaries=True, number_of_states=2)->np.ndarray:
        """
        This method, calculate tresholds based on the given config
        :param approach: "global" or "local", use_bounaries: consider upper and lower bounds for tresholds, number_of_states: 2 or 3
        :return: a ndarray of tresholds
        """
        gene_tresholds=np.zeros((self.number_of_features, 3))
        global_percentiles=[np.percentile(self.df.to_numpy(),q) for q in [25, 50, 75, 90]]
        # print(self.df.dtypes)
        # global_percentiles=np.percentile(self.df.to_numpy(),50)

        
        logger.trace("calculationg global percentiles: "+str(global_percentiles))

        if approach=="local":
            logger.trace("assiging local tresholds in gene_tresholds array with shape= "+str(gene_tresholds.shape))
            
            gene_tresholds[:,0]=global_percentiles[0]
            gene_tresholds[:,1]=self.data_statistics[1]
            gene_tresholds[:2]=global_percentiles[2]
            logger.trace("gene tresholds before applying boundaries have "+str(np.isnan(gene_tresholds[:,0]).sum())+" NaN Values.")

        elif approach=="global":
            gene_tresholds[:,1]=global_percentiles[1]

        else :
            raise NotImplementedError
        if(use_boundaries):
            
            gene_tresholds[:,0]=np.maximum(gene_tresholds[:,0],global_percentiles[0])
            logger.trace("gene tresholds after applying lower boundaries have "+str(np.isnan(gene_tresholds[:,0]).sum())+" NaN Values.")
            if number_of_states==3:
                gene_tresholds[:,0]=np.minimum(gene_tresholds[:,0],global_percentiles[2])
                logger.trace("gene tresholds after applying upper boundaries have "+str(np.isnan(gene_tresholds[:,0]).sum())+" NaN Values.")

            elif number_of_states!=2:
                raise NotImplementedError
        
        return gene_tresholds[:,0]
    
    def compare_with_threshold(row, threshold):
        """
        This method, takes a row and treshold array and return the result bolean array
        :param approach: row: an input array, treshold: array of tresholds
        :return: a boolean array
        """
        return row > threshold

    def apply_tresholding(self, approach="global", number_of_states=2,use_boundaries=True)->pd.DataFrame:
        """
        This method, calculate tresholds based on the given config and apply it to dataset and return the result dataframe
        :param approach: "global" or "local", use_bounaries: consider upper and lower bounds for tresholds, number_of_states: 2 or 3
        :return: a pandas dataframe of active genes.
        """
        gene_tresholds=self.calculate_tresholds(approach=approach,number_of_states=number_of_states, use_boundaries=use_boundaries)
        logger.trace(" gene_tresholds is calcualted in the array with shape= "+str(gene_tresholds.shape))
        logger.trace("df has "+str(self.df.isna().sum().sum())+" NaN Values.")
        active_genes=self.df.copy()
        # print(active_genes.iloc[1])
        # print(gene_tresholds)
        
        active_genes=active_genes.apply(lambda row: row>gene_tresholds,axis=1)
        # print(active_genes.iloc[1])
        logger.trace("active genes have "+str(active_genes.isna().sum().sum())+" NaN Values.")
        logger.success("active genes have been calculated with "+approach+" aproach and "+str(number_of_states)+" states "+("with considering bounds" if(use_boundaries) else "")+" in a dataframe with shape " +str(active_genes.shape))
        
        return active_genes
    def save_to_file(self,path:str):
        with open(path+"/Preproccesor_shape{}".format(str(self.df.shape),), 'wb') as f:
            pickle.dump(self, f)
        logger.success("Preproccessor Data has been seccussfully saved.")
    
def main():   
        normal_dataset = CustomTranscriptomicsDataset(
        annotations_file="./Human Tumors Dataset/Normal Samples Annotation.csv",
        dataset_dir="./Human Tumors Dataset/Normal Samples")
        gpr_info=GPRMapParser(gpr_data_filepath="./Data/Cmp_Map.txt")
        p=Preprocessor(gpr_info=gpr_info,number_of_features=1713,data=normal_dataset)
        p.save_to_file("./Data")
        # active_genes_local_3state=p.apply_tresholding("local",3,True)
        # active_genes_local_2state=p.apply_tresholding("local",2,True)
        # active_genes_global_2state=p.apply_tresholding("global",2,True)
        # active_genes_local=p.apply_tresholding("local",2,False)
        # active_genes_global=p.apply_tresholding("global",2,False)
        # print(active_genes_global_2state)
        

if __name__ == "__main__":
    main()








                




