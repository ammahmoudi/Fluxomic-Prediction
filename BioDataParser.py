"""
This script provides:
    1. A class for getting desired connections matrices from the Cmp_Map (GPRMapParser)
    2. A class for parsing and keeping the stoichiometric information (Stoichiometry)
"""
import numpy as np

#logging
import datetime
from loguru import logger
import sys        # <!- add this line
logger.remove()             # <- add this line
logger.add(sys.stdout, level="TRACE")   # <- add this line
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
log_path=".\logs\log-"+str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","-")+".log"
logger.add(log_path, level="TRACE", format=log_format, colorize=False, backtrace=True, diagnose=True)



NUM_METs = 6124
NUM_RXNs = 8593


class GPRMapParser:
    def __init__(self, gpr_data_filepath: str) -> None:
        logger.info("Initializing GPRMapParser object from data in path: "+gpr_data_filepath)
        self.gpr_data_filepath = gpr_data_filepath
        self.gpr_data = {}
        self.genes_last_id = 0  # Should be = number_of_all_genes - 1
        self.complexes_last_id = 0  # Should be = number_of_all_complexes - 1
        self.reactions_last_id = 0  # Should be  = number_of_all_reactions - 1
        self.read_gpr_data()
        # ###################################
        self.gene_associated_reactions = None
        self.find_gene_associated_reactions()
        self.g_reactions_index_map = dict(
            zip(self.gene_associated_reactions,
                range(self.get_num_g_reactions()))
        )


    def read_gpr_data(self) -> None:
        """
        This method fills the self.gpr_data dict from the file denoted by self.gpr_data_filepath.
        Finally, self.gpr_data will be like:
         {9132: {'R': 8591, 'G': [416, 1212, 418, 246]}, 9135: {'R': 8592, 'G': [1212, 418, 246, 70]}, ...}.
        Also, there are three variables of self.complexes_last_id, self.reactions_last_id, and self.genes_last_id
            are found during the filling of self.gpr_data.
        :return: -
        """
        with open(self.gpr_data_filepath, 'r') as file:
            raw_gpr_data = file.readlines()
        for raw_gpr_relation in raw_gpr_data:
            gpr_relation_items = raw_gpr_relation[:-1].split('\t')  # e.g.: ['C9126', 'R: R8588', 'G: G69,G65,G87']
            complex_id = int(gpr_relation_items[0][1:])
            corresponding_reaction_id = int(gpr_relation_items[1][4:])
            influential_genes = gpr_relation_items[2][3:].split(',')
            influential_genes_ids = [int(gene[1:]) for gene in influential_genes]
            complex_dict = {'R': corresponding_reaction_id, 'G': influential_genes_ids}
            self.gpr_data[complex_id] = complex_dict
            self.complexes_last_id = max(self.complexes_last_id, complex_id)
            self.reactions_last_id = max(self.reactions_last_id, corresponding_reaction_id)
            self.genes_last_id = max(self.genes_last_id, max(influential_genes_ids))

        logger.success("GPR data has been successfully read. Number of complexes="+str(self.complexes_last_id+1)+", Number of All Reactions="+str(self.get_num_all_reactions())+", Number of Genes="+str(self.genes_last_id+1)+".")


            


    def find_gene_associated_reactions(self) -> None:
        """
        This method finds the list of reactions that are associated with some genes (i.e., consisting of complexes).
        Note: For a metabolic network with 8593 reactions, this list can be 4942 (57.5%) of all reactions,
            which means 3651 (42.5%) are not associated with any gene or complex.
        :return: -. Filling the self.gene_associated_reactions
        """
        gene_associated_reactions = []
        for complex_id, complex_dict in self.gpr_data.items():
            reaction_id = complex_dict['R']
            gene_associated_reactions.append(reaction_id)
        self.gene_associated_reactions = list(set(gene_associated_reactions))
        logger.success("Gene-Associated Reactions have been successfully found. Number of Gene-Associated Reactions="+str(self.get_num_g_reactions()))

    def get_num_all_reactions(self) -> int:
        """
        :return: The number of all reactions
        """
        return self.reactions_last_id + 1

    def get_num_g_reactions(self) -> int:
        """
        :return: The number of gene-associated reactions
        """
        return len(self.gene_associated_reactions)

    def make_genes_to_complexes_connection_matrix(self) -> np.ndarray:
        """
        This method returns the connection_matrix for the Genes to Complexes layer in our Neural Network.
        :return: genes_to_complexes_connection_matrix, the np.ndarray connection_matrix of the shape
            (num_complexes, num_genes)
        """
        num_genes = self.genes_last_id + 1
        num_complexes = self.complexes_last_id + 1
        genes_to_complexes_connection_matrix = np.zeros((num_complexes, num_genes))
        for complex_id, complex_dict in self.gpr_data.items():
            for gene_id in complex_dict['G']:
                genes_to_complexes_connection_matrix[complex_id, gene_id] = 1
        return genes_to_complexes_connection_matrix

    def make_complexes_to_reactions_connection_matrix(self) -> np.ndarray:
        """
        This method returns the connection_matrix for the Complexes to the G_Reactions layer in our Neural Network.
        By G_Reactions, we mean the reactions associated with some genes or complexes,
            around 57.5% of all reactions, defined in self.gene_associated_reactions.
        :return: complexes_to_g_reactions connection_matrix, the np.ndarray connection_matrix of the shape
            (num_g_reactions, num_complexes)
        """
        num_complexes = self.complexes_last_id + 1
        num_g_reactions = len(self.gene_associated_reactions)
        g_reactions_index_map = dict(
            zip(self.gene_associated_reactions,
                range(num_g_reactions))
        )
        complexes_to_g_reactions_connection_matrix = np.zeros((num_g_reactions, num_complexes))
        for complex_id, complex_dict in self.gpr_data.items():
            reaction_id = complex_dict['R']
            g_reaction_id = g_reactions_index_map[reaction_id]
            complexes_to_g_reactions_connection_matrix[g_reaction_id, complex_id] = 1
        return complexes_to_g_reactions_connection_matrix


class Stoichiometry:
    def __init__(self,
                 a_matrix_filepath: str,
                 b_vector_filepath: str,
                 lb_filepath: str,
                 ub_filepath: str,
                 projection_matrix_filepath: str) -> None:
        """
        Stoichiometric Data keeper
        :param a_matrix_filepath: file path to the matrix A information
        :param b_vector_filepath: file path to the vector b information
        :param lb_filepath: file path to the lb information
        :param ub_filepath: file path to the ub information
        :param projection_matrix_filepath: Filepath to the projection_matrix.npy
        """
        self.a_matrix_filepath = a_matrix_filepath
        self.b_vector_filepath = b_vector_filepath
        self.lb_filepath = lb_filepath
        self.ub_filepath = ub_filepath
        self.a_weights = np.zeros((NUM_METs, NUM_RXNs))  # A matrix
        self.b_weights = None
        self.lb = None
        self.ub = None
        self.parse_a_weights()
        self.parse_b_vector()
        self.parse_lb()
        self.parse_ub()
        self.projection_matrix = np.load(projection_matrix_filepath)

    def parse_a_weights(self) -> None:
        """
        This method reads matrix A from self.a_matrix_filepath, and fills the self.a_weights
        :return: -
        """
        with open(self.a_matrix_filepath, 'r') as f:
            a_data = f.readlines()
            for a_line in a_data:
                a_line_splt = a_line[:-1].split(':\t')
                a_indices = a_line_splt[0].split(',')
                i_ind = int(a_indices[0][1:])
                j_ind = int(a_indices[1][1:])
                a_val = float(a_line_splt[1])
                self.a_weights[i_ind, j_ind] = a_val

    def parse_b_vector(self):
        """
        This method reads vector b from self.b_vector_filepath, and fills the self.b_weights
        :return: -
        """
        b_weights = []
        with open(self.b_vector_filepath, 'r') as f:
            b_data = f.readlines()
            for b_line in b_data:
                b_line_txt = b_line[:-1].split(':\t')
                b_weights.append(float(b_line_txt[1]))
        self.b_weights = np.array(b_weights)

    def parse_lb(self):
        """
        This method reads ub from self.lb_filepath, and fills the self lb
        :return: -
        """
        lb = []
        with open(self.lb_filepath, 'r') as f:
            lb_data = f.readlines()
            for lb_line in lb_data:
                lb_line_txt = lb_line[:-1].split(':\t')
                lb.append(float(lb_line_txt[1]))
        self.lb = np.array(lb)

    def parse_ub(self):
        """
        This method reads ub from self.ub_filepath, and fills the self.ub
        :return: -
        """
        ub = []
        with open(self.ub_filepath, 'r') as f:
            ub_data = f.readlines()
            for ub_line in ub_data:
                ub_line_txt = ub_line[:-1].split(':\t')
                ub.append(float(ub_line_txt[1]))
        self.ub = np.array(ub)

    def get_a_matrix(self) -> np.ndarray:
        """
        Matrix A getter
        :return: self.a_weights
        """
        return self.a_weights

    def get_b_vector(self) -> np.ndarray:
        """
        Vector b getter
        :return: self.b_weights
        """
        return self.b_weights

    def get_lb(self) -> np.ndarray:
        """
        Vector lb getter
        :return: self.lb
        """
        return self.lb

    def get_ub(self) -> np.ndarray:
        """
        Vector ub getter
        :return: self.ub
        """
        return self.ub

    def get_projector(self) -> np.ndarray:
        """
        Projection matrix getter
        :return: self.projection_matrix
        """
        return self.projection_matrix

