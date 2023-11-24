"""
This script, takes a Metabolic Model (SBML) file,
and saves the following maps based on it:
1. Cmp_Map.txt
2. A.txt, b.txt
3. A_fva.txt, b_fva.txt
4. Genes_Map.txt, Reactions_Map.txt, Metabolites_Map.txt
"""

import os
import math
import cobra
import numpy as np
from scipy.linalg import null_space

LOG_MODE=True

def parse_gpr_rule(gpr_rule_str: str) -> list:
    """
    This method, parses the gpr_rule_str and returns a list of c (#complexes) lists each of them including
        gene_ids of that complex
    :param gpr_rule_str: A string representing the gpr_rule
    :return: list_of_complexes_genes = [[complex1 genes], [complex2 genes], ...]
    """
    list_of_complexes_genes = []
    if gpr_rule_str:
        complexes_str = gpr_rule_str.split(' or ')
        for complex_raw_str in complexes_str:
            complex_str = complex_raw_str
            if complex_raw_str[0] == '(':
                complex_str = complex_raw_str[1:-1]
            complex_genes = complex_str.split(' and ')
            list_of_complexes_genes.append(complex_genes)
    return list_of_complexes_genes


class MetabolicModelParser:
    def __init__(self, filepath_to_model: str,M=300) -> None:
        """
        :param filepath_to_model: Filepath for the metabolic_model.xml (SBML) file
        """
        # ########################################
        self.filepath_to_model = filepath_to_model
        self.metabolic_model = None
        self.M=M
        self.load_metabolic_model()
        # ############################
        self.s_matrix=None
        self.b_vector=None
        # ############################
        self.reactions_map = {}
        self.make_reactions_map()
        # ############################
        self.genes_map = {}
        self.make_genes_map()
        # ############################
        self.complexes_map = {}
        self.make_complexes_map()
        # ############################
        self.metabolites_map = {}
        self.make_metabolites_map()
        # ############################
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.fill_bounds()
        # ############################
        



    def load_metabolic_model(self) -> None:
        """
        This method, loads the sbml model from self.filepath_to_model into the self.metabolic_model
        :return: -
        """
        config = cobra.Configuration()
        config.solver = "glpk"
        self.metabolic_model = cobra.io.read_sbml_model(self.filepath_to_model)
        if (LOG_MODE):
            print("SBML model loaded.")

    def make_reactions_map(self) -> None:
        """
        This method, fills the self.reactions_map in the format of {'rxn0_id': 'R0', 'rxn1_id': 'R1', ...}
        :return: -
        """
        all_reactions = self.metabolic_model.reactions
        for _reaction_index in range(len(all_reactions)):
            _reaction_id = all_reactions[_reaction_index].id
            self.reactions_map[_reaction_id] = 'R' + str(_reaction_index)
        if (LOG_MODE):
             print("Recations Map created.")

    def make_genes_map(self) -> None:
        """
        This method, fills the self.genes_map in the format of {'gene0_id': 'G0', 'gene1_id': 'G1', ...}
        :return: -
        """
        all_genes = self.metabolic_model.genes
        for _gene_index in range(len(all_genes)):
            _gene_id = all_genes[_gene_index].id
            self.genes_map[_gene_id] = 'G' + str(_gene_index)
        if (LOG_MODE):
            print("Genes Map created.")

    def make_complexes_map(self) -> None:
        """
        This method, fills the self.complex_map in the format of {'C0': 'R: R729	G: G1',
                                                                  'C1': 'R: R729	G: G0', ...}
        :return: -
        """
        complex_index = 0
        for _reaction in self.metabolic_model.reactions:
            for complex_genes in parse_gpr_rule(_reaction.gene_reaction_rule):
                genes_text = ','.join([self.genes_map[_gene_id] for _gene_id in complex_genes])
                complex_text = 'C' + str(complex_index)
                self.complexes_map[complex_text] = 'R: ' + self.reactions_map[_reaction.id] + '\tG: ' + genes_text
                complex_index += 1
        if (LOG_MODE):
            print("Complexes Map created.")

    def make_metabolites_map(self) -> None:
        """
        This method, fills the self.metabolites_map in the format of {'met0_id': 'M0', 'met1_id': 'M1', ...}
        :return: -
        """
        all_metabolites = self.metabolic_model.metabolites
        for _metabolite_index in range(len(all_metabolites)):
            _metabolite_id = all_metabolites[_metabolite_index].id
            self.metabolites_map[_metabolite_id] = 'M' + str(_metabolite_index)
        if (LOG_MODE):
            print("Metabolites Map created.")

    def save_maps(self, folder_to_save: str) -> None:
        """
        This method, saves self.genes_map, self.reactions_map, self.complexes_map, and self.metabolites_map
            in the folder_to_save
        :param folder_to_save: Folder path to save these maps
        :return: -
        """
        with open(folder_to_save + '/Genes_Map.txt', 'w') as file:
            for _gene_id, _gene_identifier in self.genes_map.items():
                writing_text = _gene_identifier + ':\t' + _gene_id + '\n'
                file.writelines([writing_text])
        with open(folder_to_save + '/Metabolites_Map.txt', 'w') as file:
            for _met_id, _met_identifier in self.metabolites_map.items():
                writing_text = _met_identifier + ':\t' + _met_id + '\n'
                file.writelines([writing_text])
        with open(folder_to_save + '/Reactions_Map.txt', 'w') as file:
            for _rxn_id, _rxn_identifier in self.reactions_map.items():
                writing_text = _rxn_identifier + ':\t' + _rxn_id + '\n'
                file.writelines([writing_text])
        with open(folder_to_save + '/Cmp_Map.txt', 'w') as file:
            for complex_text, r_g_text in self.complexes_map.items():
                writing_text = complex_text + '\t' + r_g_text + '\n'
                file.writelines([writing_text])
        if (LOG_MODE):
            print("Maps are saved.")

    def save_gene_free_reactions(self):
        """
        This method, finds the reactions which are not associated with any genes
            (i.e., their gene_reaction_rule are blank)
        Note: As saving is not necessary at the moment, we only print the list of these reactions
            (also filepath_to_save argument is inactive)
        :return: -
        """
        gene_free_reactions = []
        for _reaction in self.metabolic_model.reactions:
            if not _reaction.gene_reaction_rule:
                gene_free_reactions.append(self.reactions_map[_reaction.id])
        print(len(gene_free_reactions))
        # with open(filepath_to_save, 'w') as file:
        #     for gene_free_reaction in gene_free_reactions:
        #         file.writelines([gene_free_reaction + '\n'])
        if (LOG_MODE):
            print("Free reactions saved.")

    def fill_bounds(self) -> None:
        """
        This method, fills self.lower_bounds and self.upper_bounds dict.
        Note that -math.inf and math.inf bounds are replaced with -M and M in the model during this process
        :return: -
        """
        for _reaction in self.metabolic_model.reactions:
            _rxn_id = self.reactions_map[_reaction.id]
            # ###### -math.inf and math.inf --> -M and M ######
            if _reaction.lower_bound == -math.inf:
                _reaction.lower_bound = -self.M
            if _reaction.upper_bound == math.inf:
                _reaction.upper_bound = self.M
            # ################################################
            self.lower_bounds[_rxn_id] = _reaction.lower_bound
            self.upper_bounds[_rxn_id] = _reaction.upper_bound
        if (LOG_MODE):
            print("Bounds assigned.")

    def override_bounds_with_fva(self,
                                 f: float = 0.9,
                                 biomass_reaction_id: str = "biomass_reaction") -> None:
        """
        This method, override self.lower_bounds and self.upper_bounds with FVA bounds.
        Note: Because of some errors in cobra.flux_analysis import flux_variability_analysis,
              FVA is implemented manually, but it can be changed later
        Caution: Time consuming method (2-3 hours)
        :param f: Fraction of the biomass in the FVA
        :param biomass_reaction_id: The reaction_id for the biomass reaction in the self.metabolic_model
        :return: -
        """
        # fva_df = flux_variability_analysis(self.metabolic_model,
        #                                    reaction_list=self.metabolic_model.reactions[:10],
        #                                    fraction_of_optimum=f)
        v_max = self.metabolic_model.optimize().objective_value
        biomass_reaction = self.metabolic_model.reactions.get_by_id(biomass_reaction_id)
        biomass_reaction.lower_bound = f * v_max
        # fva_cons = self.metabolic_model.problem.Constraint(biomass_reaction.flux_expression, lb=f * v_max)
        for _reaction in self.metabolic_model.reactions:
            _rxn_id = self.reactions_map[_reaction.id]
            self.metabolic_model.objective = _reaction
            # #################### Max #####################
            self.metabolic_model.objective_direction = "max"

            max_solution = self.metabolic_model.optimize()
            if max_solution.status != "optimal":
                raise RuntimeError("FVA on the reaction " + _reaction.id + " does not proceed to OPTIMAL")
            self.upper_bounds[_rxn_id] = max_solution.objective_value
            # #################### Min ####################
            self.metabolic_model.objective_direction = "min"
            min_solution = self.metabolic_model.optimize()
            if min_solution.status != "optimal":
                raise RuntimeError("FVA on the reaction " + _reaction.id + " does not proceed to OPTIMAL")
            self.lower_bounds[_rxn_id] = min_solution.objective_value
        if (LOG_MODE):
            print("Bounds overrided by FVA bounds.")

    def make_and_save_stoichiometric_data(self,
                                          folder_to_save: str,
                                          use_fva: bool = False, save_kernel_projector=False, save=True, mode=1) -> None:
        """
        This method, makes A = S.diag(u-l) sparse matrix and b = Sl vector,
            and saves them.
        Vectors l and u are read from self.lower_bounds and self.upper_bounds, which are initialized by
            default model's bounds, but can be overridden with FVA bounds using override_bounds_with_fva method first
        :param folder_to_save: Folder path to save A.txt and b.txt
        :param use_fva: If False (default), uses model's default bounds,
                        and if True, overrides them with FVA bound by calling self.override_bounds_with_fva
        :return: -
        """
        # ################# FVA Init #######################
        if use_fva:
            self.override_bounds_with_fva()
            filepath_to_save_a = os.path.join(folder_to_save, "A_fva.txt")
            filepath_to_save_b = os.path.join(folder_to_save, "b_fva.txt")
            filepath_to_save_l = os.path.join(folder_to_save, "lb_fva.txt")
            filepath_to_save_u = os.path.join(folder_to_save, "ub_fva.txt")
        else:
            filepath_to_save_a = os.path.join(folder_to_save, "A.txt")
            filepath_to_save_b = os.path.join(folder_to_save, "b.txt")
            filepath_to_save_l = os.path.join(folder_to_save, "lb.txt")
            filepath_to_save_u = os.path.join(folder_to_save, "ub.txt")
        filepath_to_save_projector = os.path.join(folder_to_save, "projector.npy")
        # #################################################
        all_metabolites_ids = self.metabolites_map.values()
        self.b_vector = dict(zip(all_metabolites_ids, [0]*len(all_metabolites_ids)))
        a_info = []
        self.s_matrix = np.zeros((len(all_metabolites_ids), len(self.reactions_map.keys())))
        for _reaction in self.metabolic_model.reactions:
            _rxn_id = self.reactions_map[_reaction.id]
            _lb = self.lower_bounds[_rxn_id]
            _ub = self.upper_bounds[_rxn_id]
            _diff_bound = _ub - _lb
            _metabolites_dict = _reaction.metabolites
            for _metabolite, _coeff in _metabolites_dict.items():
                _met_id = self.metabolites_map[_metabolite.id]
                if(mode==1):
                    a_text = _met_id + ',' + _rxn_id + ':\t' + str(_coeff * _diff_bound) + '\n'
                    a_info.append(a_text)
                    self.b_vector[_met_id] += _coeff * _lb
                    self.s_matrix[int(_met_id[1:]), int(_rxn_id[1:])] = _coeff  # Reading indexes from 'Mxxx' and 'Rxxx'

                elif(mode==2):
                    a_text = _met_id + ',' + _rxn_id + ':\t' + str(_coeff) + '\n'
                    a_info.append(a_text)
                    # self.b_vector[_met_id] += _coeff * _lb
                    self.s_matrix[int(_met_id[1:]), int(_rxn_id[1:])] = _coeff  # Reading indexes from 'Mxxx' and 'Rxxx'

        s_kernel = null_space(self.s_matrix)
        kernel_projector = np.matmul(s_kernel, s_kernel.T
                                     )
        # ################## Saving #####################
        if save:
            # A
            with open(filepath_to_save_a, 'w') as file:
                file.writelines(a_info)
            # b
            with open(filepath_to_save_b, 'w') as file:
                for _met_id, _value in self.b_vector.items():
                    saving_text = _met_id + ':\t' + str(_value) + '\n'
                    file.writelines([saving_text])
            # lb
            with open(filepath_to_save_l, 'w') as file:
                for _rxn_id, _value in self.lower_bounds.items():
                    saving_text = _rxn_id + ':\t' + str(_value) + '\n'
                    file.writelines([saving_text])
            # ub
            with open(filepath_to_save_u, 'w') as file:
                for _rxn_id, _value in self.upper_bounds.items():
                    saving_text = _rxn_id + ':\t' + str(_value) + '\n'
                    file.writelines([saving_text])
            # kernel_projector
            np.save(filepath_to_save_projector, kernel_projector)
            # ########################################################
            if (LOG_MODE):
                print("Stochiometric data saved.")

def main():
    mmp = MetabolicModelParser(filepath_to_model="./Data/recon_2.2.xml")
    mmp.save_maps(folder_to_save='./Data')
    mmp.save_gene_free_reactions()
    mmp.make_and_save_stoichiometric_data(folder_to_save='./Data', use_fva=False,save_kernel_projector=True)
    

if __name__ == "__main__":
    main()