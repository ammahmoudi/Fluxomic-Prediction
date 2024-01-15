"""
This script, organizes the tissues_data into distinct files (to be further read by torch.Dataset),
    and translates the gene_ids into the format defined in the MetabolicModelParser.py
"""

from gtfparse import read_gtf
import json
import pandas as pd
import numpy as np
import warnings
import os

"""
hgnc_complete_set.json 
A dict with keys=['responseHeader', 'response']
'responseHeader'
      |--> 'status': 0
      |--> 'QTime': 25
'response'
    |--> 'numFound': 42843
    |--> 'docs': list([dict0, dict1, ..., dict42842]), len = 42843
            |--> each dict:
                     |--> ['date_approved_reserved', 'alias_name', 'vega_id', 'locus_group', 'mane_select', 
                           'status', *'alias_symbol'*, '_version_', 'uuid', 'refseq_accession', 'locus_type', 
                           'agr', *'hgnc_id'*, 'rgd_id', 'ensembl_gene_id', 'entrez_id', 'gene_group', 'omim_id', 
                           *'symbol'*, 'location', 'name', 'date_modified', 'mgd_id', 'ucsc_id', 'enzyme_id', 
                           'uniprot_ids', 'ccds_id', 'ena', 'gene_group_id', 'pubmed_id', 'location_sortable']
                     Example: {'date_approved_reserved': '2003-06-02', 
                               'alias_name': ['acyl-CoA synthetase family member 1'], 
                               'vega_id': 'OTTHUMG00000168550', 'locus_group': 'protein-coding gene', 
                               'mane_select': ['ENST00000316519.11', 'NM_023928.5'], 'status': 'Approved', 
                               'alias_symbol': ['FLJ12389', 'SUR-5', 'ACSF1'], '_version_': 1714387633296113665, 
                               'uuid': '5300a76c-885a-4184-9173-c8f3da9da950', 'refseq_accession': ['NM_023928'], 
                               'locus_type': 'gene with protein product', 'agr': 'HGNC:21298', 'hgnc_id': 'HGNC:21298', 
                               'rgd_id': ['RGD:708522'], 'ensembl_gene_id': 'ENSG00000081760', 'entrez_id': '65985', 
                               'gene_group': ['Acyl-CoA synthetase family'], 'omim_id': ['614364'], 'symbol': 'AACS', 
                               'location': '12q24.31', 'name': 'acetoacetyl-CoA synthetase', 
                               'date_modified': '2015-08-24', 'mgd_id': ['MGI:1926144'], 'ucsc_id': 'uc001uhc.4', 
                               'enzyme_id': ['6.2.1.16'], 'uniprot_ids': ['Q86V21'], 'ccds_id': ['CCDS9263'], 
                               'ena': ['AK022451'], 'gene_group_id': [40], 'pubmed_id': [12623130, 17762044], 
                               'location_sortable': '12q24.31'}
    |--> 'start': 0
"""

"""
df = read_gtf("./gencode.v38.annotation.gtf")

print(df.columns)
> Index(['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand',
       'frame', 'gene_id', 'gene_type', 'gene_name', 'level', 'hgnc_id',
       'havana_gene', 'transcript_id', 'transcript_type', 'transcript_name',
       'transcript_support_level', 'tag', 'havana_transcript', 'exon_number',
       'exon_id', 'ont', 'protein_id', 'ccdsid'],
      dtype='object')
      
print(df.iloc[5])
> seqname                                                   chr1
source                                                  HAVANA
feature                                             transcript
start                                                    12010
end                                                      13670
score                                                      NaN
strand                                                       +
frame                                                        0
gene_id                                      ENSG00000223972.5
gene_type                   transcribed_unprocessed_pseudogene
gene_name                                              DDX11L1
level                                                        2
hgnc_id                                             HGNC:37102
havana_gene                               OTTHUMG00000000961.2
transcript_id                                ENST00000450305.2
transcript_type             transcribed_unprocessed_pseudogene
transcript_name                                    DDX11L1-201
transcript_support_level                                    NA
tag                                    basic,Ensembl_canonical
havana_transcript                         OTTHUMT00000002844.2
exon_number                                                   
exon_id                                                       
ont                                    PGO:0000005,PGO:0000019
protein_id                                                    
ccdsid                                                        
Name: 5, dtype: object
"""

"""
dataset = openpyxl.load_workbook("./1260419__excel_tabless1-s18.xlsx")
Sheets: "S1. All genes", "S2. Tissue enriched", "S3. Group enriched", "S4. Enhanced", "S5. Missing genes",
        "S6. GO analysis", "S7. SP-TM", "S8. Transcription factors", "S9. Druggable proteins",
        "S10. Cancer genes", "S11. FPKM Cell-lines", "S12. Genes with different splic",
        "S13.", "S14.", "S15.", "S16", "S17.", "S18. Full FPKM dataset, tissues"
        
Tissues data: "S18. Full FPKM dataset, tissues" --> "Data/Tissue_Data.csv"
column zero key: 'ensg_id'
tissues column keys = ['adipose tissue fpkm', 'adrenal gland fpkm', 'appendix fpkm', 'bone marrow fpkm', 'brain fpkm',
                       'colon fpkm', 'duodenum fpkm', 'endometrium fpkm', 'esophagus fpkm', 'fallopian tube fpkm',
                       'gallbladder fpkm', 'heart muscle fpkm', 'kidney fpkm', 'liver fpkm', 'lung fpkm', 
                       'lymph node fpkm', 'ovary fpkm', 'pancreas fpkm', 'placenta fpkm', 'prostate fpkm', 
                       'rectum fpkm', 'salivary gland fpkm', 'skeletal muscle fpkm', 'skin fpkm', 
                       'small intestine fpkm', 'smooth muscle fpkm', 'spleen fpkm', 'stomach fpkm', 'testis fpkm', 
                       'thyroid gland fpkm', 'tonsil fpkm', 'urinary bladder fpkm']
Saved in "./GenesRegData.pkl" (with df.to_pickle) with columns:
         ['genes', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14',
          'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28',
          'T29', 'T30', 'T31', 'T32']
"""


class GeneTranslator:
    def __init__(self,
                 gtf_filepath: str = None,
                 json_filepath: str = None,
                 source_key: str = 'gene_name',
                 destination_key: str = 'hgnc_id') -> None:
        """
        Note: At least one of the gtf_filepath ar json_filepath should be assigned,
              and gtf_filepath has the priority
        :param gtf_filepath: Filepath for genes .gtf file
        :param json_filepath: Filepath for hgnc_complete_set.json file
        :param source_key: The column name in .gtf file representing the source of translation
        :param destination_key: The column name in .gtf file representing the destination of translation
        """
        self.gtf_filepath = gtf_filepath
        self.json_filepath = json_filepath
        self.source_key = source_key
        self.destination_key = destination_key
        self.translation_dict = {}
        if self.gtf_filepath:
            self.make_translation_dict_from_gtf()
        elif self.json_filepath:
            self.make_translation_dict_from_json()
        else:
            raise AttributeError("None of the gtf_filepath and json_filepath are assigned")

    def make_translation_dict_from_gtf(self) -> None:
        """
        This method, fills self.translation_dict based on self.source_key --> self.destination_key columns
            in the self.gtf_filepath file
        :return: -
        """
        genes_df = read_gtf(self.gtf_filepath,result_type="pandas")
        if self.source_key not in genes_df.columns:
            raise KeyError("Source key " + self.source_key + " is not a key of your file " + self.gtf_filepath)
        if self.destination_key not in genes_df.columns:
            raise KeyError("Destination key " + self.source_key + " is not a key of your file " + self.gtf_filepath)
        filtered_df = genes_df[[self.source_key, self.destination_key]]
        for _index, _row in filtered_df.iterrows():
            self.translation_dict[_row[self.source_key]] = _row[self.destination_key]

    def make_translation_dict_from_json(self) -> None:
        """
        This method, fills self.translation_dict based on self.source_key --> self.destination_key corresponding keys
            in the self.json_filepath file
        :return: -
        """
        with open(self.json_filepath, 'r',encoding="utf8") as json_file:
            genes_data = json.load(json_file)
        genes_info_list = genes_data['response']['docs']
        genes_keys = genes_info_list[0].keys()
        if self.source_key not in genes_keys:
            raise KeyError("Source key " + self.source_key + " is not a key of your file " + self.json_filepath)
        if self.destination_key not in genes_keys:
            raise KeyError("Destination key " + self.source_key + " is not a key of your file " + self.json_filepath)
        for gene_info in genes_info_list:
            self.translation_dict[gene_info[self.source_key]] = gene_info[self.destination_key]

    def get_translation_keys(self):
        """
        This method, returns self.translation_dict keys for probable pre-translation checks
        :return: self.translation_dict.keys()
        """
        return self.translation_dict.keys()

    def translate(self, gene_source_id: str) -> str:
        """
        This method, translated the gene_source_id using self.translation_dict
        Note: KeyError is not handled in this method
        :param gene_source_id: Input of translation
        :return: self.translation_dict[gene_source_id]
        """
        return self.translation_dict[gene_source_id]

    def save_translation_dict(self, filepath_to_save: str) -> None:
        """
        This method, provides facilities to save self.translation_dict
            to prevent regularly working with big .gtf files in special cases.
        :param filepath_to_save: Filepath to save self.translation_dict
        :return: -
        """
        with open(filepath_to_save, 'w') as file:
            json.dump(self.translation_dict, file)


def translate_and_map_genes(list_of_genes: list,
                            filepath_to_mapping_dict: str,
                            filepath_to_translation_dict: str = None,
                            gene_translator: GeneTranslator = None) -> list:
    """
    This function, translates list_of_genes from source_ids into destination_ids using a translation_dict
        or a GeneTranslator (with a priority on the second one), the maps them from destination_ids into
        gene_identifiers (G0 to G1713 nomenclature) using Genes_Map.txt
    :param list_of_genes: List of genes (with source_ids) to be translated and mapped (order is kept)
    :param filepath_to_mapping_dict: Filepath for Genes_Map.txt (destination_id --> gene_identifier)
    :param filepath_to_translation_dict: Filepath for translation_dict.json (source_id --> destination_id)
    :param gene_translator: A GeneTranslator object
    :return: mapped_genes_identifiers
    """
    # ############## Loading Translation ##############
    if gene_translator:
        translation_keys = []
    elif filepath_to_translation_dict:
        with open(filepath_to_translation_dict, 'r') as file:
            translation_dict = json.load(file)
        translation_keys = translation_dict.keys()
    else:
        raise AttributeError("None of the gene_translator and filepath_to_translation_dict are assigned")
    # ################ Loading Mapping #################
    mapping_dict = {}
    with open(filepath_to_mapping_dict, 'r') as file:
        mapping_file = file.readlines()
    for mapping_item in mapping_file:
        print(mapping_item)
        gene_mapping_info = mapping_item[:-1].split('\t:')
        print(gene_mapping_info)
        print(len(mapping_dict))
        print(len(mapping_file))
        mapping_dict[gene_mapping_info[1]] = gene_mapping_info[0]
    # ################# Translation and Mapping ##################
    mapped_genes_identifiers = []
    for gene_source_id in list_of_genes:
        if gene_source_id in translation_keys:
            # ### Translation ###
            if gene_translator:
                gene_destination_id = gene_translator.translate(gene_source_id=gene_source_id)
            else:
                gene_destination_id = translation_dict[gene_source_id]
            if gene_destination_id in mapping_dict.keys():
                # ### Mapping ###
                mapped_gene_identifier = mapping_dict[gene_destination_id]
                mapped_genes_identifiers.append(mapped_gene_identifier)
            else:
                # ### Unable to map ###
                warning_text = gene_destination_id + " does not exist in your mapping file"
                warnings.warn(warning_text)
                mapped_genes_identifiers.append(np.nan)
        else:
            # ### Unable to translate ###
            warning_text = gene_source_id + " does not exist in your translation file"
            warnings.warn(warning_text)
            mapped_genes_identifiers.append(np.nan)
    # ###########################################################
    return mapped_genes_identifiers


def parse_cancer_type_dict(cancer_type_identifier_filepath: str) -> dict:
    """
    This function, reads cancer type identifier file and provides a corresponding dictionary
    :param cancer_type_identifier_filepath: Filepath for cancer type identifier .txt file
    :return: cancer_type_identifier
    """
    cancer_type_identifier = {}
    with open(cancer_type_identifier_filepath, 'r') as file:
        cancer_type_identifier_info = file.readlines()
    for cancer_type_identifier_item in cancer_type_identifier_info:
        sample_id, cancer_type = cancer_type_identifier_item[:-1].split('\t')
        cancer_type_identifier[sample_id] = cancer_type
    return cancer_type_identifier

gene_translator = GeneTranslator(gtf_filepath="./gencode.v38.annotation.gtf",
                                 source_key='gene_name', destination_key='hgnc_id')
gene_translator.save_translation_dict(filepath_to_save="./Data/gene_name_to_hgnc.json")

gene_translator = GeneTranslator(json_filepath="./hgnc_complete_set.json",
                                 source_key='symbol', destination_key='hgnc_id')
gene_translator.save_translation_dict(filepath_to_save="./Data/gene_name_to_hgnc_2.json")


def organize_and_save_transcriptomics_data(transcriptomics_data_filepath: str,
                                           samples_name: str,
                                           filepath_to_translation_dict: str,
                                           filepath_to_mapping_dict: str,
                                           cancer_type_identifier_filepath: str,
                                           base_saving_dir: str) -> None:
    """

    :param transcriptomics_data_filepath: Filepath for the main Transcriptomics.txt file
    :param samples_name: "Normal Samples" or "Tumor Samples"
    :param filepath_to_translation_dict: Filepath for genes translation_dict.json (source_id --> destination_id)
    :param filepath_to_mapping_dict: Filepath for Genes_Map.txt (destination_id --> gene_identifier)
    :param cancer_type_identifier_filepath: Filepath for CancerType_Samples.txt
    :param base_saving_dir: The folder to save organized samples (in /samples_name sub-folder) and annotations
    :return: -
    """
    # ################## Loading Main Transcriptomics Data #####################
    transcriptomics_data = pd.read_csv(transcriptomics_data_filepath, delimiter='\t')
    columns_keys = transcriptomics_data.columns
    # ######## Translating and Mapping Genes into the Desirable Format #######
    column_zero = transcriptomics_data[columns_keys[0]]  # list of all genes
    mapped_genes = translate_and_map_genes(
        list_of_genes=column_zero.to_list(),
        filepath_to_translation_dict=filepath_to_translation_dict,
        filepath_to_mapping_dict=filepath_to_mapping_dict
    )
    mapped_genes_df = pd.DataFrame({'GeneID': mapped_genes})
    # ############# Reading Cancer Types (as Labes) ##############
    cancer_type_dict = parse_cancer_type_dict(
        cancer_type_identifier_filepath=cancer_type_identifier_filepath
    )
    # ############ Separating and Saving Samples One by One ############
    folder_to_save_samples = os.path.join(base_saving_dir, samples_name)
    if not os.path.exists(folder_to_save_samples):
        os.makedirs(folder_to_save_samples)
    file_names = []
    labels = []
    for sample_id in columns_keys[1:]:
        column_df = transcriptomics_data[sample_id]
        sample_df = pd.concat([mapped_genes_df, column_df], axis=1)
        pruned_sample_df = sample_df[sample_df['GeneID'].notna()]  # Ignoring surplus genes
        # Note: pruned_sample_df is *NOT* sorted based on Gi identifiers
        file_name = sample_id + '.csv'  # Can be .pkl
        filepath_for_saving_sample = os.path.join(folder_to_save_samples, file_name)
        pruned_sample_df.to_csv(filepath_for_saving_sample)  # better to set header=False and index=False
        file_names.append(file_name)
        labels.append(cancer_type_dict[sample_id])
    # ####################### Saving Annotation File ########################
    annotation_df = pd.DataFrame({'FileName': file_names, 'Label': labels})
    annotation_filepath = os.path.join(base_saving_dir, samples_name + ' Annotation.csv')
    annotation_df.to_csv(annotation_filepath)  # better to set header=False and index=False
    # #################################################################################


def main():
    rna_data_folder = "./RNA-Seq for The Cancer Genome Atlas"
    data_path = os.path.join(rna_data_folder, "Normal Samples/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_FPKM.txt")
    cancer_types_filepath = os.path.join(rna_data_folder, "GSE62944_06_01_15_TCGA_24_Normal_CancerType_Samples.txt")
    organize_and_save_transcriptomics_data(transcriptomics_data_filepath=data_path,
                                           samples_name="Normal Samples",
                                           filepath_to_translation_dict="./Data/gene_name_to_hgnc_2.json",
                                           filepath_to_mapping_dict="./Data/Genes_Map.txt",
                                           cancer_type_identifier_filepath=cancer_types_filepath,
                                           base_saving_dir="./Human Tumors Dataset")


if __name__ == "__main__":
    main()
