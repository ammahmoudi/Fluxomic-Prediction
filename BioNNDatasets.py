"""
This script, provides classes to builds a torch.utils.data.Dataset object based on
    user's transcriptimics data.
"""
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


def make_sorted_list_of_expressions(expressions_dict: dict) -> np.ndarray:
    """
    :param expressions_dict: A dictionary of genes expressions in the format of {gene_id: gene_expression}
    :return: Array of genes expressions, sorted by genes ids
    """
    _num_genes = max(list(expressions_dict.keys())) + 1  # It is an acceptable assumption :))
    _sorted_expressions = np.zeros(_num_genes)
    for _gene_id, _gene_expr in expressions_dict.items():
        _sorted_expressions[_gene_id] = _gene_expr
    return _sorted_expressions


class TissuesTranscriptomicsDataset(Dataset):
    """
    A Dataset object for Tissue Transcriptomics Data,
        suitable when all transcriptional data are saved in a single dataframe
    """
    def __init__(self,
                 annotations_file: str,
                 dataset_dir: str,
                 transform=None, target_transform=None):
        """
        :param annotations_file: The filepath for the annotation.csv file, including "T**.csv, T**" in each column (!)
        :param dataset_dir: The filepath for the .pkl file in which all expressions are saved
        :param transform: -
        :param target_transform: -
        """
        self.dataset_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.expressions_dataframe = pd.read_pickle(self.dataset_dir)

    def __len__(self):
        return len(self.dataset_labels)

    def _get_expressions(self, tissue_label: str):
        """
        This method, extract and returns the list of genes expressions, sorted by genes ids,
            of the tissue labeled with tissue_label.
        :param tissue_label: The tissue identifier, in the format of self.dataset_labels 'label' column.
        :return: tissue_sorted_expressions
        """
        tissue_expr_dict = {}
        for _index, _row in self.expressions_dataframe.iterrows():
            _gene_id = int(_row['genes'][1:])  # Extracts **** from 'G****'; e.g., 491 from 'G491'
            _gene_expr = _row[tissue_label]
            tissue_expr_dict[_gene_id] = _gene_expr
        tissue_sorted_expressions = make_sorted_list_of_expressions(expressions_dict=tissue_expr_dict)
        return tissue_sorted_expressions

    def __getitem__(self, idx):
        expressions_list = self._get_expressions(tissue_label=self.dataset_labels.iloc[idx, 1])
        label = self.dataset_labels.iloc[idx, 1]  # Dummy for our purpose
        if self.transform:
            expressions_list = self.transform(expressions_list)
        if self.target_transform:
            label = self.target_transform(label)
        return expressions_list, label


class CustomTranscriptomicsDataset(Dataset):
    """
    A Dataset object for Tumors Transcriptomics Data,
        suitable when transcriptional data are saved in separated files
    """
    def __init__(self,
                 annotations_file: str,
                 dataset_dir: str,
                 transform=None, target_transform=None):
        """
        :param :param annotations_file: The filepath for the annotation.csv file,
                                 including "SampleFile.csv, CancerType" in each column
        :param dataset_dir: The direction of the folder in which samples' expressions .csv files are saved
        :param transform: -
        :param target_transform: -
        """
        self.dataset_labels = pd.read_csv(annotations_file)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset_labels)

    def __getitem__(self, idx):
        sample_filepath = os.path.join(self.dataset_dir, self.dataset_labels.iloc[idx]['FileName'])
        sample_df = pd.read_csv(sample_filepath)
        gene_ids = [int(gene_identifier[1:]) for gene_identifier in sample_df['GeneID']]  # Extracts *** from 'G***'
        expressions = sample_df[sample_df.columns[-1]]
        expressions_list = make_sorted_list_of_expressions(expressions_dict=dict(zip(gene_ids, expressions)))
        label = self.dataset_labels.iloc[idx]['Label']  # Cancer Type
        if self.transform:
            expressions_list = self.transform(expressions_list)
        if self.target_transform:
            label = self.target_transform(label)
        return expressions_list, label
