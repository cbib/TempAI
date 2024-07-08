#!/usr/bin/env python3
import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from collections import Counter
from pathlib import Path
import os
#########################################
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#########################################

path1 = "/home/alekchiri/files/subHomo_sapiens.GRCh38.cdna_mod.fa"
path2 = "/home/alekchiri/files/sublncipedia_5_2.fasta"
path3 = "/home/alekchiri/files/humrep.fa"
path4 = "/home/alekchiri/files/sub_defam.fa"

def seq_finder(file):
    """
    Parses a FASTA file and extracts sequences with their IDs.

    This function reads a FASTA file and extracts the sequence IDs and the
    corresponding nucleotide sequences as strings. The sequences are returned
    as a list of tuples, where each tuple contains a sequence ID and the
    sequence itself.

    Parameters:
    file (str): The path to the input FASTA file.

    Returns:
    List[Tuple[str, str]]: A list of tuples where each tuple contains a
                           sequence ID and its corresponding sequence.
    """
    sequences = list()
    for sequence in SeqIO.parse(file, "fasta"):
        sequences.append((sequence.id, str(sequence.seq)))
    return  sequences

def one_hot(seq):
 """
    Encode a DNA sequence into a one-hot representation.

    This function takes a string representing a nucleotide sequence, where each nucleotide is 
    represented by 'A', 'C', 'G', 'T'. Nucleotides that are not recognized or ambiguous are 
    represented by 'N'. Each nucleotide is converted into a list of four elements, where each 
    element corresponds to 'A', 'C', 'G', 'T', respectively. The nucleotide present in the sequence 
    is represented by a 1 in the list, and the others by 0.

    Parameters:
    seq (str): The DNA sequence to be encoded. The sequence should be a string composed of the characters
               'A', 'C', 'G', 'T', and optionally 'N' for any ambiguous or unknown nucleotides.

    Returns:
    list of list of int: The encoded list where each nucleotide is represented by a one-hot encoded list.
                          For example, 'A' is encoded as [1, 0, 0, 0], 'C' as [0, 1, 0, 0], etc.

    Example:
    >>> one_hot('ACGTN')
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    """
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    seq_encoded = list()
    for nuc in seq.upper():
        if nuc in mapping:
            seq_encoded.append(mapping[nuc])
        else:
            seq_encoded.append(mapping['N'])
    return seq_encoded


def fragmentor(sequence, id='', maxseq=75, overlap=275, max_gap=5):
    """
    Splits a sequence into fragments with the specified overlap and size.

    This function only fragments sequences longer than maxseq.

    Parameters:
    sequence (str): The input sequence to fragment.
    id (str): Identifier to associate with each fragment.
    maxseq (int): Maximum length of each fragment.
    overlap (int): Overlap between consecutive fragments.
    max_gap (int): Maximum percentage of gap '-' allowed in a fragment.

    Returns:
    list: A list of tuples where each tuple contains a fragment and its identifier.
    """
    fragments = list()
    step_size = maxseq - overlap
    if len(sequence) <= maxseq:
        return fragments
    else:
        for i in range(0, len(sequence), step_size):
            fragment = sequence[i:i + maxseq]
            if len(fragment) < maxseq:
                fragment += 'N' * (maxseq - len(fragment))
            num_ = fragment.count('N')
            if ( num_ * 100 / maxseq) < max_gap:
                fragments.append((fragment,id))
    return fragments



def preparing_seq(file, maxseq):
    """
    Prepares and processes sequences from a given file.

    This function reads sequences from a file, fragments longer sequences
    using the `fragmentor` function, and pads shorter sequences to ensure
    uniform sequence length. It also labels the sequences based on the
    prefix of the sequence ID.

    Parameters:
    file (str): The path to the file containing sequences in FASTA format.
    maxseq (int): The maximum sequence length. Sequences longer than this
                  value will be fragmented, and shorter ones will be padded.

    Returns:
    tuple: A tuple containing:
        - res (list): A list of tuples where each tuple contains a sequence
                      and its associated ID.
        - labels (list): A list of labels assigned to each sequence based
                         on its ID prefix. The labels are:
                             - "protein_coding": for IDs starting with "ENS"
                             - "lncRNA": for IDs starting with "LIN"
                             - "rep": for all other IDs

    Raises:
    ValueError: If the lengths of `res` and `labels` lists are not the same.
    """
    res = list()
    labels = list()
    sequences = seq_finder(file)
    for id, sequence in sequences:
        if len(sequence) > maxseq:
            fragments = fragmentor(sequence,id=id,maxseq=maxseq)
            res.extend(fragments)
        elif len(sequence) < maxseq:
            padded_sequence = sequence + "N" * (maxseq - len(sequence))
            res.append((padded_sequence, id))
        else:
            res.append((sequence, id))
    return res
path_test = "/home/aladdine_lekchiri/Téléchargements/DB/fasta_test.fa"

all_sequences = list()
all_labels = list()
sequence_ens,labels_ens = preparing_seq(path1,550)
labels_ens = ["protein_coding" for _ in sequence_ens]
sequence_lnc = preparing_seq(path2,550)
labels_lnc = ["lncRNA" for _ in sequence_lnc]
sequence_repbase = preparing_seq(path3,550)
labels_repbase = ["rep" for _ in sequence_repbase]
sequence_defam = preparing_seq(path4,550)
labels_defam = ["rep" for _ in sequence_defam]
sequence_rep = sequence_repbase + sequence_defam
labels_rep = labels_repbase + labels_defam
print(f"mra : {len(sequence_ens)}, lncrna : {len(sequence_lnc)}, repbase : {len(sequence_rep)}")
print("Pre_processing done !")

all_sequences = sequence_ens + sequence_lnc + sequence_rep
all_labels = labels_ens + labels_lnc + labels_rep
seq_one_hot_code = list()
for seq, id in all_sequences:
    encoded = one_hot(seq)
    seq_one_hot_code.append(encoded)

def convert_labels(labels):
"""
    Converts a list of textual class labels into numerical labels.

    This function maps each text label from a list to a corresponding integer based on a predefined dictionary. 
    It is particularly useful for preparing categorical data for machine learning models where numerical input 
    is required. The current mapping is defined as {'protein_coding': 0, 'lncRNA': 1, 'rep': 2}, which are common 
    categories for gene classification tasks.

    Parameters:
    labels (list of str): A list of text labels to convert. Each label must be one of the keys in the predefined
                          dictionary (i.e., 'protein_coding', 'lncRNA', 'rep').

    Returns:
    list of int: A list of integers representing the converted labels.

    Example:
    >>> convert_labels(['protein_coding', 'lncRNA', 'rep', 'lncRNA'])
    [0, 1, 2, 1]
"""
    dico = {'protein_coding': 0, 'lncRNA' : 1, 'rep' : 2}
    num_labels = list()
    for label in labels:
        num_labels.append(dico[label])
    return num_labels

labels = convert_labels(all_labels)
print(f'labels_number {len(all_labels)}, sequences_number {len(seq_one_hot_code)}')
