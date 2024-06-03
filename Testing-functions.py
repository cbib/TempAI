import unittest
from Bio import SeqIO
from Bio.Seq import Seq
#################fasta_test_file###################
'''
>chr1
AGTCCGATAGCTACGATCGATCGATCGATCGATACGATCGATCGATCGATCGATCGATCGATCGAAGCTATCGATCGTAGCTAGCTGACTGACTGACTGATGCTAGCTAGCTAGCTAGCTAGCTAGCTGATCGATCGATCGACTGACTGACTGATG
>chr2
AGTCCGATAGCTACGATCGATCGATCGATCGATA
>chr3
AGTCCGATAGCTACGATCGATCGATCGATCGATACGATCGATCGATCGATCGATCGATCGATCGAAGAGC
'''
######################################################################

###############################################
#################Functions_to_test#############
###############################################
def seq_finder(file):
    
    sequences = []
    for sequence in SeqIO.parse(file, "fasta"):
        sequences.append((sequence.id, str(sequence.seq)))  
    return sequences


def fragmentor(sequence, id='', maxseq=75, overlap=10, max_gap=5):
    fragments = []
    step_size = maxseq - overlap
    if len(sequence) <= maxseq:
        return fragments
    else:
        for i in range(0, len(sequence), step_size):
            fragment = sequence[i:i + maxseq]
            if len(fragment) < maxseq:
                fragment += '-' * (maxseq - len(fragment))
            num_ = fragment.count('-')
            if ( num_ * 100 / maxseq) < max_gap:
                fragments.append((fragment,id))
    return fragments

def one_hot(seq):
    
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    seq_encoded = list()
    for nuc in seq.upper():
        if nuc in mapping:
            seq_encoded.append(mapping[nuc])
        else:
            seq_encoded.append(mapping['N'])
    return seq_encoded

def convert(labels):
    dico = {'protein_coding': 0, 'lncRNA' : 1, 'rep' : 2}
    num_labels = list()
    for lab in labels:
        num_labels.append(dico[lab])
    return num_labels



###############################################################################################
#################################Testing_functions#############################################


class TestSeqFinder(unittest.TestCase):
    def test_seq_finder(self):
        # Test input file path
        path_test = "/home/aladdine_lekchiri/Téléchargements/DB/fasta_test.fa"
        expected = [
            ("chr1", "AGTCCGATAGCTACGATCGATCGATCGATCGATACGATCGATCGATCGATCGATCGATCGATCGAAGCTATCGATCGTAGCTAGCTGACTGACTGACTGATGCTAGCTAGCTAGCTAGCTAGCTAGCTGATCGATCGATCGACTGACTGACTGATG"),
            ("chr2", "AGTCCGATAGCTACGATCGATCGATCGATCGATA"),
            ("chr3", "AGTCCGATAGCTACGATCGATCGATCGATCGATACGATCGATCGATCGATCGATCGATCGATCGAAGAGC")
        ]
        
        result = seq_finder(path_test)
        self.assertEqual(result, expected)


class TestFragmentor(unittest.TestCase):
    def test_basic_case(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = []
        self.assertEqual(fragmentor(seq, maxseq=75, overlap=10, max_gap=5), expected)
    
    def test_exact_match(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = []
        self.assertEqual(fragmentor(seq, maxseq=80, overlap=0, max_gap=5), expected)

    def test_less_than_fragment_size(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = [("ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACT","")]
        self.assertEqual(fragmentor(seq, maxseq=75, overlap=10, max_gap=5), expected)

    def test_overlap_and_gaps(self):
        seq = "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
        expected = []
        self.assertEqual(fragmentor(seq, maxseq=68, overlap=10, max_gap=50), expected)

class TestOneHotEncoding(unittest.TestCase):
    def test_normal_sequence(self):
        seq = "ACGT"
        expected = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.assertEqual(one_hot(seq), expected)

    def test_sequence_with_non_standard_nucleotides(self):
        seq = "ACGTXN"
        expected = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(one_hot(seq), expected)

    def test_empty_sequence(self):
        seq = ""
        expected = []
        self.assertEqual(one_hot(seq), expected)

    def test_lower_case_sequence(self):
        seq = "acgt"
        expected = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.assertEqual(one_hot(seq), expected)

    def test_sequence_with_only_n(self):
        seq = "NNNN"
        expected = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(one_hot(seq), expected)

class TestConvertFunction(unittest.TestCase):
    def test_valid_inputs(self):
        labels = ['protein_coding', 'lncRNA', 'rep']
        expected = [0, 1, 2]
        result = convert(labels)
        self.assertEqual(result, expected)

    def test_empty_input(self):
        labels = []
        expected = []
        result = convert(labels)
        self.assertEqual(result, expected)

    def test_invalid_input(self):
        labels = ['unknown_label']
        with self.assertRaises(KeyError):
            convert(labels)

    def test_mixed_input(self):
        labels = ['protein_coding', 'unknown_label', 'rep']
        with self.assertRaises(KeyError):
            convert(labels)
# Run 
if __name__ == "__main__":
    unittest.main()
