Pre-processing 

This repository contains a Python script designed to preprocess DNA sequences for machine learning tasks. The script reads sequences from FASTA files, fragments and pads sequences, encodes sequences into one-hot representations, and converts textual class labels into numerical labels. It utilizes libraries such as NumPy, Pandas, and Biopython.
Features

    Setting Up the Environment:
        Configures environment variables for CUDA to enable GPU usage.
        Selects the appropriate device (GPU or CPU) for computations.

    Reading Sequences:
        Parses FASTA files to extract sequence IDs and corresponding nucleotide sequences as strings.

    Encoding Sequences:
        Converts DNA sequences into one-hot encoded representations. Each nucleotide ('A', 'C', 'G', 'T') is represented by a list of four elements, where a '1' indicates the presence of the nucleotide and '0' otherwise.

    Fragmenting Sequences:
        Splits long sequences into smaller fragments with specified overlap and maximum size. Short sequences are padded to ensure uniform length. Fragments are created only if the sequence length exceeds a predefined maximum.

    Preparing Sequences:
        Reads sequences from a file, fragments longer sequences, and pads shorter ones. It labels the sequences based on their ID prefixes and returns the processed sequences and labels.

    Loading and Preprocessing Data:
        Aggregates sequences from multiple FASTA files.
        Prepares and processes the sequences by fragmenting and padding them.
        Encodes the sequences into one-hot representations.
        Converts textual class labels ('protein_coding', 'lncRNA', 'rep') into numerical labels.

Usage

    Set Environment Variables:
        Ensure CUDA environment variables are set correctly to enable GPU usage.

    Define Paths to Input Files:
        Provide the paths to the FASTA files containing the sequences to be processed.

    Execute the Script:
        Run the script to read, process, and encode the sequences.

    Convert Labels:
        Convert textual class labels into numerical labels for use in machine learning models.

Output

    One-Hot Encoded Sequences:
        Each nucleotide sequence is converted into a list of lists, where each inner list represents the one-hot encoding of a nucleotide.
    Numerical Labels:
        Textual labels indicating the type of sequence ('protein_coding', 'lncRNA', 'rep') are converted into numerical labels (0, 1, 2).

Dependencies

    numpy
    pandas
    biopython
    torch

Installation

Install the required libraries using pip:

sh

pip install numpy pandas biopython torch

Example

sh

python preprocess_sequences.py

This command will read the specified FASTA files, preprocess the sequences, and output the one-hot encoded sequences and numerical labels.
################################################################
CNN-model

This repository contains tools for preprocessing RNA sequences and a neural network model for classifying RNA sequences. The project includes scripts to read sequences from FASTA files, process and encode these sequences, and a PyTorch-based neural network model designed to classify them into three categories: protein_coding, lncRNA, and rep.
Features

    RNA Sequence Preprocessing:
        Reading Sequences: Parses FASTA files to extract sequence IDs and corresponding nucleotide sequences.
        Encoding Sequences: Converts DNA sequences into one-hot encoded representations.
        Fragmenting Sequences: Splits long sequences into smaller fragments with specified overlap and maximum size. Pads shorter sequences to ensure uniform length.
        Preparing Sequences: Aggregates sequences from multiple FASTA files, processes them by fragmenting and padding, and labels the sequences based on their IDs.
        Converting Labels: Converts textual class labels (protein_coding, lncRNA, rep) into numerical labels for machine learning tasks.

    Neural Network Model:
        Architecture: A convolutional neural network (CNN) that processes one-hot encoded DNA sequences and classifies them.
        Layers:
            Convolutional Layer: Applies 1D convolution to the input sequences.
            Pooling Layer: Uses max pooling to down-sample the feature maps.
            Dropout Layer: Regularizes the network by randomly setting a fraction of input units to zero.
            Fully Connected Layers: Transforms the flattened feature maps into class probabilities.

Prerequisites

Make sure you have the following Python libraries installed:

    numpy
    torch

You can install these using pip:

sh

pip install numpy torch

Usage

    Set Up Environment:
        Ensure CUDA environment variables are set correctly to enable GPU usage.

    Define Paths to Input Files:
        Provide the paths to the FASTA files containing the sequences to be processed.

    Execute the Preprocessing Script:
        Run the script to read, process, and encode the sequences, and convert labels.

    Train the Neural Network Model:
        Load the preprocessed data.
        Train the model using the one-hot encoded sequences and numerical labels.

Example

    Preprocessing Sequences:
        Run the preprocessing script to read sequences from the provided FASTA files, fragment, pad, and encode them, and convert labels.

    Training the Model:
        Initialize the RNASequenceClassifier model.
        Train the model using the preprocessed data.
    ###############################################################"
    code-execution

This repository contains tools for preprocessing RNA sequences and a neural network model for classifying RNA sequences. The project includes scripts to read sequences from FASTA files, process and encode these sequences, and a PyTorch-based neural network model designed to classify them into three categories: protein_coding, lncRNA, and rep.
Features

    RNA Sequence Preprocessing:
        Reading Sequences: Parses FASTA files to extract sequence IDs and corresponding nucleotide sequences.
        Encoding Sequences: Converts DNA sequences into one-hot encoded representations.
        Fragmenting Sequences: Splits long sequences into smaller fragments with specified overlap and maximum size. Pads shorter sequences to ensure uniform length.
        Preparing Sequences: Aggregates sequences from multiple FASTA files, processes them by fragmenting and padding, and labels the sequences based on their IDs.
        Converting Labels: Converts textual class labels (protein_coding, lncRNA, rep) into numerical labels for machine learning tasks.

    Neural Network Model:
        Architecture: A convolutional neural network (CNN) that processes one-hot encoded DNA sequences and classifies them.
        Layers:
            Convolutional Layer: Applies 1D convolution to the input sequences.
            Pooling Layer: Uses max pooling to down-sample the feature maps.
            Dropout Layer: Regularizes the network by randomly setting a fraction of input units to zero.
            Fully Connected Layers: Transforms the flattened feature maps into class probabilities.

    Training and Evaluation:
        Data Preparation: Loads and prepares the data for training, validation, and testing.
        Model Initialization: Initializes the CNN model.
        Optimization: Uses the Adam optimizer for training the model.
        Loss Calculation: Implements a custom loss function that accounts for masked (padded) data.
        Early Stopping: Stops training early if the validation accuracy does not improve for a specified number of epochs.
        Model Saving: Saves the trained model to a file.

Prerequisites

Ensure you have the following Python libraries installed:

    numpy
    torch

You can install these using pip:

sh

pip install numpy torch

Usage

    Set Up Environment:
        Ensure CUDA environment variables are set correctly to enable GPU usage.

    Define Paths to Input Files:
        Provide the paths to the FASTA files containing the sequences to be processed.

    Execute the Preprocessing Script:
        Run the script to read, process, and encode the sequences, and convert labels.

    Train the Neural Network Model:
        Load the preprocessed data.
        Train the model using the one-hot encoded sequences and numerical labels.

Training and Evaluation

    Data Loading:
        Loads data tensors for sequences and labels.
        Creates a custom dataset class to handle data loading.

    Data Splitting:
        Splits the dataset into training, validation, and test sets.

    Data Loaders:
        Creates data loaders for training, validation, and testing with specified batch sizes.

    Class Weights Calculation:
        Calculates class weights to handle imbalanced datasets.

    Model Initialization and Training:
        Initializes the RNASequenceClassifier model.
        Trains the model using the Adam optimizer.
        Applies a custom loss function that considers masked (padded) data.
        Implements early stopping based on validation accuracy.

    Model Saving:
        Saves the trained model to a specified file path.

Example

    Preprocessing Sequences:
        Run the preprocessing script to read sequences from the provided FASTA files, fragment, pad, and encode them, and convert labels.

    Training the Model:
        Initialize the RNASequenceClassifier model.
        Train the model using the preprocessed data.
