import numpy as np
import pandas as pd


def create_matrix(data, alpha: float = 0.1):
    """create_matrix: creates matrix from a dict of nodes and their neighbors.
    Parameters
    ---------
    data: dict
        output of the crawler as a dataframe
    alpha: float
        parameter to determine the jump rate of the matrix
    Returns
    ------
    numpy matrix
        a matrix showing the probabilities of each transform
    """
    matrix = np.zeros((len(data), len(data)))
    alpha = 0.1
    for index in range(len(data)):
        row = data.iloc[index]
        refs = set(int(item) for item in row['references'])
        ref_cells = data[data['id'].apply(lambda x: x in refs)].index
        matrix[index] = (alpha if len(ref_cells) else 1) / (len(data) - len(ref_cells))
        if len(ref_cells):
            matrix[index, ref_cells] = (1 - alpha) / len(ref_cells)
    return matrix


def get_eigen_vector(matrix, convergence_limit: float = 1e-5):
    """
    get_eigen_vector: creates the stable state vector which shows the probabilities of each documents
    :param M: numpy matrix of shape N*N
    :param convergence_limit: float
        determining when to stop the iterations
    :return: numpy array of shape N*1 showing the stable state probabilities of each document
    """
    v = np.random.rand(matrix.shape[0], 1)
    v = v / np.linalg.norm(v, 1)
    dist = np.inf
    while dist > convergence_limit:
        past_v = v
        v = matrix.T @ v
        dist = np.linalg.norm(v - past_v)
    return v


def page_rank(file_name: str, alpha: float = 0.1, conv_limit: float = 1e-5):
    """
    :param file_name: the name of the file with the documents information
    :param alpha: jump rate
    :param conv_limit:
    :return: list of the title of top results
    """
    data = pd.read_json(file_name)
    scores = get_eigen_vector(create_matrix(data, alpha), conv_limit)
    ranks = np.argsort(-scores.reshape(-1))
    data['page-rank'] = ranks
    return data.iloc[np.argsort(ranks)]
