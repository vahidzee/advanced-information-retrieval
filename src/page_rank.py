import numpy as np


def create_matrix(nodes: dict, alpha: float = 0.1):
    """create_matrix: creates matrix from a dict of nodes and their neighbors.
    Parameters
    ---------
    nodes: dict
        a dictionary with key of id and value of an object of information of a document
    alpha: float
        parameter to determine the jump rate of the matrix
    Returns
    ------
    numpy matrix
        a matrix showing the probabilities of each transform
    """
    ids = list(nodes.keys())
    vectors = []
    N = len(ids)
    for i in range(N):
        temp = []
        nei = nodes[i].references
        for j in range(N):
            if ids[j] in nei:
                temp.append((1 - alpha) / len(nei))
            else:
                temp.append(alpha / (N - len(nei)))
        vectors.append(temp)
    mat = np.matrix(vectors).T
    return mat
    pass


def get_eigen_vector(M, convergence_limit: float = 1):
    """
    get_eigen_vector: creates the stable state vector which shows the probabilities of each documents
    :param M: numpy matrix of shape N*N
    :param convergence_limit: float
        determining when to stop the iterations
    :return: numpy array of shape N*1 showing the stable state probabilities of each document
    """
    v = np.random.rand(M.shape[1], 1)
    v = v / np.linalg.norm(v, 1)
    dist = convergence_limit + 10
    while dist > convergence_limit:
        past_v = v
        v = M @ v
        dist = np.linalg.norm(v - past_v)
    return v


def pagerank(file_name: str, alpha: float = 0.1, conv_limit: float = 1, top_count: int = 10):
    """
    :param file_name: the name of the file with the documents information
    :param alpha: jump rate
    :param conv_limit:
    :param top_count: number of top results to return
    :return: list of the title of top results
    """
    nodes = {}
    # TODO some how get the dict of the objects
    vec = list(get_eigen_vector(create_matrix(nodes, alpha), conv_limit).T)
    sorted_vec = list(sorted(vec, reverse=True))
    top_results = []
    ids = list(nodes.keys())
    for i in range(top_count):
        top_results.append(nodes[ids[vec.index(sorted_vec[i])]].title)
    return top_results
