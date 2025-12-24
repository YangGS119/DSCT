import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def DSCT(datas, types, K=20, c=0.001):
    """
    Data-Driven Superior Connected Tree for Anomaly Detection

    Parameters:
    -----------
    datas : ndarray, shape (n_samples, n_features)
        Input data matrix, each row is a sample, each column is a feature
    types : ndarray, shape (1, n_features)
        Feature type array, 0 for numerical features, 1 for categorical features
    K : int, optional (default=20)
        Number of nearest neighbors for constructing the KNN graph
    c : float, optional (default=0.001)
        Truncation threshold for SCS similarity, prevents excessive similarity values from affecting results

    Returns:
    --------
    AS : ndarray, shape (n_samples, 1)
        Anomaly Scores, higher scores indicate greater likelihood of being an anomaly
    """
    n_samples, n_features = datas.shape

    HMOM_matrix = HMOM(datas, types)

    nbrs = NearestNeighbors(n_neighbors=K, leaf_size=10000, metric="precomputed").fit(HMOM_matrix)
    dist, index = nbrs.kneighbors()

    rho = np.sum(1 - dist / n_features, axis=1)

    adjacency = nbrs.kneighbors_graph(mode='connectivity').toarray()
    np.fill_diagonal(adjacency, 1)

    MNN = (adjacency * adjacency.T).astype(float)

    intersection = np.dot(MNN, MNN.T)
    neighbor_counts = np.sum(MNN, axis=1, keepdims=True)
    union = neighbor_counts + neighbor_counts.T - intersection

    SCS = np.divide(intersection, union)
    SCS_c = np.clip(SCS, a_min=0, a_max=c)

    AS = np.zeros(n_samples, dtype=np.float64)
    sorted_index = np.argsort(-rho)
    for i in range(n_samples):
        current_idx = sorted_index[i]
        current_rho = rho[current_idx]

        neighbor_index = index[current_idx]
        neighbor_rho = rho[neighbor_index]
        neighbor_sim = SCS_c[current_idx, neighbor_index]

        mask = neighbor_rho > current_rho
        if not mask.any():
            continue

        valid_index = neighbor_index[mask]
        valid_sim = neighbor_sim[mask]

        max_sim_idx = np.argmax(valid_sim)
        max_neighbor = valid_index[max_sim_idx]
        max_similarity = valid_sim[max_sim_idx]

        if max_similarity >= 0:
            ld_ratio = rho[max_neighbor] / (rho[current_idx] + 1e-10) - 1
            AS[current_idx] = AS[max_neighbor] + HMOM_matrix[current_idx][max_neighbor] * ld_ratio / (max_similarity + 1e-10)

    return AS[:, None]


def HMOM(datas, types):
    """
    Heterogeneous Manhattan-Overlap Metric between two objects

    Parameters:
    -----------
    datas : ndarray, shape (n_samples, n_features)
        Input data matrix
    types : ndarray, shape (1, n_features)
        Feature type array, 0 for numerical, 1 for categorical

    Returns:
    --------
    dis : ndarray, shape (n_samples, n_samples)
        Pairwise distance matrix combining numerical and categorical distances
    """
    n, m = datas.shape

    num_fea = types[0] == 0
    nom_fea = types[0] == 1

    num_dis = squareform(pdist(datas[:, num_fea], metric="cityblock")) if num_fea.any() else np.zeros((n, n))
    nom_dis = squareform(pdist(datas[:, nom_fea], metric='hamming')) if nom_fea.any() else np.zeros((n, n))

    dis = num_dis + nom_dis * np.sum(nom_fea)

    return dis
