import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    """
    Convert a sparse matrix into tuple representation.

    Parameters
    ----------
    sparse_mx : scipy.sparse matrix
        Input sparse matrix.

    Returns
    -------
    tuple
        Coordinates, values, and shape of the sparse matrix.
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape


def preprocess_graph(adj):
    """
    Normalize the adjacency matrix for graph convolution.

    The function first adds self-loops and then applies symmetric
    normalization: D^(-1/2) A D^(-1/2).
    """
    adj = sp.coo_matrix(adj)

    # Add self-loops
    adj_ = adj + sp.eye(adj.shape[0])

    # Compute D^(-1/2)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(
        np.power(rowsum, -0.5).flatten()
    )

    # Symmetric adjacency normalization
    adj_normalized = (
        adj_
        .dot(degree_mat_inv_sqrt)
        .transpose()
        .dot(degree_mat_inv_sqrt)
        .tocoo()
    )

    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, features, placeholders):
    """
    Construct the feed dictionary for TensorFlow training.

    Parameters
    ----------
    adj_normalized : tuple
        Normalized adjacency matrix in tuple format.
    features : tuple
        Node feature matrix in tuple format.
    placeholders : dict
        TensorFlow placeholders.

    Returns
    -------
    dict
        Feed dictionary used for model training.
    """
    feed_dict = dict()

    feed_dict.update({
        placeholders['features']: features
    })

    feed_dict.update({
        placeholders['adj']: adj_normalized
    })

    return feed_dict


def mask_test_edges(adj):
    """
    Generate positive and negative edge samples.

    This function removes self-loops, extracts existing edges as
    positive samples, and randomly generates an equal number of
    non-existing circRNA-disease pairs as negative samples.
    """

    # Remove diagonal elements
    adj = adj - sp.dia_matrix(
        (adj.diagonal()[np.newaxis, :], [0]),
        shape=adj.shape
    )

    adj.eliminate_zeros()

    # Ensure that all diagonal elements are zero
    assert np.diag(adj.todense()).sum() == 0

    # Extract upper-triangular edges to avoid duplicated undirected edges
    adj_triu = sp.triu(adj)

    adj_tuple = sparse_to_tuple(adj_triu)

    # Positive edges without duplication
    edges = adj_tuple[0]

    # All observed edges
    edges_all = sparse_to_tuple(adj)[0]

    # Randomly shuffle edge indices
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)

    def ismember(a, b, tol=5):
        """
        Check whether an edge already exists in the edge set.
        """
        rows_close = np.all(
            np.round(a - b[:, None], tol) == 0,
            axis=-1
        )

        return np.any(rows_close)

    # Generate negative samples
    edges_false = []

    num_positive_edges = len(edges)

    while len(edges_false) < num_positive_edges:

        # Randomly select one circRNA node and one disease node
        idx_i = np.random.randint(0, 561)
        idx_j = np.random.randint(561, adj.shape[0])

        if idx_i == idx_j:
            continue

        # Skip existing positive associations
        if ismember([idx_i, idx_j], edges_all):
            continue

        edges_false.append([idx_i, idx_j])

    # These edge lists contain only one direction of each edge.
    return edges_all, edges, edges_false
```
