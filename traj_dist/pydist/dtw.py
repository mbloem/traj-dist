import numpy as np
from basic_euclidean import eucl_dist
from basic_spherical import great_circle_distance


######################
# Euclidean Geometry #
######################

def e_dtw(t0, t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array

    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    C : len(n0) x len(n1) numpy matrix
          C[i,j] contains DTW distance that ends with the pairing 
          of point i on t0 with point j on t1
    W : len(t0) list of len(t1) lists of (t0 index, t1 index) length-2 tuples of trajectory point indices
          W[i][j] contains the indices of the previous trajectory point pairing
          on the DTW path that ends with the pairing 
          of point i on t0 with point j on t1
    """

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    W = [[None for j in range(n1)] for i in range(n0)]
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            min_is_from_idx = np.argmin([C[i, j - 1], C[i - 1, j - 1], C[i - 1, j]])
            min_from_distance = [C[i, j - 1], C[i - 1, j - 1], C[i - 1, j]][min_is_from_idx]
            min_is_from = [(i,j-1),(i-1,j-1),(i-1,j)][min_is_from_idx]
            C[i, j] = eucl_dist(t0[i - 1], t1[j - 1]) + min_from_distance
            if i==1 and j==1:
                W[i-1][j-1] = None
            else:
                W[i-1][j-1] = (min_is_from[0]-1,min_is_from[1]-1) # -1 on RHS because re-index C (to C[1:,1:]) upon return
    dtw = C[n0, n1]

    return dtw, C[1:,1:], W

def e_extract_dtw_warping_path(W):
    """
    Usage
    -----
    The Dynamic-Time Warping path between trajectory t0 and t1.

    Parameters
    ----------
    param W : len(t0) list of len(t1) lists (t0 index, t1 index) length-2 tuples returned by e_dtw

    Returns
    -------
    path : list of (t0 index, t1 index) length-2 tuples
          The Dynamic-Time Warping path between trajectory t0 and t1
          corresponding to the distance returned by e_dtw
    """
    path = []

    idxs = (len(W)-1, len(W[0])-1)

    while idxs != (0,0):
        path.append(idxs)
        idxs = W[idxs[0]][idxs[1]]

    path.append((0,0))
    
    path.reverse()

    return path


######################
# Spherical Geometry #
######################

def s_dtw(t0, t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array

    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) + \
                      min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
    dtw = C[n0, n1]
    return dtw
