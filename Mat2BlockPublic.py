import numpy as np
from scipy import sparse
from scipy import io as sio
import math, copy

def PrintBins(bins):
    for br in range(len(bins)):
        for bc in range(len(bins[0])):
            print "(", br, ",", bc, ")"
            print bins[br][bc], '\n'


def BuildBinsFromRCV(rows, cols, vals, size_mat, B, size_blk):
    print len(rows), len(cols)
    len_r, len_c = size_mat
    len_blk_r, len_blk_c = size_blk
   
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(len_r,len_c),
                              dtype=int)
    mat = sparse.csr_matrix(mat, dtype=int)
    # Split matrix into blocks
    bins = [[[] for b2 in range(B)] for b1 in range(B)]
    for br in range(B):
        for bc in range(B):
            r1 = int(len_blk_r * br)
            r2 = int(min(len_blk_r * (br+1), len_r))
            c1 = int(len_blk_c * bc)
            c2 = int(min(len_blk_c * (bc+1), len_c))
            bins[br][bc] = mat[r1:r2, c1:c2]

    return bins


def BuildBinsFromMat(mat, B, size_blk, sparse_flag):
    len_r, len_c = mat.shape
    print len_r, len_c
    len_blk_r, len_blk_c = size_blk
    bins = [[[] for b2 in range(B)] for b1 in range(B)]
    for br in range(B):
        for bc in range(B):
            r1 = int(len_blk_r * br)
            r2 = int(min(len_blk_r * (br+1), len_r))
            c1 = int(len_blk_c * bc)
            c2 = int(min(len_blk_c * (bc+1), len_c))
            if sparse_flag == 0:
                bins[br][bc] = mat[r1:r2,c1:c2]
            else:
                bins[br][bc] = sparse.csr_matrix(mat[r1:r2,c1:c2])

    return bins


def Edges2Bins(filename, B, sym_flag, shuf_flag, diag_flag):
    '''
    Read adjacency matrix from file in the form of (row, col, 1), organize it into bins.
    Output: A 2D-list of csr parse matrices
    '''
    data = np.genfromtxt(filename, delimiter=',', dtype=int)
    data = data[:,0:3]

    maxes = np.amax(data, axis=0)
    if sym_flag == 1:
        maxes = np.array([max(maxes), max(maxes)])
    num_node = maxes[0]
    size_blk = np.ceil(maxes/float(B))
    size_blk_rb = maxes - (B - 1)*size_blk

    # Shuffle
    if shuf_flag == 1:
        shuf = np.random.permutation(maxes[0])
    else:
        shuf = np.array(range(maxes[0]))

    # Construct bins
    rows = shuf[data[:,0]-1]
    cols = shuf[data[:,1]-1]
    vals = data[:,2]

    if sym_flag == 1:
        rows_tmp = copy.copy(rows)
        rows = np.hstack([rows, cols])
        cols = np.hstack([cols, rows_tmp])
        vals = np.hstack([vals, vals])
        rows_tmp = []
    
    if diag_flag == 1:
        rows = np.hstack([rows, range(maxes[0])])
        cols = np.hstack([cols, range(maxes[0])])
        vals = np.hstack([vals, np.ones(maxes[0], dtype=int)])

    print "Number of edges: ", len(rows)

    A = BuildBinsFromRCV(rows, cols, vals, maxes, B, size_blk)

    return A, shuf


def Adjmat2Bins(filename, B, sym_flag, shuf_flag, diag_flag):
    '''
        Read adjacency matrix from file in the form of adj matrix, organize it into bins.
        Output: A 2D-list of csr parse matrices
    '''
    A = sio.loadmat(filename)['A']
    size_blk = np.ceil(np.array(A.shape) / float(B))
    
    if (sym_flag == 1) & (A.shape[0] == A.shape[1]):
        A = A + A.T
    if (diag_flag == 1) & (A.shape[0] == A.shape[1]):
        A = A + np.eye(A.shape[0])

    A = np.array(A > 0, dtype='int')

    if shuf_flag == 1:
        shuf = np.random.permutation(A.shape[0])
    else:
        shuf = np.array(range(A.shape[0]))

    print "Number of edges: ", np.sum(A>0)

    A = BuildBinsFromMat(A, B, size_blk, 1)     # assume A is sparse

    return A, shuf


def X2Bins(filename, B, shuf_flag, shuf, sparse_flag):
    ext = filename.split('.')[-1]
    if ext == "csv":
        X = np.genfromtxt(filename, delimiter=',')
        if sparse_flag == 1:
            # X contains row, col, val
            size_X = np.array([max(X[:,0]), max(X[:,1])])
            X = sparse.coo_matrix((X[:,2], (X[:,0]-1,X[:,1]-1)),    # original index starts from 1
                                  shape=(size_X[0],size_X[1]))
            X = sparse.csr_matrix(X)
        else:
            # non-sparse, we already have the X
            size_X = np.array(X.shape)
    elif ext == "mat":
        X = sio.loadmat(filename)['X']
        size_X = np.array(X.shape)

    # Shuffle
    if shuf_flag == 1:
        X = X[shuf, :]
    
    size_blk = np.ceil(size_X/float(B))
    size_blk_rb = size_X - (B - 1)*size_blk

    X = BuildBinsFromMat(X, B, size_blk, sparse_flag)

    return X
