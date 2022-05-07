import torch
from scipy import sparse
import numpy as np

from util.helpers import numFaces, numEdges, geth_x, geth_gamma, geth_gamma_manifold

def spcoo(r, c, v, size=0):
    if size == 0:
        return torch.sparse_coo_tensor(indices=torch.vstack([r, c]), values=v,
                                       dtype=torch.float).coalesce()
    else:
        return torch.sparse_coo_tensor(indices=torch.vstack([r, c]), values=v, size=size,
                                       dtype=torch.float).coalesce()


def spabs(spmat):
    return torch.sparse_coo_tensor(indices=spmat.indices(), values=torch.abs(spmat.values()),
                                   size=spmat.size(), dtype=spmat.dtype).coalesce()


def spzeros(size):
    return torch.sparse_coo_tensor(size=size, dtype=torch.float).coalesce()


def speye(dim):
    return spcoo(torch.arange(dim), torch.arange(dim), torch.ones(dim))


def spdiag(vals_in, offsets, size):
    rows = torch.empty(0)
    cols = torch.empty(0)
    vals = torch.empty(0)

    for offset in offsets:
        if offset < 0:
            length2border = min(size[1], size[0] + offset)
            rows = torch.hstack((rows, torch.arange(-offset, length2border - offset)))
            cols = torch.hstack((cols, torch.arange(length2border)))
        elif offset == 0:
            rows = torch.hstack([rows, torch.arange(min(size))])
            cols = torch.hstack([cols, torch.arange(min(size))])
        elif offset > 0:
            length2border = min(size[1] - offset, size[0])
            rows = torch.hstack([rows, torch.arange(length2border)])
            cols = torch.hstack([cols, torch.arange(offset, length2border + offset)])
        vals = torch.hstack([vals, vals_in[offsets.index(offset)]])

    return spcoo(rows, cols, vals, size)


def spblockdiag(mats_list):
    v = [None] * len(mats_list)
    r = [None] * len(mats_list)
    c = [None] * len(mats_list)
    rr_, cc_ = 0, 0
    for ii in range(len(mats_list)):
        v[ii] = mats_list[ii].values()
        r[ii] = mats_list[ii].indices()[0] + rr_
        c[ii] = mats_list[ii].indices()[1] + cc_

        rr_ += mats_list[ii].shape[0]
        cc_ += mats_list[ii].shape[1]

    v = torch.cat(v)
    r = torch.cat(r)
    c = torch.cat(c)

    return spcoo(r, c, v, (rr_, cc_))


def torchSparseToScipySparse(mat):
    values = mat.coalesce().values().to('cpu')
    indices = mat.coalesce().indices().to('cpu')
    shape = mat.size()

    return sparse.coo_matrix((values, indices), shape=shape)


def scipySparseToTorchSparse(mat):
    r = torch.tensor(mat.row)
    c = torch.tensor(mat.col)
    v = torch.tensor(mat.data)
    s = torch.Size(mat.shape)
    return spcoo(r, c, v, s)


def kron(mat1, mat2):
    mat1_ = torchSparseToScipySparse(mat1)
    mat2_ = torchSparseToScipySparse(mat2)

    result_ = sparse.kron(mat1_, mat2_).tocoo()

    return scipySparseToTorchSparse(result_)


def extractRowsFromSamplingMat(mat, rows):
    i = mat.indices()
    v = mat.values()

    rows_extract = torch.arange(rows.shape[0]).repeat(2)
    
    sp_indices = torch.hstack([rows * 2, rows * 2 + 1])
    cols_extract = i[1, sp_indices]
    values_extract = v[sp_indices]
    size = (rows.numel(), mat.shape[1])

    return spcoo(rows_extract, cols_extract, values_extract, size)


# return gradient and divergence operator of size=(num_edges, num_faces) and size=(num_faces, num_edges), respectively
def nabla2d(height, width):
    # sparse helper matrices
    eye_width = speye(width)
    eye_height = speye(height)

    # rescaling
    h_x = geth_x(width, height)

    e0_h = torch.ones(height) / h_x
    e0_h[0] = 0
    e1_h = -torch.ones(height) / h_x
    e1_h[-1] = 0
    dy_tilde = spdiag([e0_h, e1_h], offsets=[0, -1], size=(height + 1, height))
    dy = kron(eye_width, dy_tilde)

    e0_w = torch.ones(width) / h_x
    e0_w[0] = 0
    e1_w = -torch.ones(width) / h_x
    e1_w[-1] = 0
    dx_tilde = spdiag([e0_w, e1_w], offsets=[0, -1], size=(width + 1, width))
    dx = kron(dx_tilde, eye_height)

    nabla_2d = torch.vstack((dx, dy)).coalesce()
    div_2d = -nabla_2d.transpose(0, 1).coalesce()

    return nabla_2d, div_2d


# diff2d is sparse matrix of size=(num_edges, num_faces)
# dims is scalar
# returns 3d gradient and divergence operator of size=(num_edges * dims , num_faces * dims) and size=(num_faces * dims, num_edges * dims), respectively
def nabla3d(nabla_2d, dims):
    eye_dims = speye(dims)
    nabla_3d = kron(eye_dims, nabla_2d)
    div_3d = kron(eye_dims, -nabla_2d.transpose(0, 1).coalesce())
    return nabla_3d, div_3d

# returns gradient operator based on FEM of size=(2 * num_vertices, num_edges)
def interpFEM(height, width):
    num_faces = numFaces(height, width)
    num_edges = numEdges(height, width)

    eye_width = speye(width)
    eye_faces = speye(num_faces)
    csr_left = spzeros((num_faces, num_edges - num_faces))
    csr_right = spzeros((num_faces, height))
    csr_right2 = spzeros((num_faces, width * (height + 1)))
    csr_up = spzeros((num_faces, (width + 1) * height))
    diag_up = spdiag([torch.ones(height)], offsets=[0], size=(height, height + 1))
    diag_down = spdiag([torch.ones(height)], offsets=[1], size=(height, height + 1))

    left = torch.hstack([eye_faces, csr_left]).coalesce()
    right = torch.hstack([csr_right, eye_faces, csr_right2]).coalesce()
    up = torch.hstack([csr_up, kron(eye_width, diag_up)]).coalesce()
    down = torch.hstack([csr_up, kron(eye_width, diag_down)]).coalesce()
    interp_fem = torch.vstack([up, left, down, right, left, down, right, up]).coalesce()
    interp_fem_t = interp_fem.transpose(0, 1).coalesce()
    return interp_fem, interp_fem_t


# nabla_fem is gradient operator based on FEM of size=(8 * num_faces, num_edges)
# return 3d nabla_fem gradient operator based on FEM of size=(8 * num_faces * num_labels, num_edges * num_labels)
def nablaGamma(interp_fem, labels, problem, num_edges):
    num_labels = labels.shape[0]

    # rescaling
    h_gamma = geth_gamma(labels)

    e0_l = -torch.ones(num_labels) / h_gamma
    e1_l = torch.ones(num_labels - 1) / h_gamma
    d_gamma = spdiag([e0_l, e1_l], offsets=[0, 1], size=(num_labels, num_labels))

    nabla_gamma = kron(d_gamma, interp_fem)
    nabla_gamma_t = nabla_gamma.transpose(0, 1).coalesce()
    if problem == "hsv":
        h_gamma_mf = geth_gamma_manifold(num_labels)
        e0_l = -torch.ones(num_labels) / h_gamma_mf
        e1_l = torch.ones(num_labels - 1) / h_gamma_mf
        d_gamma_tmp = spdiag([e0_l, e1_l, torch.ones(1)/h_gamma_mf], offsets=[0, 1, 1-num_labels], size=(num_labels, num_labels))
        nabla_gamma_manifold = kron(d_gamma_tmp, interp_fem)
        
        row_indices = nabla_gamma_manifold.indices()[0]
        col_indices = nabla_gamma_manifold.indices()[1]
        val = nabla_gamma_manifold.values()
        col_indices[col_indices >= num_edges * (num_labels - 1)] -= num_edges * (num_labels - 1)
        nabla_gamma_manifold = spcoo(row_indices, col_indices, val, nabla_gamma_manifold.shape)
        nabla_gamma_manifold_t = nabla_gamma_manifold.transpose(0, 1).coalesce()
    else:
        nabla_gamma_manifold = nabla_gamma
        nabla_gamma_manifold_t = nabla_gamma_t
    return nabla_gamma, nabla_gamma_t, nabla_gamma_manifold, nabla_gamma_manifold_t
