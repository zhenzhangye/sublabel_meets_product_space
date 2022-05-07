import math

import numpy as np
import torch

from util.helpers import vec2img, img2vec, numEdges, numVertices
from util.sparse import interpFEM, nabla2d, spabs


# return per element L1-norm, is of size(w*h,c)
def L1loss(x):
    return torch.abs(x)


# return per element squared L2-norm, is of size(w*h,c)
def sqL2loss(x):
    return x * x


# return per element truncated cost.
def truncate(x, a=0.025):
    return torch.clamp(x, max=a)


def gradientDescentStep(x, step_size, dir):
    return x - step_size * dir


def gradientAscentStep(x, step_size, dir):
    return gradientDescentStep(x, -step_size, dir)


def extrapolation(x_new, x_old, step_size):
    return x_new + step_size * (x_new - x_old)


def proxSquaredL2(x, x_prev, weight, step_size):
    return (weight * x + step_size * x_prev) / (weight + step_size)

def proxL2(x, n, weight, step_size, variable):
    result = torch.zeros(x.shape, dtype=torch.float)
    num_faces = 200
    for dd in range(x.shape[1]):
        x_vec = x[:, dd].unsqueeze(1)
        x_hat = vec2img(x_vec, 2, n).squeeze()
        x_hat_norm = torch.sqrt(torch.clamp((x_hat * x_hat).sum(dim=-1), min=1e-10))
        x_row_norm = weight / x_hat_norm.repeat(1, 2).transpose(-2, -1)
        x_sol = gradientDescentStep(torch.ones((x.shape[0], 1), dtype=torch.float), step_size,
                                    x_row_norm)  # it's not a GD step, just does the same algebra
        result[:, dd] = torch.clamp(x_sol, min=0).mul(img2vec(x_hat)).squeeze()
    return result


def proxTV(x, lambda_, step_size):
    ind1 = x > (lambda_ * step_size)
    ind2 = x < (-lambda_ * step_size)
    x_new = torch.zeros(x.shape, dtype=torch.float)
    x_new[ind1] = x[ind1] - lambda_ * step_size.repeat(1, x.shape[1])[ind1]
    x_new[ind2] = x[ind2] + lambda_ * step_size.repeat(1, x.shape[1])[ind2]
    return x_new


def proxIndicatorPositive(x):
    return torch.clamp(x, min=0)


def projSimplex(f):
    num_faces, num_labels = f.shape

    y = -(torch.sort(-f, 1)[0])
    tmpsum = torch.zeros(num_faces, dtype=torch.float)
    active = torch.ones(num_faces, dtype=torch.bool)
    tmax = torch.zeros(num_faces, dtype=torch.float)
    for ii in range(num_labels - 1):
        tmpsum[active] = tmpsum[active] + y[active, ii]
        tmax[active] = (tmpsum[active] - 1) / (ii + 1)
        active = tmax < y[:, ii + 1]

    tmax[active] = (tmpsum[active] + y[active, -1] - 1) / num_labels
    u = torch.clamp(f - tmax.unsqueeze(1).repeat(1, num_labels), min=0)

    return u


def proxIndicatorSimplex(x, dim1, dim2, problem):
    label_space_dim = x.shape[-1]
    result = torch.zeros(x.shape, dtype=torch.float)
    if problem == "hsv":
        result[:-dim2, 0] = img2vec(projSimplex(vec2img(x[:-dim2, 0], dim1-1, dim2).squeeze(0))).squeeze()
    else:
        result[:, 0] = img2vec(projSimplex(vec2img(x[:, 0], dim1, dim2).squeeze(0))).squeeze()
    for dd in range(1, label_space_dim):
        result[:, dd] = img2vec(projSimplex(vec2img(x[:, dd], dim1, dim2).squeeze(0))).squeeze()
    return result

def primalRegEnergyFixedU(u, lambda_, max_iter, eps, verbose):
    if (verbose):
        print('Compute primal energy')
    label_space_dim, height, width = u.shape
    num_edges = numEdges(height, width)
    num_vertices = numVertices(height, width)

    u_vec = img2vec(u)
    v = torch.zeros((2 * num_vertices, label_space_dim), dtype=torch.float)
    p = torch.zeros((num_edges, label_space_dim), dtype=torch.float)

    interp_p, interp_p_t = interpFEM(height, width)  # moves values on edges to values on vertices
    nabla = nabla2d(height, width)[0]

    tau_v = 0.09 / torch.sparse.sum(spabs(interp_p), dim=1).to_dense().unsqueeze(1)
    nu_p = 9.8 / torch.sparse.sum(spabs(interp_p), dim=0).to_dense().unsqueeze(1)

    if (verbose):
        print("  {:<10}  {:<8}".format('Iterations', 'Residual(s)'))
    nabla_u = nabla.mm(u_vec)
    for i in range(max_iter):
        # label_space_dim dimensional residual vector
        res = torch.sum(torch.abs(interp_p_t.mm(v) - nabla_u), dim=0) / nabla.shape[0]

        if (verbose):
            iter = str(i) + '/' + str(max_iter - 1)
            print("\r  {:<10}  {:}".format(iter, res.tolist()), end=' ', flush=True)

        if torch.all(res < eps):
            break

        v_prev = v.detach().clone()

        # primal-dual update steps
        v_hat = gradientDescentStep(v, tau_v, interp_p.mm(p))
        v = proxTV(v_hat, lambda_, tau_v)
        v_bar = extrapolation(v, v_prev, 1)
        p = gradientAscentStep(p, nu_p, interp_p_t.mm(v_bar) - nabla_u)
    #print()

    return L1loss(v)
