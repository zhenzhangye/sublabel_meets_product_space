import functools
import inspect
import math

import numpy as np
import torch
from torch.nn.functional import normalize
from torchvision import transforms
import pickle


def ASSERT(cond, message):
    assert cond, message


def getImageAttributes(images):
    num_images = len(images)
    channels = len(images[0].getbands())
    width, height = images[0].size
    return num_images, channels, width, height


# images is a list of pillow images
# dev is the device (cpu or gpu), where data shall be offloaded to
# returns image tensors of size (num_images, height * width, channels), the number of images/channels and the height and width
def pillowImage2TorchArray(images, dev, problem='rof'):
    num_images, channels, width, height = getImageAttributes(images)
    img_vecs = torch.zeros(num_images, numFaces(height, width), channels)
    for ii in range(num_images):
        if problem == "hsv":
            cur_image = images[ii].convert(mode="HSV")
        else:
            cur_image = images[ii]
        img = transforms.ToTensor()(cur_image).to(dev)
        img_vecs[ii, :, :] = img2vec(img)

    return img_vecs, num_images, channels, height, width


# number of pixels
def numFaces(height, width):
    return height * width


# number of edges for FEM
def numEdges(height, width):
    return 2 * numFaces(height, width) + height + width


# number of edges for FEM
def numVertices(height, width):
    return 4 * numFaces(height, width)


# sample all vertices for second constraint (data term)
def vertexSamples(num_faces, num_labels):
    return 4 * num_faces * num_labels


def lerp(a, b, w):
    return a + w * (b - a)


# assume square pixels
def geth_x(width, height):
    # normalize to VGA resolution
    h_x = math.sqrt(640 * 480)
    h_x /= math.sqrt(width * height)
    return h_x


# assume square search window and equidistant intervals
def geth_gamma(labels):
    h_gamma = (labels[-1, 0] - labels[0, 0]) / (labels.shape[0] - 1)
    # normalized to unit interval with 3 labels
    h_gamma /= (1 - 0) / (3 - 1)
    return h_gamma.item()

def geth_gamma_manifold(num_labels):
    p0_x, p0_y = np.cos(0), np.sin(0)
    n_x, n_y = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    p1_x, p1_y = np.cos(2 * np.pi / num_labels), np.sin(2 * np.pi / num_labels)
    n_dist = np.sqrt((p0_x - n_x) **2 + (p0_y - n_y) **2)
    p_dist = np.sqrt((p1_x - p0_x) ** 2 + (p1_y - p0_y) ** 2)
    return 1.0
    #return p_dist / n_dist


# img is assumed to have dimensions [h,w], [1,h,w], or [c,h,w]
# returns a vector of size [h*w,c] (c=1 for grayscale image)
def img2vec(img):
    if img.dim() == 2:  # 2d grayscale image
        return img.transpose(-2, -1).reshape(-1, 1)
    if img.dim() == 3 and img.shape[0] == 1:  # 2d grayscale image with unnecessary 3rd dimension
        return img2vec(img.squeeze(0))
    elif img.dim() == 3 and img.shape[0] > 1:  # Color image
        channels = img.shape[0]
        return torch.cat([img2vec(img[c, :, :]) for c in range(channels)], dim=1)
    else:
        ASSERT(False, "Image size not supported (c,h,w): {}".format(img.shape))


# vec is assumed to have dimensions [w*h,c] or [w*h]
# returns image of dimensions [c,h,w]
def vec2img(vec, width, height):
    ASSERT(vec.dim() <= 2,
           "Vector is no form of column vector [w*h,c] or [w*h]: {}".format(vec.shape))
    if vec.dim() == 1:
        vec = vec.unsqueeze(1)

    channels = vec.shape[-1]
    return vec.transpose(-1, -2).reshape(channels, width, height).transpose(-2, -1)


def dimsum(spmat, dim):
    return torch.sparse.sum(spmat, dim=dim).to_dense().reshape(-1, 1)

def recoverUManifold(u, labels):
    num_labels, labels_space_dim = labels.shape
    num_faces = int(u.shape[0] / num_labels)
    r_label = torch.max(labels[:, 0])

    labels_rep = labels[:, 0].T.repeat(num_faces, 1)
    u = vec2img(u, num_labels, num_faces).squeeze()
    h = labels[torch.argmax(u, axis=1), 0].unsqueeze(1)

    for i in range(20):
        v = log_map(h, labels_rep, r_label)
        vfrom = (u * v / u.sum(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)
        h = exp_map(h, vfrom, r_label).reshape(num_faces, 1)

    return h

# u is of size=(num_faces * num_labels, label_space_dim)
# labels is matrix of size=(num_labels, label_space_dim)
# returns tensor of size=(num_faces, label_space_dim)
def recoverU(u, labels):
    num_labels, labels_space_dim = labels.shape
    num_faces = int(u.shape[0] / num_labels)

    # make size=(label_space_dim, num_faces, num_labels)
    u_img = vec2img(u, num_labels, num_faces)
    l_img = vec2img(labels, num_labels, 1).repeat(1, num_faces, 1)

    # sum over labels and make size=(num_faces, label_space_dim)
    result = (u_img * l_img).sum(dim=-1).transpose(-2, -1)
    return result


# img_vec is of size (num_images, num_faces, channels)
# returns lifted vector of size (num_images, num_faces * samples, channels)
def liftInputImgs(img_vec, samples):
    return img_vec.repeat(1, samples, 1)


# nabla_gamma is a list of interpolation matrices to approximate continuous gamma. Each is of
# size=(num_faces * label_samples, num_faces * num_labels)
# labels is of size (num_labels, label_space_dim)
# returns gamma as tensor of size (num_faces * samples, label_space_dim)
def recoverGamma(nabla_gamma, labels):
    num_labels, label_space_dim = labels.shape
    num_faces = int(nabla_gamma[0].shape[1] / num_labels)
    samples = int(nabla_gamma[0].shape[0] / num_faces)

    labels_vec = labels.repeat_interleave(num_faces, dim=0)
    gamma = torch.zeros(num_faces * samples, label_space_dim)
    for dd in range(label_space_dim):
        gamma[:, dd] = nabla_gamma[dd].mm(labels_vec[:, dd].unsqueeze(1)).squeeze()

    return gamma


def saveImg(img_path, img, img_name):
    fn = img_path + "/" + img_name
    print("Wrote image: ", fn)
    img.save(fn)
    return


def powerIteration(spmat, n_iterations=100, eps=1e-12):
    spmat_t = spmat.t().coalesce()
    u = torch.rand(spmat_t.shape[1])
    v = torch.rand(spmat.shape[1])

    for __ in range(n_iterations):
        v = normalize(spmat_t.mv(u), dim=0, eps=eps)
        u = normalize(spmat.mv(v), dim=0, eps=eps)

    if n_iterations > 0:
        u = u.clone()
        v = v.clone()

    return torch.dot(u, spmat.mv(v))


####################################################################################################
########################################MFD-LIFT####################################################
####################################################################################################
def normalize(u, p=2, thresh=0.0):
    """ Normalizes u along the last axis with norm p.
    If  |u| <= thresh, 0 is returned (this mimicks the sign function).
    """
    ndim = u.shape[-1]
    multi = u.shape if u.ndim > 1 else None
    u = u.reshape(1, ndim) if multi is None else u.reshape(-1, ndim)
    ns = np.linalg.norm(u, ord=p, axis=1)
    fact = np.zeros_like(ns)
    fact[ns > thresh] = 1.0 / ns[ns > thresh]
    out = fact[:, None] * u
    return out[0] if multi is None else out.reshape(multi)


def gramschmidt(X):
    """ Apply Gram-Schmidt orthonormalization to given tuples of same size
    Args:
        X : ndarray of floats, shape (ntuples, nvecs, ndim)
            X[j,i,:] is the `i`-th vector of tuple `j`
    Return:
        Y : ndarray of floats, shape (nbases, nvecs, ndim)
            Orthonormalized bases with basis vectors Y[j,i,:]
    """
    nbases, nvecs, ndim = X.shape
    assert ndim >= nvecs
    Y = np.zeros_like(X)
    for i in range(nvecs):
        Y[:, i, :] = X[:, i, :]
        Y[:, i, :] -= np.einsum('jk,jmk,jml->jl', Y[:, i, :], Y[:, :i, :], Y[:, :i, :])
        Y[:, i, :] /= np.linalg.norm(Y[:, i, :], axis=1)[:, None]
    return Y


class cached_property(object):
    def __init__(self, fun):
        self._fun = fun

    def __get__(self, obj, _):
        setattr(obj, self._fun.__name__, self._fun(obj))
        return getattr(obj, self._fun.__name__)


def broadcast_io(indims, outdims):
    def decorator_func(func, indims=indims):
        params = [str(p) for p in inspect.signature(func).parameters.values()]
        ismethod = params[0] == "self"
        ninputs = len(params) - 1
        if ismethod:
            ninputs -= 1
        if type(indims) is not tuple:
            indims = (indims,) * ninputs

        @functools.wraps(func)
        def wrapped_func(*inputs, **kwargs):
            if ismethod:
                self = inputs[0]
                inputs = inputs[1:]
            assert len(inputs) == ninputs
            if len(kwargs.keys()) == 0:
                output = None
            else:
                assert len(kwargs.keys()) == 1
                output_name = list(kwargs.keys())[0]
                output = kwargs[output_name]
            multi = inputs[0].ndim - indims[0]
            inputs = list(inputs)
            if multi == 0:
                for i, el in enumerate(inputs):
                    assert el.ndim == indims[i]
                    inputs[i] = el[None, None]
                if output is not None:
                    assert output.ndim == outdims
                    output = output[(None,) * (ninputs + 1)]
            elif multi == 1:
                nbatch = inputs[0].shape[0]
                for i, el in enumerate(inputs):
                    assert el.ndim == indims[i] + 1
                    assert el.shape[0] == nbatch
                    inputs[i] = el[:, None]
                if output is not None:
                    assert output.ndim == outdims + 1
                    assert output.shape[0] == nbatch
                    output = output[(slice(None),) + (None,) * ninputs]
            else:
                nbatch = inputs[0].shape[0]
                for i, el in enumerate(inputs):
                    assert el.ndim == indims[i] + 2
                    assert el.shape[0] == nbatch
                if output is not None:
                    assert output.ndim == outdims + ninputs + 1
                    assert output.shape[0] == nbatch
            if output is not None:
                kwargs[output_name] = output
            if ismethod:
                inputs = (self,) + tuple(inputs)
            output = func(*inputs, **kwargs)
            if multi == 0:
                output = output[(0,) * (ninputs + 1)]
            elif multi == 1:
                output = output[(slice(None),) + (0,) * ninputs]
            return output

        return wrapped_func

    return decorator_func

def log_map(location, pfrom, r_label):
    out = torch.zeros_like(pfrom)
    out = pfrom - location
    fact1 = torch.heaviside(out - r_label/2, torch.tensor(0.))
    fact2 = torch.heaviside(-out - r_label/2, torch.tensor(0.))
    out -= (fact1 - fact2) *  r_label
    return out

def exp_map(location, vfrom, r_label):
    out = location + vfrom
    return circ_normalize(out.reshape(-1), r_label)

def circ_normalize(x, r_label):
    while True:
        ind1 = torch.where(x >= r_label)[0]
        x[ind1] -= r_label
        ind2 = torch.where(x < 0)[0]
        x[ind2] += r_label
        if ind1.shape[0] == 0 and ind2.shape[0] == 0:
            break
    return x

