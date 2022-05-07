import time

from util.helpers import *
from util.optimization import *
from util.sparse import *


def fixROF(label_range_left, label_range_right, num_labels, channels):
    labels = ROFLabels(label_range_left, label_range_right, num_labels, channels)
    return labels, ROFResidual, ROFVisualize, ROFSave, None, None


# returns labels of size (num_labels,label_space_dim), labels_space_dim=channels for ROF
def ROFLabels(label_range_left, label_range_right, num_labels, channels):
    label = torch.linspace(label_range_left, label_range_right, steps=num_labels)
    return label.repeat(channels, 1).transpose(-2, -1)


# f is of size=(1, num_faces * samples, channels)
# gamma is of size=(num_faces * samples, label_space_dim)
# returns residual of size=(num_faces * samples, channels)
def ROFResidual(f, gamma, width, height):
    return f.squeeze(0) - gamma


# input_img is list of pillow images - the command line input (one element list of an image in ROF case)
# result is a torch tensor (denoised image in ROF case)
def ROFVisualize(input_img, denoised):
    input_img[0].show()
    transforms.ToPILImage()(denoised).show()
    return


def ROFSave(input_img, denoised, outputpath, filetype="png"):
    saveImg(outputpath, input_img[0], "input." + filetype)
    saveImg(outputpath, transforms.ToPILImage()(denoised), "denoised." + filetype)
    return
