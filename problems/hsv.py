import colorsys
import importlib
import math

import torch.nn.functional as F

from util.helpers import *
from util.optimization import *
from util.sparse import *

def fixHSV(label_range_left, label_range_right, num_labels, channels):
    labels = HSVLabels(label_range_left, label_range_right, num_labels)
    residual = HSVResidual
    visualize = HSVVisualize
    saveOutput = HSVSave
    return labels, residual, visualize, saveOutput, None, None


def HSVLabels(label_range_left, label_range_right, num_labels):
    label_space_dim = 3
    label = torch.linspace(label_range_left, label_range_right, steps=num_labels)
    return label.repeat(label_space_dim, 1).transpose(-2, -1)

def HSVResidual(f, gamma, width, height):
    res = f.squeeze(0) - gamma
    index = res[:, 0] > 0.5
    res[index, 0] = 1 - res[index, 0]
    return res

# input_imgs is list of pillow images - the command line input (two element list of images in OF case)
# flow is a torch tensor of size=(2,h,w)
def HSVVisualize(input_imgs, flow):
    return


# input_imgs is list of pillow images - the command line input (two element list of images in OF case)
# flow is a torch tensor of size=(2,h,w)
def HSVSave(input_img, result, outputpath, filetype='png'):
    hsv_result = transforms.ToPILImage(mode="HSV")(result)
    rgb_result = hsv_result.convert(mode="RGB")
    saveImg(outputpath, input_img[0], "input." + filetype)
    saveImg(outputpath, rgb_result, "denoised." + filetype)
    return

