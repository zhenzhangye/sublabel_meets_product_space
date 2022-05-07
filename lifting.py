import argparse
import datetime
import os
import sys
from pathlib import Path
from subprocess import call

import torch
from PIL import Image
from numpy import random
from torchvision import transforms
import matplotlib.pyplot as plt

from problems.problems import fixProblem
from util import stochastic
from util.helpers import pillowImage2TorchArray, ASSERT, saveImg


def fieldValueToString(val, acc):
    if type(val) == list:
        s = ""
        for v in val:
            s += fieldValueToString(v, acc) + "-"
        return s[:-1]  # remove unnecessary last hyphen
    elif type(val) == str:
        return val
    elif type(val) == int:
        return str(val)
    elif type(val) == bool:
        return str(val)
    elif type(val) == float:
        return format(val, acc)
    else:
        ASSERT(False, "Unknown type: " + str(type(val)))


def cliParse():
    parser = argparse.ArgumentParser(prog='StochasticLifting',
                                     description='Tool to solve the variational problems with lifting using stochastic sampling')
    parser.add_argument('-i', '--images', default=['data/debug/24004.jpg'], dest='images',
                        nargs="+", help='Image file(s)')
    parser.add_argument('-o', '--output', default="", dest='outputpath',
                        help='Path where output is being saved')
    parser.add_argument('--gt', default="", dest='ground_truth',
                        help="Calculate problem specific error metrics between result and parsed ground truth. Only if output (path) is provided as data has to be saved.")
    parser.add_argument('--folder_name', default="problem", dest='folder_name',
                        choices=['timestamp', 'problem'],
                        help='Saves output folder name with timestamp or with detailed CL input name given with --fields')

    parser.add_argument('-l', '--lambda', default=0.1, dest='lambda_', type=float,
                        help='Trade-off parameter')
    parser.add_argument('-m', '--method', default='stochastic', dest='method',
                        choices=['stochastic', 'goldluecke2013'],
                        help='Which method to use: stochastic (proposed) or goldluecke2013, see https://epubs.siam.org/doi/abs/10.1137/120862351')
    parser.add_argument('--loss', default='sqL2', dest='loss_key',
                        choices=['L1', 'trunc_L1', 'sqL2', 'trunc_sqL2'],
                        help='Which loss function to choose for the data term')
    parser.add_argument('--loss_params', default=[0.025], dest='loss_params', nargs="*", type=float,
                        help='Set of parameters for the loss function')
    parser.add_argument('-s', '--sampling', default=['random', 'exhaustive'], dest='sampling',
                        choices=['exhaustive', 'meank', 'random', 'violation'],
                        nargs="*", help='Which (combination of) sampling strategies to use')
    parser.add_argument('--meank', default=1, dest='meank', type=int,
                        help='Use meank gaussian sampled costs with mean at the current solution and variance meank_variance. Used only if meank sampling is used. (Even if value is zero at least one sample at the current solution will be used)')
    parser.add_argument('--meank_variance', default=0.1, dest='meank_variance', type=float,
                        help='Variance of Gaussian of meank sampling. Used only if meank sampling is used.')
    parser.add_argument('--mostviok', default=0, dest='mostviok', type=int,
                        help='Use the most violated k constraints for sampling. Use only if violation sampling is used.')
    parser.add_argument('--mostviok_pixel', default=False, dest='mostviok_pixel',
                        action='store_true',
                        help='mostviok per pixel (across all samples otherwise). Use only if violation sampling is used.')
    parser.add_argument('--one_sample_mean', default=True, dest='one_sample_mean',
                        action='store_true',
                        help='one sample at the mean of current u per pixel')
    parser.add_argument('-p', '--problem', default='rof', dest='problem',
                        choices=['rof', 'optical_flow', 'hsv'],
                        help='Which problem to solve')
    parser.add_argument('--max_iter_inner', default=10, dest='max_iter_inner', type=int,
                        help='Maximum number of inner iterations')
    parser.add_argument('--max_iter_outer', default=10, dest='max_iter_outer', type=int,
                        help="Maximum number of iterations of outer/sampling loop. Only used if 'stochastic_{pd,pg}' method is used")
    parser.add_argument('--labels', default=5, dest='num_labels', type=int,
                        help='Number of labels')
    parser.add_argument('--label_samples', default=10, dest='label_samples', type=int,
                        help="Number of samples per label.")
    parser.add_argument('--lrl', default=0, dest='label_range_left', type=float,
                        help='Label range left boundary')
    parser.add_argument('--lrr', default=1, dest='label_range_right', type=float,
                        help="Label range right boundary")
    parser.add_argument('--width', default=0, dest='width', type=int,
                        help="Resize image to given width")
    parser.add_argument('--height', default=0, dest='height', type=int,
                        help="Resize image to given height")

    parser.add_argument('--pruning', default=False, dest='pruning', action='store_true',
                        help="Use sample pruning (remove inactive constraints)")

    parser.add_argument('--gpu', default=False, dest='gpu', action='store_true',
                        help='Use the GPU during computation')
    parser.add_argument('--gpu_id', default=0, dest='gpu_id',
                        help="Which GPU to use (nvidia-smi). '--gpu' flag has to be set for this option to be valid")
    parser.add_argument("--random_seed", default=1989, dest='random_seed', type=int,
                        help='Set the random seed')

    cli_ = parser.parse_args()

    return cli_

def setDevice(gpu, gpu_id):
    if torch.cuda.is_available() and gpu:
        device = "cuda"
        torch.cuda.set_device(int(gpu_id))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"

    return torch.device(device)

def loadPillowImages(image_paths, w, h):
    images = [None] * len(image_paths)
    for ii in range(len(image_paths)):
        print("Load image: " + image_paths[ii])
        img = Image.open(image_paths[ii])

        if w > 0 and h > 0:
            img = img.resize((w, h))

        images[ii] = img

    # check if image sizes match
    for ii in range(len(images)):
        ASSERT(images[ii].size == images[0].size and images[ii].mode == images[0].mode,
               "Input images have to have same size and mode: {si} != {s0} or {mi} != {m0}".format(
                   si=images[ii].size, s0=images[0].size, mi=images[ii].mode, m0=images[0].mode))
        print("{m} image {i} size (wxh): {s}".format(m=images[ii].mode, i=ii, s=images[ii].size))

    return images


def runMethod(cli, img_vec, height, width, labels, residual):
    if cli.method == 'stochastic':
        (result, energy_val, runtime, inner_iter, outer_iter,
         memory, constraints) = stochastic.stochastic(
            img_vec,
            height,
            width,
            labels,
            residual,
            cli.lambda_,
            cli.max_iter_inner,
            cli.max_iter_outer,
            cli.label_samples,
            cli.sampling,
            cli.meank,
            cli.meank_variance,
            cli.mostviok, cli.mostviok_pixel,
            cli.one_sample_mean,
            cli.pruning,
            cli.loss_key,
            cli.loss_params,
            cli.problem)

    elif cli.method == 'goldluecke2013':
        (result, energy_val, runtime, inner_iter, outer_iter,
         memory, constraints) = stochastic.stochastic(
            img_vec,
            height,
            width,
            labels,
            residual,
            cli.lambda_,
            cli.max_iter_inner,
            1,
            0,
            ["exhaustive"],
            0, 0,
            0, False,
            False, False,
            cli.loss_key,
            cli.loss_params,
            cli.problem)
    else:
        ASSERT(False, "Unsupported method")

    return result, energy_val, runtime, inner_iter, outer_iter, memory, constraints


def saveData(cli, logname, pil_imgs, result, constraints, energy_val, runtime, inner_iter,
        outer_iter, memory, saveOutput, evaluate):
    # create output folder name
    path2outputdir = cli.outputpath + cli.problem
    if cli.folder_name == "timestamp":
        path2outputdir += "_" + logname

    # mkdir if it doesn't exist
    Path(path2outputdir).mkdir(parents=True, exist_ok=True)
    saveOutput(pil_imgs, result, path2outputdir)

    sys.stdout = open(path2outputdir + "/namespace.txt", "w")
    for k in cli.__dict__:
        print(k, end=": ", flush=False)
        print(cli.__dict__[k])
    sys.stdout.close()

    # save numbers
    sys.stdout = open(path2outputdir + "/result.txt", "w")
    print("Energy: ", energy_val)
    print("Runtime: ", runtime)
    print("Inner iterations: ", inner_iter)
    print("Outer iterations: ", outer_iter)
    print("Memory: ", memory)
    if cli.ground_truth != "" and evaluate is not None:
        evaluate(cli.ground_truth, path2outputdir)
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    return


def main():
    logname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    cli = cliParse()
    torch.manual_seed(cli.random_seed)  # set seed to always have same random numbers
    torch.cuda.manual_seed_all(cli.random_seed)  # set seed to always have same random numbers

    dev = setDevice(cli.gpu, cli.gpu_id)

    pil_imgs = loadPillowImages(cli.images, cli.width, cli.height)

    # preprocess image
    img_vec, num_images, channels, height, width = pillowImage2TorchArray(pil_imgs, dev, cli.problem)

    # fix the chosen problem (get correct labels, cost function, recover function and energy function)
    labels, residual, visualize, saveOutput, evaluate, unliftedEnergy = fixProblem(cli.problem,
                                                                                   cli.label_range_left,
                                                                                   cli.label_range_right,
                                                                                   cli.num_labels,
                                                                                   channels)

    print("------------------------------------------------------------")
    (result, energy_val, runtime, inner_iter,
     outer_iter, memory, constraints) = runMethod(cli,
                                                  img_vec,
                                                  height,
                                                  width,
                                                  labels,
                                                  residual)

    print()
    print("Energy (Lifted {}): {:.2f}".format(cli.problem, energy_val))
    print("Time (Lifted {}): {:.2f}".format(cli.problem, runtime))
    print("Memory (Lifted {}): {}".format(cli.problem, memory))

    # save
    if cli.outputpath != "":
        saveData(cli, logname, pil_imgs, result, constraints, energy_val, runtime, inner_iter,
                 outer_iter, memory, saveOutput, evaluate)

    print("-------------------------END-------------------------------")

    return


if __name__ == '__main__':
    main()

