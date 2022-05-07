import colorsys
import importlib

import torch.nn.functional as F

from util.helpers import *
from util.optimization import *
from util.sparse import *

flo = importlib.import_module("data.optical_flow.flo")


def fixOpticalFlow(label_range_left, label_range_right, num_labels, channels):
    labels = opticalFlowLabels(label_range_left, label_range_right, num_labels)
    return labels, opticalFlowResidual, opticalFlowVisualize, opticalFlowSave, opticalFlowEvaluate, None


def opticalFlowLabels(label_range_left, label_range_right, num_labels):
    label_space_dim = 2
    # labels = torch.zeros((num_labels, label_space_dim), dtype=torch.float)
    # len_lrl = len(label_range_left)
    # len_lrr = len(label_range_right)
    # for d in range(label_space_dim):
    #     labels[:, d] = torch.linspace(label_range_left[min(d, len_lrl - 1)],
    #                                   label_range_right[min(d, len_lrr - 1)],
    #                                   steps=num_labels)
    label = torch.linspace(label_range_left, label_range_right, steps=num_labels)
    return label.repeat(label_space_dim, 1).transpose(-2, -1)


# This function evaluates f_2(x + (gamma_1,gamma_2)):
# x + (gamma_1,gamma_2) is a float, so we have to interpolate f_2 at x + (gamma_1,gamma_2)
# f2 is of size=(num_faces * label_samples, channels)
# gamma is of size=(num_faces * label_samples, label_space_dim)
def interpLayeredImage(f2, gamma, width, height, mode="bilinear"):
    # f2 = [x1,x2,x3,....,xN, x1,x2,x3,...xN,.....]
    # gamma_1 = [x1,x2,x3,....,xN, x1,x2,x3,...xN,.....] shift in x-direction
    # gamma_2 = [x1,x2,x3,....,xN, x1,x2,x3,...xN,.....] shift in y-direction
    num_faces = numFaces(height, width)
    label_samples = int(gamma.shape[0] / num_faces)

    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))

    x_vec = img2vec(grid_x).repeat(1, label_samples)
    y_vec = img2vec(grid_y).repeat(1, label_samples)

    # size=(1, num_faces, label_samples)
    gamma1 = vec2img(gamma[:, 0], label_samples, num_faces)
    gamma2 = vec2img(gamma[:, 1], label_samples, num_faces)

    # size=(1, num_faces, label_samples)
    x_pos = torch.clamp(x_vec + gamma1, min=0, max=width - 1)
    y_pos = torch.clamp(y_vec + gamma2, min=0, max=height - 1)

    # size=(channels, num_faces, label_samples)
    f2_ = vec2img(f2, label_samples, num_faces)
    interped_f2 = torch.zeros(f2_.shape)
    for ls in range(label_samples):
        f2_img = vec2img(f2_[:, :, ls].transpose(0, 1), width, height).unsqueeze(
            0)  # size=(1,c,h,w)
        x_img = vec2img(x_pos[:, :, ls].transpose(0, 1), width, height)  # size=(1,h,w)
        y_img = vec2img(y_pos[:, :, ls].transpose(0, 1), width, height)  # size=(1,h,w)
        x_img_normalized = 2 * x_img / (width - 1) - 1  # get grid to range [-1,1]
        y_img_normalized = 2 * y_img / (height - 1) - 1  # get grid to range [-1,1]
        grid = torch.stack([x_img_normalized, y_img_normalized], dim=-1)  # size=(1,h,w,2)
        f2_img_interp = torch.nn.functional.grid_sample(f2_img, grid, mode=mode,
                                                        align_corners=True)  # size=(1,c,h,w)

        interped_f2[:, :, ls] = img2vec(f2_img_interp.squeeze(0)).transpose(0, 1)

    return img2vec(interped_f2)


# f is of size=(2, num_faces * samples, channels)
# gamma is of size=(num_faces * samples, label_space_dim)
# returns residual of size=(num_faces * samples, channels)
def opticalFlowResidual(f, gamma, width, height):
    return f[0, :, :] - interpLayeredImage(f[1, :, :], gamma, width, height)


def makeColorWheelImage(width, height):
    radius = 30.0
    center_x, center_y = math.floor(width / 2), math.floor(height / 2)
    colorwheel = torch.zeros(3, width, height)
    n = 10

    for x in range(width):
        for y in range(height):
            rx = x - center_x
            ry = y - center_y
            s = math.sqrt(rx * rx + ry * ry) / radius
            #if s <= 1:
            h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0
            for j in range(n):
                if h >= j/n and h<(j+1)/n:
                    h = j/n
            if s <=0.5:
                h = 0
                rgb = colorsys.hsv_to_rgb(h, 0, 1)
            else:
                rgb = colorsys.hsv_to_rgb(h, 1, 1)
            colorwheel[:, y, x] = torch.tensor(rgb)

    return colorwheel


# flow is of size=(2,h,w)
def flowNormalize(flow):
    # calculate max radius of flow: max(sqrt(flow_x * flow_x + flow_y * flow_y))
    radius_max = torch.max(torch.norm(flow, dim=0))

    flow_normalized = flow.detach().clone()
    if radius_max != 0:
        flow_normalized = flow_normalized / radius_max

    return flow_normalized


def flowToColor(flow, cwheel):
    # normalize flow to unit circle
    flow_normalized = flowNormalize(flow)

    # interpolate color wheel at flow positions
    indices4d = flow_normalized.permute(1, 2, 0).reshape(1, flow_normalized.shape[-2],
                                                         flow_normalized.shape[-1], 2)
    cwheel4d = cwheel.unsqueeze(0)
    flowColor = F.grid_sample(cwheel4d, indices4d, align_corners=True).squeeze(0)

    return flowColor


def applyFlowToImage(img, flow, width, height):
    return vec2img(interpLayeredImage(img2vec(img), img2vec(flow), width, height), width, height)


# input_imgs is list of pillow images - the command line input (two element list of images in OF case)
# flow is a torch tensor of size=(2,h,w)
def opticalFlowVisualize(input_imgs, flow):
    width, height = input_imgs[1].size
    input_imgs[0].show()  # input image 1
    input_imgs[1].show()  # input image 2

    cwheel = makeColorWheelImage(257, 257)
    transforms.ToPILImage()(flowToColor(flow, cwheel)).show()  # visualize flow
    transforms.ToPILImage()(flowToColorMiddlebury(flow)).show()  # visualize flow middlebury
    transforms.ToPILImage()(cwheel).show()  # visualize colorwheel

    f2_interp = applyFlowToImage(transforms.ToTensor()(input_imgs[1]), flow, width, height)
    transforms.ToPILImage()(f2_interp).show()
    return


# input_imgs is list of pillow images - the command line input (two element list of images in OF case)
# flow is a torch tensor of size=(2,h,w)
def opticalFlowSave(input_imgs, flow, outputpath, filetype="png"):
    width, height = input_imgs[1].size

    saveImg(outputpath, input_imgs[0], "input1." + filetype)
    saveImg(outputpath, input_imgs[1], "input2." + filetype)

    # cwheel = makeColorWheelImage(257, 257)
    # saveImg(outputpath, transforms.ToPILImage()(flowToColor(flow, cwheel)), "flow." + filetype)
    # saveImg(outputpath, transforms.ToPILImage()(flowToColorMiddlebury(flow)),
    #         "flow_middlebury." + filetype)
    # saveImg(outputpath, transforms.ToPILImage()(cwheel), "colorwheel." + filetype)

    f2_interp = applyFlowToImage(transforms.ToTensor()(input_imgs[1]).to(flow.device), flow, width,
                                 height)
    saveImg(outputpath, transforms.ToPILImage()(f2_interp), "img_flowed." + filetype)

    saveFloFileMiddlebury(flow, outputpath + "/flow.flo")

    # flow640x480_nearest = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(480, 640),
    #                                                       mode='nearest').squeeze()
    # saveFloFileMiddlebury(flow640x480_nearest, outputpath + "/flow640x480_nearest.flo")

    flow640x480_bilinear = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(480, 640),
                                                           mode='bilinear',
                                                           align_corners=True).squeeze()
    saveFloFileMiddlebury(flow640x480_bilinear, outputpath + "/flow640x480_bilinear.flo")
    return


def flowImageToFlo(flow):
    return flow.permute(1, 2, 0).reshape(-1)


def floToFlowImage(flo, height, width):
    return flo.reshape(height, width, 2).permute(2, 0, 1)


def saveFloFileMiddlebury(flow, filename):
    c, h, w = flow.shape
    flow_flo_format = flowImageToFlo(flow)
    flo.writeFloFormat(w, h, flow_flo_format.tolist(), filename)
    return


def makeColorwheelMiddlebury():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = torch.zeros([ncols, 3])  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    colorwheel[col:YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    col += GC

    # CB
    colorwheel[col:CB + col, 1] = 255 - torch.floor(255 * torch.arange(0, CB) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col:BM + col, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    colorwheel[col:BM + col, 2] = 255
    col += BM

    # MR
    colorwheel[col:MR + col, 0] = 255
    colorwheel[col:MR + col, 2] = 255 - torch.floor(255 * torch.arange(0, MR) / MR)
    return colorwheel


def computeColorMiddlebury(flow):
    # normalized flow in x-direction (fx) and y-direction (fy)
    fx = flow[0, :, :]
    fy = flow[1, :, :]
    colorwheel = makeColorwheelMiddlebury()

    ncols = colorwheel.shape[0]
    radius = torch.sqrt(fx * fx + fy * fy)
    a = torch.atan2(fx, fy) / math.pi
    fk = (a.reshape(-1) + 1) / 2 * (ncols - 1)  # [-1, 1] mapped to [0, ncols-1]
    k0 = fk.type(torch.long)  # 0, 1, ..., ncols - 1
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = torch.zeros([3, flow.shape[-2], flow.shape[-1]], dtype=torch.uint8)
    nchannels = colorwheel.shape[1]
    for cc in range(nchannels):
        tmp = colorwheel[:, cc]
        col0 = tmp.index_select(0, k0) / 255
        col1 = tmp.index_select(0, k1) / 255
        col = ((1 - f) * col0 + f * col1).reshape(fx.shape)
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[2 - cc, :, :] = (255 * col).type(torch.uint8)

    return img


def flowToColorMiddlebury(flow):
    # normalize flow to unit circle
    flow_normalized = flowNormalize(flow)
    flowColor = computeColorMiddlebury(flow_normalized)

    return flowColor


def opticalFlowEvaluate(ground_truth, outputpath):
    w_gt, h_gt, f_gt = flo.readFloFormat(ground_truth)
    w_ours, h_ours, f_ours = flo.readFloFormat(outputpath + "/flow.flo")

    f_gt_img = floToFlowImage(torch.tensor(list(f_gt)), h_gt, w_gt)
    f_ours_img = floToFlowImage(torch.tensor(list(f_ours)), h_ours, w_ours)

    if w_gt != w_ours or h_gt != h_ours:
        f_ours_img = torch.nn.functional.interpolate(f_ours_img.unsqueeze(0), size=(h_gt, w_gt),
                                                     mode='bilinear',
                                                     align_corners=True).squeeze()

    f_gt_numpy = img2vec(f_gt_img).cpu().numpy()
    f_ours_numpy = img2vec(f_ours_img).cpu().numpy()

    aep = flo.averageEndpointError(f_gt_numpy, f_ours_numpy)
    aae = flo.averageAngularError(f_gt_numpy, f_ours_numpy)

    print("Average Endpoint error (AEP): ", aep)
    print("Average Angular error (AAE): ", aae)
    return

if __name__=="__main__":
    color = makeColorWheelImage(128, 128)
    img = transforms.ToPILImage()(color)  # visualize flow
    saveImg("./", img, "input." + "png")
    raise E
