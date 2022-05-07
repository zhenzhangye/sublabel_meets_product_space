import getpass
import sys

script_descriptor = open("lifting.py")
a_script = script_descriptor.read()

outputpath = "./outputs/out"

problem = "hsv"
#problem = "optical_flow"
#problem = "rof"

images = ["data/hsv/input2.png"]
#images = ["data/optical_flow/frame10.png",
#          "data/optical_flow/frame11.png"]
#images = ["data/rof/lenna_64_gray_noise.jpg"]

if problem == "hsv":
    width = 0
    height = 0
    lambda_ = 0.05
    lrr = 1
    lrl = 0
    loss_func = "trunc_sqL2"
    loss_params = [0.015]
elif problem == "optical_flow":
    scaling_factor = 2 
    width = int(640 / scaling_factor)
    height = int(480 / scaling_factor)
    gt = "data/optical_flow/flow10.flo"
    lambda_ = 0.04
    lrr = 7.5
    lrl = -2.5
    loss_func = "L1"
elif problem == "rof":
    width = 0
    height = 0
    lambda_ = 0.25
    lrr = 1
    lrl = 0
    loss_func = "trunc_sqL2"
    loss_params = [0.025]

# base_args are applicable for goldluecke2013 and our method
base_args = ["lifting.py",
             "--gpu", "--gpu_id", "0",
             "--problem", problem,
             "-o", outputpath,
             "--width", str(width),
             "--height", str(height),
             "--lambda", str(lambda_),
             "--lrr", str(lrr),
             "--lrl", str(lrl),
             "--loss", loss_func]
base_args += ["-i"] + images

if 'loss_params' in locals():
    base_args += ["--loss_params"] + [str(p) for p in loss_params]

if 'gt' in locals():
    base_args += ["--gt", gt]

# define layout of our algorithm
proposed = base_args + ["--method", "stochastic"]

labels = 7
max_iter_inner = 30000
max_iter_outer = 1
label_samples = 30
meank = 30
meank_variance = 0.1
mostviok = 5
mostviok_pixel = True
pruning = True

# define our sampling strategy
proposed += ["--sampling", "violation", "random", "meank",
             "--labels", str(labels),
             "--max_iter_inner", str(max_iter_inner),
             "--max_iter_outer", str(max_iter_outer),
             "--label_samples", str(label_samples),
             "--meank", str(meank),
             "--meank_variance", str(meank_variance),
             "--mostviok", str(mostviok)]

if mostviok_pixel:
    proposed += ["--mostviok_pixel"]
if pruning:
    proposed += ["--pruning"]

sys.argv = proposed
exec(a_script)

script_descriptor.close()
