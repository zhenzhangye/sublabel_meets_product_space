from problems.hsv import fixHSV
from problems.optical_flow import fixOpticalFlow
from problems.rof import fixROF
from util.helpers import ASSERT


def fixProblem(problem, label_range_left, label_range_right, num_labels, channels):
    if problem == 'hsv':
        fix_problem = fixHSV
    elif problem == 'optical_flow':
        fix_problem = fixOpticalFlow
    elif problem == 'rof':
        fix_problem = fixROF
    else:
        ASSERT(False, "Unknown problem: " + problem)

    return fix_problem(label_range_left, label_range_right, num_labels, channels)

