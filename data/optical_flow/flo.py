import struct

import numpy as np


#  bytes  contents
#
#  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
#          (just a sanity check that floats are represented correctly)
#  4-7     width as an integer
#  8-11    height as an integer
#  12-end  data (width*height*2*4 bytes total)
#          the float values for u and v, interleaved, in row order, i.e.,
#          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
def readFloFormat(filename):
    assert filename.lower().endswith('.flo'), f"File is not a .flo file: {filename}"

    with open(filename, "rb") as f:
        bytes_read = f.read()

    float_check = struct.unpack('f', bytes_read[0:4])[0]
    assert float_check == 202021.25, "Float check failed"

    width = int.from_bytes(bytes_read[4:8], "little")
    height = int.from_bytes(bytes_read[8:12], "little")

    vals = struct.unpack(str(width * height * 2) + 'f', bytes_read[12:])

    return width, height, vals


#  bytes  contents
#
#  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
#          (just a sanity check that floats are represented correctly)
#  4-7     width as an integer
#  8-11    height as an integer
#  12-end  data (width*height*2*4 bytes total)
#          the float values for u and v, interleaved, in row order, i.e.,
#          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
def writeFloFormat(width, height, flow, filename):
    bytes_write = b'PIEH'  # for sanity check

    bytes_write += width.to_bytes(4, "little")
    bytes_write += height.to_bytes(4, "little")

    bytes_write += struct.pack(str(width * height * 2) + 'f', *flow)

    if not filename.lower().endswith('.flo'):
        filename += '.flo'

    f = open(filename, 'wb')
    f.write(bytes_write)
    f.close()
    print("Wrote file: ", filename)

    return


def normalize(vec, axis):
    return vec / np.expand_dims(np.linalg.norm(vec, axis=axis), axis=axis)


# f1 and f2 have size=(num_faces,2)
def averageAngularError(f1, f2):
    assert f1.shape == f2.shape, "Size doesn't match"

    # add third dimension (with 1)
    f1 = np.hstack([f1, np.ones((f1.shape[0], 1))])
    f2 = np.hstack([f2, np.ones((f2.shape[0], 1))])

    # normalize along columns
    f1_u = normalize(f1, axis=1)
    f2_u = normalize(f2, axis=1)

    # calculate dot product
    dot = (f1_u * f2_u).sum(axis=1)

    # calculate angle
    err = np.arccos(np.clip(dot, -1.0, 1.0))

    # get mean
    return np.mean(err)


# f1 and f2 have size=(num_faces,2)
def averageEndpointError(f1, f2):
    assert f1.shape == f2.shape, "Size doesn't match"

    diff = f1 - f2
    err = np.sqrt((diff * diff).sum(axis=1))

    return np.mean(err)

# if __name__ == '__main__':
#     filename_in = '/home/haefner/code/lifting_ups/data/middlebury/flow/other-gt-flow/Grove3/flow10.flo'
#     filename_out = '/home/haefner/code/lifting_ups/data/middlebury/flow/other-gt-flow/Grove3/flow10_out.flo'
#
#     width, height, vals = readFloFormat(filename_in)
#     writeFloFormat(width, height, vals, filename_out)
#     width_, height_, vals_ = readFloFormat(filename_out)
#
#     print("Width: ", width)
#     print("Height: ", height)
#     print("Width_: ", width_)
#     print("Height_: ", height_)
#
#     result = np.array_equal(np.array(vals), np.array(vals_))
#
#     print("Input == Output: ", result)
