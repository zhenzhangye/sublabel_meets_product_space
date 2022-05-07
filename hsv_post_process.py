import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tikzplotlib

row_id = 59

png_file = "outputs/outhsv/denoised.png"
image = Image.open(png_file)
image_array = np.copy(np.array(image))[:, :, 0] / 255.0
width, height = image.size
draw = ImageDraw.Draw(image)
draw.line((0, row_id, width, row_id), fill=(0, 0, 255, 128), width=1)
image.save("outputs/outhsv/denoised_withline.png")

image = Image.open(png_file).convert(mode="HSV")
image_array = np.copy(np.array(image))[:, :, 0] / 255.0

for row in range(row_id, row_id+1):
    array1d = image_array[row, :]
    for i in range(1, array1d.shape[0]):
        if np.abs(array1d[i] - array1d[i-1]) <= 0.5:
            plt.plot(np.array([i, i+1]), array1d[i-1:i+1], color='b')
        else:
            if array1d[i] > array1d[i-1]:
                plt.plot(np.array([i-1, i]), np.array([array1d[i-1], 0]), color='b')
                plt.plot(np.array([i-1, i]), np.array([1, array1d[i]]), color='b')
            else:
                plt.plot(np.array([i-1, i]), np.array([array1d[i-1], 1]), color='b')
                plt.plot(np.array([i-1, i]), np.array([0, array1d[i]]), color='b')
    plt.savefig("outputs/outhsv/denoised_hue.png")
    tikzplotlib.save("outputs/outhsv/denoise_hue.tex")
plt.clf()

png_file = "data/hsv/input2.png"
image = Image.open(png_file)
image_array = np.copy(np.array(image))[:, :, 0] / 255.0
width, height = image.size
draw = ImageDraw.Draw(image)
draw.line((0, row_id, width, row_id), fill=(0, 0, 255, 128), width=1)
image.save("outputs/outhsv/input_withline.png")

image = Image.open(png_file).convert(mode="HSV")
image_array = np.copy(np.array(image))[:, :, 0] / 255.0
#print(image_array[row, 110:140])
#raise E

for row in range(row_id, row_id+1):
    array1d = image_array[row, :]
    for i in range(1, array1d.shape[0]):
        if np.abs(array1d[i] - array1d[i-1]) <= 0.5:
            plt.plot(np.array([i, i+1]), array1d[i-1:i+1], color='b')
        else:
            if array1d[i] > array1d[i-1]:
                plt.plot(np.array([i-1, i]), np.array([array1d[i-1], 0]), color='b')
                plt.plot(np.array([i-1, i]), np.array([1, array1d[i]]), color='b')
            else:
                plt.plot(np.array([i-1, i]), np.array([array1d[i-1], 1]), color='b')
                plt.plot(np.array([i-1, i]), np.array([0, array1d[i]]), color='b')
    plt.savefig("outputs/outhsv/input_hue.png")
    tikzplotlib.save("outputs/outhsv/input_hue.tex")
