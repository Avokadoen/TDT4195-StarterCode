import matplotlib.pyplot as plt
import pathlib
import numpy as np
import time
from utils import read_im, save_im, normalize
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def convolve_im(im, kernel):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    assert len(im.shape) == 3

    # Rotate kernel
    cKernel = kernel.copy()
    kLen = len(kernel)
    for y in range(kLen):
        for x in range(kLen):
            cKernel[y][x] = kernel[kLen - y - 1][kLen - x - 1]

    # Will get the rgb value at the give coordinates, if we request index outside of
    # list it will return a zero vector
    def zero_pad(l, y, x):
        # TODO: this is kind of abusing exceptions, use length instead to do this
        try:
            return l[y][x]
        except IndexError:
            return [0, 0, 0]

    # copy image
    imCopy = im.copy()

    # We assume we always perform the convolution in the center of the kernel
    # and that the kernel length is odd
    def convolve(y, x): 
        startY = y - kLen // 2
        startX = x - kLen // 2

        output = [0, 0, 0]
        for i in range(kLen):
            for j in range(kLen):
                im_rgb = zero_pad(im, startY + i, startX + j)
                output[0] += im_rgb[0] * cKernel[i][j]
                output[1] += im_rgb[1] * cKernel[i][j]
                output[2] += im_rgb[2] * cKernel[i][j]

        return output    

    for y in range(len(im)):
        for x in range(len(im[y])):
            imCopy[y][x] = convolve(y, x)

    return imCopy


# Define the convolutional kernels
h_b = 1 / 256 * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Convolve images
im_smoothed = convolve_im(im.copy(), h_b)
save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
im_sobel = convolve_im(im, sobel_x)
save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

# DO NOT CHANGE. Checking that your function returns as expected
assert isinstance(
    im_smoothed, np.ndarray),     f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
assert im_smoothed.shape == im.shape,     f"Expected smoothed im ({im_smoothed.shape}" + \
    f"to have same shape as im ({im.shape})"
assert im_sobel.shape == im.shape,     f"Expected smoothed im ({im_sobel.shape}" + \
    f"to have same shape as im ({im.shape})"


plt.subplot(1, 2, 1)
plt.imshow(normalize(im_smoothed))

plt.subplot(1, 2, 2)
plt.imshow(normalize(im_sobel))
plt.show()
