import matplotlib.pyplot as plt
import pathlib
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)

def mutate_image(im, callback):
    """ Iteratates each pixel and reassigns pixel using callback function

    Args: 
        im
        callback: A function that takes in a pixel and returns any
    """
    
    for y in range(len(im)):
        for x in range(len(im[y])):
            im[y][x] = callback(im[y][x])

def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """

    grey_callback = lambda p : 0.212 * p[0] + 0.7152 *  p[1] + 0.0722 * p[2]
    mutate_image(im, grey_callback)

    return im


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """

    inverse_callback = lambda p : 1 - p
    mutate_image(im, inverse_callback)

    return im

im_inverse = inverse(im_greyscale)
save_im(output_dir.joinpath("lake_inverse.jpg"), im_inverse, cmap="gray")
plt.imshow(im_inverse, cmap="gray")
