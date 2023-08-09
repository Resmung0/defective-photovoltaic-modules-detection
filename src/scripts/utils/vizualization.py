"""File to define visualizations about pieces of code, like Masks and 
Thermograms.
"""

import matplotlib.pyplot as plt
import numpy as np
from flyr import FlyrThermogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import color
from skimage.segmentation import find_boundaries

from scripts.image_processing.mask import Mask


def show_thermal_image(
    thermogram: FlyrThermogram,
    figure_size: tuple[int, int] = (10, 5),
    title: str = "Thermography image",
    **kwargs
) -> None:
    """Vizualization of thermographic image.

    Args:
        thermogram (FlyrThermogram): Thermogram mask.
        figure_size (tuple[int, int]): Size of the figure showed. Defaults to (10, 5).
        title (str): Image title. Defaults to "Thermography image".
        **kwargs: Thermogram related arguments.
    """
    image = thermogram.render_pil(**kwargs)

    mask, palette = None, "gray"
    if isinstance(kwargs.get("palette"), str):
        palette = kwargs["palette"]
    if isinstance(kwargs.get("mask"), np.ndarray):
        mask = kwargs["mask"]

        assert_message = """
        There is a mask, but no palette was specified! Please choose a palette other than grayscale.
        """
        assert palette != "gray", assert_message

    unique_values = np.unique(np.ma.masked_array(np.array(image)[:, :, 0], mask=mask))

    _, axe = plt.subplots(1, 1, figsize=figure_size)
    image = axe.imshow(image, cmap=palette)
    cax = make_axes_locatable(axe).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=cax, values=unique_values, label="Temperature (ºC)")
    axe.set_title(title)
    axe.set_axis_off()
    plt.show()


def show_label_mask(mask: Mask, image: np.ndarray):
    """Vizualization of label mask.

    Args:
        mask (Mask): Thermogram mask.
        image (np.ndarray): Grayscale image that will be transformed.
    """
    label_mask = mask.label_mask

    masked_image = color.label2rgb(label_mask, bg_label=0, image=image)
    masked_image[find_boundaries(label_mask) is True] = (0, 0, 0)

    _, axes = plt.subplots(1, 3, figsize=(20, 20))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].set_axis_off()

    axes[1].imshow(label_mask, cmap="viridis")
    axes[1].set_title("Label Mask")
    axes[1].set_axis_off()

    axes[2].imshow(masked_image)
    axes[2].set_title("Masked Image")
    axes[2].set_axis_off()

    plt.tight_layout()
    plt.show()


def inspect_transformed_planes(
    points: dict[str, np.ndarray[int, int]],
    image: np.ndarray,
    figure_size: tuple[int, int] = (10, 5),
    color_map: str = "gray",
):
    """Vizualization of transformed planes by Perspective transformation.

    Args:
        points (dict[str, np.ndarray[int, int]]): Dictionary storing "source" and "destination"
        points.
        image (np.ndarray): Grayscale image that will be transformed.
        figure_size (tuple[int, int], optional): Size of the figure showed. Defaults
        to (10, 5).
        color_map (str, optional): Color map to colorize the image. Defaults to "gray".
    """
    _, axis = plt.subplots(1, 2, figsize=figure_size)
    for index in range(2):
        if index == 0:
            points = points["source"]
            show_image = image
            title = "Area of interest"
        else:
            points = points["destination"]
            show_image = np.zeros_like(image)
            title = "Area of projection"
        x_show = [val[0] for val in points] + [points[0][0]]
        y_show = [val[1] for val in points] + [points[0][1]]

        axis[index].imshow(show_image, cmap=color_map)
        axis[index].plot(x_show, y_show, "r--")
        axis[index].set_title(title)
        axis[index].set_axis_off()
    plt.tight_layout()
    plt.show()
