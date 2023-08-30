"""File that defines functions that read different kind of data."""

from json import load
from pathlib import Path

import numpy as np
from flyr import FlyrThermogram, unpack
from skimage import draw

from .thermograms import TIFFThermogram


def read_thermogram(
    file_path: str, tiff_info: dict[str, str] | None = None
) -> FlyrThermogram | TIFFThermogram:
    """Read thermogram data from RJPG and TIFF files.

    Args:
        file_path (str): Thermogram image file path.
        tiff_info (dict[str, str] | None): Informations needed to read tiff thermograms,
        like metadata and optical image paths. Default to None.

    Returns:
        FlyrThermogram | TIFFThermogram: Thermogram data.
    """
    file_format = Path(file_path).suffix
    if file_format in (".jpg", ".JPG"):
        thermogram = unpack(file_path)
    elif file_format in (".tif", ".tiff"):
        if tiff_info is None:
            tiff_info = {
                "metadata_path": None,
                "optical_path": None,
                "method": "thermomap",
            }
        thermogram = TIFFThermogram(file_path, **tiff_info)
    return thermogram


def read_annotation(
    file_path: str, shape: tuple[int, int] = (512, 640)
) -> tuple[np.ndarray[int], np.ndarray[dict[str, float]], np.ndarray[str]]:
    """Read data from JSON annotation file.

    Args:
        file_path (str): JSON file path
        shape (tuple[int, int], optional): Dimensions of corresponding image mask.
        Defaults to (512, 640).

    Returns:
        tuple[np.ndarray[int], np.ndarray[dict[str, float]], np.ndarray[str]]: Label
        mask, centers of each label and corresponding PV module class.
    """

    #  Get data from annotation JSON file
    with open(file_path, "r", encoding="utf-8") as annotation_file:
        annotation_data = load(annotation_file)

    masks, centers, classes = [], [], []
    for instance in annotation_data["instances"]:
        # Store PV modules classes
        roi_class = "defected" if instance["defected_module"] else "non-defected"
        classes.append(roi_class)

        # Store mask center
        centers.append(instance["center"])

        # Create and store the label mask
        polygon = np.array(
            [[coords["y"], coords["x"]] for coords in instance["corners"]]
        )
        masks.append(draw.polygon2mask(shape, polygon))

    return tuple(map(np.array, [masks, centers, classes]))
