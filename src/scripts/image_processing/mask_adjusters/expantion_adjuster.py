"""File to define thermogram masks adjusters."""

import numpy as np
from skimage.morphology import binary_closing
from skimage.segmentation import expand_labels
from skimage.util import img_as_ubyte

from scripts.image_processing.mask import Mask

from .interface import Adjuster


class ExpantionMaskAdjuster(Adjuster):
    """Adjust the mask by expanding each instance domain."""

    def __init__(self, distance: int = 7) -> None:
        self.distance = distance

    def __call__(self, image: Mask) -> Mask:
        """Method that expand each instance domain in the mask.

        Args:
            image (Mask): Mask that needs to be adjusted.

        Returns:
            Mask: Adjusted Mask.
        """
        # get binary mask from original label mask
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = img_as_ubyte(binary_closing(image.binary_mask, kernel))

        # Expanding each instance domain from label mask
        label_mask = expand_labels(image.label_mask, distance=self.distance)

        # convert expanded label mask to binary
        binary_expanded_mask = label_mask.astype("bool").copy()

        # Subtract binary mask from expanded binary one
        lines = binary_expanded_mask - binary_mask

        # Erase overlapping lines
        label_mask[lines == 1] = 0
        return Mask.from_label_mask(label_mask)

    def __repr__(self) -> str:
        """Display the transformation params.

        Returns:
            str: Params.
        """
        return f"Parameters: \tdistance -> {self.distance}"
