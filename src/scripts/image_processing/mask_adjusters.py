"""File that implements the Adjuster interface."""

from abc import ABC, abstractmethod

import numpy as np
from skimage.morphology import binary_closing
from skimage.segmentation import expand_labels
from skimage.util import img_as_ubyte

from .mask import Mask


class MaskAdjuster(ABC):
    """Interface class that is responsible for change the thermogram
    mask in some kind way."""

    @abstractmethod
    def __call__(self, mask: Mask) -> Mask:
        """Method that modifies the mask in some way.

        Args:
            mask (Mask): Thermogram mask that needs to be adjusted.

        Returns:
            Mask: Adjusted mask.
        """
        return NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        """Display the transformation params."""
        return ""


class OverlapMaskAdjuster(MaskAdjuster):
    """Adjust the mask by eliminating overlap between instances.
    Based on https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/279995.
    """

    def __init__(self) -> None:
        self.have_overlap = False

    def __check_overlap(self, mask: Mask) -> bool:
        """Check if overlap exist in the mask.

        Args:
            mask (Mask): Thermogram mask.

        Returns:
            bool: Confirmation if overlap exists.
        """
        mask = mask.data.astype("bool").astype("uint8")
        return np.any(np.sum(mask, axis=-1) > 1)

    def __fix_overlap(self, mask: Mask) -> Mask:
        """Correct the overlapped mask.

        Args:
            mask (Mask): Thermogram mask.

        Returns:
            Mask: Corrected thermogram mask.
        """
        # Change instance index
        new_label_mask = np.stack(mask.data, axis=-1)

        # Correct overlap
        new_label_mask = np.pad(new_label_mask, [[0, 0], [0, 0], [1, 0]])
        new_label_mask = np.argmax(new_label_mask, axis=-1)
        new_label_mask = np.eye(new_label_mask.shape[-1], dtype="uint8")[new_label_mask]
        new_label_mask = new_label_mask[..., 1:]
        new_label_mask = new_label_mask[..., np.any(new_label_mask, axis=(0, 1))]

        # Bring back instance index
        new_label_mask = [
            new_label_mask[:, :, i] for i in range(new_label_mask.shape[-1])
        ]
        new_label_mask = np.array(new_label_mask, dtype="bool")
        return Mask(new_label_mask)

    def __call__(self, mask: Mask) -> Mask:
        """Method that remove instances overlaping in the mask.

        Args:
            mask (Mask): Thermogram mask that needs to be adjusted.

        Returns:
            Mask: Adjusted Mask.
        """
        return self.__fix_overlap(mask) if self.__check_overlap(mask) else mask

    def __repr__(self) -> str:
        """Display the transformation params.

        Returns:
            str: Params.
        """
        return "Parameters: \tNone"


class ExpantionMaskAdjuster(MaskAdjuster):
    """Adjust the mask by expanding each instance domain."""

    def __init__(self, distance: int = 7) -> None:
        self.distance = distance

    def __call__(self, mask: Mask) -> Mask:
        """Method that expand each instance domain in the mask.

        Args:
            mask (Mask): Thermogram mask that needs to be adjusted.

        Returns:
            Mask: Adjusted Mask.
        """
        # get binary mask from original label mask
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = img_as_ubyte(binary_closing(mask.binary_mask, kernel))

        # Expanding each instance domain from label mask
        label_mask = expand_labels(mask.label_mask, distance=self.distance)

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
