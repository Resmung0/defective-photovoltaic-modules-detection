"""File that define the adjuster that fix overlapped masks."""

import numpy as np

from scripts.image_processing.mask import Mask

from .interface import Adjuster


class OverlapMaskAdjuster(Adjuster):
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

    def __call__(self, image: Mask) -> Mask:
        """Method that remove instances overlaping in the mask.

        Args:
            image (Mask): Mask that needs to be adjusted.

        Returns:
            Mask: Adjusted Mask.
        """
        return self.__fix_overlap(image) if self.__check_overlap(image) else image

    def __repr__(self) -> str:
        """Display the transformation params.

        Returns:
            str: Params.
        """
        return "Parameters: \tNone"
