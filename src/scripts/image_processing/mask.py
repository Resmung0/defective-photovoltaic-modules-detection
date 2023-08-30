""""File to define the thermograms masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table

from .utils import read_annotation


class Mask:
    """Class that establish all mask features and methods, like:
    * Different type of mask reading (from JSON annotation or model output);
    * Mask type calculation;
    """

    def __init__(self, data) -> None:
        self.data = data

    @staticmethod
    def from_label_mask(label_mask: np.ndarray) -> Mask:
        """Method for mask creation from label mask array.

        Args:
            label_mask (np.ndarray): Label mask array.

        Returns:
            Mask: Mask class.
        """
        labels = np.unique(label_mask)[1:]
        masks = [
            np.where(label_mask == label_name, True, False) for label_name in labels
        ]
        return Mask(np.array(masks))

    @staticmethod
    def from_annotation(file_path: str) -> Mask:
        """Method for mask creation from JSON annotation file.

        Args:
            file_path (str): JSON annotation file path.

        Returns:
            Mask: Mask class.
        """
        masks, _, _ = read_annotation(file_path)
        return Mask(masks)

    @property
    def label_mask(self) -> np.ndarray:
        """Label mask creation from data (JSON annotation or model output).

        Returns:
            np.ndarray: Label mask array.
        """
        label_mask = np.zeros(self.data.shape[1:])
        for index, label_name in enumerate(self.data, start=1):
            label_mask[label_name] = index
        return label_mask.astype("uint8")

    @property
    def binary_mask(self) -> np.ndarray:
        """Binary mask creation from label mask.

        Returns:
            np.ndarray: Binary mask array.
        """
        return np.where(self.label_mask != 0, 1, 0).astype(np.uint8)

    def calculate_properties(self, mask_type: str = "label") -> pd.DataFrame:
        """Mask properties calculation.

        Args:
            mask_type (str, optional): Type of mask to calculate properties from.
            Can be "label" or "binary". Defaults to "label".

        Returns:
            pd.DataFrame: Calculated properties table.
        """

        def get_mask_scale(
            area: float, min_limit: int = 1024, max_limit: int = 9216
        ) -> str:
            """Calculate the mask scale based on area.

            Args:
                area (float): Mask area.
                min_limit (int, optional): Minimal area value. Defaults to 1024.
                max_limit (int, optional): Maximal area value. Defaults to 9216.

            Returns:
                str: Mask type.
            """

            mask_type = "medium"
            if area < min_limit:
                mask_type = "small"
            elif area > max_limit:
                mask_type = "large"
            return mask_type

        def get_mask_corners(binary_mask: np.ndarray) -> np.ndarray:
            """Calculate the mask extreme points.

            Args:
                binary_mask (np.ndarray): Mask to calculate extreme points.

            Returns:
                np.ndarray: Mask extreme points.
            """

            # Find mask coordinates
            cnt = regionprops(binary_mask)[0].coords[:, ::-1]

            # Find extreme points from diagonal left
            sum_coords = cnt.sum(axis=1)
            top_left = cnt[np.argmin(sum_coords)]
            bottom_right = cnt[np.argmax(sum_coords)]

            # Find extreme points from diagonal right
            diff_coords = np.squeeze(np.diff(cnt, axis=1))
            top_right = cnt[np.argmin(diff_coords)]
            bottom_left = cnt[np.argmax(diff_coords)]

            # return corner points
            return np.array([top_left, top_right, bottom_right, bottom_left])

        # Choose which mask to calculate properties
        mask = self.label_mask if mask_type == "label" else self.binary_mask

        # Calculate standard mask properties
        properties = pd.DataFrame(
            regionprops_table(mask, properties=["area", "centroid"])
        )
        properties.columns = ["area", "centroid-height", "centroid-width"]

        # Calculate the mask type based on area
        properties = properties.assign(scale=properties["area"].map(get_mask_scale))

        # Calculate the mask extreme points (corners)
        corners = []
        for label in np.unique(mask)[1:]:
            instance_mask = np.where(mask == label, 1, 0)
            mask_corners = get_mask_corners(instance_mask).reshape(-1)
            corners.append(mask_corners)
        corners = pd.DataFrame(corners)
        corners.columns = [
            "top-left-height",
            "top-left-width",
            "top-right-height",
            "top-right-width",
            "bottom-right-height",
            "bottom-right-width",
            "bottom-left-height",
            "bottom-left-width",
        ]

        # Merge all properties
        return pd.concat([properties, corners], axis=1)
