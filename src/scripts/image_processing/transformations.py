"""File that implements all transformations applied to thermograms
using their correspondence mask."""


from abc import ABC, abstractmethod

import numpy as np
from skimage import exposure
from skimage.transform import ProjectiveTransform, warp
from skimage.util import img_as_ubyte

from .mask import Mask
from .thermograms import Thermogram


class Transformation(ABC):
    """Interface class that is responsible to transform the thermogram
    image utilizing the correspondence mask in some kind of way."""

    @abstractmethod
    def __call__(self, thermogram: Thermogram, mask: Mask) -> np.ndarray:
        """Method that modifies the mask in some way.

        Args:
            thermogram (Thermogram): Thermogram image.
            mask (Mask): Thermogram mask that highlights each image region
            where to apply the transformation.


        Returns:
            np.ndarray: Transformed image.
        """
        return NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        """Display the transformation params.

        Returns:
            str: Params.
        """
        return ""


class HistogramEqualizationTransformation(Transformation):
    """Transformation responsible for apply histogram equalization on a
    thermogram image, enhancing it's contrast."""

    def __init__(
        self,
        clip_limit: float = 0.04,
        kernel_size: tuple[int, int] | None = None,
        method: str = "local",
    ) -> None:
        """
        Args:
            clip_limit (float, optional): Maximum value to clip the histogram
            equalization. Defaults to 0.04,.
            kernel_size (tuple[int, int] | None, optional): Size of the kernel used.
            Defaults to None.
            method (str, optional): Histogram equalization methodology
            applied ("global" or "local"). Defaults to "local".
        """
        self.params = {
            "clip_limit": clip_limit,
            "kernel_size": kernel_size,
            "method": method,
        }

    @staticmethod
    def __aply_mask(image: np.ndarray, mask: Mask, invert_mask: bool) -> np.ndarray:
        """Apply the mask in the thermogram, highlighting only the region of interest.

        Args:
            image (np.ndarray): Rendered thermogram image (8-bit).
            mask (Mask): Thermogram mask.
            invert_mask (bool): Trigger to invert the mask intensities.

        Returns:
            np.ndarray: Masked image.
        """
        # Get thermogram binary mask
        mask = mask.binary_mask.astype("bool")
        mask = np.invert(mask) if invert_mask else mask

        # Apply the mask in the image
        image[mask] = 0
        return image

    def __call__(self, thermogram: Thermogram, mask: Mask, **kwargs) -> np.ndarray:
        """Apply the respective histogram equalization methodology to
        thermogram region highlighted by mask.

        Args:
            thermogram (Thermogram): Thermogram image.
            mask (Mask): Thermogram mask that highlights each image region
            where to apply the transformation.

        Returns:
            np.ndarray: Enhanced thermogram.
        """
        # Render thermogram
        image = thermogram.render(**kwargs)

        # Find the foreground and background regions
        foreground = self.__aply_mask(image, mask, True)
        background = self.__aply_mask(image, mask, False)

        # Equalize the foreground region
        match self.params["method"]:
            case "local":
                foreground = exposure.equalize_adapthist(
                    foreground, self.params["kernel_size"], self.params["clip_limit"]
                )
            case "global":
                foreground = exposure.equalize_hist(
                    foreground, mask=mask.binary_mask.astype("bool")
                )
        foreground = img_as_ubyte(foreground)
        foreground[foreground == foreground.min()] = 0

        # Sum the regions
        return foreground + background

    def __repr__(self) -> str:
        """Display the transformations params.

        Returns:
            str: Params.
        """
        return f"Parameters: {self.params['method']}"


class PerspectiveTransformation(Transformation):
    """Transformation responsible for image perspective correction by applying
    homography to masked images.
    """

    def __init__(self) -> None:
        self.homography = ProjectiveTransform()
        self.source_points, self.destination_points = [], []

    @staticmethod
    def __get_destination_values(
        source_points: np.ndarray[int, int],
    ) -> tuple[np.ndarray[tuple[int, int]], tuple[int, int]]:
        """Calculation of values related to the destination image.

        Args:
            source_points (np.ndarray[int, int]): Points to start the transformation.

        Returns:
            tuple[np.ndarray[tuple[int, int]], tuple[int, int]]: Calculated values by the
            transformation. First is the destination points and then the projected warped
            image shape.
        """
        top_left, top_right, bottom_right, bottom_left = source_points
        w_1 = np.sqrt(
            (bottom_right[0] - bottom_left[0]) ** 2
            + (bottom_right[1] - bottom_left[1]) ** 2
        )
        w_2 = np.sqrt(
            (top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2
        )
        h_1 = np.sqrt(
            (top_right[0] - bottom_right[0]) ** 2
            + (top_right[1] - bottom_right[1]) ** 2
        )
        h_2 = np.sqrt(
            (top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2
        )
        approx_w = max(int(w_1), int(w_2))
        approx_h = max(int(h_1), int(h_2))
        destination_points = np.float32(
            [(0, 0), (approx_w - 1, 0), (approx_w - 1, approx_h - 1), (0, approx_h - 1)]
        )
        return destination_points, (approx_h, approx_w)

    def __call__(
        self, thermogram: Thermogram, mask: Mask, area_threshold: float = 1000, **kwargs
    ) -> np.ndarray:
        """Method to correct the perspective of each region of interest (ROI) in the image
        based on the label mask given. First, gets extreme points of each instance of the
        mask. Then, for each of them estimate the Homography matrix of the ROI and warp it.

        Args:
            thermogram (Thermogram): Thermogram image.
            mask (Mask): Thermogram mask that highlights each image region
            where to apply the transformation.
            area_threshold (float): Threshold area to filter instances. Default to 1000.

        Returns:
            np.ndarray: Transformed image.
        """

        # Render thermogram
        image = thermogram.render(**kwargs)

        # Get mask properties
        mask_properties = mask.calculate_properties()

        # Get only the extreme points of each instance of the label mask
        self.source_points = mask_properties.iloc[:, 4:].values.reshape(-1, 4, 2)

        # Iterate for each label mask instance
        warped_images = []
        for index, src_points in enumerate(self.source_points):
            # Get the projected values (destination points and projected shape of the warped image)
            dst_points, projected_shape = self.__get_destination_values(src_points)
            self.destination_points.append(dst_points)

            # Estimate the homography matrix and warp the related region of the image
            self.homography.estimate(src_points, dst_points)

            # Filter only the instanvces that agree to the following conditons
            filter_conditions = [
                # The homography matrix is inversible (determinant different from 0)
                np.linalg.det(self.homography.params) != 0,
                # The instance have a area larger then threshold
                mask_properties.loc[index, "area"] > area_threshold,
            ]

            if all(filter_conditions):
                warped_image = warp(
                    image,
                    self.homography.inverse,
                    output_shape=projected_shape,
                    preserve_range=True,
                )
                warped_images.append(warped_image.astype("uint8"))
        return warped_images

    def __repr__(self) -> str:
        """Display the transformation params.

        Returns:
            str: Params.
        """
        return f"Parameters: \tHomography matrix: {self.homography.params}"
