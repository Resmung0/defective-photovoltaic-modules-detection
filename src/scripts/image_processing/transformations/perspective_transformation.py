"""File that defines the perspective transformation."""

import numpy as np
from skimage.transform import ProjectiveTransform, warp

from scripts.image_processing.mask import Mask

from .interface import Transformation


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
        self, mask: Mask, image: np.ndarray, area_threshold: float = 1000
    ) -> np.ndarray:
        """Method to correct the perspective of each region of interest (ROI) in the image
        based on the label mask given. First, gets extreme points of each instance of the
        mask. Then, for each of them estimate the Homography matrix of the ROI and warp it.

        Args:
            mask (Mask): Mask that highlights each image ROI.
            image (np.ndarray):Rendered thermogram image (8-bit).
            area_threshold (float): Threshold area to filter instances. Default to 1000.

        Returns:
            np.ndarray: Transformed image.
        """

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
