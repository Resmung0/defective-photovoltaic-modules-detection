"""File that define the transformation responsible to enhanced image's 
contrast."""

import numpy as np
from skimage import exposure
from skimage.util import img_as_ubyte

from scripts.image_processing.mask import Mask

from .interface import Transformation


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

    def __call__(self, image: np.ndarray, mask: Mask) -> np.ndarray:
        """Apply the respective histogram equalization methodology to
        thermogram region highlighted by mask.

        Args:
            image (np.ndarray): Rendered thermogram image (8-bit).
            mask (Mask): Thermogram mask.

        Returns:
            np.ndarray: Enhanced thermogram.
        """
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
