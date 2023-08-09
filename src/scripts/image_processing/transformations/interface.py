"""File that implements the Transform interface."""

from abc import ABC, abstractmethod

import numpy as np

from scripts.image_processing.mask import Mask
from scripts.image_processing.thermograms.interface import Thermogram


class Transformation(ABC):
    """Interface class that is responsible to transform the thermogram
    image utilizing the correspondence mask in some kind of way."""

    @abstractmethod
    def __call__(self, mask: Mask, thermogram: Thermogram) -> np.ndarray:
        """Method that modifies the mask in some way.

        Args:
            thermogram (Thermogram): Thermogram image to transform.
            mask (Mask): Thermogram mask that highlights the region where
            to transform the image.

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
