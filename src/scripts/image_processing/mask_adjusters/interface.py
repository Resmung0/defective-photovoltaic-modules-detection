"""File that implements the Adjuster interface."""

from abc import ABC, abstractmethod

from scripts.image_processing.mask import Mask
from scripts.image_processing.thermograms.interface import Thermogram


class Adjuster(ABC):
    """Interface class that is responsible for change the thermogram
    or the mask in some kind way."""

    @abstractmethod
    def __call__(self, image: Mask | Thermogram) -> Mask | Thermogram:
        """Method that modifies the mask in some way.

        Args:
            image (Mask | Thermogram): Image that needs to be adjusted.

        Returns:
            Mask | Thermogram: Adjusted image.
        """
        return NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        """Display the transformation params."""
        return ""
