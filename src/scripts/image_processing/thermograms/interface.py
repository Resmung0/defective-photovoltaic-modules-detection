"""File that define the base class that reads thermograms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Thermogram(ABC):
    """Base class that implements thermogram image."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Thermogram file name.

        Returns:
            str: Thermogram file name.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def kelvin(self) -> np.ndarray:
        """Thermogram's temperature in Kelvin (K).

        Returns:
            np.ndarray: Thermogram temperature in Kelvin.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def celsius(self) -> np.ndarray:
        """Thermogram's temperature in celsius (°C).

        Returns:
            np.ndarray: Thermogram temperature in Celsius.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def fahrenheit(self) -> np.ndarray:
        """Thermogram's temperature in Fahrenheit (°F).

        Returns:
            np.ndarray: Thermogram temperature in Fahrenheit.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def optical(self) -> np.ndarray:
        """The thermogram's embedded photo.

        Returns:
            np.ndarray: Thermogram embedded photo.
        """
        return NotImplementedError()

    @property
    @abstractmethod
    def metadata(self) -> dict[str, str | int]:
        """Metadata related to thermogram temperature data.

        Returns:
            dict[str, str | int]: Metadata that build the thermogram.
        """
        return NotImplementedError()

    @abstractmethod
    def render(self) -> np.ndarray:
        """Renders the thermogram to RGB with the given settings.

        Returns:
            np.ndarray: Thermogram rendered to a 8 bit image.
        """
        return NotImplementedError()

    @abstractmethod
    def adjust_metadata(self) -> Thermogram:
        """Adjust the metadata that build the thermogram."""
        return NotImplementedError()
