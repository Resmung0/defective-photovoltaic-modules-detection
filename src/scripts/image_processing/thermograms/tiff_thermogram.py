""""File to define the thermograms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flyr import palettes
from skimage import io

from .interface import Thermogram


class TIFFThermogram(Thermogram):
    """Class to read and process tiff thermograms."""

    def __init__(
        self,
        image_path: str,
        metadata_path: str | None = None,
        optical_path: str | None = None,
        method: str = "thermomap",
    ) -> None:
        """

        Args:
            image_path (str): Thermogram file path.
            metadata_path (Optional[str  |  None], optional): Thermogram
            metadata file path. Defaults to None.
            optical_path (Optional[str  |  None], optional): Thermogram
            optical photo file path. Defaults to None.
            method (Optional[str], optional): Type of method to process
            tiff thermograms ('thermomap', 'other'). Defaults to "ThermoMAP".
        """
        self.__path = Path(image_path)
        self.__metadata_path = Path(metadata_path) if metadata_path else None
        self.__optical_path = Path(optical_path) if optical_path else None

        match method:
            case "thermomap":
                self.__kelvin_factor = 0.01
                self.__celsius_factor = 100
            case "other":
                self.__kelvin_factor = 0.04
                self.__celsius_factor = 273.15

    @property
    def identifier(self) -> str:
        """Thermogram file name.

        Returns:
            str: Thermogram file name.
        """
        return self.__path.name

    @property
    def raw(self) -> np.ndarray:
        """Thermogram raw values.

        Returns:
            np.ndarray: Thermogram 16 bit image.
        """
        return io.imread(self.__path.resolve())

    @property
    def kelvin(self) -> np.ndarray:
        """Thermogram's temperature in Kelvin (K).

        Returns:
            np.ndarray: Thermogram temperature in Kelvin.
        """
        return self.raw * self.__kelvin_factor

    @property
    def celsius(self) -> np.ndarray:
        """Thermogram's temperature in celsius (°C).

        Returns:
            np.ndarray: Thermogram temperature in Celsius.
        """
        return self.kelvin - self.__celsius_factor

    @property
    def fahrenheit(self) -> np.ndarray:
        """Thermogram's temperature in Fahrenheit (°F).

        Returns:
            np.ndarray: Thermogram temperature in Fahrenheit.
        """
        return (self.celsius * 1.8) + 32.0

    @property
    def optical(self):
        """The thermogram's embedded photo.

        Returns:
            np.ndarray: Thermogram embedded photo.
        """
        if self.__optical_path:
            return io.imread(self.__optical_path.resolve())
        return None

    @property
    def metadata(self) -> dict[str, str | int]:
        """Metadata related to thermogram temperature data.

        Returns:
            dict[str, str | int]: Metadata that build the thermogram.
        """
        if self.__metadata_path:
            return pd.read_csv(self.__metadata_path.resolve())
        return None

    def render(
        self, min_v: float = None, max_v: float = None, palette: str = "grayscale"
    ) -> np.ndarray:
        """Renders the thermogram to RGB with the given settings.

        Args:
            min_v (float, optional): Minimal value to consider the
            thermogram's temperature range. Defaults to None.
            max_v (float, optional): Maximal value to consider the
            thermogram's temperature range. Defaults to None.
            palette (str, optional): Palette to render the thermogram.
            Default to "grayscale".

        Returns:
            np.ndarray: A three dimensional array of integers between 0 and 255,
            representing an RGB render of the thermogram.
        """
        # Normalize the raw image
        max_value = np.max(self.kelvin) if max_v else max_v
        min_value = np.min(self.kelvin) if min_v else min_v
        normalized = (self.kelvin - min_value) / (max_value - min_value)

        # Apply the chosen palette
        if palette in ["grayscale", "grayscale-inverted"]:
            normalized = (normalized * 255.0).astype("uint8")
            normalized = np.broadcast_to(normalized[..., None], normalized.shape + (3,))
            if palette == "grayscale-inverted":
                normalized = np.invert(normalized)
        else:
            normalized = palettes.map_colors(normalized, palette)
        return normalized

    def adjust_metadata(self) -> TIFFThermogram:
        """Adjust the metadata that build the thermogram.

        Returns:
            TIFFThermogram: Thermogram.
        """
        return NotImplementedError()
