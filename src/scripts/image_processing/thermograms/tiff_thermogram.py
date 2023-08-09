""""File to define the thermograms."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pandas import read_csv
from skimage.io import imread

from .interface import Thermogram


class TIFFThermogram(Thermogram):
    """Class to read and process tiff thermograms."""

    def __init__(
        self,
        image_path: str,
        metadata_path: str | None = None,
        optical_path: str | None = None,
        method: str = "ThermoMAP",
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
        self.path = Path(image_path)
        self.metadata_path = Path(metadata_path)
        self.optical_path = Path(optical_path)

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
        return self.path.name

    @property
    def raw(self) -> np.ndarray:
        """Thermogram raw values.

        Returns:
            np.ndarray: Thermogram 16 bit image.
        """
        return imread(self.path.resolve())

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
        return imread(self.optical_path.resolve())

    @property
    def metadata(self) -> dict[str, str | int]:
        """Metadata related to thermogram temperature data.

        Returns:
            dict[str, str | int]: Metadata that build the thermogram.
        """
        return read_csv(self.metadata_path.resolve())

    def render(
        self,
        min_v: Optional[float] = None,
        max_v: Optional[float] = None,
    ) -> np.ndarray:
        """Renders the thermogram to RGB with the given settings.

        Args:
            min_v (Optional[float], optional): Minimal value to consider the
            thermogram's temperature range. Defaults to None.
            max_v (Optional[float], optional): Maximal value to consider the
            thermogram's temperature range. Defaults to None.

        Returns:
            np.ndarray: A three dimensional array of integers between 0 and 255,
            representing an RGB render of the thermogram.
        """
        max_value = np.max(self.kelvin) if max_v is None else max_v
        min_value = np.min(self.kelvin) if min_v is None else min_v
        thermal_img = (self.kelvin - min_value) / (max_value - min_value)
        return (thermal_img * 255.0).astype("uint8")

    def adjust_metadata(self) -> TIFFThermogram:
        """Adjust the metadata that build the thermogram.

        Returns:
            TIFFThermogram: Thermogram.
        """
        return NotImplementedError()
