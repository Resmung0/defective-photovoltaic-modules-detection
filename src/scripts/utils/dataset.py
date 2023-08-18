"""File that define data reading methods."""

from json import load
from pathlib import Path

import numpy as np
import pandas as pd
from flyr import FlyrThermogram, unpack
from skimage import color, draw, io
from skimage.transform import resize
from torch.utils.data import Dataset

from scripts.image_processing.thermograms.tiff_thermogram import TIFFThermogram


def read_thermogram(
    file_path: str, tiff_info: dict[str, str] | None = None
) -> FlyrThermogram | TIFFThermogram:
    """Read thermogram data from RJPG and TIFF files.

    Args:
        file_path (str): Thermogram image file path.
        tiff_info (dict[str, str] | None): Informations needed to read tiff thermograms,
        like metadata and optical image paths. Default to None.

    Returns:
        FlyrThermogram | TIFFThermogram: Thermogram data.
    """
    file_format = Path(file_path).suffix
    if file_format in (".jpg", ".JPG"):
        thermogram = unpack(file_path)
    elif file_format in (".tif", ".tiff"):
        if tiff_info is None:
            tiff_info = {
                "metadata_path": None,
                "optical_path": None,
                "method": "thermomap",
            }
        thermogram = TIFFThermogram(file_path, **tiff_info)
    return thermogram


def read_annotation(
    file_path: str, shape: tuple[int, int] = (512, 640)
) -> tuple[np.ndarray[int], np.ndarray[dict[str, float]], np.ndarray[str]]:
    """Read data from JSON annotation file.

    Args:
        file_path (str): JSON file path
        shape (tuple[int, int], optional): Dimensions of corresponding image mask.
        Defaults to (512, 640).

    Returns:
        tuple[np.ndarray[int], np.ndarray[dict[str, float]], np.ndarray[str]]: Label
        mask, centers of each label and corresponding PV module class.
    """

    #  Get data from annotation JSON file
    with open(file_path, "r", encoding="utf-8") as annotation_file:
        annotation_data = load(annotation_file)

    masks, centers, classes = [], [], []
    for instance in annotation_data["instances"]:
        # Store PV modules classes
        roi_class = "defected" if instance["defected_module"] else "non-defected"
        classes.append(roi_class)

        # Store mask center
        centers.append(instance["center"])

        # Create and store the label mask
        polygon = np.array(
            [[coords["y"], coords["x"]] for coords in instance["corners"]]
        )
        masks.append(draw.polygon2mask(shape, polygon))

    return tuple(map(np.array, [masks, centers, classes]))


class InstanceSegmentationDataset(Dataset):
    """Dataset that process thermograms and masks to be used in instance
    segmentation task.
    """

    def __init__(
        self,
        data_dir: str,
        method: str = "thermal",
        transform=None,
        target_transform=None,
    ) -> None:
        """
        Args:
            data_dir (str): Data directory path.
            method (str, optional): Method to determine the image type to
            process. Defaults to "thermal".
            transform (_type_, optional): Transformations to apply to the
            images. Defaults to None.
            target_transform (_type_, optional): Transformations to apply to
            the masks. Defaults to None.
        """
        # Get all thermograms (RJPG or TIFF)
        self.data = Path(data_dir, "thermograms").iterdir()
        self.data = pd.DataFrame(
            [str(value) for value in sorted(list(self.data))],
            columns=["image_filepath"],
        )

        # Get all thermograms masks
        self.data["mask_filepath"] = self.data["image_filepath"].str.replace(
            "thermograms", "annotations"
        )
        self.data["mask_filepath"] = self.data["mask_filepath"].str.replace(
            "jpg|tiff", "json", regex=True
        )

        # Get all optical images and inspection/camera metadata, if thermograms are in TIFF format
        if Path(data_dir, "opticals").is_dir():
            self.data["optical_filepath"] = self.data["image_filepath"].str.replace(
                "thermograms", "opticals"
            )
            self.data["optical_filepath"] = self.data["optical_filepath"].str.replace(
                "tiff", "jpg"
            )

        if Path(data_dir, "metadata.csv").is_file():
            self.data["metadata_filepath"] = ["metadata.csv"] * self.data.shape[0]

        self.transform, self.target_transform = transform, target_transform
        self.method = method

    def __len__(self) -> int:
        """Method to know the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Method that iterate in the dataset.

        Args:
            index (int): Dataset instance index.

        Returns:
            tuple[np.ndarray, np.ndarray]: Number of images in the dataset.
        """
        # Get extra data required by TIFF images
        extra_files = {}
        if "optical_filepath" in self.data.columns:
            extra_files["optical_path"] = self.data.loc[index, "optical_filepath"]
        if "metadata_filepath" in self.data.columns:
            extra_files["metadata_path"] = self.data.loc[index, "metadata_filepath"]

        # Read thermogram
        thermogram = read_thermogram(
            self.data.loc[index, "image_filepath"], extra_files
        )

        # Choose the image type to be processed
        match self.method:
            case "optical":
                msg = """There's no optical information avaible! If your thermograms
                are in TIFF format, please inform where the optical images are located.
                If they are in R-JPG format, probably your thermal camera don't have
                a optical camera attach to it. So, there's nothing to be done.
                """
                assert thermogram.optical is not None, msg
                image = color.rgb2gray(thermogram.optical)
                image = resize(image, (512, 640))
            case "thermal":
                image = thermogram.render(palette="grayscale")
                image = image[:, :, 0] if image.ndim == 3 else image

        # Read mask
        mask, _, _ = read_annotation(self.data.loc[index, "mask_filepath"], image.shape)

        # Apply transformations on image and mask
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


class ClassificationDataset(Dataset):
    """Dataset that process images and labels to be used in classification
    task.
    """

    def __init__(self, data_dir: str, transform=None, target_transform=None) -> None:
        """
        Args:
            data_dir (str): Data directory path.
            transform (_type_, optional): Transformations to apply to the
            images. Defaults to None.
            target_transform (_type_, optional): Transformations to apply to
            the masks. Defaults to None.
        """
        metadata_path = next(Path(data_dir).glob("*.json"))
        self.metadata = pd.read_json(metadata_path, orient="index")
        self.metadata = self.metadata.sort_index()
        self.metadata["image_filepath"] = self.metadata.image_filepath.apply(
            lambda x: str(Path(data_dir, x))
        )

        self.transform, self.target_transform = transform, target_transform

    def __len__(self) -> int:
        """Method to know the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.metadata.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Method that iterate in the dataset.

        Args:
            index (int): Dataset instance index.

        Returns:
            tuple[np.ndarray, np.ndarray]: Number of images in the dataset.
        """
        image = io.imread(self.metadata.loc[index, "image_filepath"])
        label = self.metadata.loc[index, "anomaly_class"]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
