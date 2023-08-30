"""File that defines functions to retrieve data from remote storage."""

from os import environ
from pathlib import Path

from dotenv import load_dotenv
from kaggle import api
from prefect import flow, get_run_logger, task


@task
def authenticate_to_kaggle() -> None:
    """Athentification on Kaggle API. First the folder that will store
    Kaggle json file is created inside the project directory and the
    environment variable KAGGLE_CONFIG_DIR is set. Than, the JSON file
    is created inside this folder.
    """
    load_dotenv()
    logger = get_run_logger()

    # Folder and environment variable creation
    logger.info("Creating Kaggle credentials")
    kaggle_folder = Path(".kaggle").resolve()
    environ["KAGGLE_CONFIG_DIR"] = str(kaggle_folder)

    if not kaggle_folder.is_dir():
        kaggle_folder.mkdir()

        # Writting User's credentials to Kaggle Json file
        kaggle_file_path = Path(kaggle_folder, "kaggle.json")
        kaggle_file_path.touch(0o600)
        credentials = {
            "username": environ["KAGGLE_USERNAME"],
            "key": environ["KAGGLE_KEY"],
        }
        kaggle_file_path.write_text(str(credentials), encoding="utf-8")

    # Authentication to Kaggle API
    logger.info("Authenticating to Kaggle")
    api.authenticate()


@task
def download_infrared_solar_modules_dataset() -> None:
    """Download Infrared Solar Modules from Kaggle."""

    path = Path("data/classification")
    api.dataset_download_files(
        "marcosgabriel/infrared-solar-modules", str(path), quiet=False, unzip=True
    )

    path = Path(path, "archive")
    for file in ["README.md", "LICENSE"]:
        Path(path, file).unlink()

    file_path = Path(path, "2020-02-14_InfraredSolarModules/InfraredSolarModules")
    file_path.rename("data/classification/infrared-solar-modules")
    path.rmdir()


@task
def download_photovoltaic_system_thermography_dataset() -> None:
    """Download Photoboltaic System Thermography from Kaggle."""
    api.dataset_download_files(
        "marcosgabriel/photovoltaic-system-thermography",
        "data/segmentation/photovoltaic-system-thermography",
        quiet=False,
        unzip=True,
    )


@task
def download_photovoltaic_system_o_and_m_inspection_dataset() -> None:
    """Download Photovoltaic System O&M Inspection from Kaggle."""
    api.dataset_download_files(
        "marcosgabriel/photovoltaic-system-o-and-m-inspection",
        "data/segmentation/photovoltaic-system-o-and-m-inspection",
        quiet=False,
        unzip=True,
    )


@flow
def download_datasets():
    """Download all datasets used by this project from Kaggle."""
    authenticate_to_kaggle()
    download_infrared_solar_modules_dataset()
    download_photovoltaic_system_thermography_dataset()
    download_photovoltaic_system_o_and_m_inspection_dataset()
