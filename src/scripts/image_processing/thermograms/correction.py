"""File that define all methods that correct thermograms based on external values."""

import numpy as np


def predict_distance(altitude: float, alpha: int, beta: int) -> float:
    """
    Calculates an approximation of the distance
    between PV modules fixed in a position and
    the drone that's making the thermal inspection.

    Parameters
    ----------
    alpha : float or int
        Tilt angle of PV modules.
    beta : float or int
        Drone's angle in relation to the glass surface
        of PV modules during thermal inspection.
    altitude : float or int
        Drone's altitude relative to a flat terrain.

    Returns
    -------
    float
        Distance between the drone and PV modules.
    """
    return abs(altitude) / (np.sin(90 - alpha - beta))
