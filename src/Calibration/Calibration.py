import numpy as np
import cv2
import yaml
from src.Common.utils import load_config


class Calibration:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.K = None
        self.R = None
        self.T = None
        self.P = None
        self.p_inv = None
        self.load_calibration_params()

    # -----------------------------------------------------------------------------
    # load_calibration_params
    # -----------------------------------------------------------------------------
    def load_calibration_params(self):
        params = load_config(self.yaml_path)

        self.K = np.array(params['K']['data']).reshape(params['K']['rows'], params['K']['cols'])
        r_vector = np.array(params['R']['data']).reshape(params['R']['rows'], params['R']['cols'])
        self.T = np.array(params['T']['data']).reshape(params['T']['rows'], params['T']['cols'])

        self.R = cv2.Rodrigues(r_vector)[0]
        self.P = self.K @ np.hstack((self.R, self.T))

    # -----------------------------------------------------------------------------
    # estimate_3d_point_pinv
    # -----------------------------------------------------------------------------
    def estimate_3d_point_pinv(self, x_2d, y_2d):
        """
        Estimates the 3D world coordinates given 2D image coordinates (x, y),
        assuming Y = 0 (altitude is zero).

        Parameters:
        x_2d (float): x-coordinate in the image.
        y_2d (float): y-coordinate in the image.

        Returns:
        np.array: Estimated 3D world coordinates [X, 0, Z].
        """

        # Construct matrix A
        mat_a = np.array([
            [self.P[0, 0], self.P[0, 2], -x_2d],
            [self.P[1, 0], self.P[1, 2], -y_2d],
            [self.P[2, 0], self.P[2, 2], -1]
        ])
        b = np.array([-self.P[0, 3], -self.P[1, 3], -self.P[2, 3]])

        # Solving system
        solution = np.linalg.solve(mat_a, b)

        # Extract X, Z
        X, Z, _ = solution  # Ignore scale factor w
        Z += 2.95  # Calibration correction

        return np.array([X, 0, Z])
