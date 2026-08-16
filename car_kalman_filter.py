"""
Luenberger/Kalman observer for the car+trailer sway model.

Estimates the full 6-state vector from tractor-only sensing, for the case
where the rig has no hitch angle sensor (the MuJoCo model does have one, so
rc_sway_experiment.py feeds theta straight to the controller instead - this
is here for the real vehicle).

    state:        [v1, r1, r2, theta, psi1, y]
    measurements: [v1, r1, psi1, y]

The dynamics come from trailer_model.build_models, so this observer and the
LQR in rc_sway_experiment.py are always designed on the same plant.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from trailer_model import Params, build_models, rig_params  # noqa: F401

# which states the tractor can actually measure
C_MEAS = np.array([
    [1.0, 0, 0, 0, 0, 0],   # v1   - lateral accelerometer, integrated
    [0, 1.0, 0, 0, 0, 0],   # r1   - yaw gyro
    [0, 0, 0, 0, 1.0, 0],   # psi1 - heading vs the lane
    [0, 0, 0, 0, 0, 1.0],   # y    - lane offset
])

# Process / measurement noise intensities. W is per-state: the two yaw-rate
# and articulation states are the ones the model gets wrong, so they get the
# most process noise. V is per-measurement, sized from the sensors: the
# integrated lateral velocity is by far the worst, heading and lane offset
# come from a wall fit over hundreds of LiDAR returns and are good.
W_DEFAULT = np.diag([0.05, 0.5, 0.5, 0.1, 0.01, 0.01])
V_DEFAULT = np.diag([0.5, 0.002, 0.002, 0.001])


def kalman_gain(V_speed, p: Params, W=None, Vn=None, C=C_MEAS):
    """Steady-state continuous-time observer gain L (6x4)."""
    W = W_DEFAULT if W is None else W
    Vn = V_DEFAULT if Vn is None else Vn
    A, _ = build_models(V_speed, p)
    # dual of the LQR Riccati equation
    P = solve_continuous_are(A.T, C.T, W, Vn)
    return P @ C.T @ np.linalg.inv(Vn)


class KalmanFilter:
    """Continuous-time observer integrated with explicit Euler at dt.

    Usage per control step:
        kf.predict(u)          # u = steer angle just commanded [rad]
        kf.update(y_meas)      # y_meas = [v1, r1, psi1, y]
    """

    def __init__(self, V_speed, p: Params, dt, L=None, C=C_MEAS, x0=None):
        self.A, self.B = build_models(V_speed, p)
        self.C = C
        self.L = kalman_gain(V_speed, p, C=C) if L is None else L
        self.dt = dt
        self.reset(x0)

    def reset(self, x0=None):
        self.xhat = np.zeros(6) if x0 is None else np.array(x0, dtype=float)

    def predict(self, u):
        self.xhat = self.xhat + (self.A @ self.xhat + self.B * u) * self.dt

    def update(self, y_meas):
        innovation = np.asarray(y_meas) - self.C @ self.xhat
        self.xhat = self.xhat + (self.L @ innovation) * self.dt

    def is_stable(self):
        """True if the estimation error dynamics A - L C are stable."""
        return np.linalg.eigvals(self.A - self.L @ self.C).real.max() < 0


if __name__ == "__main__":
    # sanity check on the rearward-loaded (open-loop unstable) trailer
    p = rig_params(-0.04, 3.5, frame_mass=0.8, frame_com_x=-0.24,
                   wheel_mass=0.05, hitch_to_axle=0.2475)
    kf = KalmanFilter(4.0, p, dt=0.001)
    np.set_printoptions(precision=4, suppress=True)
    print("L =\n", kf.L)
    print("error eigenvalues =", np.linalg.eigvals(kf.A - kf.L @ kf.C))
    print("observer stable:", kf.is_stable())
