"""
Linear car+trailer sway model, LQR design, and Kalman propagation.

Single source of truth for the analytical model that rc_sway_experiment.py
and car_kalmaan_filter.py both use.

State (matches the Julia design script):
    x = [v1, r1, r2, theta, psi1, y]
      v1    car lateral velocity, body frame          [m/s]
      r1    car yaw rate                              [rad/s]
      r2    trailer yaw rate                          [rad/s]
      theta articulation angle, psi1 - psi2           [rad]
      psi1  car heading relative to the lane          [rad]
      y     car lateral offset from lane center       [m]
Input:
    u = delta, front road-wheel steer angle           [rad]
Control law:
    delta = -K @ x
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

G = 9.81

# ---------------------------------------------------------------------
# Rig geometry (must match rc-truck-trailer.xml / rc_sway_experiment.py)
# ---------------------------------------------------------------------
CAR_MASS = 3.02          # kg, chassis + 4 wheels + 2 knuckles (from the XML)
CAR_IZZ = 0.109          # kg m^2, car yaw inertia about its own CG
CAR_A = 0.1439           # m, car CG -> front axle
CAR_B = 0.1439           # m, car CG -> rear axle
HITCH_H = 0.0418         # m, rear axle -> hitch pivot (0.1857 - 0.1439)

# MuJoCo has no tire model - lateral force comes from regularized Coulomb
# friction, so "cornering stiffness" is an effective slope, not a parameter
# of the sim. Estimate it from the static normal load: a tire saturates at
# mu*N once the slip angle reaches roughly SLIP_SAT, so C ~ mu*N/SLIP_SAT.
# The LQR below is checked for stability at 0.5x and 2x this value, which
# covers the uncertainty in the estimate.
TIRE_MU = 0.9
SLIP_SAT = 0.15          # rad, slip angle at which the tire saturates
C_PER_NEWTON = TIRE_MU / SLIP_SAT   # N/rad of cornering stiffness per N of load


@dataclass
class Params:
    """Bicycle-with-trailer parameters at one loading condition."""
    m1: float = CAR_MASS     # tractor mass                       [kg]
    I1: float = CAR_IZZ      # tractor yaw inertia                [kg m^2]
    a: float = CAR_A         # tractor CG -> front axle           [m]
    b: float = CAR_B         # tractor CG -> rear axle            [m]
    h: float = HITCH_H       # rear axle -> hitch                 [m]
    Cf: float = 100.0        # front axle cornering stiffness     [N/rad]
    Cr: float = 200.0        # rear axle cornering stiffness      [N/rad]
    m2: float = 4.4          # trailer mass                       [kg]
    I2: float = 0.023        # trailer yaw inertia about its CG   [kg m^2]
    a2: float = 0.181        # hitch -> trailer CG                [m]
    L2: float = 0.2475       # hitch -> trailer axle              [m]
    C2: float = 240.0        # trailer axle cornering stiffness   [N/rad]


def build_models(V, p: Params):
    """Continuous-time A (6x6) and B (6,) for the linear model at speed V."""
    m1, I1, a, b, h = p.m1, p.I1, p.a, p.b, p.h
    Cf, Cr = p.Cf, p.Cr
    m2, I2, a2, L2, C2 = p.m2, p.I2, p.a2, p.L2, p.C2

    M = np.array([
        [m1 + m2,       -m2 * (b + h),         -m2 * a2],
        [-m2 * (b + h),  I1 + m2 * (b + h)**2,  m2 * a2 * (b + h)],
        [-m2 * a2,       m2 * a2 * (b + h),     I2 + m2 * a2**2],
    ])

    Ct = np.array([
        [-Cf / V, -Cf * a / V,       0.0,         0.0],
        [-Cr / V,  Cr * b / V,       0.0,         0.0],
        [-C2 / V,  C2 * (b + h) / V, C2 * L2 / V, -C2],
    ])
    Cd = np.array([Cf, 0.0, 0.0])

    T = np.array([
        [1.0, 1.0, 1.0],
        [a,  -b,  -(b + h)],
        [0.0, 0.0, -L2],
    ])

    Gv = np.array([
        [0.0, -(m1 + m2) * V,   0.0, 0.0],
        [0.0,  m2 * (b + h) * V, 0.0, 0.0],
        [0.0,  m2 * a2 * V,      0.0, 0.0],
    ])

    M_inv = np.linalg.inv(M)
    A = np.zeros((6, 6))
    A[0:3, 0:4] = M_inv @ (T @ Ct + Gv)
    A[3, 1], A[3, 2] = 1.0, -1.0     # theta_dot = r1 - r2
    A[4, 1] = 1.0                    # psi1_dot  = r1
    A[5, 0], A[5, 4] = 1.0, V        # y_dot     = v1 + V*psi1

    B = np.zeros(6)
    B[0:3] = M_inv @ (T @ Cd)
    return A, B


def rig_params(cargo_offset, cargo_mass, frame_mass, frame_com_x,
               wheel_mass, hitch_to_axle):
    """Params for one payload placement, derived from the same numbers
    rc_sway_experiment.py feeds into the generated trailer XML.

    cargo_offset is measured from the axle, + = ahead of the axle.
    Cornering stiffnesses are scaled from the resulting static axle loads.
    """
    ax = -hitch_to_axle                       # axle x in the hitch frame
    cx = ax + cargo_offset                    # payload x in the hitch frame

    m2 = frame_mass + 2 * wheel_mass + cargo_mass
    # trailer CG behind the hitch (wheels sit on the axle line)
    a2 = -(frame_mass * frame_com_x
           + 2 * wheel_mass * ax
           + cargo_mass * cx) / m2

    # yaw inertia about the trailer CG (point masses in the yaw plane; the
    # frame's own 0.014 kg m^2 zz term is carried along)
    I2 = (0.014
          + frame_mass * (frame_com_x + a2)**2
          + 2 * wheel_mass * (ax + a2)**2
          + cargo_mass * (cx + a2)**2)

    # static loads: trailer moments about the hitch, car moments about its axles
    trailer_axle_load = m2 * G * a2 / hitch_to_axle
    tongue_load = m2 * G - trailer_axle_load
    wheelbase = CAR_A + CAR_B
    rear_load = CAR_MASS * G * CAR_A / wheelbase \
        + tongue_load * (wheelbase + HITCH_H) / wheelbase
    front_load = CAR_MASS * G + tongue_load - rear_load

    return Params(
        m2=m2, a2=a2, I2=I2, L2=hitch_to_axle,
        Cf=C_PER_NEWTON * max(front_load, 1.0),
        Cr=C_PER_NEWTON * rear_load,
        C2=C_PER_NEWTON * trailer_axle_load,
    )


# LQR weights. Only the states the driver actually cares about are
# penalised: articulation angle, heading, and lane offset. R is large
# because the steering authority is only +-0.61 rad and the servo is slow.
Q_DEFAULT = np.diag([0.0, 0.0, 0.0, 50.0, 20.0, 25.0])
R_DEFAULT = 100.0


def design_lqr(V, p: Params, Q=None, R=None):
    """Continuous-time LQR gain K such that delta = -K @ x."""
    Q = Q_DEFAULT if Q is None else Q
    R = R_DEFAULT if R is None else R
    A, B = build_models(V, p)
    Bc = B.reshape(6, 1)
    P = solve_continuous_are(A, Bc, Q, np.atleast_2d(R))
    return (np.linalg.solve(np.atleast_2d(R), Bc.T @ P)).ravel()


def closed_loop_poles(V, p: Params, K):
    A, B = build_models(V, p)
    return np.linalg.eigvals(A - np.outer(B, K))
