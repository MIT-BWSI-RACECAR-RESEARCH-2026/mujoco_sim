"""
RC car + CAD trailer sway experiment.
RC-SCALE port of the full-size block-truck script, for use with
rc-truck-trailer.xml (CAD meshes in ./assets).

- Lane: 0.60 m wide, centered on y=0, road runs along +x. Lane center = y=0.
  Walls at y = +-0.95 bound the corridor; the LiDAR follows those walls.
- AIR RESISTANCE: MuJoCo's fluid model is enabled (AIR_DENSITY /
  AIR_VISCOSITY knobs). Every body feels quadratic drag + viscous damping
  based on its equivalent-inertia box, measured relative to the ambient
  WIND vector - so WIND=(0,-2,0) gives a steady 2 m/s crosswind on top of
  the impulsive swerve/gust disturbances. AIR_DENSITY=0 restores vacuum.
- IMUs on car and trailer (accel, gyro, orientation quat) + hitch sensor
  (trailer angle relative to car).
- control_function() below is YOUR hook: it gets the sensor readings (plus
  the model/data handles the simulated LiDAR needs) and returns
  (steer, speed). Preloaded with two LQR controllers, selected by
  CONTROLLER_MODE: a 2-state car-only lane tracker, and a 6-state
  car+trailer controller that also damps hitch articulation. The 6-state
  gain is solved at import time from trailer_model.py, so it always
  matches the geometry and cruise speed configured here.

MODES
- PIVOT_MODE: controller OFF. A single massless "tow point" is attached to
  the center of the car's front axle by a ball-type connect constraint:
  the car is completely free in pitch, yaw, and roll about the point.
  The point moves freely along z, is driven forward along x at PIVOT_SPEED,
  and moves side to side against a proportional spring (PIVOT_SPRING_K)
  plus damper (PIVOT_SPRING_C) that always pulls it back to lane center.
- PLANAR_MODE: kills ALL vertical-axis dynamics. Car root becomes
  x/y/yaw only, the hitch becomes a yaw-only hinge, so there is no pitch,
  roll, or heave anywhere: zero weight transfer, constant normal forces,
  and one common friction coefficient on all six tires (PLANAR_TIRE_MU).
  Can be combined with PIVOT_MODE.

- N_RUNS simulations. Each run i uses:
    disturbance magnitude = DISTURB_START + i * DISTURB_STEP
    payload offset        = CARGO_OFFSET  + i * CARGO_OFFSET_STEP
    payload mass          = CARGO_MASS    + i * CARGO_MASS_STEP
  Swerve units: rad. Gust units: N.
- At the end, one figure per run: hitch angle, lateral hitch force on the
  car, car yaw vs road, car lane offset, trailer tire grip (L/R),
  car tire grip (all 4), front vs rear car axle weight, and car
  speed. Blue bands = trailer at max swing (hitting the hitch stop /
  car); red bands = car flipped sideways.

TRAILER GEOMETRY NOTE
  The trailer frame comes from the CAD mesh, so hitch-to-axle distance
  (0.2475 m) is FIXED - there is no TONGUE_LEN knob anymore. Weight
  shifting is done exactly as before, by moving/re-massing the payload
  box on the deck: CARGO_OFFSET is measured from the AXLE
  (+ = ahead of axle / stable, - = behind axle / sway-prone) and may
  range over roughly -0.10 .. +0.11 m (payload must stay on the deck).
  Useful range is narrower than that: past about -0.06 m the negative
  tongue load levers the car's REAR AXLE off the ground entirely, and no
  steering controller can recover a car with no rear traction. build_model
  prints the rear axle load per run and warns when it gets that low.
"""

import csv
import math
import os
import platform
import re
import time
if platform.system() == "Darwin": # fix plotting for running on macos
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

import trailer_model


XML_PATH = "rc-truck-trailer.xml"

# ---------------- lidar knobs (RichBeam LakiBeam 1L specs) ----------------
SIMULATE_LIDAR = True
LIDAR_FOV_DEG = 270
LIDAR_PTS_PER_DEG = 4        # matches the 0.25° angular resolution setting (1/0.25 = 4)
LIDAR_MAX_RANGE = 20.0       # m — datasheet: >=40m @ 70% reflectivity, >=20m @ 10% (conservative/worst-case)
LIDAR_RANGE_ACCURACY = 0.02  # m, +/-2cm range accuracy -> use as noise std dev
LIDAR_SCAN_HZ = 20           # rotation frequency at 0.25 deg resolution (options: 20/25/30 Hz)



# ---------------- geometry knobs ----------------
CARGO_OFFSET = 0.08      # m relative to axle (+ = ahead/stable, - = behind/sway)
CARGO_MASS = 3.5         # kg
CARGO_HALF = (0.050, 0.048, 0.035)   # payload box half-sizes (fits the rails)

TRAILER_TIRE_MU = 0.9
HITCH_DAMPING = 0.002

# ---------------- aerodynamics ----------------
# MuJoCo's built-in fluid model: every body gets quadratic drag + viscous
# damping computed from its equivalent-inertia box, relative to WIND.
AIR_DENSITY = 1.204      # kg/m^3 (sea-level air; set 0.0 for vacuum = no drag; set 1.204 for drag)
AIR_VISCOSITY = 1.8e-5   # Pa*s   (air; set 0 for no damping; set 1.8e-5 for air damping)
WIND = (0.0, 0.0, 0.0)   # m/s ambient wind, world frame. e.g. (0, -2, 0) is
                         # a steady 2 m/s crosswind from the left - a
                         # continuous alternative to the impulsive "gust"

# ---------------- experiment knobs ----------------
# rad/s wheel target; cruise speed is SPEED_CTRL * WHEEL_RADIUS.
# 88 rad/s -> 4.0 m/s. Keep this in step with DISTURB_START: a step steer
# of delta rad pulls V^2 * delta / WHEELBASE of lateral acceleration, and
# the tires can only supply TRAILER_TIRE_MU * g ~ 8.8 m/s^2. At 4.0 m/s the
# grip-limited steer angle is 0.159 rad; at the old 160 rad/s (7.28 m/s) it
# was 0.048 rad, so the 0.15 rad swerve below demanded 2.8 g and spun the
# car into the wall before the controller ever engaged.
SPEED_CTRL = 88.0
DISTURBANCE = "swerve"   # "swerve" or "gust"

N_RUNS = 3               # how many simulations to run
DISTURB_START = 0.10     # first-run magnitude: rad (swerve) or N (gust)
DISTURB_STEP = 0.00      # added to the magnitude after every run
# m added to CARGO_OFFSET after every run. -0.06 sweeps +0.08 -> +0.02 ->
# -0.04, i.e. from a solidly nose-heavy trailer through neutral tongue
# load to a negative one (sway-prone) while the car's rear tires still
# carry load. Anything past about -0.06 lifts the car's rear axle clean
# off the ground, which no steering controller can recover from - the
# per-run printout below warns when a loading gets that far.
CARGO_OFFSET_STEP = -0.06
CARGO_MASS_STEP = 0.0    # kg added to CARGO_MASS after every run

GUST_TIME = 0.5          # s
SETTLE_TIME = 1.0        # s
SPINUP_TIME = 5.0        # s
MAX_RECORD = 30.0        # s
REALTIME = True
# ---------------------------------------------------

# fixed trailer constants (from the CAD / rc-truck-trailer.xml)
FRAME_MASS = 0.8         # kg, trailer frame (explicit inertial in the XML) TODO change to 1 to make more accurate
FRAME_COM_X = -0.24      # m, frame COM behind the hitch
TRAILER_WHEEL_MASS = 0.05
HITCH_TO_AXLE = 0.2475   # m, hitch pivot -> axle line (FIXED by the CAD)
DECK_X_MIN, DECK_X_MAX = -0.398, -0.088   # deck extent behind the hitch
DECK_TOP_Z = -0.040      # deck top surface in the trailer (hitch) frame
TRAILER_WHEEL_R = 0.037  # 74 mm diameter wheels
TRAILER_WHEEL_HW = 0.013
TRAILER_TRACK_Y = 0.113
G = 9.81
WHEEL_RADIUS = 0.0455    # car tire radius, for speed dead reckoning
MAX_STEER = 0.61         # rad

HITCH_LIMIT_DEG = 45.0   # hitch joint range from the XML
HITCH_HIT_MARGIN = 1.0   # deg, within this of the limit counts as contact
FLIP_ROLL_DEG = 60.0     # deg of car roll that counts as flipped sideways

# ------------------------MODE SETTINGS---------------------------
PD_WALL_FOLLOW_MODE = True  # True: PD wall following controller, False: LQR controller
PIVOT_MODE = False        # True: controller OFF, car towed by a tow point. OVERRIDES PD MODE AND LQR MODE
PLANAR_MODE = False      # True: no vertical-axis motion at all. No fore/aft


# ---------------- pivot (towed oscillation) mode ----------------
TOW_EYE = (0.1439, 0.0, -0.0076)   # car frame: center of the front axle
CAR_Z0 = 0.0531          # car body height at qpos0 (from the XML)
PIVOT_SPEED = SPEED_CTRL * WHEEL_RADIUS   # m/s tow speed (customizable)
PIVOT_SPRING_K = 25.0    # N/m, lateral spring pulling the tow point to y=0
PIVOT_SPRING_C = 8.0     # N*s/m, lateral damping on the tow point

# ---------------- planar / no-weight-shift mode ----------------
                         # or side-to-side weight shift, constant normal
                         # forces, same friction coefficient on every tire.
PLANAR_TIRE_MU = 1.0     # friction coefficient applied to ALL six tires

TRUCK_TIRES = ["fl_tire", "fr_tire", "rl_tire", "rr_tire"]
TRAILER_TIRES = ["tl_tire", "tr_tire"]

SENSOR_NAMES = {
    "car_accel": "car_imu_accel",
    "car_gyro": "car_imu_gyro",
    "car_quat": "car_imu_quat",
    "trailer_accel": "trailer_imu_accel",
    "trailer_gyro": "trailer_imu_gyro",
    "trailer_quat": "trailer_imu_quat",
    "hitch_quat": "hitch_angle",
}

WHEELBASE = 0.288        # m, car wheelbase
CRUISE_SPEED = SPEED_CTRL * WHEEL_RADIUS   # m/s, design speed for the LQR

# rad/s, how fast the steering servo can actually move (a digital RC servo
# doing 60 deg in ~0.08 s). Keeps the commanded angle continuous; without
# it the tire-load traces show a comb at LIDAR_SCAN_HZ and its harmonics
# that is purely command discontinuity, not vehicle dynamics.
STEER_RATE_MAX = 13.0
# NOTE: apart from the servo rate limit, control_function() is an
# instantaneous full-state LQR. No reaction delay, no deadband - don't
# describe its results as including human-driver effects.


#GAINS
# lqr gains
LQR_K = np.array([0.8742, 0.5])   # [y_error, heading_error] -> steering_angle
#standard wall following gains
PD_KP = 1.2
PD_KD = 0.2

# NOTE: this K was designed on a 2-state [lane offset, heading] model of the
# CAR ONLY (BWSI racecar wall-following lab). It has no hitch/trailer term,
# so it stabilizes the car's lane tracking but does NOT actively damp
# trailer sway - it only helps indirectly, by keeping the tow vehicle's
# path smoother and its steering less abrupt than the human-driver model.
# Kept here for A/B comparison against LQR_K_SWAY below.

CONTROLLER_MODE = "sway"   # "lane" = original 2-state car-only LQR above
                           # "sway" = full 6-state car+trailer LQR below

# Full car+trailer LQR gain, x = [v1, r1, r2, theta, psi1, y]:
#   v1    - car lateral velocity (body frame)           [m/s]
#   r1    - car yaw rate                                [rad/s]
#   r2    - trailer yaw rate                            [rad/s]
#   theta - articulation angle, psi1 - psi2             [rad]
#   psi1  - car heading error relative to the lane      [rad]
#   y     - car lateral offset from lane center         [m]
# steering_angle = -LQR_K_SWAY @ x
#
# Solved here rather than pasted in as a literal, so it always matches the
# rig constants above. Designed at the *worst* loading the run sweep
# reaches (the most rearward CARGO_OFFSET), because that is the only one
# whose open loop is actually unstable and a gain designed on the neutral
# loading does not stabilise it. See trailer_model.py.
LQR_DESIGN_OFFSET = CARGO_OFFSET + (N_RUNS - 1) * CARGO_OFFSET_STEP


def _design_params(cargo_offset, cargo_mass):
    return trailer_model.rig_params(
        cargo_offset, cargo_mass,
        frame_mass=FRAME_MASS, frame_com_x=FRAME_COM_X,
        wheel_mass=TRAILER_WHEEL_MASS, hitch_to_axle=HITCH_TO_AXLE)


LQR_K_SWAY = trailer_model.design_lqr(
    CRUISE_SPEED, _design_params(LQR_DESIGN_OFFSET, CARGO_MASS),
    Q=np.diag([0.0, 0.0, 1.0, 300.0, 50.0, 25.0]), R=20.0)


def get_wall_line(ranges, angles, start_idx, end_idx):
    """Fit a line y = m*x + b to a slice of a LiDAR scan, in the site frame.
    Returns (slope, intercept), or None if too few rays hit anything."""
    r = ranges[start_idx:end_idx]
    a = angles[start_idx:end_idx]

    # Filter out max-range/invalid hits
    valid = r < (LIDAR_MAX_RANGE - 0.5)
    if np.sum(valid) < 5:
        return None

    x = r[valid] * np.cos(a[valid])   # along the site's x axis
    y = r[valid] * np.sin(a[valid])   # perpendicular to it

    # m = wall angle relative to the car, b = perpendicular distance to it
    m, b = np.polyfit(x, y, 1)
    return m, b


def get_heading_position_wall(ranges, angles):
    """Car heading error (rad) and lane offset (m) from the two walls.

    1080 rays spanning -135..+135 deg of the LiDAR site frame; index 540 is
    the site's own +x. The site carries euler="0 0 180", so that direction
    points AFT along the car and the two windows below actually straddle the
    rear quarters. Both the x and the y axis flip with that 180 deg rotation,
    which leaves the fitted slope unchanged and swaps the two walls, so the
    "left"/"right" naming and both output signs still come out in the car's
    own frame: +heading = nose toward +y, +offset = car left of center.
    (Verified against ground truth over +-0.2 m and +-10 deg.)
    """
    # -100..-20 deg (indices 140..460) and +20..+100 deg (620..940). The
    # +-20 deg gap around index 540 keeps the trailer out of the fit.
    right_line = get_wall_line(ranges, angles, 140, 460)
    left_line = get_wall_line(ranges, angles, 620, 940)

    if right_line is None or left_line is None:
        return 0.0, 0.0  # Fallback if walls are lost

    m_right, b_right = right_line
    m_left, b_left = left_line

    # Heading error is the average angle of the walls
    heading_error = math.atan((m_right + m_left) / 2.0)

    # Track center offset: average of left wall (+) and right wall (-) offsets
    y_error = (b_left + b_right) / 2.0
    return -heading_error, y_error


def get_hitch_yaw(hitch_sensor):
    """Articulation angle theta = psi1 - psi2 (car heading minus trailer
    heading), radians. Reads the ball-joint quaternion in normal mode and
    the hinge angle in PLANAR_MODE.

    Both sensors report the trailer *relative to the car*, i.e. psi2 - psi1,
    so the sign is flipped to match the state vector the LQR was designed
    on (trailer_model.py, where theta_dot = r1 - r2).
    """
    if len(hitch_sensor) == 1:                     # jointpos, PLANAR_MODE
        return -float(hitch_sensor[0])
    hw, hx, hy, hz = hitch_sensor                  # ballquat
    return -math.atan2(2 * (hw * hz + hx * hy), 1 - 2 * (hy * hy + hz * hz))


# Leak time constant on the lateral-velocity integrator: long compared with
# the ~1 s sway transient, short enough to bleed off accelerometer bias
# instead of letting it drift over the whole MAX_RECORD window.
V_EST_LEAK_TAU = 2.0  # s


def control_function(sensors, dt, state, model, data, site_id, car_id):
    # 1. Initialize state variables on the first run
    if "step_count" not in state:
        state["step_count"] = 0
        state["steer_target"] = 0.0   # newest LQR command
        state["last_steer"] = 0.0     # what the servo has reached
        state["last_speed"] = SPEED_CTRL
        state["v_est"] = 0.0     # car lateral velocity estimate [m/s]
        state["heading_err"] = 0.0
        state["y_err"] = 0.0
        state["scan"] = None     # newest (angles, ranges), for logging
        # How many physics steps make up one LiDAR scan (50 at 20 Hz, 1 kHz)
        state["lidar_interval"] = max(1, round(1 / (LIDAR_SCAN_HZ * dt)))

    # r1 (car yaw rate) and r2 (trailer yaw rate) are both directly measured
    # every step - gyro z-component in each body's own frame.
    r1 = float(sensors["car_gyro"][2])
    r2 = float(sensors["trailer_gyro"][2])

    # theta is directly measured every step too - this rig has a real hitch
    # sensor, so no estimator is needed for it.
    theta = get_hitch_yaw(sensors["hitch_quat"])

    # v1 has no direct sensor, so integrate the car IMU's body-frame lateral
    # accelerometer. That accelerometer measures specific force, which in a
    # turn is v1_dot + V*r1, so the centripetal term has to be subtracted
    # before integrating or the estimate just tracks yaw rate. The leak
    # keeps residual bias from accumulating.
    state["v_est"] += (float(sensors["car_accel"][1])
                       - CRUISE_SPEED * r1) * dt
    state["v_est"] *= (1.0 - dt / V_EST_LEAK_TAU)

    # 2. Only the expensive LiDAR scan runs at LIDAR_SCAN_HZ. It is the sole
    #    source of psi1 and y, so those two states are held between scans.
    if state["step_count"] % state["lidar_interval"] == 0:
        angles, ranges = simulate_lidar(model, data, site_id, car_id)
        state["scan"] = (angles, ranges)
        state["heading_err"], state["y_err"] = get_heading_position_wall(
            ranges, angles)

    # 3. Control law, recomputed every physics step. v1/r1/r2/theta come
    #    from the IMUs and the hitch sensor at the full rate, so holding the
    #    whole command at the LiDAR rate would throw away the fast states
    #    the sway damping depends on - and the resulting stair-step command
    #    costs enough phase margin to destabilise the rearmost loading.
    if CONTROLLER_MODE == "sway":
        x_state = np.array([state["v_est"], r1, r2, theta,
                            state["heading_err"], state["y_err"]])
        steering_angle = -float(LQR_K_SWAY @ x_state)
    else:
        x_err = np.array([state["y_err"], state["heading_err"]])
        steering_angle = -float(LQR_K @ x_err)

    # +0.61 rad is full left, -0.61 rad full right, 0 straight
    state["steer_target"] = max(-MAX_STEER, min(MAX_STEER, steering_angle))

    # 4. Slew the servo toward the target at its real rate limit.
    step = STEER_RATE_MAX * dt
    error = state["steer_target"] - state["last_steer"]
    state["last_steer"] += max(-step, min(step, error))

    state["step_count"] += 1
    return state["last_steer"], state["last_speed"]


def pd_control_function(sensors, dt, state, model, data, site_id, car_id):
    # 1. Initialize state variables on the first run
    if "step_count" not in state:
        state["step_count"] = 0
        state["steer_target"] = 0.0   # newest command
        state["last_steer"] = 0.0     # what the servo has reached
        state["last_speed"] = SPEED_CTRL
        state["heading_err"] = 0.0
        state["y_err"] = 0.0
        state["last_y_err"] = 0.0     # Added to track previous error for the derivative
        state["scan"] = None          # newest (angles, ranges), for logging
        
        # How many physics steps make up one LiDAR scan (50 at 20 Hz, 1 kHz)
        state["lidar_interval"] = max(1, round(1 / (LIDAR_SCAN_HZ * dt)))

    # 2. Update LIDAR at the specified frequency
    if state["step_count"] % state["lidar_interval"] == 0:
        angles, ranges = simulate_lidar(model, data, site_id, car_id)
        state["scan"] = (angles, ranges)
        
        # Save previous error for the derivative term
        state["last_y_err"] = state["y_err"]
        
        # Extract the new lateral offset (y_err) and heading error from the walls
        state["heading_err"], state["y_err"] = get_heading_position_wall(ranges, angles)

    # 3. Control law, recomputed every physics step.
    # We calculate the derivative using the time elapsed between LIDAR updates.
    pd_dt = state["lidar_interval"] * dt
    
    # Calculate the rate of change of the error (derivative)
    derivative = (state["y_err"] - state["last_y_err"]) / pd_dt
    
    # Apply the PD formula
    # A positive y_err means the car is left of center, requiring a negative steering angle (right).
    steering_angle = -(PD_KP * state["y_err"] + PD_KD * derivative)

    # Cap the steering angle to the physical limits of the vehicle (+/- 0.61 rad)
    state["steer_target"] = max(-MAX_STEER, min(MAX_STEER, steering_angle))

    # 4. Slew the servo toward the target at its real rate limit.
    step = STEER_RATE_MAX * dt
    error = state["steer_target"] - state["last_steer"]
    state["last_steer"] += max(-step, min(step, error))

    state["step_count"] += 1
    return state["last_steer"], state["last_speed"]

# =====================================================================
# >>> END CONTROLLER <<<
# =====================================================================

# =====================================================================
# >>> END CONTROLLER <<<
# =====================================================================

def parallel_axis(I_own, m, com, target_com):
    d = com - target_com
    return I_own + m * np.array([
        d[1] ** 2 + d[2] ** 2,
        d[0] ** 2 + d[2] ** 2,
        d[0] ** 2 + d[1] ** 2,
    ])

def trailer_xml(cargo_offset, cargo_mass):
    """Generate the CAD-trailer subtree for the given payload placement."""
    ax = -HITCH_TO_AXLE                    # axle x in the hitch frame
    cx = ax + cargo_offset                 # payload center x
    cz = DECK_TOP_Z + CARGO_HALF[2]        # payload sits on the deck

    lo = DECK_X_MIN + CARGO_HALF[0] - ax   # payload must stay on the deck
    hi = DECK_X_MAX - CARGO_HALF[0] - ax
    if not lo <= cargo_offset <= hi:
        raise ValueError(
            f"cargo offset must be within {lo:+.3f} .. {hi:+.3f} m of the axle")

    total = FRAME_MASS + 2 * TRAILER_WHEEL_MASS + cargo_mass
    moment = (FRAME_MASS * FRAME_COM_X
              + 2 * TRAILER_WHEEL_MASS * ax
              + cargo_mass * cx)
    axle_load = G * moment / ax
    tongue_load = G * total - axle_load
    # the tongue hangs HITCH_H behind the car's rear axle, so a negative
    # tongue load levers weight off that axle - and once it reaches zero
    # the driven wheels leave the ground and the car is unrecoverable
    rear_load = (trailer_model.CAR_MASS * G * trailer_model.CAR_A / WHEELBASE
                 + tongue_load * (WHEELBASE + trailer_model.HITCH_H) / WHEELBASE)
    print(f"Axle at x = {ax:.3f} m | payload at x = {cx:.3f} m "
          f"({cargo_offset:+.3f} m from axle), {cargo_mass:.2f} kg")
    print(f"Static tongue load = {1000 * tongue_load / G:.0f} g "
          f"({100 * tongue_load / (G * total):.0f}% of trailer weight)"
          + ("  << NEGATIVE: sway-prone!" if tongue_load < 0 else ""))
    print(f"Car rear axle load = {rear_load:.1f} N"
          + ("  << REAR WHEELS LIFTING: no controller can recover this"
             if rear_load <= 2.0 else ""))
    # ---- combined frame+cargo inertial (parallel-axis theorem) ----
    frame_com = np.array([FRAME_COM_X, 0.0, -0.045])
    frame_I = np.array([0.0035, 0.012, 0.014])

    cargo_com = np.array([cx, 0.0, cz])
    hx, hy, hz = CARGO_HALF
    # solid-box inertia about its own COM
    cargo_I_own = cargo_mass / 3.0 * np.array([
        hy ** 2 + hz ** 2,
        hx ** 2 + hz ** 2,
        hx ** 2 + hy ** 2,
    ])

    body_mass = FRAME_MASS + cargo_mass
    body_com = (FRAME_MASS * frame_com + cargo_mass * cargo_com) / body_mass
    body_I = (parallel_axis(frame_I, FRAME_MASS, frame_com, body_com)
          + parallel_axis(cargo_I_own, cargo_mass, cargo_com, body_com))

    if PLANAR_MODE:
        hitch = (f'<joint name="hitch" type="hinge" axis="0 0 1" '
                 f'limited="true" range="-{HITCH_LIMIT_DEG:.0f} '
                 f'{HITCH_LIMIT_DEG:.0f}" damping="{HITCH_DAMPING}"/>')
        mu = PLANAR_TIRE_MU
        # Planar mode forbids pitch, but this trailer normally pitches
        # ~4.2 deg nose-up to bring its (undersized vs the CAD) 74 mm
        # wheels onto the ground. With a yaw-only hinge they would float
        # 18 mm up and carry no load, so drop the axle to ground height:
        # hitch is at CAR_Z0 + 0.065, so wheel center rel z = r - that.
        wheel_z = TRAILER_WHEEL_R - (CAR_Z0 + 0.065)
    else:
        hitch = (f'<joint name="hitch" type="ball" limited="true" '
                 f'range="0 {HITCH_LIMIT_DEG:.0f}" '
                 f'damping="{HITCH_DAMPING}"/>')
        mu = TRAILER_TIRE_MU
        wheel_z = -0.063   # CAD axle line; ball hitch lets the trailer pitch

    return f"""<!-- TRAILER_START -->
      <body name="trailer" pos="-0.1857 0 0.065">
        {hitch}
        <inertial pos="{body_com[0]:.5f} {body_com[1]:.5f} {body_com[2]:.5f}" mass="{body_mass:.5f}" diaginertia="{body_I[0]:.6f} {body_I[1]:.6f} {body_I[2]:.6f}"/>
        <site name="hitch_force_site" pos="0 0 0" size="0.008" rgba="0 0 1 0.5"/>
        <site name="imu_trailer" pos="-0.25 0 -0.03" size="0.008" rgba="1 0 0 0.5"/>
        <geom name="trailer_frame_vis" class="visual" type="mesh" mesh="trailer_frame" material="trailer_mat"/>
        <geom name="trailer_deck_col" type="box" size="0.155 0.100 0.0125" pos="-0.243 0 -0.0525" mass="0" group="3"/>
        <geom name="trailer_tongue_col" type="capsule" size="0.008" fromto="0.005 0 -0.005  -0.100 0 -0.050" mass="0" contype="0" conaffinity="0"/>
        <geom name="cargo" type="box" size="{CARGO_HALF[0]} {CARGO_HALF[1]} {CARGO_HALF[2]}" pos="{cx:.4f} 0 {cz:.4f}" mass="0" material="payload_mat"/>
        <body name="tl_wheel" pos="{ax:.4f} {TRAILER_TRACK_Y} {wheel_z:.4f}">
          <joint name="tl_hinge" class="spin"/>
          <inertial pos="0 0 0" mass="{TRAILER_WHEEL_MASS}" diaginertia="2.0e-5 3.4e-5 2.0e-5"/>
          <geom name="tl_tire" class="tire" size="{TRAILER_WHEEL_R} {TRAILER_WHEEL_HW}" zaxis="0 1 0" friction="{mu} 0.005 0.0001"/>
        </body>
        <body name="tr_wheel" pos="{ax:.4f} -{TRAILER_TRACK_Y} {wheel_z:.4f}">
          <joint name="tr_hinge" class="spin"/>
          <inertial pos="0 0 0" mass="{TRAILER_WHEEL_MASS}" diaginertia="2.0e-5 3.4e-5 2.0e-5"/>
          <geom name="tr_tire" class="tire" size="{TRAILER_WHEEL_R} {TRAILER_WHEEL_HW}" zaxis="0 1 0" friction="{mu} 0.005 0.0001"/>
        </body>
      </body>
      <!-- TRAILER_END -->"""


def build_model(cargo_offset, cargo_mass):
    xml = open(XML_PATH).read()
    start = xml.index("<!-- TRAILER_START -->")
    end = xml.index("<!-- TRAILER_END -->") + len("<!-- TRAILER_END -->")
    xml = xml[:start] + trailer_xml(cargo_offset, cargo_mass) + xml[end:]

    # aerodynamics: override the fluid attributes on the <option> tag with
    # the knobs above (works regardless of the values written in the XML)
    # FIX #3: assert a substitution actually happened instead of silently
    # leaving the XML's original value in place if the attribute pattern
    # ever stops matching.
    xml, n = re.subn(r'density="[^"]*"', f'density="{AIR_DENSITY}"', xml, count=1)
    assert n == 1, "density=\"...\" attribute not found on <option> tag"
    xml, n = re.subn(r'viscosity="[^"]*"', f'viscosity="{AIR_VISCOSITY}"', xml, count=1)
    assert n == 1, "viscosity=\"...\" attribute not found on <option> tag"
    xml, n = re.subn(r'wind="[^"]*"',
                      f'wind="{WIND[0]} {WIND[1]} {WIND[2]}"', xml, count=1)
    assert n == 1, "wind=\"...\" attribute not found on <option> tag"

    # from_xml_string resolves meshdir against the CWD, not the XML file:
    # make it absolute so the script works from anywhere
    assets_abs = os.path.join(
        os.path.dirname(os.path.abspath(XML_PATH)), "assets")
    assert 'meshdir="assets"' in xml, "meshdir=\"assets\" not found in XML"
    xml = xml.replace('meshdir="assets"', f'meshdir="{assets_abs}"')

    # force sensor at the hitch: lateral force the trailer puts on the car
    assert "</sensor>" in xml, "</sensor> closing tag not found in XML"
    xml = xml.replace(
        "</sensor>",
        '  <force name="hitch_force" site="hitch_force_site"/>\n  </sensor>')

    if PIVOT_MODE:
        # single tow point, starting exactly at the front-axle center
        # (tow eye). Slides: x = driven forward, y = spring+damper to lane
        # center, z = free. MuJoCo needs positive mass on jointed bodies;
        # 20 g vs a 5.5 kg car is effectively massless.
        eye = (TOW_EYE[0], TOW_EYE[1], CAR_Z0 + TOW_EYE[2])   # world at qpos0
        leader = f"""<body name="leader" pos="{eye[0]:.4f} {eye[1]:.4f} {eye[2]:.4f}">
      <joint name="leader_x" type="slide" axis="1 0 0"/>
      <joint name="leader_y" type="slide" axis="0 1 0" stiffness="{PIVOT_SPRING_K}" damping="{PIVOT_SPRING_C}"/>
      <joint name="leader_z" type="slide" axis="0 0 1"/>
      <geom name="leader_marker" type="sphere" size="0.015" mass="0.02"
            contype="0" conaffinity="0" rgba="0 1 0 0.8"/>
    </body>
    <!-- CAR_ROOT_START -->"""
        assert "<!-- CAR_ROOT_START -->" in xml, "CAR_ROOT_START marker not found in XML"
        xml = xml.replace("<!-- CAR_ROOT_START -->", leader, 1)

        # ball-type attachment: tow point and front-axle center coincide,
        # car rotation about the point is completely free (pitch/yaw/roll)
        eq = """<equality>
    <connect name="tow_ball" body1="leader" body2="car" anchor="0 0 0"/>
  </equality>
  <actuator>"""
        assert "<actuator>" in xml, "<actuator> tag not found in XML"
        xml = xml.replace("<actuator>", eq, 1)
        assert "</actuator>" in xml, "</actuator> tag not found in XML"
        xml = xml.replace("</actuator>",
            '  <velocity name="leader_drive" joint="leader_x" kv="200" '
            'ctrlrange="0 100"/>\n  </actuator>')

    if PLANAR_MODE:
        # car root: x / y / yaw only -> no heave, pitch, or roll anywhere
        assert "<!-- CAR_ROOT_START -->" in xml, "CAR_ROOT_START marker not found in XML"
        assert "<!-- CAR_ROOT_END -->" in xml, "CAR_ROOT_END marker not found in XML"
        rs = xml.index("<!-- CAR_ROOT_START -->")
        re_ = xml.index("<!-- CAR_ROOT_END -->") + len("<!-- CAR_ROOT_END -->")
        planar_root = f"""<body name="car" pos="0 0 {CAR_Z0}">
      <joint name="car_x" type="slide" axis="1 0 0"/>
      <joint name="car_y" type="slide" axis="0 1 0"/>
      <joint name="car_yaw" type="hinge" axis="0 0 1"/>"""
        xml = xml[:rs] + planar_root + xml[re_:]
        # hitch is a hinge in planar mode: swap the ball sensor for jointpos
        assert '<ballquat name="hitch_angle" joint="hitch"/>' in xml, \
            "ballquat hitch_angle sensor not found in XML"
        xml = xml.replace('<ballquat name="hitch_angle" joint="hitch"/>',
                          '<jointpos name="hitch_angle" joint="hitch"/>')
        # one common friction coefficient on every tire: this string lives
        # in the "tire" default class in the XML, so one replace covers all
        # four car tires (the trailer tires already got PLANAR_TIRE_MU
        # explicitly in trailer_xml)
        tire_friction = f'friction="{TRAILER_TIRE_MU} 0.005 0.0001"'
        assert tire_friction in xml, (
            f'default tire {tire_friction} not found in XML — '
            'PLANAR_TIRE_MU would silently NOT be applied to the car tires')
        xml = xml.replace(tire_friction,
                          f'friction="{PLANAR_TIRE_MU} 0.005 0.0001"')
        # NOTE: unlike the full-size model, the car COM is already centered
        # between the axles in the XML, so no inertial patch is needed here.

    return mujoco.MjModel.from_xml_string(xml)


def quat_to_euler(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def read_sensors(model, data):
    out = {}
    for key, name in SENSOR_NAMES.items():
        s = model.sensor(name)
        adr, dim = s.adr[0], s.dim[0]
        out[key] = data.sensordata[adr:adr + dim].copy()
    return out


def simulate_lidar(model, data, site_id, exclude_body_id,
                    fov_deg=LIDAR_FOV_DEG, pts_per_deg=LIDAR_PTS_PER_DEG,
                    max_range=LIDAR_MAX_RANGE):
    """Horizontal fan of rays from the lidar site. Angles are in the site
    frame; the site carries euler="0 0 180" in the XML, so angle 0 points
    aft along the car. Returns (angles [rad], ranges [m])."""
    n_rays = int(fov_deg * pts_per_deg)
    angles = np.linspace(-fov_deg / 2, fov_deg / 2, n_rays) * np.pi / 180

    origin = data.site_xpos[site_id].copy()
    xmat = data.site_xmat[site_id].reshape(3, 3)

    ranges = np.full(n_rays, max_range)
    geomid = np.zeros(1, dtype=np.int32)
    for i, a in enumerate(angles):
        direction = xmat @ np.array([np.cos(a), np.sin(a), 0.0])
        dist = mujoco.mj_ray(model, data, origin, direction,
                              None, 1, exclude_body_id, geomid)
        if dist >= 0:
            ranges[i] = min(dist, max_range)
    if LIDAR_RANGE_ACCURACY:
        hit = ranges < max_range
        ranges[hit] += np.random.normal(0.0, LIDAR_RANGE_ACCURACY, hit.sum())
    return angles, ranges

def tire_normal_loads(model, data, tire_ids, floor_id):
    """Total ground contact normal force per tire geom [N].
    Grip capacity is proportional to mu * normal load."""
    loads = {gid: 0.0 for gid in tire_ids.values()}
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        other = None
        if c.geom1 == floor_id:
            other = c.geom2
        elif c.geom2 == floor_id:
            other = c.geom1
        if other in loads:
            mujoco.mj_contactForce(model, data, i, f)
            loads[other] += f[0]
    return {name: loads[gid] for name, gid in tire_ids.items()}


def mode_tag():
    tag = "pivot" if PIVOT_MODE else "drive"
    if PLANAR_MODE:
        tag += "_planar"
    return tag


def next_free_path(template):
    """First path of the form template.format(simNum=n) that doesn't exist,
    so repeated runs of the same configuration don't overwrite each other."""
    n = 1
    while os.path.exists(template.format(simNum=n)):
        n += 1
    return template.format(simNum=n)


def run_simulation(magnitude, cargo_offset, cargo_mass, run_idx):
    model = build_model(cargo_offset, cargo_mass)
    # fix the clipping making it impossible to view
    model.stat.extent = 5.0
    model.vis.map.znear = 0.01
    data = mujoco.MjData(model)

    drive_ids = [model.actuator(n).id for n in
                 ("drive_left", "drive_right",
                  "drive_left_front", "drive_right_front")]
    sl = model.actuator("steer_left").id
    sr = model.actuator("steer_right").id
    trailer_id = model.body("trailer").id
    hitch_joint = model.joint("hitch")
    hitch_adr = hitch_joint.qposadr[0]
    hitch_is_ball = int(hitch_joint.type[0]) == mujoco.mjtJoint.mjJNT_BALL
    hf_adr = model.sensor("hitch_force").adr[0]
    dt = model.opt.timestep
    lidar_save_every = max(1, round(1 / (LIDAR_SCAN_HZ * dt)))  # 50 steps at 20 Hz, dt=1 ms
    trailer_rear_x = DECK_X_MIN            # gust acts on the trailer tail

    pv = None
    if PIVOT_MODE:
        pv = model.actuator("leader_drive").id
        # freewheel the drive wheels: the tow point pulls, not the wheels
        for a in drive_ids:
            model.actuator_gainprm[a, 0] = 0
            model.actuator_biasprm[a, 2] = 0

    tire_ids = {n: model.geom(n).id for n in TRUCK_TIRES + TRAILER_TIRES}
    floor_id = model.geom("floor").id

    lidar_site_id = model.site("lidar_site").id
    car_body_id = model.body("car").id
    lidar_scans = []   # list of (time, angles, ranges)

    swerve_steps = int(0.4 / dt)
    if DISTURBANCE == "swerve":
        disturb = [("swerve_r", swerve_steps), ("swerve_l", swerve_steps),
                   ("swerve_c", swerve_steps)]
    else:
        disturb = [("gust", int(GUST_TIME / dt))]
    schedule = ([("settle", int(SETTLE_TIME / dt)),
                 ("spinup", int(SPINUP_TIME / dt))]
                + disturb
                + [("record", int(MAX_RECORD / dt))])

    os.makedirs("csvs", exist_ok=True)
    csv_filename = next_free_path(
        f"csvs/RC_Sway_{mode_tag()}_{DISTURBANCE}_run{run_idx + 1}"
        f"_mag{magnitude:g}_v{SPEED_CTRL:.0f}"
        f"_cargo{cargo_mass:g}kg_off{cargo_offset:+.3f}"
        f"_mu{TRAILER_TIRE_MU}_simNum{{simNum}}.csv")
    headers = ["time", "hitch_yaw_deg", "hitch_lat_force", "trailer_y",
               "car_yaw_deg", "car_roll_deg", "car_y", "tl_grip", "tr_grip",
               "fl_grip", "fr_grip", "rl_grip", "rr_grip", "front_weight",
               "rear_weight", "car_speed"]
    log = {h: [] for h in headers}

    ctrl_state = {}          # persists across steps, passed to control_function
    peak_yaw = 0.0

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            phase_idx, step_in_phase = 0, 0
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("car").id
            viewer.cam.azimuth = 0
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -25
            while viewer.is_running() and phase_idx < len(schedule):
                step_start = time.time()
                phase = schedule[phase_idx][0]

                if PIVOT_MODE:
                    # controller OFF; the tow point pulls the car forward,
                    # its lane-center spring handles everything lateral
                    if phase != "settle":
                        data.ctrl[pv] = PIVOT_SPEED
                    steer = {"swerve_r": magnitude,
                             "swerve_l": -magnitude}.get(phase, 0.0)
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer
                elif phase == "record":
                    # ---- your controller drives steering AND speed ----
                    if PD_WALL_FOLLOW_MODE:
                        steer, speed = pd_control_function(
                        read_sensors(model, data), dt, ctrl_state,
                        model=model, data=data,
                        site_id=lidar_site_id, car_id=car_body_id)
                    else:
                        steer, speed = control_function(
                            read_sensors(model, data), dt, ctrl_state,
                            model=model, data=data,
                            site_id=lidar_site_id, car_id=car_body_id)
                    speed = max(0.0, min(300.0, speed))
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer
                    for a in drive_ids:
                        data.ctrl[a] = speed
                else:
                    if phase == "spinup":
                        ramp = min(1.0, step_in_phase / schedule[phase_idx][1])
                        target = ramp * SPEED_CTRL
                    elif phase == "settle":
                        target = None
                    else:
                        target = SPEED_CTRL
                    if target is not None:
                        for a in drive_ids:
                            data.ctrl[a] = target
                    steer = {"swerve_r": magnitude,
                             "swerve_l": -magnitude}.get(phase, 0.0)
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer

                if phase == "gust":
                    rear = data.body("trailer").xpos + data.body(
                        "trailer").xmat.reshape(3, 3) @ np.array(
                        [trailer_rear_x, 0, 0])
                    qfrc = np.zeros(model.nv)
                    mujoco.mj_applyFT(model, data, np.array([0, magnitude, 0]),
                                      np.zeros(3), rear, trailer_id, qfrc)
                    data.qfrc_applied[:] = qfrc
                else:
                    data.qfrc_applied[:] = 0

                mujoco.mj_step(model, data)

                # log the scan the controller just used, rather than firing
                # another 1080 rays for the same instant
                if (SIMULATE_LIDAR and phase == "record"
                        and step_in_phase % lidar_save_every == 0
                        and ctrl_state.get("scan") is not None):
                    lidar_scans.append((data.time, ctrl_state["scan"][1]))

                if phase not in ("settle", "spinup"):
                    if hitch_is_ball:
                        q = data.qpos[hitch_adr:hitch_adr + 4]
                        _, _, hitch_yaw = quat_to_euler(*q)
                    else:
                        hitch_yaw = math.degrees(data.qpos[hitch_adr])
                    peak_yaw = max(peak_yaw, abs(hitch_yaw))
                    cw, cx_, cy_, cz_ = data.body("car").xquat
                    car_roll, _, car_yaw = quat_to_euler(cw, cx_, cy_, cz_)
                    grips = tire_normal_loads(model, data, tire_ids, floor_id)

                    # hitch force: sensor reports the force from the car on
                    # the trailer, in the sensor-site (trailer) frame. Rotate
                    # to world, take the lane-perpendicular (y) component,
                    # and negate -> force the trailer pushes on the car.
                    # Positive = trailer shoving the car toward +y (left).
                    f_local = data.sensordata[hf_adr:hf_adr + 3]
                    f_world = data.site(
                        "hitch_force_site").xmat.reshape(3, 3) @ f_local
                    hitch_lat = -float(f_world[1])

                    row = {
                        "time": data.time,
                        "hitch_yaw_deg": hitch_yaw,
                        "hitch_lat_force": hitch_lat,
                        "trailer_y": data.body("trailer").xipos[1],
                        "car_yaw_deg": car_yaw,
                        "car_roll_deg": car_roll,
                        "car_y": data.body("car").xipos[1],
                        "tl_grip": grips["tl_tire"],
                        "tr_grip": grips["tr_tire"],
                        "fl_grip": grips["fl_tire"],
                        "fr_grip": grips["fr_tire"],
                        "rl_grip": grips["rl_tire"],
                        "rr_grip": grips["rr_tire"],
                        "front_weight": grips["fl_tire"] + grips["fr_tire"],
                        "rear_weight": grips["rl_tire"] + grips["rr_tire"],
                        "car_speed": float(np.linalg.norm(
                            data.body("car").cvel[3:6])),
                    }
                    writer.writerow([row[h] for h in headers])
                    for h in headers:
                        log[h].append(row[h])

                step_in_phase += 1
                if step_in_phase >= schedule[phase_idx][1]:
                    phase_idx += 1
                    step_in_phase = 0
                    if (phase_idx < len(schedule)
                            and schedule[phase_idx][0] == "record"):
                        print(f"Disturbance done at t={data.time:.2f}s. "
                              + ("Towed, no controller. Recording..."
                                 if PIVOT_MODE else
                                 "Controller engaged, recording..."))

                viewer.sync()
                if REALTIME:
                    leftover = dt - (time.time() - step_start)
                    if leftover > 0:
                        time.sleep(leftover)

    final_speed = float(np.linalg.norm(data.body("car").cvel[3:6]))
    outcome = ("LOST CONTROL (sway divergence / jackknife)"
               if final_speed < 1.0 else "recovered")
    print(f"Run {run_idx + 1} ({DISTURBANCE} mag {magnitude:g}, "
          f"cargo {cargo_mass:g} kg at {cargo_offset:+.3f} m): {outcome}. "
          f"Peak |hitch yaw| = {peak_yaw:.1f} deg, "
          f"final speed = {final_speed:.2f} m/s")
    print(f"CSV saved at {csv_filename}")
    log = {h: np.array(v) for h, v in log.items()}
    log["magnitude"] = magnitude
    log["cargo_offset"] = cargo_offset
    log["cargo_mass"] = cargo_mass
    log["speed_ctrl"] = SPEED_CTRL


    if SIMULATE_LIDAR and lidar_scans:
        os.makedirs("lidar", exist_ok=True)
        times = np.array([s[0] for s in lidar_scans])
        scans = np.array([s[1] for s in lidar_scans])
        np.savez(f"lidar/run{run_idx + 1}_{mode_tag()}_scans.npz",
                times=times, scans=scans, fov_deg=LIDAR_FOV_DEG,
                pts_per_deg=LIDAR_PTS_PER_DEG)
        print(f"Lidar scans saved: {scans.shape[0]} scans, {scans.shape[1]} rays each")


    return log


def mask_spans(t, mask):
    """Convert a boolean mask over time samples into (start, end) spans."""
    spans, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = t[i]
        elif not m and start is not None:
            spans.append((start, t[i]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def plot_results(runs):
    os.makedirs("plots", exist_ok=True)
    tag = mode_tag()
    for i, r in enumerate(runs):
        t = r["time"]
        # blue: trailer at max swing, hitting the hitch stop (= the car)
        hit_mask = np.abs(r["hitch_yaw_deg"]) >= (HITCH_LIMIT_DEG
                                                  - HITCH_HIT_MARGIN)
        # red: car flipped sideways
        flip_mask = np.abs(r["car_roll_deg"]) >= FLIP_ROLL_DEG
        hit_spans = mask_spans(t, hit_mask)
        flip_spans = mask_spans(t, flip_mask)

        fig, ax = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
        fig.suptitle(
            f"Run {i + 1} ({tag}) — {DISTURBANCE} "
            f"magnitude {r['magnitude']:g}, "
            f"cargo {r['cargo_mass']:g} kg @ {r['cargo_offset']:+.3f} m\n"
            "blue = trailer at max swing / contacting car, "
            "red = car flipped sideways")

        ax[0, 0].plot(t, r["hitch_yaw_deg"])
        ax[0, 0].set_ylabel("deg")
        ax[0, 0].set_title("Trailer angle relative to car (hitch yaw)")

        ax[0, 1].plot(t, r["hitch_lat_force"], color="tab:red")
        ax[0, 1].axhline(0, color="gray", lw=0.5)
        ax[0, 1].set_ylabel("N")
        ax[0, 1].set_title("Lateral force trailer pushes on car "
                           "(+ = toward +y / left)")

        ax[1, 0].plot(t, r["car_yaw_deg"])
        ax[1, 0].set_ylabel("deg")
        ax[1, 0].set_title("Car angle relative to road")

        ax[1, 1].plot(t, r["car_y"], color="tab:red")
        ax[1, 1].set_ylabel("m")
        ax[1, 1].set_title("Car displacement from lane center")

        ax[2, 0].plot(t, r["tl_grip"], label="trailer L")
        ax[2, 0].plot(t, r["tr_grip"], label="trailer R")
        ax[2, 0].set_ylabel("N (normal load)")
        ax[2, 0].set_title("Trailer tire grip")
        ax[2, 0].legend()

        for k, lbl in [("fl_grip", "FL"), ("fr_grip", "FR"),
                       ("rl_grip", "RL"), ("rr_grip", "RR")]:
            ax[2, 1].plot(t, r[k], label=lbl)
        ax[2, 1].set_ylabel("N (normal load)")
        ax[2, 1].set_title("Car tire grip")
        ax[2, 1].legend()

        ax[3, 0].plot(t, r["front_weight"], label="front axle")
        ax[3, 0].plot(t, r["rear_weight"], label="rear axle")
        ax[3, 0].set_ylabel("N")
        ax[3, 0].set_xlabel("time [s]")
        ax[3, 0].set_title("Car weight: front vs rear")
        ax[3, 0].legend()

        ax[3, 1].plot(t, r["car_speed"], color="tab:green")
        ax[3, 1].set_ylabel("m/s")
        ax[3, 1].set_xlabel("time [s]")
        ax[3, 1].set_title("Car speed")

        # shade highlight bands across every subplot
        for a in ax.flat:
            for s, e in hit_spans:
                a.axvspan(s, e, color="tab:blue", alpha=0.3, lw=0)
            for s, e in flip_spans:
                a.axvspan(s, e, color="red", alpha=0.3, lw=0)

        fig.tight_layout()
        out = next_free_path(
            f"plots/run{i + 1}_{tag}_{DISTURBANCE}_mag{r['magnitude']:g}"
            f"_speed{r['speed_ctrl']:g}_simNum{{simNum}}.png")
        fig.savefig(out, dpi=120)
        print(f"Plot saved at {out}")
    plt.show()


if __name__ == "__main__":
    results = []
    for i in range(N_RUNS):
        mag = DISTURB_START + i * DISTURB_STEP
        off = CARGO_OFFSET + i * CARGO_OFFSET_STEP
        mass = CARGO_MASS + i * CARGO_MASS_STEP
        print(f"\n=== Run {i + 1}/{N_RUNS} — {DISTURBANCE} mag {mag:g}, "
              f"cargo {mass:g} kg @ {off:+.3f} m ===")
        results.append(run_simulation(mag, off, mass, i))
        time.sleep(.5)
    plot_results(results)