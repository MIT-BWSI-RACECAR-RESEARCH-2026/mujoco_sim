"""
RC car + CAD trailer sway experiment (no suspension).
RC-SCALE port of the full-size block-truck script, for use with
rc-truck-trailer.xml (CAD meshes in ./assets).

- Lane: 0.60 m wide, centered on y=0, road runs along +x. Lane center = y=0.
- AIR RESISTANCE: MuJoCo's fluid model is enabled (AIR_DENSITY /
  AIR_VISCOSITY knobs). Every body feels quadratic drag + viscous damping
  based on its equivalent-inertia box, measured relative to the ambient
  WIND vector - so WIND=(0,-2,0) gives a steady 2 m/s crosswind on top of
  the impulsive swerve/gust disturbances. AIR_DENSITY=0 restores vacuum.
- IMUs on car and trailer (accel, gyro, orientation quat) + hitch sensor
  (trailer angle relative to car).
- control_function() below is YOUR hook: it receives only the sensor data
  and returns (steer, speed). Preloaded with a human-driver model
  (preview steering, reaction delay, rate limit) that always corrects
  toward the middle of the lane but is blind to the trailer.

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


XML_PATH = "rc-truck-trailer.xml"

# ---------------- geometry knobs ----------------
CARGO_OFFSET = 0.09      # m relative to axle (+ = ahead/stable, - = behind/sway)
CARGO_MASS = 3.5         # kg
CARGO_HALF = (0.050, 0.048, 0.035)   # payload box half-sizes (fits the rails)

TRAILER_TIRE_MU = 0.9
HITCH_DAMPING = 0.002

# ---------------- aerodynamics ----------------
# MuJoCo's built-in fluid model: every body gets quadratic drag + viscous
# damping computed from its equivalent-inertia box, relative to WIND.
AIR_DENSITY = 0      # kg/m^3 (sea-level air; set 0.0 for vacuum = no drag; set 1.204 for drag)
AIR_VISCOSITY = 0   # Pa*s   (air; set 0 for no damping; set 1.8e-5 for air damping)
WIND = (0.0, 0.0, 0.0)   # m/s ambient wind, world frame. e.g. (0, -2, 0) is
                         # a steady 2 m/s crosswind from the left - a
                         # continuous alternative to the impulsive "gust"

# ---------------- experiment knobs ----------------
SPEED_CTRL = 175.0        # rad/s wheel target (~4.0 m/s)
DISTURBANCE = "swerve"   # "swerve" or "gust"

N_RUNS = 3               # how many simulations to run
DISTURB_START = 0.05     # first-run magnitude: rad (swerve) or N (gust), default .15
DISTURB_STEP = 0.00      # added to the magnitude after every run
CARGO_OFFSET_STEP = -0.08  # m added to CARGO_OFFSET after every run
CARGO_MASS_STEP = 0.0    # kg added to CARGO_MASS after every run

GUST_TIME = 0.5          # s
SETTLE_TIME = 1.0        # s
SPINUP_TIME = 5.0        # s
MAX_RECORD = 25.0        # s
REALTIME = True
# ---------------------------------------------------

# fixed trailer constants (from the CAD / rc-truck-trailer.xml)
FRAME_MASS = 0.8         # kg, trailer frame (explicit inertial in the XML)
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

# ---------------- pivot (towed oscillation) mode ----------------
PIVOT_MODE = True        # True: controller OFF, car towed by a tow point
TOW_EYE = (0.1439, 0.0, -0.0076)   # car frame: center of the front axle
CAR_Z0 = 0.0531          # car body height at qpos0 (from the XML)
PIVOT_SPEED = SPEED_CTRL * WHEEL_RADIUS   # m/s tow speed (customizable)
PIVOT_SPRING_K = 25.0    # N/m, lateral spring pulling the tow point to y=0
PIVOT_SPRING_C = 8.0     # N*s/m, lateral damping on the tow point

# ---------------- planar / no-weight-shift mode ----------------
PLANAR_MODE = False      # True: no vertical-axis motion at all. No fore/aft
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

# ---------------- human driver parameters ----------------
WHEELBASE = 0.288        # m, car wheelbase (pure-pursuit geometry)
LOOKAHEAD_TIME = 0.8     # s, how far down the road the driver looks
LOOKAHEAD_MIN = 0.6      # m, minimum preview distance at low speed
REACTION_DELAY = 0.25    # s, perception + neuromuscular delay
STEER_RATE_MAX = 6.0     # rad/s max front-wheel rate (RC servo limit)
Y_DEADBAND = 0.01        # m, offsets smaller than this are ignored


# =====================================================================
# >>> CONTROLLER — EDIT THIS FUNCTION <<<
# (unchanged — never called in PIVOT_MODE)
# =====================================================================
def control_function(sensors, dt, state):
    # heading from the car IMU
    w, x, y, z = sensors["car_quat"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    if "v_est" not in state:
        state["v_est"] = SPEED_CTRL * WHEEL_RADIUS   # speed at handoff
        state["y_est"] = 0.0
        state["steer"] = 0.0
        state["cmd_buf"] = []                        # reaction-delay queue

    # dead-reckoned speed and lateral offset from lane center
    state["v_est"] += float(sensors["car_accel"][0]) * dt
    state["y_est"] += state["v_est"] * math.sin(yaw) * dt

    # driver ignores tiny offsets
    y_err = state["y_est"] if abs(state["y_est"]) > Y_DEADBAND else 0.0

    # aim at a point on the centerline, lookahead distance ahead
    ld = max(LOOKAHEAD_MIN, state["v_est"] * LOOKAHEAD_TIME)
    alpha = math.atan2(-y_err, ld) - yaw             # angle to preview point
    target = math.atan2(2 * WHEELBASE * math.sin(alpha), ld)  # pure pursuit
    target = max(-MAX_STEER, min(MAX_STEER, target))

    # reaction delay: act on the command from REACTION_DELAY seconds ago
    state["cmd_buf"].append(target)
    n_delay = max(1, int(REACTION_DELAY / dt))
    delayed = (state["cmd_buf"].pop(0)
               if len(state["cmd_buf"]) > n_delay else 0.0)

    # steering-rate limit: hands can only turn the wheel so fast
    max_step = STEER_RATE_MAX * dt
    delta = max(-max_step, min(max_step, delayed - state["steer"]))
    state["steer"] += delta

    return state["steer"], SPEED_CTRL
# =====================================================================
# >>> END CONTROLLER <<<
# =====================================================================


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
    print(f"Axle at x = {ax:.3f} m | payload at x = {cx:.3f} m "
          f"({cargo_offset:+.3f} m from axle), {cargo_mass:.2f} kg")
    print(f"Static tongue load = {1000 * tongue_load / G:.0f} g "
          f"({100 * tongue_load / (G * total):.0f}% of trailer weight)"
          + ("  << NEGATIVE: sway-prone!" if tongue_load < 0 else ""))

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
        <inertial pos="{FRAME_COM_X} 0 -0.045" mass="{FRAME_MASS}" diaginertia="0.0035 0.012 0.014"/>
        <site name="hitch_force_site" pos="0 0 0" size="0.008" rgba="0 0 1 0.5"/>
        <site name="imu_trailer" pos="-0.25 0 -0.03" size="0.008" rgba="1 0 0 0.5"/>
        <geom name="trailer_frame_vis" class="visual" type="mesh" mesh="trailer_frame" material="trailer_mat"/>
        <geom name="trailer_deck_col" type="box" size="0.155 0.100 0.0125" pos="-0.243 0 -0.0525" mass="0" group="3"/>
        <geom name="trailer_tongue_col" type="capsule" size="0.008" fromto="0.005 0 -0.005  -0.100 0 -0.050" mass="0" contype="0" conaffinity="0"/>
        <geom name="cargo" type="box" size="{CARGO_HALF[0]} {CARGO_HALF[1]} {CARGO_HALF[2]}" pos="{cx:.4f} 0 {cz:.4f}" mass="{cargo_mass}" material="payload_mat"/>
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
    xml = re.sub(r'density="[^"]*"', f'density="{AIR_DENSITY}"', xml, count=1)
    xml = re.sub(r'viscosity="[^"]*"',
                 f'viscosity="{AIR_VISCOSITY}"', xml, count=1)
    xml = re.sub(r'wind="[^"]*"',
                 f'wind="{WIND[0]} {WIND[1]} {WIND[2]}"', xml, count=1)

    # from_xml_string resolves meshdir against the CWD, not the XML file:
    # make it absolute so the script works from anywhere
    assets_abs = os.path.join(
        os.path.dirname(os.path.abspath(XML_PATH)), "assets")
    xml = xml.replace('meshdir="assets"', f'meshdir="{assets_abs}"')

    # force sensor at the hitch: lateral force the trailer puts on the car
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
        xml = xml.replace("<!-- CAR_ROOT_START -->", leader, 1)

        # ball-type attachment: tow point and front-axle center coincide,
        # car rotation about the point is completely free (pitch/yaw/roll)
        eq = """<equality>
    <connect name="tow_ball" body1="leader" body2="car" anchor="0 0 0"/>
  </equality>
  <actuator>"""
        xml = xml.replace("<actuator>", eq, 1)
        xml = xml.replace("</actuator>",
            '  <velocity name="leader_drive" joint="leader_x" kv="200" '
            'ctrlrange="0 100"/>\n  </actuator>')

    if PLANAR_MODE:
        # car root: x / y / yaw only -> no heave, pitch, or roll anywhere
        rs = xml.index("<!-- CAR_ROOT_START -->")
        re_ = xml.index("<!-- CAR_ROOT_END -->") + len("<!-- CAR_ROOT_END -->")
        planar_root = f"""<body name="car" pos="0 0 {CAR_Z0}">
      <joint name="car_x" type="slide" axis="1 0 0"/>
      <joint name="car_y" type="slide" axis="0 1 0"/>
      <joint name="car_yaw" type="hinge" axis="0 0 1"/>"""
        xml = xml[:rs] + planar_root + xml[re_:]
        # hitch is a hinge in planar mode: swap the ball sensor for jointpos
        xml = xml.replace('<ballquat name="hitch_angle" joint="hitch"/>',
                          '<jointpos name="hitch_angle" joint="hitch"/>')
        # one common friction coefficient on every tire: this string lives
        # in the "tire" default class in the XML, so one replace covers all
        # four car tires (the trailer tires already got PLANAR_TIRE_MU
        # explicitly in trailer_xml)
        xml = xml.replace('friction="1.3 0.005 0.0001"',
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


def run_simulation(magnitude, cargo_offset, cargo_mass, run_idx):
    model = build_model(cargo_offset, cargo_mass)
    data = mujoco.MjData(model)

    dl = model.actuator("drive_left").id
    dr = model.actuator("drive_right").id
    sl = model.actuator("steer_left").id
    sr = model.actuator("steer_right").id
    trailer_id = model.body("trailer").id
    hitch_joint = model.joint("hitch")
    hitch_adr = hitch_joint.qposadr[0]
    hitch_is_ball = int(hitch_joint.type[0]) == mujoco.mjtJoint.mjJNT_BALL
    hf_adr = model.sensor("hitch_force").adr[0]
    dt = model.opt.timestep
    trailer_rear_x = DECK_X_MIN            # gust acts on the trailer tail

    pv = None
    if PIVOT_MODE:
        pv = model.actuator("leader_drive").id
        # freewheel the drive wheels: the tow point pulls, not the wheels
        for a in (dl, dr):
            model.actuator_gainprm[a, 0] = 0
            model.actuator_biasprm[a, 2] = 0

    tire_ids = {n: model.geom(n).id for n in TRUCK_TIRES + TRAILER_TIRES}
    floor_id = model.geom("floor").id

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
    simNum = 1
    while os.path.exists(f"csvs/RC_Sway_{mode_tag()}_{DISTURBANCE}_run{run_idx + 1}"
            f"_mag{magnitude:g}_v{SPEED_CTRL:.0f}"
            f"_cargo{cargo_mass:g}kg_off{cargo_offset:+.3f}"
            f"_mu{TRAILER_TIRE_MU}_simNum{simNum}.csv"):
        simNum += 1
    csv_filename = (
        f"csvs/RC_Sway_{mode_tag()}_{DISTURBANCE}_run{run_idx + 1}"
        f"_mag{magnitude:g}_v{SPEED_CTRL:.0f}"
        f"_cargo{cargo_mass:g}kg_off{cargo_offset:+.3f}"
        f"_mu{TRAILER_TIRE_MU}_simNum{simNum}.csv"
    )
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
            viewer.cam.distance = 2.5
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
                    steer, speed = control_function(
                        read_sensors(model, data), dt, ctrl_state)
                    steer = max(-MAX_STEER, min(MAX_STEER, steer))
                    speed = max(0.0, min(300.0, speed))
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer
                    data.ctrl[dl] = speed
                    data.ctrl[dr] = speed
                else:
                    if phase != "settle":
                        data.ctrl[dl] = SPEED_CTRL
                        data.ctrl[dr] = SPEED_CTRL
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
        simNum = 1
        while os.path.exists(f"plots/run{i + 1}_{tag}_{DISTURBANCE}"
                              f"_mag{r['magnitude']:g}_speed{r['speed_ctrl']:g}_simNum{simNum}.png"):
            simNum += 1
        out = (f"plots/run{i + 1}_{tag}_{DISTURBANCE}"
               f"_mag{r['magnitude']:g}_speed{r['speed_ctrl']:g}_simNum{simNum}.png")
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
