import ctypes
import platform
import shutil
import subprocess
from typing import Any

from egga.config import REPO_ROOT

LIB_NAME = "libegga_c.dll" if platform.system() == "Windows" else "libegga_c.so"
DLL_PATH = REPO_ROOT / "src" / "c" / LIB_NAME


def build_c_lib() -> None:
    c_dir = REPO_ROOT / "src" / "c"
    sources = [
        str(c_dir / "actor.c"),
        str(c_dir / "supervisor.c"),
        str(c_dir / "supervisor_envelope_data.c"),
    ]
    compiler = "gcc" if shutil.which("gcc") else ("clang" if shutil.which("clang") else None)
    if compiler is None:
        raise RuntimeError("Neither gcc nor clang found on PATH to build C library.")
    cmd = [compiler, "-O3", "-shared", "-std=c99", "-o", str(DLL_PATH)] + sources + ["-lm"]
    subprocess.run(cmd, check=True, cwd=str(c_dir))


class CEggaConfig(ctypes.Structure):
    _fields_ = [
        ("dt", ctypes.c_float),
        ("gain_hop_period", ctypes.c_float),
        ("min_dwell", ctypes.c_float),
        ("recover", ctypes.c_float),
        ("stale_grace", ctypes.c_float),
        ("mu_floor", ctypes.c_float),
        ("tau_prior", ctypes.c_float),
        ("mass_prior_max", ctypes.c_float),
        ("tau_fallback", ctypes.c_float),
        ("mass_fallback", ctypes.c_float),
        ("tau_degraded_enter", ctypes.c_float),
        ("tau_degraded_exit", ctypes.c_float),
        ("mu_low_enter", ctypes.c_float),
        ("mu_low_exit", ctypes.c_float),
        ("quality_enter", ctypes.c_float),
        ("quality_exit", ctypes.c_float),
        ("speed_scale", ctypes.c_float * 6),
        ("yaw_lpf_tau", ctypes.c_float),
        ("yaw_abs", ctypes.c_float),
        ("yaw_rel", ctypes.c_float),
        ("yaw_persist", ctypes.c_float),
        ("yaw_min_speed", ctypes.c_float),
        ("actuator_abs", ctypes.c_float),
        ("actuator_persist", ctypes.c_float),
        ("saturation_level", ctypes.c_float),
        ("saturation_persist", ctypes.c_float),
        ("rl_rate", ctypes.c_float * 4),
        ("rl_persist_rejects", ctypes.c_int32),
        ("lane_margin", ctypes.c_float),
        ("margin_speed_scale", ctypes.c_float),
        ("min_speed_for_limit", ctypes.c_float),
        ("decel", ctypes.c_float),
        ("accel", ctypes.c_float),
        ("wheelbase", ctypes.c_float),
        ("lf", ctypes.c_float),
        ("lr", ctypes.c_float),
        ("cf", ctypes.c_float),
        ("cr", ctypes.c_float),
        ("gravity", ctypes.c_float),
        ("mass_nominal", ctypes.c_float),
        ("steer_hw_max", ctypes.c_float),
        ("steer_rate_hw_max", ctypes.c_float),
    ]


class CEggaInputs(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_float),
        ("speed", ctypes.c_float),
        ("speed_request", ctypes.c_float),
        ("curvature_ahead", ctypes.c_float),
        ("mu_lo", ctypes.c_float),
        ("mu_hi", ctypes.c_float),
        ("tau_bar", ctypes.c_float),
        ("mass_hi", ctypes.c_float),
        ("quality", ctypes.c_float),
        ("est_status", ctypes.c_int32),
        ("yaw_rate", ctypes.c_float),
        ("steer_meas", ctypes.c_float),
        ("steer_cmd", ctypes.c_float),
        ("e_abs", ctypes.c_float),
        ("e_rate_abs", ctypes.c_float),
        ("rl_valid", ctypes.c_bool),
        ("rl_gain", ctypes.c_float * 4),
        ("reference_gain", ctypes.c_float * 4),
        ("request_reset", ctypes.c_bool),
    ]


class CEggaOutputs(ctypes.Structure):
    _fields_ = [
        ("gains", ctypes.c_float * 4),
        ("speed_cmd", ctypes.c_float),
        ("steer_limit", ctypes.c_float),
        ("rate_limit", ctypes.c_float),
        ("mode", ctypes.c_int32),
        ("rl_applied", ctypes.c_bool),
        ("forced_gain_jump", ctypes.c_bool),
        ("verified", ctypes.c_bool),
        ("flags", ctypes.c_int32),
    ]


class CEggaState(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("mode_entry_t", ctypes.c_float),
        ("last_t", ctypes.c_float),
        ("healthy_for", ctypes.c_float),
        ("yaw_pred", ctypes.c_float),
        ("yaw_bad", ctypes.c_float),
        ("act_bad", ctypes.c_float),
        ("sat_bad", ctypes.c_float),
        ("est_bad", ctypes.c_float),
        ("rl_rejects", ctypes.c_int32),
        ("last_rl_valid", ctypes.c_bool),
        ("speed_cmd", ctypes.c_float),
        ("gain_idx", ctypes.c_int32 * 4),
        ("have_gain", ctypes.c_bool),
        ("last_hop_t", ctypes.c_float),
        ("last_rl", ctypes.c_float * 4),
        ("hist", ctypes.c_float * 32),
        ("hist_n", ctypes.c_int32),
        ("prev_flags", ctypes.c_int32),
        ("log_t", ctypes.c_float * 256),
        ("log_code", ctypes.c_int32 * 256),
        ("log_val", ctypes.c_float * 256),
        ("log_n", ctypes.c_int32),
        ("mode_changes", ctypes.c_int32),
    ]


def load_c_lib() -> ctypes.CDLL:
    if not DLL_PATH.exists():
        build_c_lib()
    lib = ctypes.CDLL(str(DLL_PATH))

    lib.egga_actor_predict.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.egga_actor_predict.restype = None

    lib.egga_actor_proposal.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.egga_actor_proposal.restype = None

    lib.egga_supervisor_init.argtypes = [ctypes.POINTER(CEggaState)]
    lib.egga_supervisor_init.restype = None

    lib.egga_supervisor_step.argtypes = [
        ctypes.POINTER(CEggaState),
        ctypes.POINTER(CEggaConfig),
        ctypes.POINTER(CEggaInputs),
        ctypes.POINTER(CEggaOutputs),
    ]
    lib.egga_supervisor_step.restype = None

    return lib


def py_config_to_c(py_cfg: Any) -> CEggaConfig:
    c_cfg = CEggaConfig()
    c_cfg.dt = float(py_cfg.dt)
    c_cfg.gain_hop_period = float(py_cfg.gain_hop_period)
    c_cfg.min_dwell = float(py_cfg.min_dwell)
    c_cfg.recover = float(py_cfg.recover)
    c_cfg.stale_grace = float(py_cfg.stale_grace)
    c_cfg.mu_floor = float(py_cfg.mu_floor)
    c_cfg.tau_prior = float(py_cfg.tau_prior)
    c_cfg.mass_prior_max = float(py_cfg.mass_prior_max)
    c_cfg.tau_fallback = float(py_cfg.tau_fallback)
    c_cfg.mass_fallback = float(py_cfg.mass_fallback)
    c_cfg.tau_degraded_enter = float(py_cfg.tau_degraded_enter)
    c_cfg.tau_degraded_exit = float(py_cfg.tau_degraded_exit)
    c_cfg.mu_low_enter = float(py_cfg.mu_low_enter)
    c_cfg.mu_low_exit = float(py_cfg.mu_low_exit)
    c_cfg.quality_enter = float(py_cfg.quality_enter)
    c_cfg.quality_exit = float(py_cfg.quality_exit)
    for i in range(6):
        c_cfg.speed_scale[i] = float(py_cfg.speed_scale[i])
    c_cfg.yaw_lpf_tau = float(py_cfg.yaw_lpf_tau)
    c_cfg.yaw_abs = float(py_cfg.yaw_abs)
    c_cfg.yaw_rel = float(py_cfg.yaw_rel)
    c_cfg.yaw_persist = float(py_cfg.yaw_persist)
    c_cfg.yaw_min_speed = float(py_cfg.yaw_min_speed)
    c_cfg.actuator_abs = float(py_cfg.actuator_abs)
    c_cfg.actuator_persist = float(py_cfg.actuator_persist)
    c_cfg.saturation_level = float(py_cfg.saturation_level)
    c_cfg.saturation_persist = float(py_cfg.saturation_persist)
    for i in range(4):
        c_cfg.rl_rate[i] = float(py_cfg.rl_rate[i])
    c_cfg.rl_persist_rejects = int(py_cfg.rl_persist_rejects)
    c_cfg.lane_margin = float(py_cfg.lane_margin)
    c_cfg.margin_speed_scale = float(py_cfg.margin_speed_scale)
    c_cfg.min_speed_for_limit = float(py_cfg.min_speed_for_limit)
    c_cfg.decel = float(py_cfg.decel)
    c_cfg.accel = float(py_cfg.accel)
    c_cfg.wheelbase = float(py_cfg.wheelbase)
    c_cfg.lf = float(py_cfg.lf)
    c_cfg.lr = float(py_cfg.lr)
    c_cfg.cf = float(py_cfg.cf)
    c_cfg.cr = float(py_cfg.cr)
    c_cfg.gravity = float(py_cfg.gravity)
    c_cfg.mass_nominal = float(py_cfg.mass_nominal)
    c_cfg.steer_hw_max = float(py_cfg.steer_hw_max)
    c_cfg.steer_rate_hw_max = float(py_cfg.steer_rate_hw_max)
    return c_cfg
