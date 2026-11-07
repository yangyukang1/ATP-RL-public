from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def readonly(value, dtype=np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class RISActuatorState:
    group_amplitudes: np.ndarray
    group_weights: np.ndarray
    output_power_watt: float
    output_limit_watt: float
    max_amplitude: float

    def __post_init__(self):
        for name in ("group_amplitudes", "group_weights"):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        if not np.all(np.isfinite(self.group_amplitudes)) or np.any(self.group_amplitudes < 0):
            raise ValueError("Non-finite or negative actuator amplitude")
        if (
            not np.all(np.isfinite(self.group_weights))
            or np.any(self.group_weights < 0)
            or not np.isclose(self.group_weights.sum(), 1.0)
        ):
            raise ValueError("Invalid actuator power weights")
        if self.group_weights.shape != self.group_amplitudes.shape:
            raise ValueError("Actuator group dimension mismatch")
        if np.any(
            self.group_amplitudes > self.max_amplitude
        ) or self.output_power_watt > self.output_limit_watt * (1 + 1e-12):
            raise ValueError("Actuator budget violation")


def physical_actuator_state(env) -> RISActuatorState:
    """Use incident physical CSI only to compute fixed actuator amplitudes and weights."""
    groups = env.ris_groups
    weights = np.full(len(groups), 1.0 / len(groups))
    pin = np.array(
        [
            env.satellite_max_power_watt
            * env.large_scale_gain_sat_to_ris
            * np.sum(np.abs(env.small_scale_sat_to_ris[group]) ** 2)
            for group in groups
        ]
    )
    if env.ris_power_control_mode == "power_budget":
        amplitudes = np.zeros(len(groups))
        valid = pin > 1e-12
        amplitudes[valid] = np.minimum(
            env.ris_elem_max_amp_linear,
            np.sqrt(env.ris_reflection_power_limit_watt * weights[valid] / pin[valid]),
        )
    else:
        amplitudes = np.full(len(groups), env.ris_element_amplitude)
    return RISActuatorState(
        amplitudes,
        weights,
        float(np.dot(amplitudes**2, pin)),
        float(env.ris_reflection_power_limit_watt),
        float(env.ris_elem_max_amp_linear),
    )
