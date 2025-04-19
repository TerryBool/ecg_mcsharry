import numpy as np
import scipy as sp
from typing import Tuple
from numpy.typing import NDArray
from parameter_generator import AbstractParameterGenerator, NormalParamGen


class ECGGenerator:
    @staticmethod
    def generate_signal_custom(
        fs: float = 360.0,
        t_range: Tuple[float, float] = (0.0, 60.0),
        param_gen: AbstractParameterGenerator | None = None,
        # Scaling needs to be reconsidered
    ):
        if param_gen is None:
            state = np.array([1.0, 0.0, 0.4])
            param_gen = NormalParamGen(state)
        
        ts, te = t_range
        tlen = te - ts
        dt = tlen / fs

    # Should be returning time and state (x, y, z)
    @staticmethod
    def generate_pulse(t: float, xs: NDArray, param_gen: AbstractParameterGenerator) -> Tuple[float, NDArray]:
        pass

    @staticmethod
    def generate_signal_scipy():
        pass

    @staticmethod
    def _ecg_model(t: float, xs: NDArray, param_gen: AbstractParameterGenerator):
        x, y, z = xs

        params = param_gen.get_parameters(xs[:2])
        theta, a, b = params.offsets, params.scales, params.widths
        hr = 60.0 / params.heart_rate
        A, fresp = params.resp_peak, params.resp_freq

        alpha = 1.0 - np.sqrt(x * x + y * y)

        i_theta = np.arctan2(y, x)
        d_theta = (i_theta - theta) - np.round(
            (i_theta - theta) / 2 / np.pi
        ) * 2 * np.pi
        # d_theta = (i_theta - theta) % (2*np.pi)

        omega = hr
        z0 = A * np.sin(2 * np.pi * fresp * t)

        dx = alpha * x - omega * y
        dy = alpha * y + omega * x
        dz = -np.sum(a * d_theta * np.exp(-0.5 * (d_theta / b) ** 2)) - (z - z0)

        return [dx, dy, dz]


class RK4:
    @staticmethod
    def step(fn, t, y, dt, *params):
        scalers = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        result = np.zeros_like(y)

        for i in range(result.shape[0]):
            ks = np.zeros(4)

            # RK4 Step 1
            yn = y.copy()
            xn = fn(t, yn, *params)
            ks[0] = xn[i]

            # RK4 Step 2
            yn[i] = y[i] + 0.5 * dt * xn[i]
            xn = fn(t + 0.5 * dt, yn, *params)
            ks[1] = xn[i]

            # RK4 Step 3
            yn[i] = y[i] + 0.5 * dt * xn[i]
            xn = fn(t + 0.5 * dt, yn, *params)
            ks[2] = xn[i]

            # RK4 Step 4
            yn[i] = y[i] + dt * xn[i]
            xn = fn(t + dt, yn, *params)
            ks[3] = xn[i]

            result[i] = y[i] + (scalers @ ks) * dt

        return result
