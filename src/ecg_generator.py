import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple
from numpy.typing import NDArray
from parameter_generator import AbstractParameterGenerator, NormalParamGen, VentricularExtrasystoleParamGen


class ECGGenerator:
    @staticmethod
    def generate_signal_custom(
        fs: float = 360.0,
        t_range: Tuple[float, float] = (0.0, 60.0),
        param_gen: AbstractParameterGenerator | None = None,
        # Scaling needs to be reconsidered
    ) -> Tuple[NDArray, NDArray]:
        if param_gen is None:
            state = np.array([1.0, 0.0, 0.4])
            param_gen = NormalParamGen(state)
        
        ts, te = t_range
        dt = 1.0 / fs

        result = np.array([param_gen.last_state])
        t = ts
        while t < te:
            ti, pulse = ECGGenerator.generate_pulse(t, dt, result[-1, :], param_gen)
            result = np.vstack([result, pulse])
            t = ti
        times = np.linspace(ts, t, result.shape[0])
        return times, result.T

    # Should be returning time and state (x, y, z)
    @staticmethod
    def generate_pulse(t: float, dt: float, xs: NDArray, param_gen: AbstractParameterGenerator) -> Tuple[float, NDArray]:
        result = []
        state = RK4.step(ECGGenerator._ecg_model, t, xs, dt, param_gen)
        result.append(state)
        nt = t + dt
        while not param_gen._is_new_cycle(state):
            state = RK4.step(ECGGenerator._ecg_model, nt, state, dt, param_gen)
            result.append(state)
            nt += dt
        print(f"Cycle ended at: {nt - dt}")
        return nt, np.array(result)


    @staticmethod
    def generate_signal_scipy(
        fs: float = 360.0,
        t_range: Tuple[float, float] = (0.0, 60.0),
        param_gen: AbstractParameterGenerator | None = None,
    ):
        if param_gen is None:
            state = np.array([1.0, 0.0, 0.4])
            param_gen = NormalParamGen(state)

        ts, te = t_range
        tlen = te - ts
        times = np.linspace(ts, te, int(fs * tlen))
        state = param_gen.last_state

        result = solve_ivp(
            ECGGenerator._ecg_model,
            [0, te],
            state,
            method="RK45",
            t_eval=times,
            args=(param_gen,)
        )   
        return times, result
    

    @staticmethod
    def _ecg_model(t: float, xs: NDArray, param_gen: AbstractParameterGenerator):
        x, y, z = xs

        params = param_gen.get_parameters(xs[:2])
        theta, a, b = params.offsets, params.scales, params.widths
        hr = params.heart_rate / 60.0
        A, fresp = params.resp_peak, params.resp_freq

        alpha = 1.0 - np.sqrt(x * x + y * y)

        i_theta = np.arctan2(y, x)
        d_theta = (i_theta - theta) - np.round(
            (i_theta - theta) / 2 / np.pi
        ) * 2 * np.pi
        # d_theta = (i_theta - theta) % (2*np.pi)

        omega = 2 * np.pi * hr
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


if __name__ == "__main__":

    generator = VentricularExtrasystoleParamGen(np.array([1.0, 0.0, 0.4]))
    # generator = NormalParamGen(np.array([1.0, 0.0, 0.4]))

    # times, signal = ECGGenerator.generate_signal_custom(param_gen=generator)
    # print(signal.shape)
    # plt.plot(times, signal[2, :])
    # plt.show()

    t, result = ECGGenerator.generate_signal_scipy(param_gen=generator)

    plt.figure(figsize=(16, 6))
    plt.plot(t, result.y[2])
    plt.title("Generated ECG Signal")
    plt.title("Generated ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()