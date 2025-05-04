import numpy as np
from scipy.integrate import solve_ivp
from numpy.typing import NDArray

from parameter_factory import ParameterFactory


def _ecg_model(t, Z, theta, a, b, A, f2, fs, rrs, w):
    x, y, z = Z

    alpha = 1.0 - np.sqrt(x * x + y * y)

    i_theta = np.arctan2(y, x)
    d_theta = (i_theta - theta) - np.round((i_theta - theta) / 2 / np.pi) * 2 * np.pi
    # d_theta = (i_theta - theta) % (2*np.pi)

    hr = rrs[min(len(rrs) - 1, int(np.floor(t * fs)))]

    omega = w / hr
    z0 = A * np.sin(2 * np.pi * f2 * t)

    dx = alpha * x - omega * y
    dy = alpha * y + omega * x
    dz = -np.sum(a * d_theta * np.exp(-0.5 * (d_theta / b) ** 2)) - (z - z0)

    return [dx, dy, dz]


def _rrprocess(
    flo=0.1,
    fhi=0.25,
    flostd=0.1,
    fhistd=0.1,
    lfhfratio=0.5,
    hrmean=60.0,
    hrstd=1.0,
    sfrr=1.0,
    n=256,
):
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)

    df = sfrr / n
    w = np.arange(n) * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / np.sqrt(2 * np.pi * c1**2)
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / np.sqrt(2 * np.pi * c2**2)
    Hw = Hw1 + Hw2
    Hw0 = np.concatenate((Hw[0 : int(n / 2)], Hw[int(n / 2) - 1 :: -1]))
    Sw = (sfrr / 2) * np.sqrt(Hw0)

    ph0 = 2 * np.pi * np.random.uniform(size=int(n / 2 - 1))
    ph = np.concatenate([[0], ph0, [0], -np.flipud(ph0)])
    SwC = Sw * np.exp(1j * ph)
    x = (1 / n) * np.real(np.fft.ifft(SwC))

    xstd = np.std(x)
    ratio = rrstd / xstd
    return rrmean + x * ratio


def generate_ecg(
    fs: float = 360.0,
    t_max: float = 60.0,
    heart_beat: float = 60.0,
    heart_beat_mode: str = "variable",
    f_resp: float = 0.25,
    theta: NDArray = np.array([-1.0 / 3.0, -1.0 / 12.0, 0, 1.0 / 12.0, 1.0 / 2.0])
    * np.pi,
    a: NDArray = np.array([1.2, -5.0, 30.0, -7.5, 0.75]),
    b: NDArray = np.array([0.25, 0.1, 0.1, 0.1, 0.4]),
    w: float = 2 * np.pi,
    scale_low: float = -0.4,
    scale_high: float = 1.2,
    scale_mean: float = 0.0010,
    scale_std: float = 0.1164,
):
    shr = np.sqrt(heart_beat / 60.0)
    shr2 = np.sqrt(shr)
    b = shr * b
    theta = np.array([shr2, shr, 1.0, shr, shr2]) * theta
    A = 0.005

    t_eval = np.linspace(0, t_max, int(t_max * fs))

    rrs = None
    if heart_beat_mode == "variable":
        rrs = _rrprocess(hrmean=heart_beat, n=t_eval.shape[0])
    elif heart_beat_mode == "constant":
        rrs = np.ones(t_eval.shape[0]) * (60.0 / heart_beat)
    else:
        raise ValueError("Unsupported hear beat mode")

    result = solve_ivp(
        _ecg_model,
        [0, t_max],
        [1.0, 0.0, 0.04],
        method="RK45",
        t_eval=t_eval,
        args=(theta, a, b, A, f_resp, fs, rrs, w),
    )

    # Scaling the signal
    signal = result.y[2].copy()
    smin = signal.min()
    smax = signal.max()
    srange = smax - smin
    rrange = scale_high - scale_low
    rsignal = scale_low + ((signal - smin) * rrange) / srange

    # smean = np.mean(signal)
    # sstd = np.std(signal)
    # rsignal = ((signal - smean) / sstd) * scale_std + scale_mean

    return t_eval, rsignal, result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 360.0
    t_max = 60.0
    heart_beat = 50.0

    t, signal, result = generate_ecg(fs, t_max, heart_beat, heart_beat_mode="constant")

    plt.figure(figsize=(6, 6))
    plt.plot(result.y[0], result.y[1])
    plt.title("Phase Plane Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 6))
    plt.plot(t[-5000:], signal[-5000:])
    plt.title("Generated ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
    
    params = ParameterFactory.flutter()
    theta, a, b, w = params.offsets, params.scales, params.widths, params.hr_mult * 2 * np.pi

    t, signal, result = generate_ecg(
        fs,
        t_max,
        heart_beat,
        f_resp=0.1,
        heart_beat_mode="constant",
        theta=theta,
        a=a,
        b=b,
        # w=w
    )

    plt.figure(figsize=(6, 6))
    plt.plot(result.y[0], result.y[1])
    plt.title("Phase Plane Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 6))
    plt.plot(t[-5000:], signal[-5000:])
    plt.title("Generated ECG Signal")
    plt.title("Generated ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
