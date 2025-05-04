import numpy as np
from numpy.typing import NDArray

class ECGParameters:
    def __init__(self, offsets: NDArray, scales: NDArray, widths: NDArray, hr_mult: float, resp_freq: float = 0.25, resp_peak: float = 0.005) -> None:
        self.scales = scales
        self.widths = widths
        self.offsets = offsets
        self.hr_mult = hr_mult
        self.resp_freq = resp_freq
        self.resp_peak = resp_peak
        # Default heart rate
        self.heart_rate = 60.0 * hr_mult

class ParameterFactory:
    @staticmethod
    def normal_signal():
        theta = np.array([-1.0 / 3.0, -1.0 / 12.0, 0, 1.0 / 12.0, 1.0 / 2.0]) * np.pi
        a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
        b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 1.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def sinus_bradycardia():
        theta = np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 2.0])*np.pi
        a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
        b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 13.0 / 20.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def atrial_extrasystole():
        theta=np.array([-3.0 / 4.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 2.0])*np.pi
        a=np.array([-1.2, -5.0, 30.0, -7.5, 0.75])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 3.0 / 2.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def junctional_bradycardia():
        theta=np.array([-2.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, -8.0 / 10.0])*np.pi
        a=np.array([1.0, 3.0, 30.0, -7.5, 1.0])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 3.0 / 4.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def ventricular_extrasystole():
        theta=np.array([-1.0 / 2.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 5.0])*np.pi
        a=np.array([0.2, -5.0, -20.0, 0.2, 0.31])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 7.0 / 4.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def tachycardia():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 2.0])*np.pi
        a=np.array([1.2, -5.0, 30.0, -7.5, 0.75])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 5.0 / 2.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def left_branch_block():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, -3.0 / 15.0, 1.0 / 2.0])*np.pi
        a=np.array([-0.82, -5.0, -30.0, -7.5, 0.75])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 1.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def right_branch_block():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, -0.4, -0.6])*np.pi
        a=np.array([1.2, -5.0, 30.0, -0.6, -0.3])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        hr_mult = 1.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)

    @staticmethod
    def flutter():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 0.8])*np.pi
        a=np.array([10.0, 20.0, 25.0, 19.5, 10.0])
        b=np.array([0.25, 0.1, 0.1, 0.17, 0.25])
        hr_mult = 1.0
        return ECGParameters(offsets=theta, scales=a, widths=b, hr_mult=hr_mult)