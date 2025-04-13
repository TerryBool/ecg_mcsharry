import numpy as np

class ParameterFactory:
    @staticmethod
    def sinus_bradycardia():
        theta = np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 2.0])*np.pi
        a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
        b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        w = (13.0/10.0) * np.pi
        return theta, a, b, w

    @staticmethod
    def atrial_extrasystole():
        theta=np.array([-3.0 / 4.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 2.0])*np.pi
        a=np.array([-1.2, -5.0, 30.0, -7.5, 0.75])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        w = 3.0*np.pi
        return theta, a, b, w

    @staticmethod
    def junctional_bradycardia():
        theta=np.array([-2.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, -8.0 / 10.0])*np.pi
        a=np.array([1.0, 3.0, 30.0, -7.5, 1.0])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        w = (3.0/2.0)*np.pi
        return theta, a, b, w

    @staticmethod
    def ventricular_extrasystole():
        theta=np.array([-1.0 / 2.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 5.0])*np.pi
        a=np.array([0.2, -5.0, -20.0, 0.2, 0.31])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        w = (7.0/2.0)*np.pi
        return theta, a, b, w

    @staticmethod
    def tachycardia():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 1.0 / 2.0])*np.pi
        a=np.array([1.2, -5.0, 30.0, -7.5, 0.75])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        w = 5.0*np.pi
        return theta, a, b, w

    @staticmethod
    def left_branch_block():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, -3.0 / 15.0, 1.0 / 2.0])*np.pi
        a=np.array([-0.82, -5.0, -30.0, -7.5, 0.75])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        w = 2.0*np.pi
        return theta, a, b, w

    @staticmethod
    def right_branch_block():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, -0.4, -0.6])*np.pi
        a=np.array([1.2, -5.0, 30.0, -0.6, -0.3])
        b=np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        w = 2.0*np.pi
        return theta, a, b, w

    @staticmethod
    def flutter():
        theta=np.array([-1.0 / 3.0, -1.0 / 12.0, 0.0, 1.0 / 12.0, 0.8])*np.pi
        a=np.array([10.0, 20.0, 25.0, 19.5, 10.0])
        b=np.array([0.25, 0.1, 0.1, 0.17, 0.25])
        w = 2.0*np.pi
        return theta, a, b, w