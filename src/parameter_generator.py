import numpy as np
from numpy.typing import NDArray
from typing import Tuple, override
from abc import ABC, abstractmethod
from src.parameter_factory import ParameterFactory, ECGParameters

class AbstractParameterGenerator(ABC):

    def __init__(self, initial_state: NDArray, base_heart_rate: float = 60.0) -> None:
        super().__init__()
        self.last_state = initial_state
        self.base_heart_rate = base_heart_rate

    def get_parameters(self, state: NDArray) -> ECGParameters:
        ps = self._get_parameters(state)
        self._apply_heart_rate(ps)
        self.last_state = state
        return ps
    
    def _apply_heart_rate(self, params: ECGParameters):
        params.heart_rate = self.base_heart_rate * params.hr_mult
        # Recalculation of scales and offsets
        shr = np.sqrt(params.heart_rate / 60.0)
        shr2 = np.sqrt(shr)
        params.widths = shr * params.widths
        params.offsets = np.array([shr2, shr, 1.0, shr, shr2]) * params.offsets

    
    def _is_new_cycle(self, state: NDArray) -> bool:
        # Checks for (1, 0) crossing
        a1 = np.atan2(self.last_state[1], self.last_state[0])
        a2 = np.atan2(state[1], state[0])
        p_angle = a1 if a1 >= 0 else a1 + 2.0*np.pi
        n_angle = a2 if a2 >= 0 else a2 + 2.0*np.pi
        return n_angle >= p_angle

    @abstractmethod
    def _get_parameters(self, state: NDArray) -> ECGParameters:
        raise NotImplementedError("Abstract method not implemented")


class NormalParamGen(AbstractParameterGenerator):
    
    def _get_parameters(self, state: NDArray) -> ECGParameters:
        return ParameterFactory.normal_signal()

class VentricularExtrasystoleParamGen(AbstractParameterGenerator):

    def __init__(self, initial_state: NDArray, base_heart_rate = 60.0, current_params: str = "Normal") -> None:
        super().__init__(initial_state, base_heart_rate)
        self.current_params = current_params
    
    @override
    def _get_parameters(self, state: NDArray) -> ECGParameters:
        if self._is_new_cycle(state):
            self.current_params = "Anomaly" if np.random.random() < 0.2 else "Normal"

        ps = None
        if self.current_params == "Normal":
            ps = ParameterFactory.normal_signal()
        else:
            ps = ParameterFactory.ventricular_extrasystole()
        return ps
