# ECG generation using McSharry method

## Theoretical description

You can read our report or [this article](https://lcp.mit.edu/pdf/McSharryTBE03.pdf)

## Files and usage

### Files

- signal_generator.py - Initial implementation, kept for experimentation and as a baseline
- playground.ipynb - Initial messing around, some changes may have broken parts of it
- data.py - Used for simpler of reading wfdb records 
- parameter_factory.py - Contains class which is used to get parameters for different types of signals
- parameter_generator.py - Contains abstract class and its children for generating parameters during simulation
- ecg_generator.py - Main file used for generating ECG signal

### Usage

#### Basic usage

```python
from ecg_generator import ECGGenerator

t, result = ECGGenerator.generate_signal_scipy(param_gen=generator)
```

Generates normal ECG signal, result contains following signals in order of index

- X axis value of cyclical part
- Y axis value of cyclical part
- ECG signal

#### Using different parameter generator

In order to generate different type of ECG signal you need to provide a parameter generator which inherits ```AbstractParameterGenerator```. You may find some in *parameter_generator.py*

```python
from ecg_generator import ECGGenerator
from parameter_generator import RightBranchBundleBlockGenerator

generator = RightBranchBundleBlockGenerator(np.array([1.0, 0.0, 0.04]))
t, result = ECGGenerator.generate_signal_scipy(param_gen=generator)
```

The values provided in the parameter generator are the initial conditions

#### Implementing custom parameter generator

In order to implement custom parameter generator you need to inherit from ```AbstractParameterGenerator``` and implement ```_get_parameters(self, state)``` method. Here is an example of implementation of ```RightBranchBundleBlockGenerator```.

```python
from parameter_generator import AbstractParameterGenerator
from parameter_factory import ParameterFactory
import numpy as np

class RightBranchBundleBlockGenerator(AbstractParameterGenerator):
    def __init__(self, initial_state: NDArray, base_heart_rate = 60.0, current_params: str = "Normal") -> None:
        super().__init__(initial_state, base_heart_rate)
        self.current_params = current_params
        self.num_cycles = 0

    @override
    def _get_parameters(self, state: NDArray) -> ECGParameters:
        new_cycle = self._is_new_cycle(state)
        if new_cycle and self.num_cycles == 0:
            rng = np.random.random()
            if rng < 0.2:
                self.current_params = "RightBlock"
                self.num_cycles = np.random.randint(1, 3)
            else:
                self.current_params = "Normal"
        
        result = None
        if self.current_params == "RightBlock":
            result = ParameterFactory.right_branch_block()
        else:
            result = ParameterFactory.normal_signal()

        if new_cycle and self.num_cycles > 0:
            self.num_cycles -= 1

        return result
```

Here we can see that a ```self._is_new_cycle(state)``` is used to switch between different parameters based on which part of the cycle is being generated. Returned parameters are taken from ```ParameterFactory```.