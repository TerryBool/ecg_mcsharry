import os
import numpy as np
import wfdb
from numpy.typing import NDArray

DATA_PATH = "Data"

class Record:
    def __init__(self, clean_signal: NDArray, noisy_signal: NDArray, sampling_frequency: float, duration: float):
        """
        Represents an ECG record.

        :param clean_signal: Clean ECG signal (numpy array)
        :param noisy_signal: Noisy ECG signal (numpy array)
        :param sampling_frequency: Sampling frequency of the signal (Hz)
        :param duration: Duration of the signal (seconds)
        """
        self.clean_signal = clean_signal
        self.noisy_signal = noisy_signal
        self.sampling_frequency = sampling_frequency
        self.duration = duration

class RecordBuilder:
    @staticmethod
    def fetch_record(person_id: int, record_id: int) -> Record:
        """
        Builds a Record instance by reading the corresponding file.

        :param person_id: Identifier for the person
        :param record_id: Identifier for the record
        :return: Record instance
        """
        # Construct the file path
        cwd = os.getcwd()
        file_path = os.path.join(os.path.dirname(cwd), DATA_PATH, f"Person_{person_id:02d}", f"rec_{record_id}")
        # file_path = os.path.join(DATA_PATH, f"Person_{person_id:02d}", f"rec_{record_id}")

        # Check if the file exists
        if not os.path.exists(file_path + ".dat"):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the data from the file
        data = wfdb.rdrecord(file_path)

        # Extract required fields
        noisy_signal = data.p_signal[:, 0]
        clean_signal = data.p_signal[:, 1]
        sampling_frequency = data.fs
        duration = noisy_signal.shape[0] / sampling_frequency

        # Create and return a Record instance
        return Record(clean_signal, noisy_signal, sampling_frequency, duration)