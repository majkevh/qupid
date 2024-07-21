import numpy as np
from functools import partial
import pywt 
from numpy.fft import fft2
from typing import Any, Dict, Tuple, List, Optional



def quantize(pd, resolution, global_max, global_min, alpha):
    """
    Quantizes a persistence diagram.

    Parameters:
    - pd (array): An array of persistence diagrams with birth and death pairs.
    - resolution (tuple): The resolution of the lattice (number of bins in each dimension).
    - global_max (tuple): The maximum values for scaling the quantization lattice.
    - global_min (tuple): The minimum values for scaling the quantization lattice.
    - alpha (tuple): Scaling parameters for the logarithmic transformation of persistence diagram.

    Returns:
    - measure (numpy.ndarray): A quantized measure of the persistence diagram.
    """
    measure = np.zeros(resolution)
    max_transformed_x = np.log(1 + alpha[0]) if alpha[0] != 0 else None
    max_transformed_y = np.log(1 + alpha[1]) if alpha[1] != 0 else None

    for birth, death in pd:
        x = 0  # Default position if global_max[0] is 0 or birth <= 0
        y = 0  # Default position for y if death <= 0, adjusted below
        
        if birth > 0 and global_max[0] != 0:
            transformed_birth = np.log(1 + alpha[0] * ((birth -global_min[0]) / global_max[0]))
            x = min(int((transformed_birth / max_transformed_x) * (resolution[0] - 1)), resolution[0]-1) if alpha[0] != 0 else min(int(((birth -global_min[0]) / global_max[0]) * (resolution[0] - 1)), resolution[0]-1)

        if death > 0 and global_max[1] != 0:
            transformed_death = np.log(1 + alpha[1] * ((death-global_min[1]) / global_max[1]))
            y = min(int((transformed_death / max_transformed_y) * (resolution[1] - 1)),(resolution[1] - 1))  if alpha[1] != 0 else min(int(((death-global_min[1]) / global_max[1]) * (resolution[1] - 1)), resolution[1]-1)
        measure[x, y] += 1
    return measure

def wavelet_functional(pd, resolution, global_max, global_min, wave, alpha):
    """
    Applies a Wavelet Transform to the persistence diagram.
    
    Parameters:
    - pd (array): An array of persistence diagrams.
    - resolution (tuple): The resolution of the grid.
    - global_max (tuple): The maximum values for scaling the quantization lattice.
    - wave (str): The wavelet type.
    
    Returns:
    - (numpy.ndarray, float): A tuple containing the flattened wavelet coefficients.
    """
    discretized_pd = quantize(pd, resolution, global_max, global_min, alpha) 
    cA, (cH, cV, cD) = pywt.dwt2(discretized_pd, wave)
    return np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])


def fourier_functional(pd, resolution, global_max, global_min, alpha):
    """
    Applies a Fourier Transform to the persistence diagram.
    
    Parameters:
    - pd (array): An array of persistence diagrams.
    - resolution (tuple): The resolution of the grid.
    - global_max (tuple): The maximum values for scaling the quantization lattice.
    
    Returns:
    - (numpy.ndarray, float): A tuple containing concatenated magnitude and phase arrays.
    """
    discretized_pd = quantize(pd, resolution, global_max, global_min, alpha)
    fft_output = (fft2(discretized_pd))
    magnitude = np.abs(fft_output).flatten() 
    phase = np.angle(fft_output).flatten()
    return np.concatenate([magnitude, phase])

def identity(pd, resolution, global_max, global_min, alpha):
    """
    Applies a Identity Transform to the persistence diagram.
    
    Parameters:
    - pd (array): An array of persistence diagrams.
    - resolution (tuple): The resolution of the grid.
    - global_max (tuple): The maximum values for scaling the quantization lattice.
    
    Returns:
    - (numpy.ndarray, float): A tuple containing quantized measure.
    """
    return quantize(pd, resolution, global_max, global_min, alpha).flatten()


class QuPID:
    """
    A class for embedding persistence diagrams using various functional representations such as identity, Fourier, and Wavelet.
    
    Attributes:
        resolution (Optional[Tuple[int, int]]): The resolution of the grid.
        global_max (Optional[Tuple[float, float]]): The maximum values for scaling the grid.
        wave (str): The wavelet type.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the QuPID class with default or specified parameters.

        Parameters:
            **kwargs: Variable keyword arguments for class properties.
        """
        self._defaults = {
            "function": "wft",
            "resolution": (20, 20),
            "global_max": None,
            "global_min": None,
            "wave": "coif1",
            "alpha": (0, 0)
        }
        self._update_properties(kwargs)
        self._set_embedding_function()
        self.fitted = False

    def _update_properties(self, properties: Dict[str, Any]) -> None:
        """
        Updates class properties based on provided arguments.

        Parameters:
            properties (Dict[str, Any]): Property values to update.
        """
        for key, value in properties.items():
            if key in self._defaults and value is not None:
                self._defaults[key] = value
        self.__dict__.update(self._defaults)

    def _set_embedding_function(self) -> None:
        """
        Sets the embedding function based on the specified method.
        """
        function_mappings = {
            "fft": partial(fourier_functional, 
                               resolution=self.resolution,
                               global_max=self.global_max,
                               global_min = self.global_min,
                               alpha = self.alpha),
            "wvt": partial(wavelet_functional, 
                               resolution=self.resolution, 
                               global_max=self.global_max,
                               global_min = self.global_min,
                               wave=self.wave,
                               alpha = self.alpha),
            "id": partial(identity, 
                             resolution=self.resolution, 
                             global_max=self.global_max,
                             global_min = self.global_min,
                             alpha = self.alpha)
        }
        self.embedding = function_mappings.get(self.function, None)
        if not self.embedding:
            raise ValueError(f"Invalid function specified: {self.function}")

    def __call__(self, pd: np.ndarray) -> np.ndarray:
        """
        Calls the embedding function on a persistence diagram.

        Parameters:
            pd (np.ndarray): A persistence diagram.

        Returns:
            np.ndarray: The result of applying the embedding function to the persistence diagram.
        """
        if not self.embedding:
            raise ValueError("No embedding function set. Please specify a valid function.")
        return self.embedding(pd)

    def fit(self, pds: List[np.ndarray]) -> 'QuPID':
        """
        Fits the model to a set of persistence diagrams.

        Parameters:
            pds (List[np.ndarray]): A list of persistence diagrams.

        Returns:
            QuPID: The fitted QuPID object.
        """
        if not self.embedding:
            raise ValueError("No embedding function set. Please specify a valid function.")

        concatenated_vectorization = []
        for pd in pds:
            vectors = self.embedding(pd)
            concatenated_vectorization.extend(vectors)

        self.concatenated_vector = np.array(concatenated_vectorization)
        self.fitted = True
        return self

    def transform(self, pds: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms persistence diagrams after fitting.

        Parameters:
            pds (Optional[List[np.ndarray]]): A list of persistence diagrams to transform.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: The transformed data.
        """
        if not self.fitted:
            raise RuntimeError("Transform called before fit. Please fit the model first.")

        return self.concatenated_vector

