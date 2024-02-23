"""
The Scaler class offering the normalization capabilities
"""
from typing import Union
import pathlib
import json
import numpy as np

from lips.dataset.scaler import Scaler
from ...utils import NpEncoder

def iterative_fit(data,chunk_sizes):
    start_index = 0
    mean = 0
    for chunk_size in chunk_sizes:
        data_chunk_indices = range(start_index,start_index+chunk_size)
        data_chunk_x = data[data_chunk_indices,:]
        start_index += chunk_size
        lenght = start_index
        mean += (np.sum(data_chunk_x,axis = 0) - mean*chunk_size)/lenght
    start_index = 0
    standard_dev = 0
    for chunk_size in chunk_sizes:
        data_chunk_indices = range(start_index,start_index+chunk_size)
        data_chunk_x = data[data_chunk_indices,:]
        start_index+= chunk_size
        lenght = start_index  
        standard_dev +=(np.sum((data_chunk_x - mean)**2,axis = 0) - standard_dev*chunk_size)/lenght
    standard_dev = np.sqrt(standard_dev)
    return mean,standard_dev

class StandardScalerIterative(Scaler):
    """Standard scaler
    for large dataset, mean/std values computation at once is not very accurate so we do it iteratively
    - X - mean(X) / std(X)
    """
    def __init__(self,chunk_sizes,no_norm_x=None,no_norm_y=None):
        super().__init__()
        self.chunk_sizes=chunk_sizes
        self._m_x = None
        self._m_y = None
        self._std_x = None
        self._std_y = None
        self.no_norm_x = no_norm_x
        self.no_norm_y = no_norm_y

    def fit(self, x, y):
        self._m_x,self._std_x = iterative_fit(data=x, chunk_sizes=self.chunk_sizes)
        self._m_y,self._std_y = iterative_fit(data=y, chunk_sizes=self.chunk_sizes)
        # to avoid division by 0.
        self._std_x[np.abs(self._std_x) <= 1e-6] = 1
        self._std_y[np.abs(self._std_y) <= 1e-6] = 1
        if self.no_norm_x is not None:
            self._m_x[self.no_norm_x]=0
            self._std_x[self.no_norm_x]=1

        if self.no_norm_y is not None:
            self._m_y[self.no_norm_y]=0
            self._std_y[self.no_norm_y]=1

    def transform(self, x, y):
        x -= self._m_x
        x /= self._std_x
        y -= self._m_y
        y /= self._std_y

        return x, y

    def fit_transform(self, x, y):
        self.fit(x, y)

        x -= self._m_x
        x /= self._std_x
        y -= self._m_y
        y /= self._std_y

        return x, y

    def inverse_transform(self, y):
        y *= self._std_y
        y += self._m_y
        return y

    def save(self, path: Union[str, pathlib.Path]):
        """Save the scaler parameters to a file

        Parameters
        ----------
        path : Union[str, pathlib.Path]
            the path where the parameters should be saved

        Raises
        ------
        RuntimeError
            the fit function is not yet called
        """
        res_json = {}
        res_json["_m_x"] = self._m_x
        res_json["_m_y"] = self._m_y
        res_json["_std_x"] = self._std_x
        res_json["_std_y"] = self._std_y

        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        with open((path / "scaler_params.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def load(self, path: Union[str, pathlib.Path]):
        """Load the scaler parameters from file

        Parameters
        ----------
        path : Union[``str``, pathlib.Path]
            The path to the file where the scaler will be saved.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "scaler_params.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)

        self._m_x = np.array(res_json["_m_x"], dtype=np.float32)
        self._m_y = np.array(res_json["_m_y"], dtype=np.float32)
        self._std_x = np.array(res_json["_std_x"], dtype=np.float32)
        self._std_y = np.array(res_json["_std_y"], dtype=np.float32)

if __name__ == '__main__':
    x = np.arange(10000).reshape((2000,5))
    simulation_sizes= [200]*10
    truemean,truestd=np.mean(x,axis=0),np.std(x, axis=0)
    testmean,testStd=iterative_fit(x,simulation_sizes)
    np.testing.assert_almost_equal(truemean,testmean)
    np.testing.assert_almost_equal(truestd,testStd)