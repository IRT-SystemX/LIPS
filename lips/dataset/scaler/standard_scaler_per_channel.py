"""
The Scaler class offering the normalization capabilities
"""
from typing import Union
import pathlib
import numpy as np
import json

from lips.dataset.scaler.scaler import Scaler
from lips.utils import NpEncoder

#from . import Scaler

def put_along_axis_per_channel(channel,channel_index,channel_data,overall_data):
    indices = [slice(None)]*channel_data.ndim
    indices[channel_index] = channel
    overall_data[tuple(indices)] = channel_data
    return overall_data


class StandardScalerPerChannel():
    """Standard scaler per channel

    On each channel, apply
    X - mean(X) / std(X)
    """
    def __init__(self,channel_index_x=1,channel_index_y=1):
        self.channel_index_x = channel_index_x
        self.channel_index_y = channel_index_y
        self._m_x = []
        self._m_y = []
        self._std_x = []
        self._std_y = []

    def fit(self, x, y):
        for channel in range(x.shape[self.channel_index_x]):
            x_channel_axis=x.take(channel,axis=self.channel_index_x)    
            x_channel=x_channel_axis.reshape(x.shape[0], -1)
            m_x_channel= np.mean(x_channel, axis=0)
            self._m_x.append(m_x_channel)
            std_x_channel = np.std(x_channel, axis=0)
            std_x_channel[np.abs(std_x_channel) <= 1e-1] = 1
            self._std_x.append(std_x_channel)

        for channel in range(y.shape[self.channel_index_y]):    
            y_channel_axis=y.take(channel,axis=self.channel_index_y)
            y_channel=y_channel_axis.reshape(y.shape[0], -1)
            m_y_channel= np.mean(y_channel, axis=0)
            self._m_y.append(m_y_channel)
            std_y_channel = np.std(y_channel, axis=0)
            std_y_channel[np.abs(std_y_channel) <= 1e-1] = 1
            self._std_y.append(std_y_channel)

    def transform(self, x, y):
        shapeBychannel_x=x.shape[:self.channel_index_x]+x.shape[self.channel_index_x+1:]
        transformed_x=np.zeros_like(x)
        for channel in range(x.shape[self.channel_index_x]):
            x_channel_axis=x.take(channel,axis=self.channel_index_x)        
            x_channel=x_channel_axis.reshape(x_channel_axis.shape[0], -1)
            x_channel-= self._m_x[channel]
            x_channel/= self._std_x[channel]
            x_channel_reshaped=x_channel.reshape(shapeBychannel_x)
            transformed_x=put_along_axis_per_channel(channel=channel,
                                                     channel_index=self.channel_index_x,
                                                     channel_data=x_channel_reshaped,
                                                     overall_data=transformed_x)

        shapeBychannel_y=y.shape[:self.channel_index_y]+y.shape[self.channel_index_y+1:]
        transformed_y=np.zeros_like(y)
        for channel in range(y.shape[self.channel_index_y]):
            y_channel_axis=y.take(channel,axis=self.channel_index_y)        
            y_channel=y_channel_axis.reshape(y_channel_axis.shape[0], -1)
            y_channel-= self._m_y[channel]
            y_channel/= self._std_y[channel]
            y_channel_reshaped=y_channel.reshape(shapeBychannel_y)
            transformed_y=put_along_axis_per_channel(channel=channel,
                                                     channel_index=self.channel_index_y,
                                                     channel_data=y_channel_reshaped,
                                                     overall_data=transformed_y)

        return transformed_x, transformed_y

    def fit_transform(self, x, y):
        self.fit(x, y)
        transformed_x, transformed_y=self.transform(x,y)
        return transformed_x, transformed_y

    def inverse_transform(self, y):
        shapeBychannel_y=y.shape[:self.channel_index_y]+y.shape[self.channel_index_y+1:]
        transformed_y=np.zeros_like(y)
        for channel in range(y.shape[self.channel_index_y]):
            y_channel_axis=y.take(channel,axis=self.channel_index_y)        
            y_channel=y_channel_axis.reshape(y_channel_axis.shape[0], -1)
            y_channel*= self._std_y[channel]
            y_channel+= self._m_y[channel]
            y_channel_reshaped=y_channel.reshape(shapeBychannel_y)
            transformed_y=put_along_axis_per_channel(channel=channel,
                                                     channel_index=self.channel_index_y,
                                                     channel_data=y_channel_reshaped,
                                                     overall_data=transformed_y)
        return transformed_y

    def save(self, path: Union[str, pathlib.Path]):
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
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "scaler_params.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)

        self._m_x = np.array(res_json["_m_x"], dtype=np.float32)
        self._m_y = np.array(res_json["_m_y"], dtype=np.float32)
        self._std_x = np.array(res_json["_std_x"], dtype=np.float32)
        self._std_y = np.array(res_json["_std_y"], dtype=np.float32)


if __name__ =="__main__":
    np.random.seed(42)
    featuresRandom=100*np.random.rand(10,4,2,2)
    labelsRandom=2000*np.random.rand(10,8,2,2)

    #Check scaler reversibility
    myScaler=StandardScalerPerChannel()
    transformed_x, transformed_y=myScaler.fit_transform(x=featuresRandom, y=labelsRandom)
    assert transformed_x.shape==featuresRandom.shape
    assert transformed_y.shape==labelsRandom.shape

    myOtherScaler=StandardScalerPerChannel()
    double_transformed_x,double_transformed_y=myOtherScaler.fit_transform(x=transformed_x, y=transformed_y)
    np.testing.assert_almost_equal(actual=transformed_x,desired=double_transformed_x)
    np.testing.assert_almost_equal(actual=transformed_y,desired=double_transformed_y)

    myInverseCheckScaler=StandardScalerPerChannel()
    transformed_x, transformed_y=myInverseCheckScaler.fit_transform(x=featuresRandom, y=labelsRandom)
    inverse_transformed_y=myInverseCheckScaler.inverse_transform(y=transformed_y)
    np.testing.assert_almost_equal(labelsRandom,inverse_transformed_y)

    #CheckSaveLoad
    myScaler_to_save=StandardScalerPerChannel()
    transformed_x, transformed_y=myScaler_to_save.fit_transform(x=featuresRandom, y=labelsRandom)
    m_x_original = myScaler_to_save._m_x
    m_y_original = myScaler_to_save._m_y
    std_x_original = myScaler_to_save._std_x
    std_y_original = myScaler_to_save._std_y
    myScaler_to_save.save(path=".")

    myScaler_to_load = StandardScalerPerChannel()
    myScaler_to_load.load(path=".")

    m_x_load = myScaler_to_load._m_x
    m_y_load = myScaler_to_load._m_y
    std_x_load = myScaler_to_load._std_x
    std_y_load = myScaler_to_load._std_y

    np.testing.assert_almost_equal(actual=m_x_original,desired=m_x_load,decimal=4)
    np.testing.assert_almost_equal(actual=m_y_original,desired=m_y_load,decimal=4)
    np.testing.assert_almost_equal(actual=std_x_original,desired=std_x_load,decimal=4)
    np.testing.assert_almost_equal(actual=std_y_original,desired=std_y_load,decimal=4)



