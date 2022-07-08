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

class RollingWheelScaler():
    """Pneumatic specific scaler
    """
    def __init__(self,wheel_velocity=0.0):
        self.wheel_velocity = wheel_velocity

    def fit(self, x, y):
        pass

    def transform(self, x, y):
        squeezed_x=np.squeeze(x)
        scale=self.wheel_velocity*np.einsum('i,ij->ij',squeezed_x,np.ones_like(y[:,0::2]))

        transformed_y = np.zeros_like(y)
        transformed_y[:,0::2] = y[:,0::2] - scale
        transformed_y[:,1::2] = y[:,1::2]
        return x, transformed_y

    def fit_transform(self, x, y):
        self.fit(x, y)
        transformed_x, transformed_y=self.transform(x,y)
        return transformed_x, transformed_y

    def inverse_transform(self, x, y):
        squeezed_x=np.squeeze(x)
        scale=self.wheel_velocity*np.einsum('i,ij->ij',squeezed_x,np.ones_like(y[:,0::2]))

        transformed_y = np.zeros_like(y)
        transformed_y[:,0::2] = y[:,0::2] + scale
        transformed_y[:,1::2] = y[:,1::2]
        return transformed_y

    def save(self, path: Union[str, pathlib.Path]):
        res_json = {}
        res_json["_wheel_velocity"] = self.wheel_velocity

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

        self.wheel_velocity = res_json["_wheel_velocity"]

if __name__ =="__main__":
    np.random.seed(42)

    origin_x,origin_y=(0.,0.)
    lenght_x,lenght_y=(10.,10.)
    nb_line,nb_column=(5,5)
    coord_x,coord_y=np.meshgrid(np.arange(origin_x,origin_x+lenght_x,lenght_x/nb_line),np.arange(origin_y,origin_y+lenght_y,lenght_y/nb_column))
    coord_x,coord_y=coord_x.flatten(),coord_y.flatten()

    block_coords=np.empty((coord_x.shape[0]+coord_y.shape[0],))
    block_coords[0::2]=coord_x
    block_coords[1::2]=coord_y

    timestep=0.1
    wheel_velocity=1.0
    displacement_one_timestep=np.zeros((2*nb_line*nb_column))
    displacement_one_timestep[0::2]=wheel_velocity*timestep

    nb_timesteps=20
    time_instants=timestep*np.arange(nb_timesteps)
    block_positions=np.array([block_coords+step*displacement_one_timestep for step in range(nb_timesteps)])

    #Check scaler reversibility
    myScaler=RollingWheelScaler(wheel_velocity=wheel_velocity)
    transformed_x, transformed_y=myScaler.fit_transform(x=time_instants, y=block_positions)
    assert transformed_x.shape==time_instants.shape
    assert transformed_y.shape==block_positions.shape

    np.testing.assert_almost_equal(transformed_x,time_instants)
    np.testing.assert_almost_equal(transformed_y,np.repeat(block_coords[None,:],transformed_y.shape[0],axis=0))

    myInverseCheckScaler=RollingWheelScaler(wheel_velocity=wheel_velocity)
    transformed_x, transformed_y=myInverseCheckScaler.fit_transform(x=time_instants, y=block_positions)
    inverse_transformed_y=myInverseCheckScaler.inverse_transform(x=time_instants,y=transformed_y)
    np.testing.assert_almost_equal(block_positions,inverse_transformed_y)

    #CheckSaveLoad
    myScaler_to_save=RollingWheelScaler(wheel_velocity=wheel_velocity)
    transformed_x, transformed_y=myScaler_to_save.fit_transform(x=time_instants, y=block_positions)
    wheel_velocity_original = myScaler_to_save.wheel_velocity
    myScaler_to_save.save(path=".")

    myScaler_to_load = RollingWheelScaler()
    myScaler_to_load.load(path=".")

    wheel_velocity_reloaded = myScaler_to_load.wheel_velocity
    np.testing.assert_almost_equal(actual=wheel_velocity_original,desired=wheel_velocity_reloaded)



