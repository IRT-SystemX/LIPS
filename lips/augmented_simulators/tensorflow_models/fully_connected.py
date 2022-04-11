"""
Tensorflow fully connected Model
"""
import pathlib
from typing import Union

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.models import Model

from ...logger import CustomLogger
from ...config import ConfigManager

class TfFullyConnected(object):
    """_summary_

    Parameters
    ----------
    input_shape : tuple
        _description_
    output_size : int
        _description_
    name : Union[str, None], optional
        _description_, by default None
    log_path : Union[None, str], optional
        _description_, by default None
    """
    def __init__(self,
                 input_shape: tuple,
                 output_size: int,
                 name: Union[str, None]=None,
                 config_name: Union[str, None]=None,
                 log_path: Union[None, str]=None,
                 **kwargs):
        self.output_size = output_size
        self.input_shape = input_shape
        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # the config file associoated to this model
        if config_name is None:
            config_name = "DEFAULT"
        config = ConfigManager(section_name=config_name,
                               path=pathlib.Path(__file__).parent.parent /
                                    "configurations" / "tf_fc.ini")
        if name is None:
            self.name = config.get_option("name")
        else:
            self.name = name

        self.layers = {
            "linear": keras.layers.Dense,
        }

        if "layer" in kwargs:
            if not isinstance(kwargs["layer"], keras.layers.Layer):
                self.layer = self.layers[kwargs["layer"]]
            else:
                self.layer: keras.layers.Layer = kwargs["layer"]
        else:
            self.layer = keras.layers.Dense

        self.params = config.get_options_dict()
        self.params.update(kwargs)

        self._model = None
        self.__build_model()

    def __build_model(self) -> Model:
        """_summary_

        Returns
        -------
        Model
            _description_
        """
        input_ = Input(shape=self.input_shape, name="input")
        x = input_
        x = Dropout(rate=self.params["input_dropout"], name="input_dropout")(x)
        for layer_id, layer_size in enumerate(self.params["layers"]):
            x = self.layer(layer_size, name=f"layer_{layer_id}")(x)
            x = Activation(self.params["activation"], name=f"activation_{layer_id}")(x)
            x = Dropout(rate=self.params["dropout"], name=f"dropout_{layer_id}")(x)
        output_ = Dense(self.output_size)(x)
        self._model = Model(inputs=input_,
                            outputs=output_,
                            name=f"{self.name}_model")
        return self._model