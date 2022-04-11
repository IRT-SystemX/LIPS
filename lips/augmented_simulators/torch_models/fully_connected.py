"""
Torch fully connected model
"""
import pathlib
from typing import Union

from torch import nn
import torch.nn.functional as F

from ...logger import CustomLogger
from ...config import ConfigManager

class TorchFullyConnected(nn.Module):
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
                 input_size: int,
                 output_size: int,
                 name: Union[str, None]=None,
                 config_name: Union[str, None]=None,
                 log_path: Union[None, str]=None,
                 **kwargs):
        super(TorchFullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # the config file associoated to this model
        if config_name is None:
            config_name = "DEFAULT"
        config = ConfigManager(section_name=config_name,
                               path=pathlib.Path(__file__).parent.parent /
                                    "configurations" / "torch_fc.ini")
        if name is None:
            self.name = config.get_option("name")
        else:
            self.name = name

        self.params = config.get_options_dict()
        self.params.update(kwargs)

        self.activation = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh
        }

        self.input_layer = None
        self.input_dropout = None
        self.fc_layers = None
        self.dropout_layers = None
        self.output_layer = None

        self.__build_model()

    def __build_model(self):
        """Build the model flow
        """
        # Linear layers
        linear_sizes = list(self.params["layers"])

        self.input_layer = nn.Linear(self.input_size, linear_sizes[0])
        self.input_dropout = nn.Dropout(p=self.params["input_dropout"])

        self.fc_layers = nn.ModuleList([nn.Linear(in_f, out_f) \
            for in_f, out_f in zip(linear_sizes[:-1], linear_sizes[1:])])

        self.dropout_layers = nn.ModuleList([nn.Dropout(p=self.params["dropout"]) \
            for _ in range(len(self.fc_layers))])

        self.output_layer = nn.Linear(linear_sizes[-1], self.output_size)

    def forward(self, data):
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        x = self.input_layer(data)
        x = self.input_dropout(x)
        for _, (fc, dropout) in enumerate(zip(self.fc_layers, self.dropout_layers)):
            x = fc(x)
            x = self.activation[self.params["activation"]](x)
            x = dropout(x)
        x = self.output_layer(x)
        return x