import torch
from torch import Tensor
from torch import nn
from torch import optim

def rename(newname):
    """
    Decorator function to change function names
    """
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

class CustomMAELoss(nn.L1Loss):
    """
    Custom MAE loss
    """
    def __init__(self,
                 *args,
                 **kwargs
                ):
        super(CustomMAELoss, self).__init__(**kwargs)
        if len(args) > 1:
            self.seq_len = args[0]
            self.device = args[1]
        elif len(args) > 0:
            self.seq_len = args[0]
            self.device = "cpu"
        else:
            self.seq_len = None
            self.device = None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of CustomMAELoss

        Parameters
        ==========
          pred: torch.tensor
            Predictions of shape: (`batch_size`, `num_steps`, 1)
          label: torch.tensor
            Label of shape: (`batch_size`, `num_steps`, 1)
          valid_len: torch.tensor
            Valid Length of shape (`batch_size`,)

        Returns
        =======
          weighted_loss: float
            Weighted Loss
        """
        if self.seq_len is not None:
            self.reduction = 'none'
            mae_loss = super(CustomMAELoss,
                            self).forward(input, target)
            weights = torch.ones_like(target)
            for i, len_ in enumerate(self.seq_len):
                weights[i, len_:, :] = 0
            #weights = sequence_pad(weights, seq_len, 1)

            # weighted_loss = (unweighted_loss * weights).mean()
            #weighted_loss = (unweighted_loss * weights)
            mae_loss = torch.sum((mae_loss * weights).sum(dim=1) /
                                  self.seq_len.view(-1,1).to(self.device))
        else:
            mae_loss = super(CustomMAELoss,
                            self).forward(input, target)
        return mae_loss



class CustomMSELoss(nn.MSELoss):
    """
    The Mean Squared Error Loss with possibility to use masks for RNN

    .. code-block:: python

        from lips.augmented_simulators.torch_models.utils import CustomMSELoss
        loss = CustomMSELoss(size_average=None, reduce=None, reduction = 'mean')

    Parameters
    ----------
    *args: sequence lengths and device to be given in order

    """
    def __init__(self,
                 *args,
                 **kwargs
                ):
        super(CustomMSELoss, self).__init__(**kwargs)
        if len(args) > 1:
            self.seq_len = args[0]
            self.device = args[1]
        elif len(args) > 0:
            self.seq_len = args[0]
            self.device = "cpu"
        else:
            self.seq_len = None
            self.device = None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass of CustomMSELoss

        Parameters
        ----------
        input : Tensor
            Predictions of shape: (`batch_size`, `num_steps`, 1) or (`batch_size`, 1)
        target : Tensor
            Label of shape: (`batch_size`, `num_steps`, 1) or (`batch_size`, 1)

        Returns
        -------
        Tensor
            Weighted Loss if seq_len is given else unweighted loss
        """
        if self.seq_len is not None:
            self.reduction = 'none'
            mse_loss = super(CustomMSELoss,
                            self).forward(input, target)
            weights = torch.ones_like(target)
            for i, len_ in enumerate(self.seq_len):
                weights[i, len_:, :] = 0
            #weights = sequence_pad(weights, seq_len, 1)

            # weighted_loss = (unweighted_loss * weights).mean()
            #weighted_loss = (unweighted_loss * weights)
            mse_loss = torch.sum((mse_loss * weights).sum(dim=1) /
                                  self.seq_len.view(-1,1).to(self.device))
        else:
            mse_loss = super(CustomMSELoss,
                            self).forward(input, target)
        return mse_loss

LOSSES = {"MAELoss": CustomMAELoss,
          "MaskedMAELoss": lambda seq_len, device: CustomMAELoss(seq_len, device),
          "MSELoss": CustomMSELoss,
          "MaskedMSELoss": lambda seq_len, device: CustomMSELoss(seq_len, device)
          }
OPTIMIZERS = {"adam": optim.Adam, "sgd": optim.SGD}


'''
@rename("MAELoss")
def mae_loss(input: Tensor, target: Tensor, *args) -> Tensor:
    """
    Compute the Weighted MAE criteria by masking the paddings

    Parameters
    ==========
    input: The predictions
    target: the ground truth values
    *args: sequence lengths (in the case of variable length sequences) and device in order

    Return
    ======
    MAE loss computed on input and target
    """
    if len(args) > 1:
        seq_len = args[0]
        device = args[1]
    elif len(args) > 0:
        seq_len = args[0]
        device = "cpu"
    else:
        seq_len = None

    mae = torch.abs(input - target)
    #weighted_MAE = (unweighted_MAE * weights).mean()
    if seq_len is not None:
        weights = torch.ones_like(target)
        for i, len_ in enumerate(seq_len):
            weights[i, len_:, :] = 0
        mae = torch.sum((mae * weights).sum(dim=1) /
                         seq_len.view(-1,1).to(device))
    else:
        mae = mae.mean()
    return mae
'''