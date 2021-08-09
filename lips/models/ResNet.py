import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import multiply as tfk_multiply
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input

from lips.models.layers import ResNetLayer


def ResNet(sizes_enc=(20,),
           sizes_main=(150, 150),
           sizes_out=(40,),
           lr=3e-4,
           scale_main_layer=None,
           scale_input_dec_layer=None,
           scale_input_enc_layer=None,
           layer=ResNetLayer,
           layer_act=None,
           variable_size=None,
           attr_x=("prod_p", "prod_v", "load_p", "load_q"),
           attr_tau=("line_status", "topo_vect"),
           attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                   "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
           optimizer=tf.optimizers.Adam,
           loss="mse",
           metrics=["mean_absolute_error"]
           ):
    pass
