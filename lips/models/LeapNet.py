import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import multiply as tfk_multiply
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input

from lips.models.layers import ResNetLayer, LtauNoAdd


def LeapNet(sizes_enc=(20,),
            sizes_main=(150, 150),
            sizes_out=(40,),
            lr=3e-4,
            scale_main_layer=None,
            scale_input_dec_layer=None,
            scale_input_enc_layer=None,
            layer_act=None,
            variable_size=None,
            attr_x=("prod_p", "prod_v", "load_p", "load_q"),
            attr_tau=("line_status",),  # or "topo_vect"
            attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                    "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
            optimizer="adam",
            loss="mse",
            metrics=["mean_absolute_error"]
            ):

    layer = ResNetLayer

    _sz_x = [variable_size.get(nm_) for nm_ in attr_x]
    _sz_tau = [variable_size.get(nm_) for nm_ in attr_tau]
    _sz_y = [variable_size.get(nm_) for nm_ in attr_y]

    _line_attr = {"a_or", "a_ex", "p_or",
                  "p_ex", "q_or", "q_ex", "v_or", "v_ex"}

    _model = Sequential()
    inputs_x = [Input(shape=(el,), name="x_{}".format(nm_))
                for el, nm_ in zip(_sz_x, attr_x)]
    inputs_tau = [Input(shape=(el,), name="tau_{}".format(nm_))
                  for el, nm_ in zip(_sz_tau, attr_tau)]

    # line status is encoded: 1 disconnected, 0 connected
    # I invert it here
    # these two lines are replace with a Lambda layer which is compatible with plot_model of keras
    #tensor_line_status = inputs_tau[0]
    #tensor_line_status = 1.0 - tensor_line_status
    # tensor_line_status = tf.keras.layers.Lambda(
    #    function=lambda x: 1.0 - x, name='1.-tau')(inputs_tau[0])

    if ("line_status" in attr_tau) | (attr_tau == "line_status"):
        tensor_line_status = inputs_tau[0]
        tensor_line_status = 1.0 - tensor_line_status
    else:
        tensor_line_status = None

    encs_out = []
    for init_val, nm_ in zip(inputs_x, attr_x):
        lay = init_val

        if scale_input_enc_layer is not None:
            # scale up to have higher dimension
            lay = Dense(scale_input_enc_layer,
                        name=f"scaling_input_encoder_{nm_}")(lay)
        for i, size in enumerate(sizes_enc):
            lay_fun = layer(size,
                            name="enc_{}_{}".format(nm_, i),
                            activation=layer_act)
            lay = lay_fun(lay)
            if layer_act is None:
                # add a non linearity if not added in the layer
                lay = Activation("relu")(lay)
        encs_out.append(lay)

    # concatenate all that
    lay = tf.keras.layers.concatenate(encs_out)

    if scale_main_layer is not None:
        # scale up to have higher dimension
        lay = Dense(scale_main_layer, name="scaling_inputs")(lay)

    # i do a few layer
    for i, size in enumerate(sizes_main):
        lay_fun = layer(size,
                        name="main_{}".format(i),
                        activation=layer_act)
        lay = lay_fun(lay)
        if layer_act is None:
            # add a non linearity if not added in the layer
            lay = Activation("relu")(lay)

    encoded_state = lay
    for input_tau, nm_ in zip(inputs_tau, attr_tau):
        tmp = LtauNoAdd(name=f"leap_{nm_}")([lay, input_tau])
        encoded_state = tf.keras.layers.add(
            [encoded_state, tmp], name=f"adding_{nm_}")

    # TODO : implement for other models (resnet and fc)

    # i predict the full state of the grid given the input variables
    outputs_gm = []
    model_losses = {}
    # model_losses = []
    # lossWeights = {}  # TODO
    for sz_out, nm_ in zip(_sz_y, attr_y):
        lay = encoded_state
        if scale_input_dec_layer is not None:
            # scale up to have higher dimension
            lay = Dense(scale_input_dec_layer,
                        name=f"scaling_input_decoder_{nm_}")(lay)
            lay = Activation("relu")(lay)

        for i, size in enumerate(sizes_out):
            lay_fun = layer(size,
                            name="{}_{}".format(nm_, i),
                            activation=layer_act)
            lay = lay_fun(lay)
            if layer_act is None:
                # add a non linearity if not added in the layer
                lay = Activation("relu")(lay)

        # predict now the variable
        name_output = "{}_hat".format(nm_)
        # force the model to output 0 when the powerline is disconnected
        if tensor_line_status is not None and nm_ in _line_attr:
            pred_ = Dense(sz_out, name=f"{nm_}_force_disco")(lay)
            pred_ = tfk_multiply(
                (pred_, tensor_line_status), name=name_output)
        else:
            pred_ = Dense(sz_out, name=name_output)(lay)

        outputs_gm.append(pred_)
        model_losses[name_output] = loss

    _model = Model(inputs=(inputs_x, inputs_tau),
                   outputs=outputs_gm,
                   name="model")

    # TODO : add other optimizers
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam
    optimizer_ = optimizer(lr=lr)

    _model.compile(optimizer=optimizer_, loss=model_losses, metrics=metrics)
    return _model
