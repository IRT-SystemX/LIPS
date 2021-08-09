# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking
import os
import numpy as np
from tqdm import tqdm
import time
import copy
import shutil
import tempfile
import json
import importlib

from matplotlib import pyplot as plt

import tensorflow as tf

from lips.models import *

# for DC solver
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action import CompleteAction
from grid2op.Backend import PandaPowerBackend


class Benchmark():
    """
    Benchmark class which allow to build, train, evaluate different models on various power grid configurations corresponding to Benchmark 1

    params
    ------
        benchmark_name : ``str``
            a name attributed to the corresponding experiment

        generator : ``GenerateData`` object
            an object of GenerateData class which contains the data for training, validation and testing

        attr_x : ``tuple``
            the set of input variables to be used

        attr_y : ``tuple``
            the set of output variables which should be predicted

        attr_tau : ``tuple``
            the set of topology vectors that should be considered
    """

    def __init__(self,
                 benchmark_name,
                 generator,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                         "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 attr_tau=("line_status",),
                 path="lips\\Outputs",
                 limit_gpu_memory=True
                 ):

        self.benchmark_name = benchmark_name
        # attributes used for input and outputs of a model
        self.attr_x = attr_x
        self.attr_y = attr_y
        self.attr_tau = attr_tau
        # object of class GenerateData contianing datasets for training and validation
        self.generator = generator

        # model and parameters
        self._model = None
        self.model = None
        self.model_name = None
        self.loss = None
        self.metrics = None
        self._my_x = None
        self._my_tau = None
        self._my_y = None
        self.sizes_enc = None
        self.sizes_main = None
        self.sizes_out = None
        self.lr = None
        self.scale_main_layer = None
        self.scale_input_dec_layer = None
        self.scale_input_enc_layer = None
        self.layer_act = None
        self.optimizer = None

        # predictions dictionary containing the predictions for each variable
        self.predictions = dict()
        self.observations = dict()

        # Training and evaluation time
        self.training_time = 0
        self.prediction_time = 0

        # creating a dictionary store train and validation losses and metrics
        self.loss_dicts_init = False
        self.loss_metric_dict_train = None
        self.loss_metric_dict_valid = None
        # the flag indicating if the model is trained
        self._model_trained = False

        # create directories
        benchmark_path = os.path.join(
            path, self.generator.experiment_name, "Benchmarks")
        if not os.path.exists(benchmark_path):
            os.mkdir(benchmark_path)

        self.benchmark_path = os.path.join(benchmark_path, self.benchmark_name)
        if not os.path.exists(self.benchmark_path):
            os.mkdir(self.benchmark_path)

        self.model_path = None
        self.dc_path = None

        if limit_gpu_memory:
            self.limit_gpu_usage()

    def build_model(self,
                    model=LeapNet,
                    sizes_enc=(20,),
                    sizes_main=(150, 150),
                    sizes_out=(40,),
                    lr=3e-4,
                    scale_main_layer=None,
                    scale_input_dec_layer=None,
                    scale_input_enc_layer=None,
                    layer_act=None,
                    optimizer="adam",
                    loss="mse",
                    metrics=["mean_absolute_error"]
                    ):
        """
        A model with model_name is created and returned with respect to indicated parameters

        TODO : the parameters should be more flexible in future, using **kargs
        params
        ------
            model : tensorflow ``Model``
                A tensorflow model that can be trained and evaluated 

            sizes_enc : ``tuple``
                a tuple where the length indicates the number of encoding layers and the values indicate number of neurons

            sizes_main : ``tuple``
                a tuple where the length indicates the number of main layers and the values indicate number of neurons

            sizes_out : ``tuple``
                a tuple where the length indicates the number of decoding layers and the values indicate number of neurons

            layer_act : Tensorflow ``Activation``
                the activation function to be used. The default is Relu when the `layer_act=None`

            lr : ``float``
                learning rate of the optimizer

            optimizer : ``str``
                a optimizer to be used for estimation of neural network parameters

            loss : ``str``
                a loss criteria to be used for learning ("mse", "mae", etc.)

            metrics : ``list``
                a list  of metrics that should be calculated alongside of the loss metric
        """
        self.model = model
        self.model_name = model.__name__
        self.sizes_enc = sizes_enc
        self.sizes_main = sizes_main
        self.sizes_out = sizes_out
        self.lr = lr
        self.scale_main_layer = scale_main_layer
        self.scale_input_dec_layer = scale_input_dec_layer
        self.scale_input_enc_layer = scale_input_enc_layer
        self.layer_act = layer_act
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        # import the model from models folder
        self._model = model(sizes_enc=sizes_enc,
                            sizes_main=sizes_main,
                            sizes_out=sizes_out,
                            lr=lr,
                            scale_main_layer=scale_main_layer,
                            scale_input_dec_layer=scale_input_dec_layer,
                            scale_input_enc_layer=scale_input_enc_layer,
                            layer_act=layer_act,
                            variable_size=self.generator.variable_size,
                            attr_x=self.attr_x,
                            attr_tau=self.attr_tau,
                            attr_y=self.attr_y,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)

    def train_model(self,
                    epochs=20,
                    batch_size=32,
                    save_model=True,
                    tf_writer=None,
                    verbose=0):
        """
        Function to train the selected model on the training dataset already generated.

        Parameters
        -----------
            epochs: ``int``
                Value indicating the number of epochs required for the learning of the built model

            batch_size: ``int``
                Value indicating the batch size used during the learning phase

            save_model : ``bool``
                if True, save the model at self.benchmark_path

            tf_writer : ``tensorboard`` writer
                # TODO : not yet implemented

            verbose: ``bool``
                Whether to show or not the loss and accuracy information during training. The progress is managed by TQDM, it is advised to not use it.
        """
        # verify if the model is built before training
        if self._model is None:
            raise RuntimeError(
                "You asked to train a model that is not still built !")

        self.training_time = 0

        # creating batches of data for training phase
        _x_batches, _tau_batches, _y_batches, steps = self.get_batches(
            tag="training", batch_size=batch_size)
        # creating the batches for validation if validation dataset is provided
        if self.generator.dataset_available["validation"]:
            _x_batches_valid, _tau_batches_valid, _y_batches_valid, _ = self.get_batches(
                tag="validation", batch_size=batch_size)

        # whether to show the progress bar
        if verbose:
            pbar = tqdm(total=epochs*steps)

        # training and validation loop
        for epoch in range(epochs):
            train_losses_step = list()
            ################### training ##################
            for step, _x_train, _tau_train, _y_train in zip(np.arange(steps), _x_batches, _tau_batches, _y_batches):
                data = (_x_train, _tau_train), _y_train
                start_time = time.time()
                # train on a single batch
                train_batch_loss = self._model.train_on_batch(*data)
                self.training_time += time.time() - start_time
                # self._save_tensorboard(train_batch_loss)
                train_losses_step.append(train_batch_loss)
                #print("Epoch number {} / {}, step number {} / {} and losses : {}".format(epoch, epochs, step, steps, train_batch_loss))
                if verbose:
                    pbar.update(1)
            if not(self.loss_dicts_init):
                self.init_loss_dicts()
            # compute the mean of losses per epoch for training
            mean_epoch_train = np.mean(np.vstack(train_losses_step), axis=0)
            for key_, val_ in zip(self.loss_metric_dict_train.keys(), mean_epoch_train):
                self.loss_metric_dict_train[key_].append(val_)
            if verbose == 2:
                print("Epoch number {} / {} with loss {}".format(epoch,
                      epochs, self.loss_metric_dict_train["loss"][-1]))

            ################### validation ##################
            if self.generator.dataset_available["validation"]:
                valid_losses_step = list()
                for _x_valid, _tau_valid, _y_valid in zip(_x_batches_valid, _tau_batches_valid, _y_batches_valid):
                    data_valid = (_x_valid, _tau_valid), _y_valid
                    valid_batch_loss = self._model.test_on_batch(*data_valid)
                    # self._save_tensorboard(valid_batch_loss)
                    valid_losses_step.append(valid_batch_loss)
                # compute the mean of losses per epoch
                mean_epoch_valid = np.mean(
                    np.vstack(valid_losses_step), axis=0)
                for key_, val_ in zip(self.loss_metric_dict_valid.keys(), mean_epoch_valid):
                    self.loss_metric_dict_valid[key_].append(val_)
                if verbose == 2:
                    print("validation loss {}".format(
                        self.loss_metric_dict_valid["loss"][-1]))

            # Save the model and logs in tensorboard
            # if epcohs % save_freq == 0:
            if tf_writer is not None:
                # TODO
                pass

            # if verbose:
            #    pbar.update(1)

        if verbose:
            pbar.close()

        if save_model:
            self.model_path = os.path.join(
                self.benchmark_path, self.model_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            self._save_model(path=self.model_path)

    def init_loss_dicts(self):
        """
        initialize the losses dictionary from the built and trained model

        a loss value is associated to each output variable
        """
        # initialize loss dictionaries with losses from model
        self.loss_metric_dict_train = dict.fromkeys(self._model.metrics_names)
        self.loss_metric_dict_valid = dict.fromkeys(self._model.metrics_names)
        for key_ in self._model.metrics_names:
            self.loss_metric_dict_train[key_] = list()
            self.loss_metric_dict_valid[key_] = list()
        self.loss_dicts_init = True

    def predict(self,
                load_path=None,
                batch_size=32,
                tag="test",
                verbose=0,
                save_predictions=True
                ):
        """
        this module will perform prediction on test sets, and the results could be sent to evaluation class to measure the performance
        of the experimented model

        params
        ------
            load_path : ``str``
                path where the fitted model is located, if it is not in memory

            batch_size : ``int``
                inference batch size

            tag : ``str``
                the tag of a dataset on which inference should be performed using the trained model

            verbose : ``bool``
                whether to show the inference progress

            save_predictions : ``bool``
                if True save the predictions

        """
        if (self._model is None) & (load_path is None):
            raise RuntimeError(
                "The model is not in memory nor a path is indicated")
        elif self._model is None:
            # TODO to implement load_model
            self._load_model()

        #sizes = self._get_output_sizes()
        #total_evaluation_step = self.generator.dataset_size[tag]

        # a key for each tag and each variable
        self.predictions[tag] = {}
        for _var_nm in self.attr_y:
            # np.zeros((total_evaluation_step, self.generator.variable_size[_var_nm]), dtype=np.float32)
            self.predictions[tag][_var_nm] = list()
            #self.observations[tag] = [np.zeros((total_evaluation_step, el), dtype=np.float32) for el in sizes]

        # creating batches of data
        _x_batches, _tau_batches, _y_batches, steps = self.get_batches(
            tag=tag, batch_size=batch_size)
        if verbose:
            pbar = tqdm(total=steps)

        # predicting over the batches
        for step, _x_test, _tau_test, _y_test in zip(np.arange(steps), _x_batches, _tau_batches, _y_batches):
            data = (_x_test, _tau_test), _y_test
            start_time = time.time()
            res = self._model(data, training=False)
            self.prediction_time += time.time() - start_time
            # post process only if the input data is preprocessed
            if self.generator.preprocessed[tag]:
                res = self._post_process(res)

            for _var_nm, _pred in zip(self.attr_y, res):
                self.predictions[tag][_var_nm].append(_pred)
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

        for _var_nm in self.attr_y:
            self.predictions[tag][_var_nm] = np.asarray(
                np.vstack(self.predictions[tag][_var_nm]), dtype=np.float32)

        self.observations[tag] = copy.deepcopy(
            self.generator.dataset_original[tag])
        # TODO : save the predictions
        if save_predictions:
            self.save_predictions(save_path=self.model_path)

    def evaluate_dc(self,
                    name="dc_approx",
                    is_dc=True,
                    # input that will be given to the proxy
                    attr_x=("prod_p", "prod_v", "load_p",
                            "load_q", "topo_vect"),
                    # output that we want the proxy to predict
                    attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                            "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                    tag="test",
                    save_evaluation=True,
                    verbose=1
                    ):
        """
        Evaluate the DC approximation which is used as a baseline for the comparison with other models.

        params
        ------
            name : ``str``
                a name for the DC solver

            is_dc : ``bool``
                whether the solver use DC approximation

            attr_x : ``tuple``
                the set of input variables used for DC approximator, topology vector should be appreard in this tuple

            attr_y : ``tuple``
                the set of output variables used for DC approximator. 

            tag : ``str``
                which dataset should be used for evaluation using DC approximator

            save_evaluation : ``bool``
                if True save the DC predictions

            verbose : ``bool``
                if true, the progress bar will be visualized
        """
        self.model_name = "DC"

        self.predictions[tag] = {}
        nb_observations = self.generator.dataset_size[tag]

        for _var_nm in attr_y:
            self.predictions[tag][_var_nm] = list()

        _sz_x = [self.generator.dataset_original[tag].get(
            nm_).shape[1] for nm_ in attr_x]
        #_sz_y = [self.generator.dataset_original[tag].get(nm_).shape[1] for nm_ in attr_y]

        input_array = []
        for sz in _sz_x:
            input_array.append(
                np.zeros((nb_observations, sz), dtype=np.float32))

        for attr_nm, inp in zip(attr_x, input_array):
            inp[:, :] = self.generator.dataset_original[tag].get(attr_nm)

        # init the solver
        solver, _bk_act_class, _act_class, _indx_var = self._init_dc_solver(self.generator.env._init_grid_path,
                                                                            name,
                                                                            attr_x,
                                                                            attr_y)

        # iterate over test set to infer the output variables using DC approximation
        if verbose:
            pbar = tqdm(total=nb_observations)

        for idx in range(nb_observations):
            res, _pred_time = DCApproximation(solver=solver,
                                              _bk_act_class=_bk_act_class,
                                              _act_class=_act_class,
                                              _indx_var=_indx_var,
                                              idx=idx,
                                              input_array=input_array,
                                              is_dc=is_dc,
                                              attr_x=attr_x,
                                              attr_y=attr_y
                                              )

            self.prediction_time += _pred_time

            for _var_nm, _pred in zip(attr_y, res):
                self.predictions[tag][_var_nm].append(_pred)

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        for _var_nm in self.attr_y:
            self.predictions[tag][_var_nm] = np.asarray(
                np.vstack(self.predictions[tag][_var_nm]), dtype=np.float32)

        self.observations[tag] = copy.deepcopy(
            self.generator.dataset_original[tag])

        if save_evaluation:
            self.dc_path = os.path.join(self.benchmark_path, "DC")
            if not os.path.exists(self.dc_path):
                os.mkdir(self.dc_path)
            self.save_predictions(save_path=self.dc_path)

    def _init_dc_solver(self, path_grid_json, name, attr_x, attr_y):
        """
        initialize the dc solver which is used afterwards to predict the desired variables using inputs

        params
        ------
            path_grid_json : ``str``
                the path to the Grid2op environment

            name : ``str``
                a name for the DC sovler

            attr_x : ``tuple``
                the set of input variables used for DC approximator, topology vector should be appreard in this tuple

            attr_y : ``tuple``
                the set of output variables used for DC approximator.

        Returns
        --------
            solver : ``PandaPowerBackend``
                the DC solver

            _bk_act_class and _act_class

            _indx_var : ``dict``
                a dictionary comprising the indices of attr_x
        """
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            if not el in attr_x:
                raise RuntimeError(
                    f"The DC approximation need the variable \"{el}\" to be computed.")

        _supported_output = {"a_or", "a_ex", "p_or", "p_ex",
                             "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"}
        for el in attr_y:
            if not el in _supported_output:
                raise RuntimeError(f"This solver cannot output the variable \"{el}\" at the moment. "
                                   f"Only possible outputs are \"{_supported_output}\".")

        solver = PandaPowerBackend()
        solver.set_env_name(name)
        solver.load_grid(path_grid_json)
        solver.assert_grid_correct()
        _bk_act_class = _BackendAction.init_grid(solver)
        _act_class = CompleteAction.init_grid(solver)

        _indx_var = {}
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            _indx_var[el] = attr_x.index(el)

        return solver, _bk_act_class, _act_class, _indx_var

    # put in evaluation metrics and reporting class
    def _post_process(self, predicted_state):
        """
        This function post process the predictions of the model

        it returns the variable to their initial scaling if a preprocessing is performed

        params
        ------
            predicted_state :
                predictions of the model
        """
        tmp = [el.numpy() for el in predicted_state]
        resy = [arr_ * self.generator.std_dict[nm_] + self.generator.mean_dict[nm_]
                for arr_, nm_ in zip(tmp, self.attr_y)]
        return resy

    def get_batches(self, tag="training", batch_size=32):
        """
        function create batches of data which could be used for learning and prediction steps of the model

        params
        ------
            tag : ``str``
                indicate the tag of the desired dataset for which we wish have the batches

            batch_size : ``int``
                the batch sizes which indicates how many samples should the model treate at each step of learning

        returns
        -------
            inputs_batches : ``list``
                the batches of input variables stored in a list, each item in the list represents one batch of data

            output_batches : ``list``
                the batches of output variables stored in a list, each item in the list represents one batch of data

            tau_batches : ``list``
                the batches of topology vector stored in a list, each item in the list represents one batch of data

            steps : ``int``
                indicates number of batches which should be introduced to the model

        """
        self._my_x = [tf.convert_to_tensor(
            self.generator.dataset[tag][nm_]) for nm_ in self.attr_x]
        self._my_y = [tf.convert_to_tensor(
            self.generator.dataset[tag][nm_]) for nm_ in self.attr_y]
        self._my_tau = [tf.convert_to_tensor(
            self.generator.dataset[tag][nm_]) for nm_ in self.attr_tau]

        steps = np.int(
            np.ceil((self.generator.dataset_size[tag]) / batch_size))
        input_batches = list()
        tau_batches = list()
        output_batches = list()
        for step in range(steps-1):
            tmp_list = list()
            for arr_ in self._my_x:
                tmp_list.append(arr_[step*batch_size: (step+1)*batch_size])
            input_batches.append(tmp_list)

            tmp_list = list()
            for arr_ in self._my_y:
                tmp_list.append(arr_[step*batch_size: (step+1)*batch_size])
            output_batches.append(tmp_list)

            tmp_list = list()
            for arr_ in self._my_tau:
                tmp_list.append(arr_[step*batch_size: (step+1)*batch_size])
            tau_batches.append(tmp_list)

        tmp_list = list()
        for arr_ in self._my_x:
            tmp_list.append(arr_[(step+1)*batch_size:])
        input_batches.append(tmp_list)

        tmp_list = list()
        for arr_ in self._my_y:
            tmp_list.append(arr_[(step+1)*batch_size:])
        output_batches.append(tmp_list)

        tmp_list = list()
        for arr_ in self._my_tau:
            tmp_list.append(arr_[(step+1)*batch_size:])
        tau_batches.append(tmp_list)

        return input_batches, tau_batches, output_batches, steps

    def _get_output_sizes(self):
        """
        return the size of output variables in a list
        """
        return [self.generator.variable_size[key_] for key_ in self.attr_y]

    def save(self):
        """
        save the benchmark metadata and the trained model parameters for a further use

        It creates automatically a path using experiment name, benchmark name and model name to save the model
        """
        self._save_metadata(path=self.benchmark_path)
        self._save_model(path=self.model_path)

    def _save_model(self, path=None, ext=".h5"):
        """
        store the trained model

        params
        ------
            path : ``str``
                the path where the model should be stored

            ext : ``str``
                the extension of the file for saving the trained model
        """
        self._model.save(os.path.join(path, f"weights{ext}"))

    def _save_metadata(self, path=None):
        """
        Save the benchmark class metadata

        params
        ------
            path : ``str``
                the path where the benchmark metadata should be stored
        """
        res = self._get_metadata()
        json_nm = "metadata_Benchmark.json"
        with open(os.path.join(path, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=res, fp=f)

    def _get_metadata(self):
        """
        get the benchmark metadata in json serializable form
        TODO : save also the loss dicts : loss_metric_dict_train and loss_metric_dict_valid
        returns
        -------
            res : ``dict``
                a dictionary of benchmark metadata
        """
        res = dict()
        res["benchmark_name"] = self.benchmark_name
        res["model"] = self.model_name
        res["attr_x"] = self.attr_x
        res["attr_y"] = self.attr_y
        res["attr_tau"] = self.attr_tau
        res["sizes_enc"] = self.sizes_enc
        res["sizes_main"] = self.sizes_main
        res["sizes_out"] = self.sizes_out
        res["lr"] = self.lr
        res["scale_main_layer"] = self.scale_main_layer
        res["scale_input_dec_layer"] = self.scale_input_dec_layer
        res["scale_input_enc_layer"] = self.scale_input_enc_layer
        res["layer_act"] = self.layer_act
        res["optimizer"] = self.optimizer
        res["loss"] = self.loss
        res["metrics"] = self.metrics
        res["benchmark_path"] = self.benchmark_path
        res["model_path"] = self.model_path
        res["dc_path"] = self.dc_path

        return res

    def load(self, path):
        """
        load the metadata for benchmark and load the model
        TODO : consider a special case for DC approximator
        """
        self._load_metadata(path)

        for tag in self.generator.tag_list:
            self.observations[tag] = copy.deepcopy(
                self.generator.dataset_original[tag])

    def _load_metadata(self, path=None):
        """
        load metadata for the benchmark
        TODO : consider a special case for DC approximator
        """
        json_nm = "metadata_Benchmark.json"
        with open(os.path.join(path, json_nm), "r", encoding="utf-8") as f:
            res = json.load(f)

        self.benchmark_name = res["benchmark_name"]

        self.benchmark_path = res["benchmark_path"]
        self.model_path = res["model_path"]
        self.dc_path = res["dc_path"]

        if self.model_path:
            self.load_and_build_model(res)
            self._load_model(self.model_path)

    def load_and_build_model(self, res):
        model_name = res["model"]
        model = importlib.import_module("lips.models." + model_name)
        self.model = getattr(model, model_name)
        self.attr_x = res["attr_x"]
        self.attr_y = res["attr_y"]
        self.attr_tau = res["attr_tau"]
        self.sizes_enc = res["sizes_enc"]
        self.sizes_main = res["sizes_main"]
        self.sizes_out = res["sizes_out"]
        self.lr = res["lr"]
        self.scale_main_layer = res["scale_main_layer"]
        self.scale_input_dec_layer = res["scale_input_dec_layer"]
        self.scale_input_enc_layer = res["scale_input_enc_layer"]
        self.layer_act = res["layer_act"]
        self.optimizer = res["optimizer"]
        self.loss = res["loss"]
        self.metrics = res["metrics"]

        self.build_model(model=self.model,
                         sizes_enc=self.sizes_enc,
                         sizes_main=self.sizes_main,
                         sizes_out=self.sizes_out,
                         lr=self.lr,
                         scale_main_layer=self.scale_main_layer,
                         scale_input_dec_layer=self.scale_input_dec_layer,
                         scale_input_enc_layer=self.scale_input_enc_layer,
                         layer_act=self.layer_act,
                         optimizer=self.optimizer,
                         loss=self.loss,
                         metrics=self.metrics)

    def _load_model(self, path=None, ext=".h5"):
        """
        Load the model parameters. The model should be built before the loading
        """
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_file = f"weights{ext}"
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(os.path.join(path, nm_file), nm_tmp)
            # load this copy (make sure the proper file is not corrupted)
            self._model.load_weights(nm_tmp)

    def save_predictions(self, save_path=None):
        """
        save the model predictions in file
        """

        # save the data for each existing tag
        for key_, val_ in self.predictions.items():
            tag_dir = os.path.join(save_path, key_)
            if not os.path.exists(tag_dir):
                os.mkdir(tag_dir)
            for key__, val__ in val_.items():
                np.save(os.path.join(tag_dir, f"{key__}_pred.npy"), val__)
        print("Predictions saved successfully !")

    def load_predictions(self, load_path=None):
        """
        Load the predictions 
        TODO : to be implemented
        """
        if self.model:
            pass
        if self.dc_path:
            pass

    def limit_gpu_usage(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        # Currently, memory growth needs to be the same across GPUs
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def plot_convergence(self):
        """
        Visualizing the convergence of the model using training and validation losses curve in function of epochs

        """
        if self.loss_metric_dict_train:
            epochs = len(self.loss_metric_dict_train["loss"])
            # plot here if the loss metric dict is populated
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.loss_metric_dict_train["loss"], label="training loss")
            plt.plot(
                self.loss_metric_dict_valid["loss"], label="validation loss")
            plt.legend()
            plt.grid()
            plt.ylabel(f"{self.loss} value")
            plt.xlabel("Epochs")
            plt.xticks(np.arange(epochs), np.arange(epochs)+1)
            plt.show()

    def visualize_network_state(self):
        """
        TODO : integrate the visualisation tools allowing to visualize the network state over different observations
        """
        pass
