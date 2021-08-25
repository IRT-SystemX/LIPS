import numpy as np
import time
import copy
import os

from numpy.core.fromnumeric import nonzero
from lips.simulators import ProxyBackend
from lips.simulators import Simulator
from tqdm import tqdm


from grid2op.dtypes import dt_int


class DCApproximation(ProxyBackend, Simulator):
    """
    This class adapts the leap_net implementation of DC approximation for benchmarking

    """
    def __init__(self,
                 path_grid_json, # complete path where the grid is represented as a json file
                 name="dc_approx",
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),  # input that will be given to the proxy
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),  # output that we want the proxy to predict
                ):
        ProxyBackend.__init__(self,
                              path_grid_json=path_grid_json,
                              attr_x=attr_x,
                              attr_y=attr_y
                             )
        Simulator.__init__(self, name=name)

        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            if not el in self.attr_x:
                raise RuntimeError(f"The DC approximation need the variable \"{el}\" to be computed.")
        for el in self.attr_y:
            if not el in self._supported_output:
                raise RuntimeError(f"This solver cannot output the variable \"{el}\" at the moment. "
                                   f"Only possible outputs are \"{self._supported_output}\".")

        self.nb_samples = None

        # internal variables (speed optimisation)
        self._indx_var = {}
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            self._indx_var[el] = self.attr_x.index(el)

        self.predict_time = 0

        self.obs_list = None


    def init(self, obs_list):
        self.nb_samples = len(obs_list)
        self.obs_list = obs_list
        self.process_obs(obs_list)


    def process_obs(self, obs_list):
        """
        This function will process the grid2op observations and create input and outputs for the physical simulator

        """
        obs = obs_list[0]
        self._x = []
        self._sz_x = []
        self._y = []
        self._sz_y = []

        for attr_nm in self.attr_x:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_x.append(sz)
        
        for sz in self._sz_x:
            self._x.append(np.zeros((self.nb_samples, sz), dtype=self.dtype))

        for attr_nm in self.attr_y:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_y.append(sz)

        for sz in self._sz_y:
            self._y.append(np.zeros((self.nb_samples, sz), dtype=self.dtype))

        self._y_hat = copy.deepcopy(self._y)
        

        for idx in range(self.nb_samples):
            for attr_nm, arr_ in zip(self.attr_x, self._x):
                arr_[idx, :] = getattr(obs_list[idx], attr_nm)
            for attr_nm, arr_ in zip(self.attr_y, self._y):
                arr_[idx, :] = getattr(obs_list[idx], attr_nm)
        
        arr_ = self._x[self._indx_var["topo_vect"]]
        self._x[self._indx_var["topo_vect"]] = arr_.astype(dt_int)        


    def predict(self, save_path=None, verbose=True):
        """
        Compute the flow from injection using DC approximation
        The prediction in the case of DC approximator is one observation at a time
        """
        if verbose:
            pbar = tqdm(total=self.nb_samples)

        for obs_idx in range(self.nb_samples):
            self._extract_data(obs_idx)
            _beg = time.time()
            self._make_predictions()
            _pred_time = time.time() - _beg
            self.predict_time += _pred_time

            res = []
            res = self._post_process(res)

            for _pred, arr_ in zip(res, self._y_hat):
                arr_[obs_idx,:] = _pred

            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

        if save_path is not None:
            self.save_data(save_path)

        return copy.deepcopy(self._y_hat), copy.deepcopy(self._y), self.predict_time

    def _make_predictions(self):
        """
        compute the dc powerflow.

        In the formalism of grid2op backends, this is done with calling the function "runpf"
        """
        self.solver.runpf(is_dc=self.is_dc)
        return None

    def _extract_data(self, obs_idx):
        res = self._bk_act_class()
        act = self._act_class()
        act.update({"set_bus": self._x[self._indx_var["topo_vect"]][obs_idx, :],
                    "injection": {
                        "prod_p": self._x[self._indx_var["prod_p"]][obs_idx, :],
                        "prod_v": self._x[self._indx_var["prod_v"]][obs_idx, :],
                        "load_p": self._x[self._indx_var["load_p"]][obs_idx, :],
                        "load_q": self._x[self._indx_var["load_q"]][obs_idx, :],
                        }
                    })
        res += act
        self.solver.apply_action(res)
        return None, None

    def data_to_dict(self):
        observations = dict()
        predictions = dict()

        for attr_nm in (*self.attr_x, *self.attr_y, "line_status"):
            observations[attr_nm] = np.vstack([getattr(obs, attr_nm) for obs in self.obs_list])

        for attr_nm, var_ in zip(self.attr_y, self._y_hat):
            predictions[attr_nm] = var_

        return observations, predictions


    def save_data(self, path):
        observations, predictions = self.data_to_dict()

        if not os.path.exists(path):
            os.mkdir(path)
        simulator_path = os.path.join(path, self.name)
        if not os.path.exists(simulator_path):
            os.mkdir(simulator_path)

        for key_, val_ in predictions.items():
            np.save(os.path.join(simulator_path, f"{key_}_pred.npy"), val_)

        for key_, val_ in observations.items():
            np.save(os.path.join(simulator_path, f"{key_}_real.npy"), val_)

        print("Observations and Predictions are saved successfully !")

    def load_data(self, path):
        if not os.path.exists(path):
            print("the indicated path does not exist!")
            return
        
        

