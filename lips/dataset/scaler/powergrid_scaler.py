import json
import pathlib
import numpy as np

from . import Scaler
from ..powergridDataSet import PowerGridDataSet
from ...utils import NpEncoder

class PowerGridScaler(Scaler):
    """Power grids specific scaler
    """
    def __init__(self):
        self.dtype = np.float32
        self._m_x = []
        self._sd_x = []
        # for the output
        self._m_y = []
        self._sd_y = []
        # for the tau vectors
        self._m_tau = []
        self._sd_tau = []

        self._attr_x = None
        self._attr_y = None
        self._attr_tau = None
        self.obss = None
        self.dataset = None

    def fit(self, dataset: PowerGridDataSet):
        self.dataset = dataset
        self._attr_x = dataset._attr_x
        self._attr_y = dataset._attr_y
        self._attr_tau = dataset._attr_tau
        obss = self._make_fake_obs(dataset)
        self.obss = obss

        for attr_nm in self._attr_x:
            # Fetch for tau
            self._m_x.append(self._get_mean(obss, attr_nm))
            self._sd_x.append(self._get_sd(obss, attr_nm))
        for attr_nm in self._attr_y:
            self._m_y.append(self._get_mean(obss, attr_nm))
            self._sd_y.append(self._get_sd(obss, attr_nm))
        for attr_nm in self._attr_tau:
            self._m_tau.append(self._get_mean(obss, attr_nm))
            self._sd_tau.append(self._get_sd(obss, attr_nm))

    def transform(self, dataset: PowerGridDataSet):
        res_x = []
        res_tau = []
        res_y = []
        all_data = dataset.data
        for attr_nm, m_, sd_ in zip(self._attr_x, self._m_x, self._sd_x):
            res_x.append((all_data[attr_nm] - m_) / sd_)
        for attr_nm, m_, sd_ in zip(self._attr_tau, self._m_tau, self._sd_tau):
            if attr_nm == "line_status":
                res_tau.append((all_data[attr_nm] - m_) / sd_)
            elif attr_nm == "topo_vect":
                res_tau.append(all_data[attr_nm])#(all_data[attr_nm] - m_) / sd_)
                # if self.obss is None:
                #     self.obss = self._make_fake_obs(dataset)
                # res_tau.append(np.array([leap_net_model.topo_vect_handler(obs)
                #                          for obs in self.obss],
                #                          dtype=np.float32))
            else:
                raise RuntimeError(f"Unknown tau attribute : {attr_nm}")
        for attr_nm, m_, sd_ in zip(self._attr_y, self._m_y, self._sd_y):
            res_y.append((all_data[attr_nm] - m_) / sd_)

        return (res_x, res_tau), res_y

    def fit_transform(self, dataset: PowerGridDataSet):
        self.fit(dataset)
        (res_x, res_tau), res_y = self.transform(dataset)
        return (res_x, res_tau), res_y

    def inverse_transform(self, pred_y):
        res = {}
        for attr_nm, arr_, m_, sd_ in zip(self._attr_y, pred_y, self._m_y, self._sd_y):
            res[attr_nm] = (arr_ * sd_) + m_
        return res

    def transform_tau(self, tau, leap_net_model):
        """Transform only the tau vector with respect to LeapNet encodings
        """
        if self.obss is None:
            self.obss = self._make_fake_obs(self.dataset)
        tau.pop()
        tau.append(np.array([leap_net_model.topo_vect_handler(obs)
                            for obs in self.obss],
                            dtype=np.float32))
        return tau

    def _extract_obs(self, obss, attr_nm):
        return getattr(obss, attr_nm)

    def _get_mean(self, obss, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data
        This function can be overridden (for example if you want more control on how to scale the data)
        obss is a list of observation
        """

        if attr_nm =="topo_vect" : # Exclude topo_vect from scaling
            add_ = 0
        else :
            add_, _ = self._get_adds_mults_from_name(obss, attr_nm)
        return add_

    def _get_sd(self, obss, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data
        This function can be overridden (for example if you want more control on how to scale the data)
        obss is a list of observation
        """
        if attr_nm == "topo_vect": # Exclude topo_vect from scaling
            mul_ = 1
        else :
            _, mul_ = self._get_adds_mults_from_name(obss, attr_nm)
        return mul_

    def _get_adds_mults_from_name(self, obss, attr_nm):
        """
        extract the scalers (mean and std) used for the observation
        We don't recommend to overide this function, modify the function `_get_mean` and `_get_sd` instead
        obss is a list of observation obtained from running some environment with just the "actor" acting on
        the grid. The size of this list is set by `AgentWithProxy.nb_obs_init`
        Notes
        ------
        for each variables, the data are scaled with:
        data_scaled = (data_raw - add_tmp) / mult_tmp
        """
        obs = obss[0]
        add_tmp = np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
        mult_tmp = np.std([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype) + 1e-1

        if attr_nm in ["prod_p"]:
            # mult_tmp = np.array([max((pmax - pmin), 1.) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)],
            #                     dtype=self.dtype)
            # default values are good enough
            pass
        elif attr_nm in ["prod_q"]:
            # default values are good enough
            pass
        elif attr_nm in ["load_p", "load_q"]:
            # default values are good enough
            pass
        elif attr_nm in ["load_v", "prod_v"]:
            # default values are good enough
            # stds are almost 0 for loads, this leads to instability
            add_tmp = np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
            mult_tmp = 1.0  # np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
        elif attr_nm in ["v_or", "v_ex"]:
            # default values are good enough
            add_tmp = self.dtype(0.)  # because i multiply by the line status, so i don't want any bias
            mult_tmp = np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
        elif attr_nm == "hour_of_day":
            add_tmp = self.dtype(12.)
            mult_tmp = self.dtype(12.)
        elif attr_nm == "minute_of_hour":
            add_tmp = self.dtype(30.)
            mult_tmp = self.dtype(30.)
        elif attr_nm == "day_of_week":
            add_tmp = self.dtype(4.)
            mult_tmp = self.dtype(4)
        elif attr_nm == "day":
            add_tmp = self.dtype(15.)
            mult_tmp = self.dtype(15.)
        elif attr_nm in ["target_dispatch", "actual_dispatch"]:
            add_tmp = self.dtype(0.)
            mult_tmp = np.array([max((pmax - pmin), 1.) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)],
                                dtype=self.dtype)
        elif attr_nm in ["p_or", "p_ex", "q_or", "q_ex"]:
            add_tmp = self.dtype(0.)  # because i multiply by the line status, so i don't want any bias
            mult_tmp = np.array([max(np.abs(val), 1.0) for val in getattr(obs, attr_nm)], dtype=self.dtype)
        elif attr_nm in ["a_or", "a_ex"]:
            add_tmp = self.dtype(0.)  # because i multiply by the line status, so i don't want any bias
            if hasattr(obs, "thermal_limit"):
                mult_tmp = obs.thermal_limit
            elif hasattr(obs, "rho"):
                mult_tmp = np.abs(obs.a_or / (obs.rho + 1e-2))  # which is equal to the thermal limit
            else:
                # defaults would be good enough
                pass
            mult_tmp[mult_tmp <= 1.] = 1.0
        elif attr_nm == "line_status":
            # encode back to 0: connected, 1: disconnected
            add_tmp = self.dtype(1.)
            mult_tmp = self.dtype(-1.)
        return add_tmp, mult_tmp

    def _make_fake_obs(self, dataset: PowerGridDataSet):
        """
        the underlying _leap_net_model requires some 'class' structure to work properly. This convert the
        numpy dataset into these structures.

        Definitely not the most efficient way to process a numpy array...
        """
        all_data = dataset.data
        class FakeObs(object):
            pass

        if "topo_vect" in all_data:
            setattr(FakeObs, "dim_topo", all_data["topo_vect"].shape[1])

        # TODO find a way to retrieve that from data...
        # TODO maybe the "dataset" class should have some "static data" somewhere
        setattr(FakeObs, "n_sub", dataset.env_data["n_sub"])
        setattr(FakeObs, "sub_info", np.array(dataset.env_data["sub_info"]))

        nb_row = all_data[next(iter(all_data.keys()))].shape[0]
        obss = [FakeObs() for k in range(nb_row)]
        for attr_nm in all_data.keys():
            arr_ = all_data[attr_nm]
            for ind in range(nb_row):
                setattr(obss[ind], attr_nm, arr_[ind, :])
        return obss

    def save(self, path: str):
        res_json = {}
        res_json["_m_x"] = self._m_x
        res_json["_sd_x"] = self._sd_x
        res_json["_m_tau"] = self._m_tau
        res_json["_sd_tau"] = self._sd_tau
        res_json["_m_y"] = self._m_y
        res_json["_sd_y"] = self._sd_y
        res_json["_attr_x"] = self._attr_x
        res_json["_attr_tau"] = self._attr_tau
        res_json["_attr_y"] = self._attr_y

        with open((path / "scaler_params.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def load(self, path: str):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open((path / "scaler_params.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self._m_x = res_json["_m_x"]
        self._sd_x = res_json["_sd_x"]
        self._m_tau = res_json["_m_tau"]
        self._sd_tau = res_json["_sd_tau"]
        self._m_y = res_json["_m_y"]
        self._sd_y = res_json["_sd_y"]
        self._attr_x = res_json["_attr_x"]
        self._attr_tau = res_json["_attr_tau"]
        self._attr_y = res_json["_attr_y"]
