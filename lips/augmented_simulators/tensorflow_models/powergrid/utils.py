import numpy as np
import tensorflow as tf

from ....dataset import DataSet

try:
    from leap_net.proxy import ProxyLeapNet
except ImportError as err:
    raise RuntimeError("You need to install the leap_net package to use this class") from err


class TopoVectTransformation:

    def __init__(self, bench_config, params, dataset):
        self.bench_config = bench_config
        self.params = params
        self.fake_leapnet_proxy = self._init_leapnet_proxy(dataset)

    def _init_leapnet_proxy(self, dataset):
        # initialize a fake leapNet proxy
        leapnet_proxy = ProxyLeapNet(
            attr_x=self.bench_config.get_option("attr_x"),
            attr_y=self.bench_config.get_option("attr_y"),
            attr_tau=self.bench_config.get_option("attr_tau"),
            topo_vect_to_tau=self.params["topo_vect_to_tau"] if "topo_vect_to_tau" in self.params else "raw",
            kwargs_tau=self.params["kwargs_tau"] if "kwargs_tau" in self.params else None,
        )
        obss = self._make_fake_obs(dataset)
        leapnet_proxy.init(obss)
        return leapnet_proxy

    def transform_topo_vect(self, dataset: DataSet):
        """ Extract and transform topo_vect according to the processing method defined by the argument `topo_vect_to_tau`

            This function reuses either the ProxyLeapNet methods to process the tau vector or the
            self ._transform_tau_given_list.
            See https://github.com/BDonnot/leap_net/blob/master/leap_net/proxy/proxyLeapNet.py for more details.

                From the LeapNet documentation :

                    There are multiple ways to process the `tau` vector from the topology of the grid. Some of these
                    different methods have been coded in the LeapNetProxy and are controlled by the `topo_vect_to_tau`
                    argument:

                    1) `topo_vect_to_tau="raw"`: the most straightforward encoding. It transforms the `obs.topo_vect`
                     directly into a `tau` vector of the same dimension with the convention: if obs.topo_vect[i] == 2
                     for a given `i` then `tau[i] = 1` else `tau[i] = 0`. More details are given in
                     the :func:`ProxyLeapNet._raw_topo_vect`, with usage examples on how to create it.
                    2) `topo_vect_to_tau="all"`: it encodes the global topology of the grid by a one hot encoding of the
                     "local topology" of each substation. It first computes all the possible "local topologies" for
                     all the substations of the grid and then assign a number (unique ID) for each of them. The resulting
                     `tau` vector is then the concatenation of the "one hot encoded" ID of the current "local topology"
                     of each substation. More information is given in :func:`ProxyLeapNet._all_topo_encode`
                     with usage examples on how to create it.
                    3) `topo_vect_to_tau="given_list"`: it encodes the topology into a `tau` vector following the same
                     convention as method 2) (`topo_vect_to_tau="all"`) with the difference that it only considers
                     a given list of possible topologies instead of all the topologies of all the substation of the grid.
                     This list should be provided as an input in the `kwargs_tau` argument. If a topology not given
                     is encounter, it is mapped to the reference topology.
                    4) `topo_vect_to_tau="online_list"`: it encodes the topology into a `tau` vector following the same
                     convention as method 2) (`topo_vect_to_tau="all"`) and 3) (`topo_vect_to_tau="given_list"`) but does
                     not require to specify any list of topologies. Instead, each time a new "local topology" is
                     encountered during training, it will be assigned to a new ID. When encountered again, this new
                     ID will be re used. It can store a maximum of different topologies given as `kwargs_tau` argument.
                     If too much topologies have been encountered, the new ones will be encoded as the reference topology.
                Returns
                -------
                topo_vect

                """
        # if option is given_list then use the accelerated function else use the leapnet porxy method
        if "topo_vect_to_tau" in self.params and self.params["topo_vect_to_tau"] == "given_list":
            return self.transform_tau_given_list(dataset.data["topo_vect"], self.fake_leapnet_proxy.subs_index)
        else:
            obss = self._make_fake_obs(dataset)
            return np.array([self.fake_leapnet_proxy.topo_vect_handler(obs) for obs in obss])

    def _make_fake_obs(self, dataset: DataSet):
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

        setattr(FakeObs, "n_sub", dataset.env_data["n_sub"])
        setattr(FakeObs, "sub_info", np.array(dataset.env_data["sub_info"]))

        nb_row = all_data[next(iter(all_data.keys()))].shape[0]
        obss = [FakeObs() for k in range(nb_row)]
        for attr_nm in all_data.keys():
            arr_ = all_data[attr_nm]
            for ind in range(nb_row):
                setattr(obss[ind], attr_nm, arr_[ind, :])
        return obss

    def transform_tau_given_list(self, topo_vect_input, subs_index, with_tf=True):
        """Transform only the tau vector with respect to LeapNet encodings given a list of predefined topological actions
                Parameters
        ----------
        tau : list of raw topology representations (line_status, topo_vect)

        with_tf : transformation using tensorflow or numpy operations

        Returns
        -------
        tau
            list of encoded topology representations (line_status, topo_vect_encoded)
        """
        ##############
        # WARNING: TO DO
        # if we find two topology matches at a same substation, the current code attribute one bit for each
        # But only one should be choosen in the end (we are not in a quantum state, or it does not make sense to combine topologies at a same substation in the encoding here
        # This can happen when there are several lines disconnected at a substation on which we changed the topology, probably in benchmark 3, but probably not in benchmark 1 and 2

        list_topos = []
        sub_length = []
        for topo_action in self.params["kwargs_tau"]:
            topo_vect = np.zeros(topo_vect_input.shape[1], dtype=np.int32)
            sub_id = topo_action[0]
            sub_topo = np.array(topo_action[1])
            sub_index = subs_index[sub_id][0]
            n_elements = len(sub_topo)
            topo_vect[sub_index:sub_index + n_elements] = sub_topo
            list_topos.append(topo_vect)
            sub_length.append(n_elements)

        list_topos = np.array(list_topos)

        # we are here looking for the number of matches for every element of a substation topology in the predefined list for a new topo_vect observation
        # if the count is equal to the number of element, then the predefined topology is present in topo_vect observation
        # in that case, the binary encoding of that predefined topology is equal to 1, otherwise 0

        import time
        start = time.time()
        if with_tf:
            # count the number of disconnected lines for each substation of topologies in the prefdefined list.
            # These lines could have been connected to either bus_bar1 or bus_bar2, we consider it as a match for that element
            line_disconnected_sub = tf.linalg.matmul((list_topos > 0).astype(np.int32),
                                                     (np.transpose(topo_vect_input) < 0).astype(np.int32))

            # we look at the number of elements on bus_bar1 that match, same for the number of elements on bus_bar2
            match_tensor_bus_bar1 = tf.linalg.matmul((list_topos == 1).astype(np.int32),
                                                     (np.transpose(topo_vect_input) == 1).astype(np.int32))
            match_tensor_bus_bar2 = tf.linalg.matmul((list_topos == 2).astype(np.int32),
                                                     (np.transpose(topo_vect_input) == 2).astype(np.int32))

            # the number of matches is equal to the sum of those 3 category of matches
            match_tensor_adjusted = match_tensor_bus_bar1 + match_tensor_bus_bar2 + line_disconnected_sub

            # we see if all elements match by dividing by the number of elements. If this proportion is equal to one, we found a topology match
            normalised_tensor = match_tensor_adjusted / tf.reshape(np.array(sub_length).astype(np.int32), (-1, 1))

        else:  # with_numpy

            line_disconnected_sub = np.matmul((list_topos > 0), 1 * (np.transpose(topo_vect_input) < 0))

            match_tensor_bus_bar1 = np.matmul((list_topos == 1), 1 * (np.transpose(topo_vect_input) == 1))
            match_tensor_bus_bar2 = np.matmul((list_topos == 2), 1 * (np.transpose(topo_vect_input) == 2))

            match_tensor_adjusted = match_tensor_bus_bar1 + match_tensor_bus_bar2 + line_disconnected_sub

            normalised_tensor = match_tensor_adjusted / np.array(sub_length).reshape((-1, 1))

        boolean_match_tensor = np.array(normalised_tensor == 1.0).astype(np.int8)

        duration_matches = time.time() - start

        #############"
        ## do correction if multiple topologies of a same substation have a match on a given state
        # as it does not make sense to combine topologies at a same substation
        start = time.time()
        boolean_match_tensor = self._unicity_tensor_encoding(boolean_match_tensor)

        duration_correction = time.time() - start
        if (duration_correction > duration_matches):
            print("warning, correction time if longer that matches time: maybe something to better optimize there")
        topo_vect_input = np.transpose(boolean_match_tensor)

        return topo_vect_input

    def _unicity_tensor_encoding(self, tensor):
        """
        do correction if multiple topologies of a same substation have a match on a given state
        as it does not make sense to combine topologies at a same substation
        """
        sub_encoding_pos = np.array([topo_action[0] for topo_action in self.params["kwargs_tau"]])

        # in case of multiple matches of topology for a given substation, encode only one of those topologies as an active bit, not several
        def per_col(a):  # to only have one zero per row
            idx = a.argmax(0)
            out = np.zeros_like(a)
            r = np.arange(a.shape[1])
            out[idx, r] = a[idx, r]
            return out

        for sub in set(sub_encoding_pos):
            indices = np.where(sub_encoding_pos == sub)[0]
            if (len(indices) >= 2):
                tensor[indices, :] = per_col(tensor[indices, :])

        return tensor
