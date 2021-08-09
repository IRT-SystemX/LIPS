import numpy as np
import time


def DCApproximation(solver,
                    _bk_act_class,
                    _act_class,
                    _indx_var,
                    idx,
                    input_array,
                    name="dc_approx",
                    is_dc=True,
                    # input that will be given to the proxy
                    attr_x=("prod_p", "prod_v", "load_p",
                            "load_q", "topo_vect"),
                    # output that we want the proxy to predict
                    attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or",
                            "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                    ):
    """
    DC approximation

    params
    ------
        name : ``str``
            the name of the solver

        is_dc : ``bool``
            to consider the dc case in solver

        attr_x : ``set``
            the set of names of inputs variables

        attr_y : ``set``
            the set of names of output variables

        idx : ``int``
            the index of current observation to predict
    """
    arr_ = input_array[_indx_var["topo_vect"]]
    input_array[_indx_var["topo_vect"]] = arr_.astype(np.int)

    res = _bk_act_class()
    act = _act_class()
    act.update({"set_bus": input_array[_indx_var["topo_vect"]][idx, :],
                "injection": {
                    "prod_p": input_array[_indx_var["prod_p"]][idx, :],
                    "prod_v": input_array[_indx_var["prod_v"]][idx, :],
                    "load_p": input_array[_indx_var["load_p"]][idx, :],
                    "load_q": input_array[_indx_var["load_q"]][idx, :],
    }
    })
    res += act
    solver.apply_action(res)

    _beg = time.time()
    solver.runpf(is_dc=is_dc)
    _pred_time = time.time() - _beg

    predicted_state = []
    tmp = {}
    tmp["p_or"], tmp["q_or"], tmp["v_or"], tmp["a_or"] = solver.lines_or_info()
    tmp["p_ex"], tmp["q_ex"], tmp["v_ex"], tmp["a_ex"] = solver.lines_ex_info()
    tmp1, tmp2, tmp["load_v"] = solver.loads_info()
    tmp1, tmp["prod_q"], tmp2 = solver.generators_info()
    for el in attr_y:
        # the "1.0 * " is here to force the copy...
        predicted_state.append(1. * tmp[el].reshape(1, -1))
    return predicted_state, _pred_time
