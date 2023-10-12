import os, shutil
import copy
import operator
from typing import Union, Callable
import numpy as np

import airfrans as af

from lips.dataset.dataSet import DataSet
from lips.config.configmanager import ConfigManager
from lips.logger.customLogger import CustomLogger

def download_data(root_path, directory_name):
    af.dataset.download(root = root_path, file_name = directory_name, unzip = True, OpenFOAM = False)

class AirfRANSDataSet(DataSet):

    def __init__(self, 
                 config: Union[None, ConfigManager],
                 name: Union[None, str], 
                 task : str,
                 split : str,
                 attr_names: Union[tuple, None] = None,
                 log_path: Union[str, None] = None,
                 **kwargs
                ):
        super().__init__(name = name)
        assert task in ['full', 'scarce', 'reynolds', 'aoa'], "task %s not supported for this dataset.\n" %task \
        +"Available choices: full, scarce, reynolds, aoa" 
        self._task = task
        assert split in ["training","testing"], "split %s not supported for this dataset. Available choices: training, testing"
        self._split = split
        self._attr_names = copy.deepcopy(attr_names)
        self.extra_data = dict()

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        self.config = config

        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_y = None
        self._sizes_x = None  # dimension of each variable
        self._sizes_y = None  # dimension of each variable
        self._attr_x = kwargs["attr_x"] if "attr_x" in kwargs.keys() else self.config.get_option("attr_x")
        self._attr_y = kwargs["attr_y"] if "attr_y" in kwargs.keys() else self.config.get_option("attr_y")
        self.no_normalization_attr_x = ['x-normals','y-normals']

    def load(self, path: str):
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        
        variables=[
            'x-position','y-position','x-inlet_velocity','y-inlet_velocity','distance_function','x-normals','y-normals',
            'x-velocity','y-velocity','pressure','turbulent_viscosity','surface'
            ]
        indices_variables = dict(zip(variables,range(len(variables))))
        split = True if self._split=="training" else False
        dataset, simulation_names = af.dataset.load(root = path, task = self._task, train = split)
        simulation_size = np.array([data.shape[0] for data in dataset])[:, None]
        simulation_names = np.concatenate([np.array(simulation_names)[:, None], simulation_size], axis = 1)
        self.data = {}
        for key in self._attr_names:
            self.data[key] = np.concatenate([sim[:, indices_variables[key]] for sim in dataset], axis = 0)
        self.extra_data['surface'] = np.concatenate([sim[:, indices_variables['surface']] for sim in dataset], axis = 0)
        self.extra_data['simulation_names'] = simulation_names
        self._infer_sizes()

    def _infer_sizes(self):
        """Infer the data sizes"""
        self._sizes_x = np.array([1 if len(self.data[el].shape)==1 else self.data[el].shape[1] for el in self._attr_x], dtype=int)
        self._size_x = np.sum(self._sizes_x)
        self._sizes_y = np.array([1 if len(self.data[el].shape)==1 else self.data[el].shape[1] for el in self._attr_y], dtype=int)
        self._size_y = np.sum(self._sizes_y)

    def get_sizes(self):
        """Get the sizes of the dataset

        Returns
        -------
        tuple
            A tuple of size (size_x, size_y)

        """
        return self._size_x, self._size_y
    
    def get_simulations_sizes(self):
        """Get the size of each simulation

        Returns
        -------
        list
            A list of size number of simulation

        """
        return [int(simulation[1]) for simulation in self.extra_data["simulation_names"]]

    def extract_data(self) -> tuple:
        """extract the x and y data from the dataset

        Parameters
        ----------
        concat : ``bool``
            If True, the data will be concatenated in a single array.
        Returns
        -------
        tuple
            extracted inputs and outputs
        """
        # init the sizes and everything
        # data = copy.deepcopy(self.data)
        extract_x = np.concatenate([self.data[key][:, None].astype(np.single) for key in self._attr_x], axis = 1)
        extract_y = np.concatenate([self.data[key][:, None].astype(np.single) for key in self._attr_y], axis = 1)
        return extract_x, extract_y

    def get_no_normalization_axis_indices(self):
        no_normalization_indices_x=np.array([self._attr_x.index(attr) for attr in self.no_normalization_attr_x])
        return no_normalization_indices_x

    def reconstruct_output(self, data: "np.ndarray") -> dict:
        """It reconstruct the data from the extracted data

        Parameters
        ----------
        data : ``np.ndarray``
            the array that should be reconstruted

        Returns
        -------
        dict
            the reconstructed data with variable names in a dictionary
        """
        predictions = {}
        prev_ = 0
        for var_id, this_var_size in enumerate(self._sizes_y):
            attr_nm = self._attr_y[var_id]
            predictions[attr_nm] = np.squeeze(data[:, prev_:(prev_ + this_var_size)])
            prev_ += this_var_size
        return predictions


    def __str__(self)->str:
        s_info="Instance of "+type(self).__name__+"\n"
        s_info+="Dataset name: "+str(self.name)+"\n"
        for paramName, paramVal in self.data.items():
            s_info+="\t"+str(paramName)+"\n"
            s_info+="\t\t Data size: "+str(paramVal.shape)+"\n"
        return s_info

    def __eq__(self, other)->bool:
        if sorted(self._attr_names)!=sorted(other._attr_names):
            return False
        for data_attrib_name in self._attr_names:
            try:
                np.testing.assert_almost_equal(self.data[data_attrib_name],other.data[data_attrib_name])
            except AssertionError:
                self.logger.info("%s data not the same for both dataset",data_attrib_name)
                return False

        try:
            np.testing.assert_almost_equal(self.extra_data['surface'],other.extra_data['surface'])
        except AssertionError:
            self.logger.info("surface data not the same for both dataset")
            return False

        try:
            nbTerm=len(self.extra_data["simulation_names"][0])
            for idTerm in range(nbTerm):
                term=[term[idTerm] for term in self.extra_data["simulation_names"]]
                termComp=[term[idTerm] for term in other.extra_data["simulation_names"]]
                assert term==termComp
        except AssertionError:
            self.logger.info("simulation_names data not the same for both dataset")
            return False
        return True

def save_internal(dataset, path_out):
    full_path_out = os.path.join(os.path.abspath(path_out), dataset.name)
    if not os.path.exists(os.path.abspath(path_out)):
        os.mkdir(os.path.abspath(path_out))
    if os.path.exists(full_path_out):
        shutil.rmtree(full_path_out)

    os.mkdir(full_path_out)
    for attr_nm in dataset._attr_names:
        np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data = dataset.data[attr_nm])

    for extra_attr_nm,extra_attr_val in dataset.extra_data.items():
        np.savez_compressed(f"{os.path.join(full_path_out, extra_attr_nm)}.npz", data = extra_attr_val)

def reload_dataset(path_in,name,task,split,attr_x,attr_y):
    """Load the internal data

    Parameters
    ----------
    path: input path
        A str to indicate where to load the data from.
    """
    if not os.path.exists(path_in):
        raise RuntimeError(f"{path} cannot be found on your computer")
    if not os.path.isdir(path_in):
        raise RuntimeError(f"{path} is not a valid directory")
    full_path = os.path.join(path_in, name)
    if not os.path.exists(full_path):
        raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                           f"a given `path_out` ?")

    attr_names=attr_x+attr_y
    for attr_nm in attr_names:
        path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
        if not os.path.exists(path_this_array):
            raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                               f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

    dataset_from_data=AirfRANSDataSet(config = None,
                                    name=name,
                                    task = task,
                                    split = split,
                                    attr_names=attr_names,
                                    attr_x= attr_x,
                                    attr_y= attr_y)

    if dataset_from_data.data is not None:
        warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
    dataset_from_data.data = {}

    for attr_nm in dataset_from_data._attr_names:
        path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
        dataset_from_data.data[attr_nm] = np.load(path_this_array)["data"]

    extra_data_info = ['simulation_names','surface']
    for extra_attr_nm in extra_data_info:
        path_this_array = f"{os.path.join(full_path, extra_attr_nm)}.npz"
        dataset_from_data.extra_data[extra_attr_nm] = np.load(path_this_array)["data"]

    dataset_from_data._infer_sizes()
    return dataset_from_data

def extract_dataset_by_simulations(newdataset_name:str,
                                   dataset:AirfRANSDataSet,
                                   simulation_indices:list):
    simulation_sizes = dataset.get_simulations_sizes()
    sample_sizes = [None]*len(simulation_sizes)
    start_index = 0
    for simulation_Id,simulation_size in enumerate(simulation_sizes):
        sample_sizes[simulation_Id] = range(start_index,start_index+simulation_size)
        start_index+= simulation_size
    values=operator.itemgetter(*simulation_indices)(sample_sizes)
    nodes_simulation_indices = sorted([item for sublist in values for item in sublist])

    new_data={}
    for data_name in dataset._attr_names:
        new_data[data_name]=dataset.data[data_name][nodes_simulation_indices]
    new_extra_data={
                    'simulation_names':dataset.extra_data['simulation_names'][simulation_indices],
                    'surface':dataset.extra_data['surface'][nodes_simulation_indices]
                    }
    new_dataset=type(dataset)(config = dataset.config, 
                             name = newdataset_name,
                             task = dataset._task,
                             split = dataset._split,
                             attr_names = dataset._attr_names, 
                             attr_x = dataset._attr_x , 
                             attr_y = dataset._attr_y)

    new_dataset.data=new_data
    new_dataset.extra_data=new_extra_data
    new_dataset._infer_sizes()
    return new_dataset


if __name__ == '__main__':
    import time
    attr_names = (
        'x-position',
        'y-position',
        'x-inlet_velocity', 
        'y-inlet_velocity', 
        'distance_function', 
        'x-normals', 
        'y-normals', 
        'x-velocity', 
        'y-velocity', 
        'pressure', 
        'turbulent_viscosity',
        'surface'
    )
    attr_x = attr_names[:7]
    attr_y = attr_names[7:]

    directory_name='Dataset'
    if not os.path.isdir("Dataset"):
         download_data(root_path=".", directory_name=directory_name)
    my_dataset = AirfRANSDataSet(config = None, 
                                 name = 'train',
                                 task = 'scarce',
                                 split = "training",
                                 attr_names = attr_names, 
                                 log_path = 'log', 
                                 attr_x = attr_x, 
                                 attr_y = attr_y)
    start_time = time.time()
    my_dataset.load(path = directory_name)
    end_time = time.time() - start_time

    print(my_dataset, "Loaded in %.2E s" %end_time)
    save_internal(dataset=my_dataset,path_out="AirfRANSDataset")
    start_time = time.time()
    reloaded_dataset=reload_dataset(path_in = "AirfRANSDataset",
                                  name = "train",
                                  task = 'scarce',
                                  split = "training",
                                  attr_x = attr_x, 
                                  attr_y = attr_y)
    end_time = time.time() - start_time

    print(reloaded_dataset, "Loaded in %.2E s" %end_time)
    assert my_dataset==reloaded_dataset,"Datasets should be the same!"