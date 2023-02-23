from lips.dataset.dataSet import DataSet

import numpy as np
import copy

def test_dataset_iterable():
    class DummyDataset(DataSet):
        def __init__(self,name,attr_names,**kwargs):
            super(DummyDataset,self).__init__(name=name)
            
            self._attr_names = copy.deepcopy(attr_names)
            self.size = 0
            self._inputs = []
            self._size_x = None
            self._size_y = None
            self._sizes_x = None  # dimension of each variable
            self._sizes_y = None  # dimension of each variable
            self._attr_x = kwargs["attr_x"] 
            self._attr_y = kwargs["attr_y"]

        def load_data(self,data:dict):
            self.data=data
            self._infer_sizes()

        def _infer_sizes(self):
            data = copy.deepcopy(self.data)
            self.size = data[list(data.keys())[0]].shape[0]
            self._sizes_x = np.array([data[el].shape[1] for el in self._attr_x], dtype=int)
            self._sizes_y = np.array([data[el].shape[1] for el in self._attr_y], dtype=int)
            self._size_x = np.sum(self._sizes_x)
            self._size_y = np.sum(self._sizes_y)

        def get_data(self, index: tuple):
            super().get_data(index)  # check that everything is legit
            if isinstance(index, list):
                index = np.array(index, dtype=int)
            elif isinstance(index, int):
                index = np.array([index], dtype=int)

            res = {}
            nb_sample = index.size
            for el in self._attr_names:
                res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)

            for el in self._attr_names:
                res[el][:] = self.data[el][index, :]

            return res

    data={"dummyAtrib1":np.arange(20).reshape(2,10),
          "dummyAtrib2":np.arange(10,40).reshape(2,15),
          "dummyAtrib3":np.arange(30,80).reshape(2,25)
            }

    extraArguments={"attr_x":["dummyAtrib1","dummyAtrib2"],"attr_y":["dummyAtrib3"]}
    myDataset=DummyDataset(name="myDummyDataset",attr_names=list(data.keys()),**extraArguments)
    myDataset.load_data(data)
    assert len(myDataset)==2

    expectedResult=[
        {dataName:np.array([dataValue[0]]) for dataName,dataValue in data.items()},
        {dataName:np.array([dataValue[1]]) for dataName,dataValue in data.items()}
    ]

    for iterIndex,iterData in enumerate(myDataset):
        np.testing.assert_equal(iterData,expectedResult[iterIndex])