# Submission

## Submission format

A submission shall be a zip file containing at minimum:
- a Python file named `my_augmented_simulator.py` and that follows the following template:
- a configuration file `simulator_config.json`
- any optional additional files necessary to the correct execution of the benchmark for the provided augmented simulator

The Python file shall contain a class `BenchmarkedSimulator`

```
class BenchmarkedSimulator(AugmentedSimulator):
    def __init__(self, 
                 name: str ,
                 **kwargs: Optional):
    

    # optional method
    def train(self, nb_iter: int,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet] = None,
              **kwargs):
              
    def predict(self, dataset: DataSet, batch_size: int=32, 
                 save_values: bool=False):


```

The class `BenchmarkedSimulator` provided by the user must extend `AugmentedSimulator
- the constructor of the BenchmarkedSimulator may support additional arguments.
The additional arguments provided for the instantiation of the simulator must be provided in the `simulator_config.json file` so that it can be correctly instantiated by the Codabench orchestrator.

- Additonal configuration files may be provided by the user in the submission bundle, for example to define in a configurable 
way the architecture of a NN-based augmented simulator.

Example of `simulator_config.json` configuration file:

```
{
  "simulator_config": {
    "scaler_class" : "StandardScaler",
    "name": "MyAugmentedSimulator"
  },

  "requires_training" : "true",
  "trained_model_path" : "./trained_model",
  "simulator_training_config": {
    "epochs": 2
  }

}

```
- The first section `"simulator_config"` contains a list of parameters automatically passed to the constructor of the `BenchmarkedSimulator` when it is instantiated on Codabench;
- The other sections of the configuration file  are related to the management of training
  - `"requires_training"`indicates if the provided submission requires training or not
  - `"trained_model_path` indicates where to find a pre-trained model if the submitted Augmented Simulator has been pre-trained outside of Codabench
  - `"simulator_training_config"` defines all the parameters to train the augmented simulator on the Codabench platform (this will only be taken into account if no pre-trained model is provided in the submission) 

Basic examples of augmented simulator to be benchmarked are provided in the starting kit.
