# Copyright (c) 2021, IRT SystemX and RTE (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking


class HyperParameterTuner(object):
    """
    A class providing the functionality to tune the hyper-parameters of the "fullyConnectedAS"
    augmented simulator.

    params
    ------
        augmented_simulator: ``object`` of class ``AugmentedSimulator``
            an augmented simulator like fullyConnectedAs or LeapNet
    """
    def __init__(self, augmented_simulator):
        self.augmented_simulator = augmented_simulator

    def tune(self,
             dataset, 
             sizes_layer=[(150,150)], 
             layer_act=["relu"], 
             lr=[3e-4], 
             batch_size=[128], 
             epochs=[5], 
             loss=["mse"], 
             n_folds=3, 
             verbose=1, 
             n_jobs=1):
        """
        This function allows to run a grid search cross validation over a set of hyper parameters
        All the parameters of this function except for dataset, epochs and number of folds could be a list
        """
        processed_x, processed_y = self.augmented_simulator._process_all_dataset(dataset, training=True)

        from sklearn.model_selection import GridSearchCV
        from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

        model = KerasRegressor(build_fn=self.__create_model, verbose=0)

        param_grid = dict(sizes_layer=sizes_layer, layer_act=layer_act, lr=lr, batch_size=batch_size, epochs=epochs, loss=loss)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs, cv=n_folds)

        grid_result = grid.fit(processed_x, processed_y, verbose=verbose)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        return grid_result

    def __create_model(self,
                       sizes_layer=(150,150),
                       layer_act="relu",
                       lr=3e-4,
                       loss="mse"):
        import tensorflow.keras.optimizers as tfko
        self.augmented_simulator.sizes_layer = sizes_layer
        self.augmented_simulator.layer_act = layer_act
        
        self.augmented_simulator.init()
        optimizer = tfko.Adam(learning_rate=lr)
        self.augmented_simulator._model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

        return self.augmented_simulator._model
