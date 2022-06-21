Augmented Simulators
====================
Objecitves
----------
An "Augmented Simulator" is a surrogate model which aims at accelerating the physical solvers computation time,
using data driven models.

Detailed Documentation by class
-------------------------------
.. list-table:: Benchmark classes
   :widths: 50 50
   :header-rows: 1

   * - Class
     - Description
   * - :ref:`AugmentedSimulator<augmentedsimulator>`
     - Benchmark class for pneumatic use case
   * - :ref:`TensorflowSimulator<TensorflowSimulator>`
     - Tensorflow based simulators base class
   * - :ref:`Tensorflow models<tensorflow-models>`
     - Implemented models based on Tensorflow package
   * - :ref:`TorchSimulator<TorchSimulator>`
     - Pytorch based simulators base class
   * - :ref:`Pytorch models<pytorch-models>`
     - Implemented models based on Pytorch package

.. _augmentedsimulator:

AugmentedSimulator base class
-----------------------------

.. automodule:: lips.augmented_simulators
   :members:
   :undoc-members:
   :show-inheritance:
   :autosummary-exclude-members: AugmentedSimulator

.. _TensorflowSimulator:

TensorflowSimulator base class
------------------------------

.. automodule:: lips.augmented_simulators.tensorflow_simulator
   :members:
   :undoc-members:
   :show-inheritance:
   :autosummary-exclude-members: TensorflowSimulator

.. _tensorflow-models:

.. include:: lips.augmented_simulators.tensorflow_models.rst

.. _TorchSimulator:

TorchSimulator base class
------------------------------

.. automodule:: lips.augmented_simulators.torch_simulator
   :members:
   :undoc-members:
   :show-inheritance:
   :autosummary-exclude-members: TorchSimulator

.. _pytorch-models:

.. include:: lips.augmented_simulators.torch_models.rst
