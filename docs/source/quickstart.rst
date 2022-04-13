Getting started
===============

In this page, we present how the users should install the requirements for the LIPS platform.

Installation
------------
To be able to run the experiments in this repository, the users should install the last lips package from its github repository. The following steps show how to install this package and its dependencies from source.

Requirements
************
- Python >= 3.6

Setup a Virtualenv (optional)
*****************************
Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd my-project-folder
    pip3 install -U virtualenv
    python3 -m virtualenv venv_lips


Enter virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    source venv_lips/bin/activate

Install from source
*******************

.. code-block:: bash

    git clone https://github.com/Mleyliabadi/LIPS
    cd LIPS
    git checkout ml-dev
    pip3 install -U .
    cd ..

To contribute
**************
.. code-block:: bash

    pip3 install -e .[recommended]


