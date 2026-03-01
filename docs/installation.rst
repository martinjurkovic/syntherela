Installation
============

Basic Installation
------------------

To install only the benchmark package, run the following command:

.. code-block:: bash

   pip install syntherela

Development Installation
------------------------

To install the package with development dependencies:

.. code-block:: bash

   pip install syntherela[dev]

Optional Dependencies
---------------------

RDL Utility
~~~~~~~~~~~

To use the RDL utility features:

.. code-block:: bash

   pip install syntherela[rdl-utility]

This includes:

* relbench with full features
* torch_geometric
* tqdm
* lightgbm
* featuretools

Privacy Metrics
~~~~~~~~~~~~~~~

To use privacy evaluation metrics:

.. code-block:: bash

   pip install syntherela[privacy]

This includes the syntheval package.

From Source
-----------

To install from source:

.. code-block:: bash

   git clone https://github.com/martinjurkovic/syntherela.git
   cd syntherela
   pip install -e .

Requirements
------------

SyntheRela requires Python 3.10 or higher.

Main dependencies include:

* sdv>=1.9.0,<2
* seaborn==0.13.2
* xgboost==1.7.6
* scikit-learn>1.3.1,<1.5
