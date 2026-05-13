==========
Tutorials
==========

This section provides interactive tutorials demonstrating how to use FairLangProc
for bias detection and mitigation in NLP models.

.. tip::

   All tutorials are available as Jupyter notebooks. You can run them locally by
   cloning the repository or access them on Google Colab.

   .. code-block:: bash

      git clone https://github.com/arturo-perez-peralta/FairLangProc

   Then navigate to the ``notebooks/`` directory and open the notebooks in Jupyter.

Datasets
--------

**Demo Datasets** - `Open on GitHub <../notebooks/DemoDatasets.ipynb>`__

   Learn how to load and explore bias benchmark datasets.

   - Load BBQ, StereoSet, BOLD, and other datasets
   - Explore dataset structure and attributes

Metrics
-------

**Demo Metrics** - `Open on GitHub <../notebooks/DemoMetrics.ipynb>`__

   Measure bias using various fairness metrics.

   - WEAT for embedding-level bias
   - LPBS, CBS, CPS, AUL for probability-level bias
   - DR, ST, HONEST for generated text bias

Bias Mitigation
---------------

**Demo Debiasing** - `Open on GitHub <../notebooks/DemoProcessors.ipynb>`__

   Apply debiasing algorithms to reduce model bias.

   - Pre-processing with CDA, BLIND and projection-based debiasing.
   - In-procesing with ADELE and regularizers.
   - Intra-processing with MoDDiffy and EAT scaling. 


End-to-End Workflow
-------------------

**Demo Processors** - `Open on GitHub <../notebooks/DemoDebiasing.ipynb>`__

   Use fairness processor to debiase LLMs in the GLUE tasks.

   - Use WEAT to measure bias.
   - Use the different processors to mitigate embedding-level bias.
   - Measure the impact on the performance metrics of the GLUE datasets.

Notebook Index
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Notebook
     - Description
   * - `DemoDatasets.ipynb`_
     - Load and explore bias benchmark datasets
   * - `DemoMetrics.ipynb`_
     - Measure bias with WEAT, LPBS, CBS, HONEST, etc.
   * - `DemoDebiasing.ipynb`_
     - Apply debiasing algorithms (CDA, Projection, ADELE)
   * - `DemoProcessors.ipynb`_
     - Use in-processing and intra-processing bias mitigators

.. _DemoDatasets.ipynb: ../notebooks/DemoDatasets.ipynb
.. _DemoMetrics.ipynb: ../notebooks/DemoMetrics.ipynb
.. _DemoProcessors.ipynb: ../notebooks/DemoProcessors.ipynb
.. _DemoDebiasing.ipynb: ../notebooks/DemoDebiasing.ipynb

Requirements
------------

To run the notebooks locally, you need to install FairLangProc and download the datasets:

.. code-block:: bash

   pip install FairLangProc

.. note::

   Some notebooks require significant computational resources (GPU recommended).
   For GPU acceleration, ensure CUDA is installed and configured.

   .. code-block:: bash

      # Verify CUDA availability
      python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"