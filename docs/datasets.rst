========
Datasets
========

The FairLangProc datasets module provides access to standard benchmarks for evaluating gender, racial, religious, and other social biases in NLP models.

Overview
--------

The ``BiasDataLoader`` is the main entry point for loading bias evaluation datasets.
It supports multiple output formats and dataset configurations.


.. list-table:: Supported Benchmark Datasets
   :header-rows: 1
   :widths: 20 15 35 30

   * - Data Set
     - Size
     - Bias target
     - Reference
   * - BBQ
     - 58,492
     - Gender, race, religion,...
     - `(Parrish et al., 2021) <https://arxiv.org/abs/2110.08193>`_
   * - BEC-Pro
     - 5,400
     - Gender
     - `(Bartl et al., 2020) <https://arxiv.org/abs/2010.14534>`_
   * - BOLD
     - 23,679
     - Gender, race, religion,...
     - `(Dhamala et al., 2021) <https://doi.org/10.1145/3442188.3445924>`_
   * - BUG
     - 108,419
     - Gender
     - `(Levy et al., 2021) <https://arxiv.org/abs/2109.03858>`_
   * - Crow-SPairs
     - 1,508
     - Age, disability, gender, nationality,...
     - `(Nangia et al., 2020) <https://aclanthology.org/2020.emnlp-main.154/>`_
   * - GAP
     - 8,908
     - Gender
     - `(Webster et al., 2018) <https://aclanthology.org/Q18-1029>`_
   * - HolisticBias
     - 460,000
     - Age, disability, gender, nationality,...
     - `(Smith et al., 2022) <https://arxiv.org/abs/2205.09209>`_
   * - HONEST
     - 420
     - Gender
     - `(Nozza et al., 2021) <https://aclanthology.org/2021.naacl-main.191/>`_
   * - StereoSet
     - 16,995
     - Gender, race, religion,...
     - `(Nadeem et al., 2020) <https://arxiv.org/abs/2004.09456>`_
   * - UnQover
     - 30
     - Gender, nationality, race,...
     - `(Li et al., 2020) <https://arxiv.org/abs/2010.02428>`_
   * - WinoBias+
     - 1,367
     - Gender
     - `(Vanmassenhove et al., 2021) <https://arxiv.org/abs/2109.06105>`_
   * - WinoBias
     - 3,160
     - Gender
     - `(Zhao et al., 2018) <https://arxiv.org/abs/1804.06876>`_
   * - WinoGender
     - 720
     - Gender
     - `(Rudinger et al., 2018) <https://arxiv.org/abs/1804.09301>`_


API Reference and usage examples
--------------------------------

.. autofunction:: FairLangProc.datasets.fairness_datasets.BiasDataLoader
  :no-index:

.. seealso::

   - :doc:`tutorials` - Interactive Jupyter notebooks `(DemoDatasets.ipynb) <https://github.com/arturo-perez-peralta/FairLangProc/blob/main/notebooks/DemoDatasets.ipynb>` demonstrating dataset usage