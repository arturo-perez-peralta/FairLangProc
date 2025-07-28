FairLangProc.datasets package
=============================

The `BiasDataLoader` method permits the use of different datasets for bias evaluation.

The supported datasets are:

.. list-table:: Benchmark Datasets
   :header-rows: 1
   :widths: 20 10 60

   * - Data Set
     - Size
     - Reference
   * - BBQ
     - 58,492
     - `(Parrish et al., 2021) <https://arxiv.org/abs/2110.08193>`_
   * - BEC-Pro
     - 5,400
     - `(Bartl et al., 2020) <https://arxiv.org/abs/2010.14534>`_
   * - BOLD
     - 23,679
     - `(Dhamala et al., 2021) <https://doi.org/10.1145/3442188.3445924>`_
   * - BUG
     - 108,419
     - `(Levy et al., 2021) <https://arxiv.org/abs/2109.03858>`_
   * - Crow-SPairs
     - 1,508
     - `(Nangia et al., 2020) <https://aclanthology.org/2020.emnlp-main.154/>`_
   * - GAP
     - 8,908
     - `(Webster et al., 2018) <https://aclanthology.org/Q18-1029>`_
   * - HolisticBias
     - 460,000
     - `(Smith et al., 2022) <https://arxiv.org/abs/2205.09209>`_
   * - HONEST
     - 420
     - `(Nozza et al., 2021) <https://aclanthology.org/2021.naacl-main.191/>`_
   * - StereoSet
     - 16,995
     - `(Nadeem et al., 2020) <https://arxiv.org/abs/2004.09456>`_
   * - UnQover
     - 30
     - `(Li et al., 2020) <https://arxiv.org/abs/2010.02428>`_
   * - WinoBias+
     - 1,367
     - `(Vanmassenhove et al., 2021) <https://arxiv.org/abs/2109.06105>`_
   * - WinoBias
     - 3,160
     - `(Zhao et al., 2018) <https://arxiv.org/abs/1804.06876>`_
   * - WinoGender
     - 720
     - `(Rudinger et al., 2018) <https://arxiv.org/abs/1804.09301>`_

BiasDataLoader
-----------------------------------------------

.. autofunction:: FairLangProc.datasets.fairness_datasets.BiasDataLoader
