.. FairLangProc documentation master file, created by
   sphinx-quickstart on Wed Jul 16 10:11:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FairLangProc documentation
==========================

The Fair Language Processing package is a extensible open-source Python library containing techniques developed by the
research community to help detect and mitigate bias in Natural Language Processing throughout the AI application lifecycle.

The FairLangProc package includes:

1. Data sets to test for biases in NLP models.
2. Metrics based on different philosophies to quantified said biases. 
3. Algorithms to mitigate biases.

It has been created with the intention of encouraging the use of bias mitigation strategies in the NLP community, and with the hope of democratizing these tools for the ever-increasing set of NLP practitioners. We invite you to use it and improve it.

The companion paper provides a comprehensive introduction to the concepts and capabilities, with all code available in `notebooks <https://github.com/arturo-perez-peralta/FairLangProc/tree/main/notebooks>`_.

We have developed the package with extensibility in mind. This library is still in development. We encourage your contributions.

Supported datasets
------------------

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

Supported metrics
-----------------
- :ref:`Generalized association tests (WEAT) <weat>` `(Caliskan et al., 2016) <https://arxiv.org/abs/1608.07187>`_.
- :ref:`Log Probability Bias Score (LPBS) <lpbs>` `(Kurita et al., 2019) <https://arxiv.org/abs/1906.07337>`_.
- :ref:`Categorical Bias Score (CBS) <cbs>` `(Ahn et al., 2021) <https://aclanthology.org/2021.emnlp-main.42/>`_.
- :ref:`CrowS-Pairs Score (CPS) <cps>` `(Nangia et al., 2020) <https://aclanthology.org/2020.emnlp-main.154/>`_.
- :ref:`All Unmasked Score (AUL) <aul>` `(Kaneko et al., 2021) <https://arxiv.org/abs/2104.07496>`_.
- :ref:`Demographic Representation (DR) <dr>` `(Liang et al., 2022) <https://arxiv.org/abs/2211.09110>`_.
- :ref:`Stereotypical Association (SA) <sa>` `(Liang et al., 2022) <https://arxiv.org/abs/2211.09110>`_.
- :ref:`HONEST <honest>` `(Nozza et al., 2021) <https://aclanthology.org/2021.naacl-main.191/>`_.

Supported algorithms
--------------------

- :ref:`Counterfactual Data Augmentation (CDA) <cda>` `(Webster et al. 2020) <https://arxiv.org/abs/2010.06032>`_.
- :ref:`Projection based debiasing <emb>` `(Bolukbasi et al., 2023) <https://arxiv.org/abs/1607.06520>`_.
- :ref:`Bias removaL wIth No Demographics (BLIND) <blind>` `(Orgad et al., 2023) <https://aclanthology.org/2023.acl-long.490/>`_.
- :ref:`Adapter-based DEbiasing of LanguagE models (ADELE) <adele>` `(Lauscher et al., 2021) <https://arxiv.org/abs/2109.03646>`_.
- :ref:`Modular Debiasing with Diff Subnetworks <diff>` `(Hauzenberger et al., 2023) <https://aclanthology.org/2023.findings-acl.386/>`_.
- :ref:`Entropy Attention Temperature (EAT) scaling <eat>` `(Zayed et al., 2023) <https://arxiv.org/abs/2305.13088>`_.
- :ref:`Entropy Attention Regularizer (EAR) <ear>` `(Attanasio et al., 2022) <https://arxiv.org/abs/2203.09192>`_.
- :ref:`Embedding based regularizer <embreg>` `(Liu et al., 2020) <https://arxiv.org/abs/1910.10486>`_.
- :ref:`Selective unfreezing <selective>` `(Gira et al., 2024) <https://aclanthology.org/2022.ltedi-1.8/>`_.

Index
-----

.. toctree::
   :maxdepth: 6
   :caption: Contents:

   FairLangProc.datasets
   FairLangProc.metrics
   FairLangProc.algorithms

Dependencies
------------
- Python >= 3.10
- numpy >= 2.2.4
- pandas >= 2.2.3
- scikit-learn >= 1.6.1
- torch >= 2.6.0
- transformers >= 4.47.1
- datasets >= 3.4.1
- adapter-transformers >= 1.1.0
- pytest >= 8.4.1
