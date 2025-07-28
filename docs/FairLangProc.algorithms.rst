FairLangProc.algorithms package
===============================

FairLangProc supports multiple algorithms to mitigate bias in LLMs.
They can be classified depending on their position on the ML pipeline:

- Pre-processors: Fairness processors that modify the model inputs.
- In-processors: Fairness processors that modify the training process.
- Intra-processors: Fairness processors that modify the model's behavior without further training.

The supported methods are:

- :ref:`Counterfactual Data Augmentation (CDA) <cda>` `(Webster et al. 2020) <https://arxiv.org/abs/2010.06032>`_.
- :ref:`Projection based debiasing <emb>` `(Bolukbasi et al., 2023) <https://arxiv.org/abs/1607.06520>`_.
- :ref:`Bias removaL wIth No Demographics (BLIND) <blind>` `(Orgad et al., 2023) <https://aclanthology.org/2023.acl-long.490/>`_.
- :ref:`Adapter-based DEbiasing of LanguagE models (ADELE) <adele>` `(Lauscher et al., 2021) <https://arxiv.org/abs/2109.03646>`_.
- :ref:`Modular Debiasing with Diff Subnetworks <diff>` `(Hauzenberger et al., 2023) <https://aclanthology.org/2023.findings-acl.386/>`_.
- :ref:`Entropy Attention Temperature (EAT) scaling <eat>` `(Zayed et al., 2023) <https://arxiv.org/abs/2305.13088>`_.
- :ref:`Entropy Attention Regularizer (EAR) <_ear>` `(Attanasio et al., 2022) <https://arxiv.org/abs/2203.09192>`_.
- :ref:`Embedding based regularizer <embreg>` `(Liu et al., 2020) <https://arxiv.org/abs/1910.10486>`_.
- :ref:`Selective unfreezing <selective>` `(Gira et al., 2024) <https://aclanthology.org/2022.ltedi-1.8/>`_.

Subpackages
-----------

.. toctree::
   :maxdepth: 3

   FairLangProc.algorithms.inprocessors
   FairLangProc.algorithms.intraprocessors
   FairLangProc.algorithms.preprocessors