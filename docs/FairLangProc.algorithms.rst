FairLangProc.algorithms package
===============================

FairLangProc supports multiple algorithms to mitigate bias in LLMs.
They can be classified depending on their position on the ML pipeline:

- Pre-processors: Fairness processors that modify the model inputs.
- In-processors: Fairness processors that modify the training process.
- Intra-processors: Fairness processors that modify the model's behavior without further training.

The supported methods are:

- Counterfactual Data Augmentation (CDA) `(Webster et al. 2020) <https://arxiv.org/abs/2010.06032>`_.
- Projection based debiasing `(Bolukbasi et al., 2023) <https://arxiv.org/abs/1607.06520>`_.
- Bias removaL wIth No Demographics (BLIND) `(Orgad et al., 2023) <https://aclanthology.org/2023.acl-long.490/>`_.
- Adapter-based DEbiasing of LanguagE models (ADELE) `(Lauscher et al., 2021) <https://arxiv.org/abs/2109.03646>`_.
- Modular Debiasing with Diff Subnetworks `(Hauzenberger et al., 2023) <https://aclanthology.org/2023.findings-acl.386/>`_.
- Entropy Attention Temperature (EAT) scaling `(Zayed et al., 2023) <https://arxiv.org/abs/2305.13088>`_.
- Entropy Attention Regularizer (EAR) `(Attanasio et al., 2022) <https://arxiv.org/abs/2203.09192>`_.
- Embedding based regularizer `(Liu et al., 2020) <https://arxiv.org/abs/1910.10486>`_.
- Selective unfreezing `(Gira et al., 2024) <https://aclanthology.org/2022.ltedi-1.8/>`_.

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   FairLangProc.algorithms.inprocessors
   FairLangProc.algorithms.intraprocessors
   FairLangProc.algorithms.preprocessors