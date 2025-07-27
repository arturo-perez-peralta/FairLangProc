FairLangProc.algorithms.intraprocessors package
===============================================

Intra-processors are fairness processors that modify the model's behavior without further training.

The supported methods are:
* Modular Debiasing with Diff Subnetworks ([Hauzenberger et al., 2023](https://aclanthology.org/2023.findings-acl.386/)).
* Entropy Attention Temperature (EAT) scaling ([Zayed et al., 2023](https://arxiv.org/abs/2305.13088)).

FairLangProc.algorithms.intraprocessors.modular module
------------------------------------------------------

.. automodule:: FairLangProc.algorithms.intraprocessors.modular
   :members: DiffPrunDebiasing, DiffPrunBERT
   :undoc-members:
   :show-inheritance:

FairLangProc.algorithms.intraprocessors.redistribution module
-------------------------------------------------------------

.. automodule:: FairLangProc.algorithms.intraprocessors.redistribution
   :members: add_EAT_hook
   :undoc-members:
   :show-inheritance:
