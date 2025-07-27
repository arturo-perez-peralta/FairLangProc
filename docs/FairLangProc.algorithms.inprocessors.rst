FairLangProc.algorithms.inprocessors package
============================================

In-processors are fairness processors that modify the training process.

The supported methods are:
* ADELE ([Lauscher et al., 2021](https://arxiv.org/abs/2109.03646)).
* Entropy Attention Regularizer (EAR) ([Attanasio et al., 2022](https://arxiv.org/abs/2203.09192)).
* Selective unfreezing ([Gira et al., 2024](https://aclanthology.org/2022.ltedi-1.8/)).

FairLangProc.algorithms.inprocessors.adapter module
---------------------------------------------------

.. automodule:: FairLangProc.algorithms.inprocessors.adapter
   :members: DebiasAdapter
   :undoc-members:
   :show-inheritance:

FairLangProc.algorithms.inprocessors.regularizers module
--------------------------------------------------------

.. automodule:: FairLangProc.algorithms.inprocessors.regularizers
   :members: EmbeddingBasedRegularizer, BERTEmbedingReg, EARModel
   :undoc-members:
   :show-inheritance:

FairLangProc.algorithms.inprocessors.selective\_updating module
---------------------------------------------------------------

.. automodule:: FairLangProc.algorithms.inprocessors.selective_updating
   :members: selective_unfreezing
   :undoc-members:
   :show-inheritance: