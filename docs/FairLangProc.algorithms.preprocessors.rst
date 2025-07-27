FairLangProc.algorithms.preprocessors package
=============================================

Pre-processors are fairness processors that modify the model inputs.

The supported methods are:
* Counterfactual Data Augmentation (CDA) ([Webster et al. 2020](https://arxiv.org/abs/2010.06032)).
* Projection based debiasing ([Bolukbasi et al., 2023](https://arxiv.org/abs/1607.06520)).
* Bias removaL wIth No Demographics (BLIND) ([Orgad et al., 2023](https://aclanthology.org/2023.acl-long.490/)).

FairLangProc.algorithms.preprocessors.augmentation module
---------------------------------------------------------

.. automodule:: FairLangProc.algorithms.preprocessors.augmentation
   :members: CDA
   :undoc-members:
   :show-inheritance:

FairLangProc.algorithms.preprocessors.projection\_based module
--------------------------------------------------------------

.. automodule:: FairLangProc.algorithms.preprocessors.projection_based
   :members: SentDebiasModel, SentDebiasForSequenceClassification
   :undoc-members:
   :show-inheritance:

FairLangProc.algorithms.preprocessors.reweighting module
--------------------------------------------------------

.. automodule:: FairLangProc.algorithms.preprocessors.reweighting
   :members: BLINDTrainer
   :undoc-members:
   :show-inheritance:
