FairLangProc.algorithms.inprocessors package
============================================

In-processors are fairness processors that modify the training process.

The supported methods are:

- :ref:`Adapter-based DEbiasing of LanguagE models (ADELE) <adele>` `(Lauscher et al., 2021) <https://arxiv.org/abs/2109.03646>`_.
- :ref:`Entropy Attention Regularizer (EAR) <ear>` `(Attanasio et al., 2022) <https://arxiv.org/abs/2203.09192>`_.
- :ref:`Embedding based regularizer <embreg>` `(Liu et al., 2020) <https://arxiv.org/abs/1910.10486>`_.
- :ref:`Selective unfreezing <selective>` `(Gira et al., 2024) <https://aclanthology.org/2022.ltedi-1.8/>`_.

.. _ADELE:

ADELE
---------------------------------------------------

The ADELE procedure `(Lauscher et al., 2021) <https://arxiv.org/abs/2109.03646>`_ is based on the adapter framework.
A single adapter module is included to each transformer layer after the feed-forward sub-layer, where the outputs are compressed to a bottleneck dimension
:math:`m` and then decompressed back to the hidden size of the transformer, :math:`d_L`. The adapter module itself consists of a two-layer feed-forward network:

.. math::
    \text{Adapter}(\mathbf{h}, \mathbf{r}) = U \cdot g(D\cdot \mathbf{h) + \mathbf{r}}

where :math:`\mathbf{h}` and :math:`\mathbf{r}` are the hidden state and residual of the corresponding transformer layer, :math:`g` is an activation function and 
:math:`D \in \mathbb{R}^{m \times d_L}, U \in \mathbb{R}^{d_L\times m}` represent the projection matrices.
The idea behind the adapter layer is to introduce an information bottleneck which compresses the latent representation of the inputs,
forcing the model to discard all irrelevant information.

.. autoclass:: FairLangProc.algorithms.inprocessors.adapter.DebiasAdapter
   :members: __init__
   :no-index:
   
.. _embreg:

Embedding based regularizer
--------------------------------------------------------

Embedding based regularizers `(Liu et al., 2020) <https://arxiv.org/abs/1910.10486>`_ are based on the distance between the embeddings of counterfactual pairs given by $A$,

.. math::
    \mathcal{R} = \sum_{(a_i, a_j)\in A} || M(a_i) - M(a_j)||_2 .

.. autoclass:: FairLangProc.algorithms.inprocessors.regularizers.EmbeddingBasedRegularizer
   :members: __init__
   :no-index:

.. _EAR:

EAR
--------------------------------------------------------

EAR `(Attanasio et al., 2022) https://arxiv.org/abs/2203.09192`_ tries to maximize the entropy of the attention weights to encourage attention to the broader context
of the input,

.. math::
    \mathcal{R} = - \sum_{l=1}^L \text{entropy}_l(\mathbf{A})

where :math:`\text{entropy}_l(\cdot)` denotes the entropy of the l-th layer.

.. autoclass:: FairLangProc.algorithms.inprocessors.regularizers.EARModel
   :members: __init__
   :no-index:

.. _selective:

Selective unfreezing
---------------------------------------------------------------

Selective unfreezing `(Gira et al., 2024) https://aclanthology.org/2022.ltedi-1.8/`_ aims to circumvent catastrophic forgetting during fine-tuning 
by freezing a big amount of the model parameters, which also helps lessening computational expenses.

.. autofunction:: FairLangProc.algorithms.inprocessors.selective_updating.selective_unfreezing
    :no-index: