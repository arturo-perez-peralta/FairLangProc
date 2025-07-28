FairLangProc.algorithms.preprocessors package
=============================================

Pre-processors are fairness processors that modify the model inputs.

The supported methods are:

- Counterfactual Data Augmentation (CDA) `(Webster et al. 2020) <https://arxiv.org/abs/2010.06032>`_.
- Projection based debiasing `(Bolukbasi et al., 2023) <https://arxiv.org/abs/1607.06520>`_.
- Bias removaL wIth No Demographics (BLIND) `(Orgad et al., 2023) <https://aclanthology.org/2023.acl-long.490/>`_.

Counterfactual Data Augmentation (CDA)
---------------------------------------------------------

Data augmentation is the process of curating or upsampling the dataset to obtain a more representative distribution to train the model on.
In particular, Counterfactual Data Augmentation (CDA) `(Webster et al. 2020) <https://arxiv.org/abs/2010.06032>`_ consists of flipping words with demographic information
while preserving semantic correctness. This procedure can be one-sided and discard the original sentence or two-sided to consider both the original and its
augmented version.

.. autofunction:: FairLangProc.algorithms.preprocessors.augmentation.CDA

Projection-based debiasing
--------------------------------------------------------------

Projection-based debiasing methods `(Bolukbasi et al., 2023) <https://arxiv.org/abs/1607.06520>`_ operate on latent space, looking to identify a bias subspace given by an
orthogonal basis, :math:`\{v_i\}_{i=1}^{n_{bias}}`. Then, the hidden representation of any input can be debiased by removing its projection onto this space, formally

.. math::
    h_{proj} = h - \sum_{i = 1}^{n_{bias} } \langle h, v_i \rangle \, v_i.

This can be done either at the word or sentence level. In either case the bias subspace is generally identified through PCA,
and usually its dimension is one, resulting in the construction of a bias direction.

.. autoclass:: FairLangProc.algorithms.preprocessors.projection_based.SentDebiasModel
   :members: __init__
   :no-index:

BLIND debiasing
--------------------------------------------------------

BLIND `(Orgad et al., 2023) <https://aclanthology.org/2023.acl-long.490/>`_ is a debiasing procedure based on a complementary classifier 
:math:`g_{B} : \mathbb{R}^{d_L} \longrightarrow \mathbb{R}` with parameters :math:`\theta_{B}`, that takes the hidden representation vector
as inputs and outputs the success probability of the model head for the downstream task. This probability is then used as a weight for said observation
whose magnitude is controlled through a hyper-parameter :math:`\gamma \geq 0`:

.. math::
    \mathcal{L}_{BLIND} = \left(1 - \sigma \left( g_{B}(h; \theta_{B} ) \right) \right)^{\gamma} \mathcal{L}^{task}(\hat{y}, y).

The term :math:`\sigma(g_{B}(h;\theta_B))` represents the model success probability for the downstream task:
the bigger it is the less weight the observation has, while the smaller it is the more weight it carries.
This forces the model to pay special attention to observations with low probability of success during training.
Note that when :math:`\gamma = 0` the original loss function is restored, while :math:`\gamma >> 1` exacerbates the effect of the reweighting.

.. autoclass:: FairLangProc.algorithms.preprocessors.reweighting.BLINDTrainer
   :members: __init__
   :no-index:
