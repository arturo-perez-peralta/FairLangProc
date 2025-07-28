FairLangProc.algorithms.intraprocessors package
===============================================

Intra-processors are fairness processors that modify the model's behavior without further training.

The supported methods are:

- :ref:`Modular Debiasing with Diff Subnetworks <diff>` `(Hauzenberger et al., 2023) <https://aclanthology.org/2023.findings-acl.386/>`_.
- :ref:`Entropy Attention Temperature (EAT) scaling <eat>` `(Zayed et al., 2023) <https://arxiv.org/abs/2305.13088>`_.

.. _diff:

MoDDiffy
------------------------------------------------------

MoDDiffy `(Hauzenberger et al., 2023) <https://aclanthology.org/2023.findings-acl.386/>`_ creates many sparse subnetworks to address bias for different attributes
(gender, religion,...) through the idea of *Diff* prunning. Basically, they freeze the model parameters, :math:`\theta`,
and train another network with parameters, :math:`\delta`, with a loss function that promotes accuracy, sparsity and debiasing:

.. math::
   \mathcal{L}_\rho = \mathcal{L}^{task}_{\rho} + \lambda_{\rho}^0 \mathcal{L}^{0}_{\rho} + \lambda_{\rho}^{debias} \mathcal{L}_{\rho}^{debias},

where:
 - :math:`\mathcal{L}_{\rho}^{task}` represents the original loss of the downstream task with the new parameters.
 - :math:`\mathcal{L}_{\rho}^{0}` promotes sparsity through a smooth approximation to the :math:`L_0` norm of the new parameters by means of the hard-concrete distribution of parameters :math:`(\log \alpha_{\rho}, 1)` and hyper-parameters :math:`\gamma < 0, \zeta > 1`. 
 - :math:`\mathcal{L}_{\rho}^{debias}` debiases the outputs by approximating the mutual information of embeddings belonging to different demographic groups.

In particular the sparsity loss takes the form:

.. math:: 
   \mathcal{L}_{\rho}^{0} = \sum_{i=1}^{|\delta_{\rho}|} \sigma\left( \log \alpha_{\rho, i} - \log\left(- \frac{\gamma}{\zeta}\right) \right).

And the debiasing loss:

.. math::
   \mathcal{L}_{\rho}^{debias} = \left(\frac{\sum_{x_A \in X^A_\rho} \phi (M(x_A))}{|X_{\rho}^A |} - \frac{\sum_{x_B \in X^B_\rho} \phi (M(x_B))}{|X_{\rho}^B|} \right)^2,

where :math:`\phi` is a transformation kernel.
 
.. autoclass:: FairLangProc.algorithms.intraprocessors.modular.DiffPrunDebiasing
   :members: __init__
   :no-index:

.. _eat:

EAT
-------------------------------------------------------------

EAT scaling `(Zayed et al., 2023) <https://arxiv.org/abs/2305.13088>`_ modifies the distribution of the attention scores with a temperature-related parameter,
:math:`\beta \in [0, \infty)`:

.. math::
   \text{Attention}_{\beta} (\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left(\frac{\beta \mathbf{Q} \mathbf{K}}{\sqrt{d_k}} \right) \mathbf{V},

the idea being that when :math:`\beta >> 1` the head attends only to the tokens with biggest scores while :math:`\beta \approx 0` forces the head to attend equally to all tokens.
When :math:`\beta = 1` the attention head remains unmodified.

.. autofunction:: FairLangProc.algorithms.intraprocessors.redistribution.add_EAT_hook
