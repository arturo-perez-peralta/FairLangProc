.. FairLangProc documentation master file

FairLangProc |version| documentation
=====================================

The **Fair Language Processing** (FairLangProc) package is an extensible open-source Python library
containing techniques developed by the research community to help detect and mitigate bias in
Natural Language Processing throughout the AI application lifecycle.

The FairLangProc package includes:

1. **Datasets** - Benchmarks to test for biases in NLP models
2. **Metrics** - Fairness measures based on different philosophies to quantify bias
3. **Algorithms** - Bias mitigation techniques at pre-processing, in-processing, and intra-processing stages

It has been created with the intention of encouraging the use of bias mitigation strategies
in the NLP community, with the hope of democratizing these tools for the ever-increasing set
of NLP practitioners.

.. tip::

   The companion paper provides a comprehensive introduction to the concepts and capabilities,
   with all code available in the `notebooks <https://github.com/arturo-perez-peralta/FairLangProc/tree/main/notebooks>`_.

.. note::

   If you use this library in your research, please cite our `arXiv paper <https://arxiv.org/abs/2508.03677>`_.

.. toctree::
   :maxdepth: 2
   :numbered:
   :hidden:

   getting_started
   datasets
   metrics
   algorithms
   tutorials
   citation
