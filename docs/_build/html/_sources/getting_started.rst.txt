===============
Getting Started
===============

This chapter provides an introduction to FairLangProc, including installation,
quick start examples, and a comparison with other fairness tools.

.. toctree::
   :maxdepth: 2

Why FairLangProc?
-----------------

.. rst-class:: lead

   A unified framework for fairness in NLP with comprehensive coverage of datasets, metrics, and algorithms.

There are several tools available for bias mitigation in the fair ML landscape:

* **AI Fairness 360 (aif360)** - The main inspiration for this project. Provides bias mitigation methods for tabular data but lacks NLP-specific features.
* **Fairlearn** - Offers fairness tools but is not focused on NLP.
* **AllenNLP** - Provides embedding metrics and debiasing methods but development has been discontinued.
* **Dbias** - Offers a pipeline for bias evaluation but uses a custom methodology.
* **LangFair** - Allows generating custom datasets for fairness evaluation but lacks debiasing algorithms.
* **langtest** - Comprehensive evaluation library but with limited bias mitigation capabilities.

FairLangProc addresses these limitations by providing:

* **NLP-First Design** - Specifically designed for text data with HuggingFace transformers integration
* **Comprehensive Coverage** - 13+ benchmark datasets, 8+ fairness metrics, and 9+ debiasing algorithms
* **Pre/In/Intra-Processing** - Support for all stages of the ML pipeline
* **Unified Interface** - Consistent API across all modules
* **Active Development**


Overview
--------

.. list-table:: FairLangProc Main Functionalities
   :header-rows: 1
   :widths: 33 33 34

   * - **DATASETS**
     - **METRICS**
     - **ALGORITHMS**
   * - BBQ
     - WEAT
     - Pre-processors:
   * - BEC-Pro
     - LPBS
     -   - CDA
   * - BOLD
     - CBS
     -   - Projection
   * - BUG
     - CPS
     -   - BLIND
   * - CrowS-Pairs
     - AUL
     - In-processors:
   * - GAP
     - HONEST
     -   - ADELE
   * - HolisticBias
     - DR
     -   - EAR
   * - HONEST
     - SA
     -   - Regularizers
   * - StereoSet
     -
     - Intra-processors:
   * - UnQover
     -
     -   - EAT
   * - WinoBias
     -
     -   - MoDDiffy
   * - WinoBias+
     -
     -
   * - WinoGender
     -
     -

.. raw:: pdf

   \newpage

Installation
------------

Prerequisites
~~~~~~~~~~~~~

FairLangProc requires:

* Python >= 3.10
* numpy >= 2.2.4
* pandas >= 2.2.3
* scikit-learn >= 1.6.1
* torch >= 2.6.0
* transformers >= 4.47.1
* datasets >= 3.4.1
* adapter-transformers >= 1.1.0
* pytest >= 8.4.1

Install from PyPI
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install FairLangProc

Install from source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/arturo-perez-peralta/FairLangProc
   cd FairLangProc
   pip install -e .

Quick Start
-----------

Load a dataset for bias evaluation:

.. code-block:: python

   from FairLangProc.datasets import BiasDataLoader

   ageBBQ = BiasDataLoader(dataset = 'BBQ', config = 'Age')
   print(ageBBQ['data'][0])


Measure bias using common metrics:

.. code-block:: python
   from FairLangProc.metrics import CBS

   model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
   tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   target_words = [("John", "Mamadouk", "Liu"), ("white", "black", "asian"), ("white", "black", "asian")]
   sentences = ["[MASK] is a [MASK]", "The [MASK] kid got [MASK] results", "The [MASK] kid wanted to be a [MASK]"]
   fill_words = ["engineer", "outstanding", "doctor"]
   mask_indices = [0, 1, 1]
   
   CBSscore = CBS(
       model = model,
       tokenizer = tokenizer,
       sentences = sentences,
       target_words = target_words,
       fill_words = fill_words,
       mask_indices = mask_indices
   )

Mitigate bias with an algorithm:

.. code-block:: python
   from transformers import BertForSequenceClassification
   from FairLangProc.algorithms.inprocessors import EARModel

   model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

   EARRegularizer = EARModel(
      model = model,
      ear_reg_strength = 0.01
   )

Further Reading
---------------

For a more in-depth reading of the methodology and implementation consult:

- `GitHub Repository <https://github.com/arturo-perez-peralta/FairLangProc>`_ - Source code and issues
- `arXiv Paper <https://arxiv.org/abs/2508.03677>`_ - Academic publication