# Fair Language Processing (FairLangProc)

The Fair Language Processing package is a extensible open-source Python library containing techniques developed by the
research community to help detect and mitigate bias in Natural Language Processing throughout the AI application lifecycle.

The FairLangProc package includes:
1) Data sets to test for biases in NLP models.
2) Metrics based on different philosophies to quantified said biases. 
3) Algorithms to mitigate biases.
It has been created with the intention of encouraging the use of bias mitigation strategies in the NLP community, and with the hope of democratizing these tools for the ever-increasing set of NLP practitioners. We invite you to use it and improve it.

The companion paper provides a comprehensive introduction to the concepts and capabilities, with all code available in [notebooks](./notebooks). Furthermore, we are working torwards a comprehensive [documentation](https://fairlangproc.readthedocs.io/en/latest/).

We have developed the package with extensibility in mind. This library is still in development. We encourage your contributions.

## Supported fairness datasets

| Data Set       | Size     | Reference |
|----------------|----------|-----------|
| BBQ            | 58,492   | [Parrish et al., 2021](https://arxiv.org/abs/2110.08193) |
| BEC-Pro        | 5,400    | [Bartl et al., 2020](https://arxiv.org/abs/2010.14534) |
| BOLD           | 23,679   | [Dhamala et al., 2021](https://doi.org/10.1145/3442188.3445924) |
| BUG            | 108,419  | [Levy et al., 2021](https://arxiv.org/abs/2109.03858) |
| Crow-SPairs    | 1,508    | [Nangia et al., 2020](https://aclanthology.org/2020.emnlp-main.154/) |
| GAP            | 8,908    | [Webster et al., 2018](https://aclanthology.org/Q18-1029) |
| HolisticBias   | 460,000  | [Smith et al., 2022](https://arxiv.org/abs/2205.09209) |
| HONEST         | 420      | [Nozza et al., 2021](https://aclanthology.org/2021.naacl-main.191/) |
| StereoSet      | 16,995   | [Nadeem et al., 2020](https://arxiv.org/abs/2004.09456) |
| UnQover        | 30       | [Li et al., 2020](https://arxiv.org/abs/2010.02428) |
| WinoBias+      | 1,367    | [Vanmassenhove et al., 2021](https://arxiv.org/abs/2109.06105) |
| WinoBias       | 3,160    | [Zhao et al., 2018](https://arxiv.org/abs/1804.06876) |
| WinoGender     | 720      | [Rudinger et al., 2018](https://arxiv.org/abs/1804.09301) |

## Supported fairness metrics

* Generalized association tests (WEAT) ([Caliskan et al., 2016](https://arxiv.org/abs/1608.07187))
* Log Probability Bias Score (LPBS) ([Kurita et al., 2019](https://arxiv.org/abs/1906.07337))
* Categorical Bias Score (CBS) ([Ahn et al., 2021](https://aclanthology.org/2021.emnlp-main.42/))
* CrowS-Pairs Score (CPS) ([Nangia et al., 2020](https://aclanthology.org/2020.emnlp-main.154/))
* All Unmasked Score (AUL) ([Kaneko et al., 2021](https://arxiv.org/abs/2104.07496))
* Demographic Representation (DR) ([Liang et al., 2022](https://arxiv.org/abs/2211.09110))
* Stereotypical Association (SA) ([Liang et al., 2022](https://arxiv.org/abs/2211.09110))
* HONEST ([Nozza et al., 2021](https://aclanthology.org/2021.naacl-main.191/))

## Supported bias mitigation algorithms

* Counterfactual Data Augmentation (CDA) ([Webster et al. 2020](https://arxiv.org/abs/2010.06032))
* Projection based debiasing ([Bolukbasi et al., 2023](https://arxiv.org/abs/1607.06520))
* Bias removaL wIth No Demographics (BLIND) ([Orgad et al., 2023](https://aclanthology.org/2023.acl-long.490/))
* Adapter-based DEbiasing of LanguagE models ([Lauscher et al., 2021](https://arxiv.org/abs/2109.03646))
* Modular Debiasing with Diff Subnetworks ([Hauzenberger et al., 2023](https://aclanthology.org/2023.findings-acl.386/))
* Entropy Attention Temperature (EAT) scaling ([Zayed et al., 2023](https://arxiv.org/abs/2305.13088))
* Entropy Attention Regularizer (EAR) ([Attanasio et al., 2022](https://arxiv.org/abs/2203.09192))
* Embedding based regularizer ([Liu et al., 2020](https://arxiv.org/abs/1910.10486))
* Selective unfreezing ([Gira et al., 2024](https://aclanthology.org/2022.ltedi-1.8/))

## Setup

### Python

Has been tested and ran with Python 3.13. Compatibility with older versions is possible and expected, although no tests have been run to check the possible configurations.

To install the latest stable version from PyPI, run:

```bash
pip install FairLangProc
```

### Manual installation

Clone the latest version of this repository:

```bash
git clone https://github.com/arturo-perez-peralta/FairLangProc
```

## Using FairLangProc

The `notebooks` directory contains a diverse collection of jupyter notebooks that showcase how to use the different processors, metrics and data sets. If you'd like to run the examples requiring , download the data sets now and place them in a folder named `Fair-LLM-Benchmarks` inside the 'FairLangProc/datsets' path or simply clone the repository from [Gallegos et al](https://github.com/i-gallegos/Fair-LLM-Benchmark).
