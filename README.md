<p align="center">
    <img src="">
</p>

<p align="center">

[![LICENSE](https://img.shields.io/badge/license-MIT-brightgreen)](https://github.com/dean-jordan/NANSNA-Neuromorphic-Accelerators/blob/main/LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/dean-jordan/NANSNA-Neuromorphic-Accelerators)

</p>

NANSNA improves the efficiency of Neuromorphic Computing by designing a Neuromorphic Accelerator based on a novel Spiking Neural Network (SNN)-based architecture intended to improve multi-domain performance and specialization simultaneously.

---
- [About](#About)
- [Quickstart](#quickstart)
- [NANSNA Source](#nansna-source)
    - [Neurons](#neurons)
    - [Loss](#loss)
    - [Layers](#layers)
    - [Encoder](#encoder)
    - [Decoder](#decoder)
    - [Subnetwork](#subnetwork)
    - [Testing](#testing)
    - [Adapters](#adapters)
- [Models](#models)
- [Documentation](#documentation)
- [Notebooks](#notebooks)
- [Reports](#reports)
- [References](#references)
- [Training](#training)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
---

### About

### Quickstart

### NANSNA Source

#### Activation

#### Neurons

#### Loss

#### Layers

#### Encoder

#### Decoder

#### Subnetwork

#### Testing

#### Adapters

### Models

### Documentation

### Notebooks

### Reports

### References

### Training

### Dependencies

### Directory Structure
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         NANSNA and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── NANSNA   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes NANSNA a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

