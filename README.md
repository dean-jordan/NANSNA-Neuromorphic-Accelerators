<p align="center">
    <img src="">
</p>

<p align="center">

[![LICENSE](https://img.shields.io/badge/license-MIT-brightgreen)](https://github.com/dean-jordan/NANSNA-Neuromorphic-Accelerators/blob/main/LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/dean-jordan/NANSNA-Neuromorphic-Accelerators)

</p>

<h3 align="center">NANSNA: Neuromorphic Accelerators with Novel Spiking Neural Subnetwork Ensemble-Based Architecture</h3>

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
Neuromorphic computing is an industry which is experiencing a rapid onset. Initially, the introduction of the artificial neural network (ANN) introduced the idea of software algorithms inspired by neuroscience. While implementations of ANNs have seen vast applications in fields such as Natural Language Processing (NLP), language translation, and neuroimaging, there is an increasing demand for hardware algorithms which are more biologically plausible than ANNs. However, as of currently, the spiking neural network (SNN) that is utilized within neuromorphic computing has a high computational cost. In addition, SNNs excel at performing specific tasks, but are relatively unable to generalize to multi-domain tasks. As such, NANSNA proposes a novel neuromorphic accelerator, SNN-based model, and SNN architecture which contains more biological plausibility while incurring a lower computational cost and utilizing neurosymbolic programming to improve the performance in multi-domain and multimodal logical Vision and Language (VaL) tasks. The SNN utilizes an adapter-based training approach which is combined with a neural subnetwork ensemble to allow for specific domains to be utilized at once. This allows for on-the-fly specialization while additionally increasing parameter efficiency. An encoder-decoder architecture is used before and after the subnetwork ensemble to load adapters and process inputs through a muti-head attention mechanism. The resulting model outperforms current State of The Art (SOTA) methods in several key metrics.

### Quickstart
Quickstart to install all prerequisites and allow for use of model architecture, schematics for neuromorphic accelerator, and use of SNN. Use the following scripts to install the repository.

```bash
# Install Repository
git clone https://github.com/dean-jordan/NANSNA-Neuromorphic-Accelerators.git

# Install Prerequisite Packages
pip install -r requirements.txt

# View/Modify Architecture
cd ./NANSNA

# View Neuromorphic Accelerator
cd ./accelerator

# Run SNN
python ./NANSNA/NANSNA.py
```
Run all scripts from within root directory `NANSNA-Neuromorphic-Accelerators` for commands to utilize relative pathing and execute successfully.

The NANSNA accelerator, architecture, and model were developed on Windows with Python v3.11.5 and Amazon Linux with Python v3.11.5.
While Python 3.8 is the minimum requirement, the newest PyTorch-supported version is recommended for full compatibility.

### NANSNA Source
The NANSNA network definition is found in the `./NANSNA` directory. Information regarding the SpiNNaker-based neuromorphic accelerator can be found in the `./accelerator` directory. Finally, an interactive script can be used to run the model. This is found in `./NANSNA.py`. Testing can be done through `./test.py`.

Within the NANSNA directory, all files were used to develop the architecture. Key files within each directory will be described below, allowing one to browse through the architecture.

#### Activation
The activation directory contains all activation functions which are used within NANSNA. This contains spiking and non-spiking activation functions.

`relu.py`
> Contains simple ReLU activation function. Used in feedforward networks in encoder and decoder.

`softmax.py`
> Contains softmax activation function for attention mechanism.

`spiking_relu.py`
> Contains spiking version of ReLU activation function for subnetwork ensemble.

#### Neurons
The neuron directory contains the neurons used in the spiking networks.

`all_neuron_types.py`
> Contains all four neuron types used within the network as a Python module.

`residual_neuron_1.py`
> Contains the first order recurrent leaky integrate-and-fire neuron.

`residual_neuron_2.py`
> Contains the second order recurrent leaky integrate-and-fire neuron.

`subnetwork_neuron_1.py`
> Contains the first type of neuron used within the residual subnetwork ensemble.

`subnetwork_neuron_2.py`
> Contains the second type of neuron used within the residual subnetwork ensemble.

#### Loss
The loss directory contains the novel loss function within the architecture and scripts to graph and test the loss function.

`loss.py`
> The custom loss function within NANSNA.

`loss_test.py`
> The script used to test the loss function for accuracy.

`loss_graph.py`
> The script used to use matplotlib to graph the loss function.

#### Layers
the layers directory contains every custom layer used within NANSNA. This is with the exception of simple layer types such as the nn.Linear() layer.

`all_layer_types.py`
> Contains every layer type packaged into a Python package for use within NANSNA.

`integration_layer.py`
> Contains the integration layer used to create neurosymbolic primitives.

`encoder_recurrent.py`
> Contains the non-spiking recurrent layer within the encoder. The same applies to the decoder.

`symbolic_block.py`
> While not a layer, this contains the full module for neurosymbolic programming primitives within the architecture.

`adapter_subnetwork.py`
> While also not a layer, contains the module for loading and using adapter-based learning within the subnetwork ensemble.

#### Encoder
The encoder directory contains code for creating the encoder blocks. These blocks are non-spiking and contain an attention mechanism, feedforward network, and have the ability to load adapters into the subnetwork ensemble. One adapter is used per subnetwork. In addition, the encoder processes symbolic primitives.

`encoder.py`
> Contains the full encoder block.

`encoder_attention.py`
> Contains the attention mechanism for within the encoder.

`encoder_symbolic.py`
> Implements the symbolic blocks within the encoder.

`encoder_feedforward.py`
> Processes input into low-level inputs and loads adapters into subnetwork ensemble.

#### Decoder
The decoder directory contains code for creating the decoder blocks. These work similarly to the encoder blocks. However, the feedforward network does not load adapters into the subnetwork ensemble.

`decoder.py`
> Contains full decoder block.

`decoder_feedforward.py`
> Processes low-level input back into output.

`decoder_symbolic.py`
> Fully processes symbolic outputs back into high-level features.

#### Subnetwork
The subnetwork directory contains code to create a subnetwork ensemble between the encoders and decoders.

`subnetwork_block.py`
> Groups multiple layers together to create ResNet-like identity blocks for creating subnetworks.

`subnetwork.py`
> Groups multiple blocks together to create a residual subnetwork.

`subnetwork_ensemble.py`
> Groups multiple subnetworks together to create a full subnetwork ensemble.

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

