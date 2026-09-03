# DeepDPF

DeepDPF predicts protein responses to drug perturbations using drug features and a protein-protein interaction graph.

## Environment

The project was tested with:

- Linux
- Python 3.11.7
- PyTorch 2.6.0
- CUDA 12.4

## Installation

Enter the project directory:

```bash
cd DeepDPF
```

- Install the required packages.

```bash
pip install -r requirements.txt
```

- Required package list:

```text
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.2.2
torch==2.6.0
torch-geometric==2.7.0
```

## Input files

The required input files are located in `database/`:

```text
database/
├── drug_feature.npy
├── drug_protein_log2FC.csv
└── string_ppi_graph.pt
```

## Run

Run the model from the project root directory:

```bash
python DeepDPF.py
```

The program automatically uses a GPU when CUDA is available. Training logs and the best model are saved under `output/DeepDPF/`.
