# QuPID: Discrete transforms of Quantized PersIstence Diagrams
**Author**: Michael Etienne Van Huffel, Olympio Hacquard, Vadim Lebovici, Matteo Palo 

## Description
Implementation of QuPID vectorization for Persistence Diagrams.

## Overview
This repository contains the official implementation of QuPID, a method for embedding Persistence Diagrams into elements of vector spaces. It showcases the integration of such vectorization technique in the context of graph classification, dynamical particles classification, and tumor immune cells classification.

## Repository Structure
The repository is systematically organized to facilitate easy navigation and comprehensive understanding of each component.

### Jupyter Notebooks
- `graph-classy.ipynb`: Demo that demonstrates the application of QuPID Vectorization in graph classification.
- `orbit-classy.ipynb`: Demo that demonstrates the application of QuPID Vectorization in dynamical particles classification.
- `tic-classy.ipynb`: Demo that demonstrates the application of QuPID Vectorization in tumor immune cells classification.

### Python Scripts
- `quipid/qupid.py`: Core implementation of the QuPID Vectorization algorithm.
- `qupid/utils.py`: Provides essential utility functions for data processing and analysis within the notebooks.


### Other Mixed Scripts
- `other_scripts/`: Contains multiple scripts for computing timings of QuPID and creating Figures 4 and 6 in the paper.

## Installation
To reproduce the analysis environment, you will need Python 3.8 or later. Please install the required Python packages listed in `requirements.txt`.

```bash
git clone git@github.com:majkevh/qupid.git
cd qupid
pip install -r requirements.txt
```

### Data Folder 
- `data/`: Contains some of the datasets used in the notebooks. For conducting experiments involving graph data, I utilized datasets and functions sourced from the [*PersLay*](https://github.com/MathieuCarriere/perslay) and [*ATOL*](https://github.com/martinroyer/atol) repositories. 

## Citation
If you use this project, please cite: 

[![DOI](https://zenodo.org/badge/829908288.svg)](https://zenodo.org/doi/10.5281/zenodo.13827944)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
