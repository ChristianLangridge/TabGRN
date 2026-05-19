# TRACE
[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=flat-square)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Data Requests](https://img.shields.io/badge/data_requests-zcbtcl9%40ucl.ac.uk-informational?style=flat-square&logo=gmail)](https://github.com/ChristianLangridge)
[![CI](https://github.com/ChristianLangridge/TRACE/actions/workflows/tests.yml/badge.svg)](https://github.com/ChristianLangridge/TRACE/actions/workflows/tests.yml)



TRACE is a novel PFN-style model, fine-tuned with neuroepithelial brain organoid sc-RNA-seq data for cell trajectory and identity prediction tasks. 

***Joint-rotational project at Queen Mary University London, University College London and The Alan Turing Institute under Prof. Julien Gautrot, Prof. Yanlan Mao and Dr. Isabel Palacios and Dr. Federico Nanni.***

---

# Table of Contents

- [Background](#background)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Known Issues & Limitations](#known-issues--limitations)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---


## Background

---

## Repository Structure

---

## Installation 

**Prerequisites**

- Python 3.12+

### 1. Clone the repository 

```bash
git https://github.com/ChristianLangridge/TRACE.git
cd TRACE
```

### 2. Create and activate a conda environment using smt_pipeline.yml 

```bash
conda env create -f TRACE.yml 
```

---

## Data Requirements

--- 

## Usage

### 1. Package registration

Before running scripts, register the package using `pip install -e .` from the project root with `pyproject.toml`. 

### 2. Data implementation

Place all raw data within a 'data/raw/' folder so the path-finding system can retrieve it. 


---

## Known Issues & Limitations

---

## Contributing

This is an active research project. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature'`)
4. Push and open a Pull Request
5. Add a dated, detailed comment/annotation in the `CHANGELOG.md` file of the change

Please ensure any new scripts avoid hardcoded paths and include basic inline documentation.

---

## Citation

If you use this codebase or the TRACE architecture in your work, please cite:

```
Langridge, C. (2025–2026). TRACE.
Joint-rotational project, Queen Mary University London, University College London
and The Alan Turing Institute.
https://github.com/ChristianLangridge/TRACE
```

---

## License

This project is licensed under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.

---

*Developed as part of a joint rotational PhD project at Queen Mary University London, University College London and The Alan Turing Institute, under Prof. Julien Gautrot, Prof. Yanlan Mao, Dr. Isabel Palacios and Dr. Federico Nanni.*
