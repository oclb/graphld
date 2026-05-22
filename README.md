# GraphLD

This repository implements the graphREML and graphREML-ST methods described in:
> Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O'Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11. DOI: [10.1101/2024.11.04.24316716](https://doi.org/10.1101/2024.11.04.24316716)

and provides a Python API for computationally efficient linkage disequilibrium (LD) matrix operations with [LD graphical models](https://github.com/awohns/ldgm) (LDGMs), described in:
> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O’Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: [10.1038/s41588-023-01487-8](https://pubmed.ncbi.nlm.nih.gov/37640881/)

It also provides very fast utilities for simulating GWAS summary statistics, performing LD clumping, and computing polygenic risk scores.

## Installation

```bash
git clone https://github.com/oclb/graphld.git
```

`graphld` depends on [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [scikit-sparse](https://scikit-sparse.readthedocs.io/en/latest/). It also requires downloading LDGM precision matrices (a few GB). These are not required if you only wish to use the enrichment score test. 

For platform-specific installation instructions, development setup, and data download options, see [docs/installation.md](docs/installation.md).

## Documentation

The repository provides an [AGENTS.md](AGENTS.md) file and skills that your coding agent can use for installation + analyses. For human-facing documentation:

- [Installation Guide](docs/installation.md)
- [Command Line Interface](docs/cli.md)
- [Enrichment Score Test](docs/score_test.md)
- [Python Guide](docs/python_api.md)
- [File Formats](docs/file_formats.md)
- [API Reference](docs/api/graphld.md)

## See Also

- Main LDGM repository, including a MATLAB API: [https://github.com/awohns/ldgm](https://github.com/awohns/ldgm)
- Original graphREML repository, with a MATLAB implementation: [https://github.com/huilisabrina/graphREML](https://github.com/huilisabrina/graphREML) (we recommend using the Python implementation, which is much faster)
- LD score regression repository: [https://github.com/bulik/ldsc](https://github.com/bulik/ldsc)
- Giulio Genovese has implemented a LDGM-VCF file format specification and a bcftools plugin written in C with partially overlapping features, available [here](https://github.com/freeseek/score).
- `graphld` relies on sparse matrix operations implemented in [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).
