# GraphLD

This repository implements the graphREML method described in:

> Hui Li, Tushar Kamath, Rahul Mazumder, Xihong Lin, & Luke J. O'Connor (2024). _Improved heritability partitioning and enrichment analyses using summary statistics with graphREML_. medRxiv, 2024-11. DOI: [10.1101/2024.11.04.24316716](https://doi.org/10.1101/2024.11.04.24316716)

and provides a Python API for computationally efficient linkage disequilibrium (LD) matrix operations with [LD graphical models](https://github.com/awohns/ldgm) (LDGMs), described in:

> Pouria Salehi Nowbandegani, Anthony Wilder Wohns, Jenna L. Ballard, Eric S. Lander, Alex Bloemendal, Benjamin M. Neale, and Luke J. O'Connor (2023) _Extremely sparse models of linkage disequilibrium in ancestrally diverse association studies_. Nat Genet. DOI: [10.1038/s41588-023-01487-8](https://pubmed.ncbi.nlm.nih.gov/37640881/)

## Features

- **Heritability Estimation**: Run graphREML for heritability partitioning and enrichment analysis
- **Enrichment Score Test**: Fast score test for genomic annotations (no dependencies required)
- **Matrix Operations**: Efficient LD matrix operations using LDGM precision matrices
- **Simulation**: Simulate GWAS summary statistics from flexible mixture distributions
- **LD Clumping**: Fast LD-based pruning for polygenic score computation
- **BLUP**: Best Linear Unbiased Prediction for effect size estimation

## Table of Contents

- [Installation](installation.md): How to install GraphLD and its dependencies
- [Command Line Interface](cli.md): Using the `graphld` CLI
- [Enrichment Score Test](score_test.md): Fast annotation enrichment testing
- [API Reference](api/graphld.md): Python API documentation
- [File Formats](file_formats.md): Supported input/output formats
