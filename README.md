# A Kernel-Based Method for Schrödinger Bridges

[![arXiv](https://img.shields.io/badge/arXiv-2310.14522-b31b1b.svg)](https://arxiv.org/abs/2310.14522)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official implementation of the paper:

**"A Kernel-Based Method for Schrödinger Bridges"**  
Yumiharu Nakano, arXiv:2310.14522  
https://arxiv.org/abs/2310.14522

---

## Overview

This code provides a **kernel-based numerical method for Schrödinger bridge problems** using  
- Hilbert space embeddings of probability measures,  
- Maximum Mean Discrepancy (MMD) penalty terms, and  
- neural stochastic differential equations (neural SDEs).  

The experiments here reproduce the main results reported in the paper.

---

## Requirements

The code has been tested with **Python 3.10**.  
Install the dependencies with:

```bash
git clone https://github.com/username/kernelSB.git
cd kernelSB
pip install -r requirements.txt
