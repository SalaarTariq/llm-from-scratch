# LLM From Scratch

Building neural networks and language models from scratch, following Andrej Karpathy's tutorials.

## Micrograd From Scratch

A from-scratch implementation of a scalar-valued autograd engine and a small neural network library.

### What's implemented:

- **Value class** — scalar value with automatic gradient computation (backpropagation)
  - Supports `+`, `*`, `/`, `-`, `**`, `tanh`, `exp` operations
  - Full backward pass with topological sort
- **Neuron** — single neuron with weights, bias, and tanh activation
- **Layer** — collection of neurons
- **MLP** — multi-layer perceptron built from layers
- **Visualization** — computation graph rendering using Graphviz

### Training example

A simple MLP with architecture `[3, 4, 4, 1]` is trained on 4 data points using gradient descent with MSE loss.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib graphviz torch
brew install graphviz
```

## Run

```bash
python Micrograd_from_scratch/lesson_1.py
```
