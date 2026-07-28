# Deep Learning & Neural Networks Practical Repository

Welcome to this repository! This project serves as a dedicated workspace for **practical exercises, self-study, lab assignments, and skill assessment** focused on Artificial Neural Networks (ANN) and foundational Deep Learning concepts.

---

##  Repository Purpose & Acknowledgment

Please note the following regarding the scope and intent of this repository:

* **Academic & Self-Study Purpose:** This repository was constructed primarily for hands-on learning, continuous practical experimentation, and course assessments.
* **Non-Production Environment:** The source code, implementation files, and notebooks contained herein are designed for educational demonstrations, algorithm prototyping, and concept verification rather than production-level deployment.
* **Originality & Refactoring:** Code snippets and implementations are structured from scratch or refactored for clarity, readability, and modular design.

---

##  Key Topics Covered

This repository contains implementations and notebooks exploring key deep learning fundamentals:

1. **Foundational Neural Architectures**
   * Single-Layer Perceptron (SLP) implementation from scratch.
   * Multi-Layer Perceptron (MLP) architectures and backpropagation logic.
2. **Mathematical Concepts & Vectorization**
   * Vectorized forward and backward passes using `NumPy`.
   * Activation functions: Sigmoid, ReLU, Tanh, and Heaviside Step functions.
   * Loss functions and optimization techniques (e.g., Perceptron Learning Rule, Gradient Descent).
3. **Logic Gate & Toy Problem Simulations**
   * Binary classification of linearly separable functions (AND, OR gates).
   * Exploring decision boundaries and non-linear separability (XOR problem).

---

##  Repository Structure

```text
.
├── notebooks/              # Jupyter notebooks for interactive demonstrations
│   └── perceptron_and_gate.ipynb
├── src/                    # Refactored Python source modules
│   ├── perceptron.py       # Vectorized Perceptron class definition
│   └── utils.py            # Helper scripts and activation functions
├── README.md               # Repository documentation and overview
└── requirements.txt        # Python package dependencies
