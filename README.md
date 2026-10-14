# Lab 2: 3-Layer MLP from Scratch

Implemented a 3-layer neural network from scratch using NumPy to classify handwritten digits (0–9) from the `load_digits` dataset.

## What was done:
- **Architecture:** 64 inputs → Hidden 1 (ReLU) → Hidden 2 (ReLU) → Output (10, Softmax).
- **Data:** Pixels divided by 16 to scale into `[0, 1]`. Labels converted to one-hot encoding.
- **Loss:** Categorical Cross-Entropy with stable Softmax.
- **Backprop:** Implemented gradient descent using the chain rule.

## Results:

### 1. Checks
- **Initial Loss:** `2.3026` (matches theoretical random guessing: $-\ln(0.1)$).
- **Gradient Check:** `9/10 passed` with relative error $< 10^{-6}$. Check 9 had an error of `4.13e-06`, which is basically just floating-point rounding.

### 2. Training (50 epochs, batch size = 32)
- `W=16, lr=0.001`: 10.19% (too slow, stays at random guess)
- `W=16, lr=0.01`: 10.19%
- `W=16, lr=0.1`: 43.89%
- `W=64, lr=0.001`: 10.19%
- `W=64, lr=0.01`: 10.19%
- `W=64, lr=0.1`: **89.07%** *(Best performance)*

*Loss curves saved to `results/loss_curves.png`.*

### 3. Timing (Batch size = 32, W = 64)
- **Forward pass:** 19.00 µs
- **Backward pass:** 22.87 µs  
Backward takes longer because it computes gradients for both weights and activations.

## How to run:
```bash
pip install numpy matplotlib scikit-learn
python lab2.py
