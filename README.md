# Lab 2: A neural network from scratch (Ch. 3)

Implementation of a 3-layer neural network from scratch using only NumPy to classify handwritten digits (0–9) from the `load_digits` dataset.

---

### Task 1: Dataset & Preprocessing
- Loaded the `load_digits` dataset (8x8 grayscale images).
- Scaled input features by dividing pixel values by 16 so they fall into the `[0, 1]` range.
- Converted integer labels (0 to 9) into one-hot encoded vectors.
- Split the dataset into 70% train (1257 samples) and 30% test (540 samples) with `random_state=42` and stratification.

### Task 2: Network Architecture
- **Design:** Input (64) -> Hidden 1 (width $W$, ReLU) -> Hidden 2 (width $W$, ReLU) -> Output (10, Softmax).
- **Tested widths:** $W = 16$ and $W = 64$.
- **Shapes:**
  - $W_1$: `(64, W)`, $b_1$: `(W,)`
  - $W_2$: `(W, W)`, $b_2$: `(W,)`
  - $W_3$: `(W, 10)`, $b_3$: `(10,)`
- **Initialization:** Weights initialized from a normal distribution $\mathcal{N}(0, 0.01^2)$ to break symmetry; biases initialized to zeros.

### Task 3: Forward Pass & Sanity Check
- Implemented linear steps, ReLU activations, and a numerically stable Softmax (subtracting the row max before exponents to avoid overflow).
- **Check results:**
  - Output shape: `(32, 10)` for a batch of 32.
  - Row sums equal 1: `True` (verified within $10^{-6}$ tolerance).

### Task 4: Loss Function & Baseline Check
- Implemented Categorical Cross-Entropy with a small epsilon (`1e-15`) to prevent `log(0)`.
- **Sanity check:** The loss of the untrained network on the first batch was `2.3026`, which exactly matches the theoretical expectation $\ln(10) \approx 2.3026$ for 10 random classes.

### Task 5: Backpropagation & Gradient Check
- Derived gradients by hand using the chain rule:
  - Output layer: `dz3 = (a3 - targets) / batch_sz`
  - Hidden layers: backpropagated through weights and masked with the ReLU derivative indicator `(z > 0)`.
- **Numerical Gradient Checking:**
  - Tested 10 random weights using central difference with a step of $\pm 10^{-5}$.
  - **Result:** **9/10 passed** with relative error $< 10^{-6}$ (satisfying the manual requirement of at least 9/10).
  - *Check 9 Note:* Had a relative error of `4.13e-06` due to floating-point rounding on small values (`1.628631e-06` vs `1.628644e-06`), but the numbers matched up to 6 decimal places.

### Task 6: Training Loop & Experiments
- Ran mini-batch gradient descent (batch size 32, 50 epochs, shuffling dataset each epoch) across all 6 combinations of widths and learning rates:
  - `W=16, lr=0.001`: **10.19%** (learning rate too small, stuck at baseline)
  - `W=16, lr=0.01`: **10.19%** (too slow to learn within 50 epochs)
  - `W=16, lr=0.1`: **43.89%** (starts learning, but limited by layer width)
  - `W=64, lr=0.001`: **10.19%** (too slow)
  - `W=64, lr=0.01`: **10.19%** (too slow)
  - `W=64, lr=0.1`: **89.07%** (best performance: wider layers + high enough step size)
- All loss curves are plotted and saved to `results/loss_curves.png`.

### Task 7: Timing & Profiling 
- Timed forward and backward passes separately over 100 runs on a batch of 32 ($W=64$):
  - **Forward pass median time:** `19.00 µs`
  - **Backward pass median time:** `22.87 µs`
- **Why backward takes longer:**
  Forward only performs **one** matrix multiplication per layer ($X \cdot W$). Backward has to perform **two** matrix multiplications per layer: one to compute weight gradients ($A^T \cdot dZ$) and one to pass the error back to the previous layer ($dZ \cdot W^T$).

---

## How to Run
```bash
pip install numpy matplotlib scikit-learn
python lab2.py
