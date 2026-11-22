# Trying to figure out the pi0/pi0.5 policy math
# Flow matching algorithm and flow matching loss
# FAST tokenizer

# Goal: To figure out if it can be of any use for lc-act, and if so how to implement it.

import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# in pi0.5, PI uses a mix of DCT based tokenization and flow matching. During pre-training it uses DCT for tokenization (rep actions as discrete tokens) and during inference they use flow matching for smooth, high freq motor control.

## Let's try to implement DCT-2 and corresponding IDCT-3 first.

def dct2(x: np.ndarray) -> np.ndarray:
    """
    The orthonomal DCT-2 transforms a signal x[n] of length N into freq coefficients X[k].
    """
    N, K = x.shape[0], x.shape[0]
    X = np.zeros_like(x, dtype=np.float32)
    for k in range(0, K):
        alpha_k = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(0, N):
            X[k] += x[n] * np.cos((np.pi * k  * (2*n+1)) / (2 * N))
        X[k] *= alpha_k
    return X

def idct3(X: np.ndarray) -> np.ndarray:
    """
    The orthonormal IDCT-3 transforms freq coefficients X[k] of length N back into signal x[n].
    """
    N, K = X.shape[0], X.shape[0]
    x = np.zeros_like(X, dtype=np.float32)
    for n in range(0, N):
        for k in range(0, K):
            alpha_k = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            x[n] += alpha_k * X[k] * np.cos((np.pi * k * (2*n+1)) / (2 * N))
    return x

# PLotting the progressive reconstruction of a signal using DCT-2 and IDCT-3
def plot_dct_reconstruction(signal: np.ndarray):
    N = signal.shape[0]
    X = dct2(signal)
    reconstructed_signals = []
    for k in range(N):
        # Zero out all but the first k+1 coefficients
        X_k = np.zeros_like(X)
        X_k[:k+1] = X[:k+1]
        x_reconstructed = idct3(X_k)
        reconstructed_signals.append(x_reconstructed)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Original Signal', color='blue')
    plt.title('Original Signal')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    for k, x_rec in enumerate(reconstructed_signals):
        plt.plot(x_rec, label=f'Reconstructed with {k+1} coeffs')
    plt.title('Progressive Reconstruction using DCT-2 and IDCT-3')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# The next part is to design the FAST tokenizer.
# Working is pretty simple: We normalize the action chunk, and perform DCT, then quantize to create a sparse frequency matrix, which is flattened. This result is encoded using BPE to get the compressed action tokens. During inferecne, the decoder will perform the reverse operation: BPE decode, reshape, IDCT to get back the action chunk.

# For implementation we can create one single function that acts as the parent and have separate functions for each of the steps.
def fast_tokenizer(action_chunk: np.ndarray, num_coeffs: int) -> np.ndarray:
    """
    FAST tokenizer for action chunks.
    1. Normalize the action chunk.
    2. Perform DCT-2 to get frequency coefficients.
    3. Quantize the coefficients to create a sparse frequency matrix.
    4. Flatten and round the matrix.
    5. Encode using BPE to get compressed action tokens.
    """
    # Step 1: Normalize
    normalized_chunk = (action_chunk - np.mean(action_chunk)) / np.std(action_chunk)

    # Step 2: DCT-2
    dct_coeffs = dct2(normalized_chunk)

    # Step 3: Quantize (keep only the first num_coeffs coefficients)
    quantized_coeffs = np.zeros_like(dct_coeffs)
    quantized_coeffs[:num_coeffs] = dct_coeffs[:num_coeffs]

    # Step 4: Flatten 
    flattened_coeffs = quantized_coeffs.flatten().astype(np.int32)

    # Step 5: BPE encoding (placeholder, actual BPE implementation needed, or tiktoken?)
    # For now, we will just return the flattened coefficients as tokens
    return flattened_coeffs

def main():
    # Example usage
    x = np.array([0.0, 1.32e-1, 2.5e-1, 3.7e-1, 4.9e-1, 6.1e-1], dtype=np.float32)
    X = dct2(x)
    print("Input signal:", x)
    print("DCT-2 coefficients:", X)
    x_reconstructed = idct3(X)
    print("Reconstructed signal:", np.round(x_reconstructed, 5))
    plot_dct_reconstruction(x)

if __name__ == "__main__":
    main()