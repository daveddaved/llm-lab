// ============================================================================
// LLM Lab — Milestone 1: Tensors & Linear Algebra
// ============================================================================
//
// Before we can build a transformer, we need tensors. A tensor is just a
// multi-dimensional array of numbers. In deep learning, nearly everything
// is a tensor operation:
//
//   - A single number (scalar) is a 0-D tensor
//   - A list of numbers (vector) is a 1-D tensor: [768]
//   - A table of numbers (matrix) is a 2-D tensor: [512, 768]
//   - A batch of matrices is a 3-D tensor: [8, 512, 768]
//
// In a transformer, the key operations are:
//
//   MATRIX MULTIPLY (matmul): the workhorse. Attention, projections, and
//   feed-forward layers are ALL matrix multiplies. A GPU is basically a
//   machine optimized for doing enormous matmuls fast.
//
//   SOFTMAX: turns a vector of arbitrary numbers into a probability
//   distribution (all positive, sums to 1). Used in attention to decide
//   "how much should I attend to each token?"
//
//   RMSNORM (Root Mean Square Normalization): keeps values from exploding
//   or vanishing as they flow through dozens of layers. Without
//   normalization, deep networks are untrainable.
//
//   ELEMENT-WISE OPS: add, multiply, apply activation functions. These
//   are the glue between matmuls.
//
// We implement everything from scratch with f32. No BLAS, no ndarray.
// This is intentionally slow but maximally educational.
//
// ============================================================================

use std::fmt;

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
// Our tensor is dead simple: a flat Vec<f32> plus a shape vector.
//
// A [3, 4] matrix is stored as 12 contiguous floats in ROW-MAJOR order:
//   row 0: [a b c d]  row 1: [e f g h]  row 2: [i j k l]
//   memory: [a b c d e f g h i j k l]
//
// To access element [i][j] in a [rows, cols] matrix:
//   index = i * cols + j
//
// Row-major is what C, Rust, and PyTorch use. (Fortran and Julia use
// column-major.) Row-major means iterating over the last dimension is
// cache-friendly — important later when we optimize.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    /// Create a tensor from raw data and a shape.
    /// Panics if data length doesn't match the shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Tensor::new: data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected,
        );
        Tensor { data, shape }
    }

    /// Create a 1-D tensor (vector) from a slice.
    pub fn from_vec(data: &[f32]) -> Self {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    // -----------------------------------------------------------------------
    // MATRIX MULTIPLY (matmul)
    // -----------------------------------------------------------------------
    // The most important operation in all of deep learning.
    //
    // Given matrices A [M, K] and B [K, N], produce C [M, N] where:
    //   C[i][j] = sum over k of A[i][k] * B[k][j]
    //
    // Each element of C is a DOT PRODUCT of a row from A and a column from B.
    //
    // Why matmul matters for transformers:
    //   - Attention: Q @ K^T computes similarity between every pair of tokens
    //   - Projections: input @ W_q gives us the "query" for each token
    //   - Feed-forward: two matmuls with a nonlinearity in between
    //   - Output: final matmul projects hidden states to vocabulary logits
    //
    // Time complexity: O(M * N * K). For a 768-dim model with 512 tokens,
    // a single attention matmul is ~200 million multiply-adds. This is why
    // GPUs exist — they do thousands of these in parallel.
    // -----------------------------------------------------------------------

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul: self must be 2-D, got {:?}", self.shape);
        assert_eq!(other.ndim(), 2, "matmul: other must be 2-D, got {:?}", other.shape);

        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        assert_eq!(k, k2, "matmul: incompatible shapes {:?} @ {:?}", self.shape, other.shape);

        let mut result = vec![0.0f32; m * n];

        // The classic triple loop. This is O(M*N*K).
        // In a real engine, this would be a call to BLAS (sgemm) or a
        // hand-tuned SIMD kernel. For learning, the naive version is clear.
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    // A[i, ki] * B[ki, j]
                    sum += self.data[i * k + ki] * other.data[ki * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor::new(result, vec![m, n])
    }

    // -----------------------------------------------------------------------
    // MATRIX-VECTOR MULTIPLY (matvec)
    // -----------------------------------------------------------------------
    // A special case that's extremely common in inference: multiply a
    // matrix [M, K] by a vector [K] to get a vector [M].
    //
    // During inference we process one token at a time, so most "matmuls"
    // are actually matvecs. This is why LLM inference is MEMORY-BOUND,
    // not compute-bound — we load an entire weight matrix from memory
    // just to multiply it by a single vector.
    // -----------------------------------------------------------------------

    pub fn matvec(&self, vec: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matvec: matrix must be 2-D");
        assert_eq!(vec.ndim(), 1, "matvec: vector must be 1-D");

        let m = self.shape[0];
        assert_eq!(self.shape[1], vec.shape[0], "matvec: incompatible shapes {:?} @ {:?}", self.shape, vec.shape);

        // Delegate to the optimized matvec_into which uses 4-accumulator splitting.
        let mut result = Tensor::zeros(&[m]);
        self.matvec_into(vec, &mut result);
        result
    }

    // -----------------------------------------------------------------------
    // IN-PLACE MATRIX-VECTOR MULTIPLY
    // -----------------------------------------------------------------------
    // PERF: During the forward pass, matvec is called hundreds of times per
    // token. The allocating version creates a new Vec each time. This variant
    // writes into a pre-allocated output buffer, eliminating allocation in
    // the hot path. The output tensor must already have shape [M].
    // -----------------------------------------------------------------------

    pub fn matvec_into(&self, vec: &Tensor, out: &mut Tensor) {
        assert_eq!(self.ndim(), 2, "matvec_into: matrix must be 2-D");
        assert_eq!(vec.ndim(), 1, "matvec_into: vector must be 1-D");

        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(k, vec.shape[0], "matvec_into: incompatible shapes {:?} @ {:?}", self.shape, vec.shape);
        assert_eq!(out.data.len(), m, "matvec_into: output buffer size mismatch");

        // ===================================================================
        // OPTIMIZED DOT PRODUCT WITH 4-WAY ACCUMULATOR SPLITTING
        // ===================================================================
        //
        // WHY 4 ACCUMULATORS?
        // -------------------
        // A single-accumulator dot product looks like:
        //     sum += a[i] * b[i];   // iteration i
        //     sum += a[i+1] * b[i+1]; // iteration i+1 — STALL! Depends on 'sum' from above
        //
        // On modern CPUs, floating-point addition has a LATENCY of ~4 clock cycles
        // but a THROUGHPUT of 1 per cycle (pipelined). With one accumulator, each
        // add must wait for the previous add to finish (data dependency), so we
        // waste 3/4 of the CPU's add throughput.
        //
        // With 4 independent accumulators (sum0, sum1, sum2, sum3), the CPU can
        // pipeline all 4 adds simultaneously:
        //     Cycle 1: sum0 += a[0]*b[0]  (starts, finishes at cycle 4)
        //     Cycle 2: sum1 += a[1]*b[1]  (starts, finishes at cycle 5)
        //     Cycle 3: sum2 += a[2]*b[2]  (starts, finishes at cycle 6)
        //     Cycle 4: sum3 += a[3]*b[3]  (starts, finishes at cycle 7)
        //     Cycle 5: sum0 += a[4]*b[4]  (sum0 is now free — no stall!)
        //
        // This achieves ~4x better utilization of the FP add pipeline.
        //
        // WHY THIS ENABLES AUTO-VECTORIZATION
        // ------------------------------------
        // LLVM's auto-vectorizer looks for independent operations it can pack
        // into SIMD lanes. With one accumulator, the loop-carried dependency
        // prevents vectorization. With 4 independent accumulators operating on
        // groups of 4 elements, LLVM can:
        //   1. Pack 4 f32 multiplies into one SIMD multiply (vmulps on AVX)
        //   2. Pack 4 f32 adds into one SIMD add (vaddps on AVX)
        //   3. Further unroll to use 256-bit AVX (8 f32s) or 512-bit AVX-512
        //
        // The chunks_exact(4) pattern gives LLVM a strong hint that iterations
        // are independent, making auto-vectorization much more reliable.
        //
        // PERFORMANCE IMPACT
        // ------------------
        // For the stories15M model (dim=288, hidden_dim=768), matvec is called
        // ~20 times per layer x 6 layers = ~120 times per token. Each call
        // processes 288*768 = 221,184 multiply-adds for the FFN matrices.
        // The 4-accumulator pattern typically yields a 2-4x speedup over the
        // naive single-accumulator loop, depending on the CPU's SIMD width.
        // ===================================================================

        let v = &vec.data;
        for (row, o) in self.data.chunks_exact(k).zip(out.data.iter_mut()) {
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;

            // PERF: Use chunks_exact(4) on BOTH the row and vector slices so that
            // LLVM can prove both slices have exactly 4 elements per iteration.
            // This eliminates bounds checks in the inner loop, which is critical:
            // bounds checks prevent SIMD vectorization because they introduce
            // branches that break the vectorizer's stride pattern analysis.
            let row_chunks = row.chunks_exact(4);
            let row_remainder = row_chunks.remainder();
            for (r, vi) in row_chunks.zip(v.chunks_exact(4)) {
                // Each accumulator is independent — no data dependency between them.
                // This lets the CPU's out-of-order engine and SIMD units work in parallel.
                sum0 += r[0] * vi[0];
                sum1 += r[1] * vi[1];
                sum2 += r[2] * vi[2];
                sum3 += r[3] * vi[3];
            }

            // Handle remaining elements (when k is not a multiple of 4).
            // PERF: The remainder slice from chunks_exact is at most 3 elements,
            // so this loop body executes 0-3 times — negligible cost.
            let remainder_start = k - row_remainder.len();
            for i in 0..row_remainder.len() {
                sum0 += row_remainder[i] * v[remainder_start + i];
            }

            // Combine the 4 accumulators. This final reduction is just 3 adds —
            // negligible compared to the k/4 iterations in the main loop.
            *o = sum0 + sum1 + sum2 + sum3;
        }
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE OPERATIONS
    // -----------------------------------------------------------------------
    // These apply an operation to each element independently.
    // Used for residual connections (add), scaling, and activations.
    // -----------------------------------------------------------------------

    /// Element-wise addition. Shapes must match.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "add: shape mismatch {:?} vs {:?}", self.shape, other.shape);
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Element-wise multiplication (Hadamard product).
    /// Used in gating mechanisms like SwiGLU.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "mul: shape mismatch {:?} vs {:?}", self.shape, other.shape);
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Multiply every element by a scalar.
    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Tensor::new(data, self.shape.clone())
    }

    // -----------------------------------------------------------------------
    // SOFTMAX
    // -----------------------------------------------------------------------
    // Converts a vector of arbitrary real numbers into a probability
    // distribution: all values become positive and sum to 1.
    //
    //   softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    //
    // The "subtract max" trick prevents overflow. exp(1000) = infinity,
    // but exp(1000 - 1000) = exp(0) = 1. Since we divide by the sum,
    // subtracting a constant doesn't change the result (it cancels out).
    //
    // In transformers, softmax is used in attention:
    //   attention_weights = softmax(Q @ K^T / sqrt(d_k))
    //
    // This turns raw similarity scores into "what fraction of attention
    // should each token get?" Higher scores → more attention.
    // -----------------------------------------------------------------------

    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.ndim(), 1, "softmax: expected 1-D tensor, got {:?}", self.shape);

        // Step 1: find the max (for numerical stability)
        let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Step 2: exp(x_i - max)
        let exps: Vec<f32> = self.data.iter().map(|x| (x - max).exp()).collect();

        // Step 3: normalize so they sum to 1
        let sum: f32 = exps.iter().sum();
        let data: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        Tensor::new(data, self.shape.clone())
    }

    // -----------------------------------------------------------------------
    // RMSNORM (Root Mean Square Layer Normalization)
    // -----------------------------------------------------------------------
    // Normalization keeps activations from exploding or vanishing as data
    // flows through many layers. Without it, gradients either blow up
    // (making training unstable) or disappear (making learning impossible).
    //
    // RMSNorm is simpler than LayerNorm — it skips the mean subtraction:
    //
    //   rms = sqrt(mean(x_i^2) + eps)
    //   output_i = (x_i / rms) * weight_i
    //
    // The weight vector is a learned per-element scaling factor.
    // eps (epsilon) prevents division by zero.
    //
    // Llama 2 uses RMSNorm instead of LayerNorm because it's faster
    // (one fewer reduction) and works just as well in practice.
    // -----------------------------------------------------------------------

    pub fn rmsnorm(&self, weight: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 1, "rmsnorm: expected 1-D tensor");
        assert_eq!(self.shape, weight.shape, "rmsnorm: shape mismatch");

        let n = self.data.len() as f32;
        let eps = 1e-5f32;

        // Step 1: compute RMS (root mean square)
        let ss: f32 = self.data.iter().map(|x| x * x).sum::<f32>() / n;
        // PERF: Precompute 1/rms so the inner loop uses multiply instead of divide.
        // Multiply is ~4x faster than divide on most CPUs.
        let inv_rms = 1.0 / (ss + eps).sqrt();

        // Step 2: normalize and scale by learned weights
        let data: Vec<f32> = self.data.iter().zip(&weight.data)
            .map(|(x, w)| x * inv_rms * w)
            .collect();

        Tensor::new(data, self.shape.clone())
    }

    // -----------------------------------------------------------------------
    // IN-PLACE RMSNORM
    // -----------------------------------------------------------------------
    // PERF: Avoids allocating a new Vec for the normalized result. The forward
    // pass calls rmsnorm twice per layer (before attention and before FFN),
    // so eliminating these allocations matters for throughput.
    // -----------------------------------------------------------------------

    pub fn rmsnorm_into(&self, weight: &Tensor, out: &mut Tensor) {
        assert_eq!(self.ndim(), 1, "rmsnorm_into: expected 1-D tensor");
        assert_eq!(self.shape, weight.shape, "rmsnorm_into: shape mismatch");
        assert_eq!(self.data.len(), out.data.len(), "rmsnorm_into: output size mismatch");

        let n = self.data.len() as f32;
        let eps = 1e-5f32;

        let ss: f32 = self.data.iter().map(|x| x * x).sum::<f32>() / n;
        // PERF: Precompute 1/rms so the inner loop uses multiply instead of divide.
        // Multiply is ~4x faster than divide on most CPUs.
        let inv_rms = 1.0 / (ss + eps).sqrt();

        for ((x, w), o) in self.data.iter().zip(&weight.data).zip(out.data.iter_mut()) {
            *o = x * inv_rms * w;
        }
    }

    // -----------------------------------------------------------------------
    // ACTIVATION FUNCTIONS
    // -----------------------------------------------------------------------

    /// SiLU (Sigmoid Linear Unit), also called "swish":
    ///   silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    ///
    /// Used in Llama 2's feed-forward network (SwiGLU activation).
    /// SiLU is smooth and non-monotonic — it slightly suppresses small
    /// negative values instead of zeroing them like ReLU.
    pub fn silu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    // -----------------------------------------------------------------------
    // FUSED SILU-MULTIPLY (in-place)
    // -----------------------------------------------------------------------
    // PERF: The SwiGLU FFN computes silu(gate) * up, which normally requires
    // two allocations (one for silu, one for mul). This fused variant applies
    // silu to self and multiplies by other in a single pass, writing the
    // result into self. One loop, zero allocations, better cache utilization.
    // -----------------------------------------------------------------------

    pub fn silu_mul_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "silu_mul_inplace: shape mismatch");
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            let silu_a = *a * (1.0 / (1.0 + (-*a).exp()));
            *a = silu_a * b;
        }
    }

    // -----------------------------------------------------------------------
    // ARGMAX
    // -----------------------------------------------------------------------
    // Returns the index of the largest element. In inference, this is
    // "greedy decoding" — always pick the most likely next token.
    // -----------------------------------------------------------------------

    pub fn argmax(&self) -> usize {
        self.data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Get a row from a 2-D tensor as a 1-D tensor.
    /// This is how we do embedding lookups: row i of the embedding
    /// table is the vector for token i.
    pub fn row(&self, i: usize) -> Tensor {
        assert_eq!(self.ndim(), 2, "row: expected 2-D tensor");
        let cols = self.shape[1];
        let start = i * cols;
        Tensor::from_vec(&self.data[start..start + cols])
    }

    // -----------------------------------------------------------------------
    // COPY ROW INTO BUFFER
    // -----------------------------------------------------------------------
    // PERF: Copies a row from a 2-D tensor directly into an existing 1-D
    // buffer, avoiding the allocation that row() performs. Used for embedding
    // lookup in the forward pass where we already have a pre-allocated x buffer.
    // -----------------------------------------------------------------------

    pub fn copy_row_into(&self, i: usize, out: &mut Tensor) {
        assert_eq!(self.ndim(), 2, "copy_row_into: expected 2-D tensor");
        let cols = self.shape[1];
        assert_eq!(out.data.len(), cols, "copy_row_into: output size mismatch");
        let start = i * cols;
        out.data.copy_from_slice(&self.data[start..start + cols]);
    }

    /// Zero out all data in this tensor (reuse the buffer).
    /// PERF: Used to reset buffers between iterations without reallocating.
    pub fn zero_out(&mut self) {
        // PERF: fill() compiles to memset, which is heavily optimized by libc.
        self.data.fill(0.0);
    }

    // -----------------------------------------------------------------------
    // DOT PRODUCT
    // -----------------------------------------------------------------------
    // The dot product of two vectors is the sum of element-wise products.
    // In attention, we compute dot(query, key) for each token pair to get
    // the raw attention score. This is the scalar version of matmul.
    // -----------------------------------------------------------------------

    pub fn dot(&self, other: &Tensor) -> f32 {
        assert_eq!(self.ndim(), 1, "dot: expected 1-D tensors");
        assert_eq!(self.shape, other.shape, "dot: shape mismatch");
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    // -----------------------------------------------------------------------
    // IN-PLACE OPERATIONS
    // -----------------------------------------------------------------------
    // During the forward pass, we accumulate residual connections:
    //   hidden_state += attention_output
    //   hidden_state += ffn_output
    //
    // Allocating a new tensor each time wastes memory. In-place add
    // modifies the tensor directly, which is what real inference engines do.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // COSINE SIMILARITY
    // -----------------------------------------------------------------------
    // Measures the angle between two vectors, ignoring magnitude.
    //
    //   cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)
    //
    // Returns a value in [-1, 1]:
    //   +1 = vectors point in the same direction (semantically identical)
    //    0 = vectors are orthogonal (unrelated)
    //   -1 = vectors point in opposite directions (semantically opposite)
    //
    // In embedding space, cosine similarity is THE standard way to compare
    // word meanings. Two words with high cosine similarity have similar
    // representations — the model "thinks" they are related.
    //
    // WHY cosine instead of Euclidean distance? Embeddings can have very
    // different magnitudes. A word that appears often during training may
    // have a larger embedding vector than a rare word, but magnitude doesn't
    // reflect meaning — direction does. Cosine similarity normalizes away
    // the magnitude and measures only the angular relationship.
    // -----------------------------------------------------------------------

    pub fn cosine_similarity(&self, other: &Tensor) -> f32 {
        assert_eq!(self.ndim(), 1, "cosine_similarity: expected 1-D tensors");
        assert_eq!(self.shape, other.shape, "cosine_similarity: shape mismatch");

        // REVIEW: Fuse dot product, norm_a, and norm_b into a single pass over
        // both vectors. The original 3-pass version iterates the data 3 times;
        // this does it in 1 pass, which is ~3x better for cache utilization on
        // large embedding vectors (dim=288 in stories15M, 4096+ in production).
        let (dot, sq_a, sq_b) = self.data.iter().zip(&other.data).fold(
            (0.0f32, 0.0f32, 0.0f32),
            |(d, sa, sb), (&a, &b)| (d + a * b, sa + a * a, sb + b * b),
        );

        let norm_a = sq_a.sqrt();
        let norm_b = sq_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// In-place addition: self += other
    pub fn add_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "add_inplace: shape mismatch");
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// Get a mutable slice into a row of a 2-D tensor.
    /// Used for writing into KV cache positions.
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        assert_eq!(self.ndim(), 2, "row_mut: expected 2-D tensor");
        let cols = self.shape[1];
        let start = i * cols;
        &mut self.data[start..start + cols]
    }

    /// Create a tensor filled with random values in [-0.5, 0.5] scaled by 1/sqrt(dim).
    /// This is a simplified version of Xavier/Glorot initialization.
    /// Used for demo purposes when we don't have real weights.
    pub fn rand_init(shape: &[usize], seed: u64) -> Self {
        let size: usize = shape.iter().product();
        let scale = 1.0 / (*shape.last().unwrap_or(&1) as f32).sqrt();
        let mut data = Vec::with_capacity(size);
        // Simple LCG PRNG — not cryptographic, just reproducible
        let mut state = seed;
        for _ in 0..size {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((state >> 33) as f32 / u32::MAX as f32 - 0.5) * scale;
            data.push(val);
        }
        Tensor::new(data, shape.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.ndim() {
            1 => {
                write!(f, "[")?;
                for (i, v) in self.data.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    if i >= 8 && self.data.len() > 10 {
                        write!(f, "... ({} more)", self.data.len() - i)?;
                        break;
                    }
                    write!(f, "{:.4}", v)?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                writeln!(f, "Tensor [{} x {}]:", rows, cols)?;
                for i in 0..rows.min(6) {
                    write!(f, "  [")?;
                    for j in 0..cols.min(6) {
                        if j > 0 { write!(f, ", ")?; }
                        write!(f, "{:>8.4}", self.data[i * cols + j])?;
                    }
                    if cols > 6 { write!(f, ", ...")?; }
                    writeln!(f, "]")?;
                }
                if rows > 6 { writeln!(f, "  ...")?; }
                Ok(())
            }
            _ => write!(f, "Tensor {:?} ({} elements)", self.shape, self.numel()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "element {} differs: {} vs {} (tol={})", i, x, y, tol,
            );
        }
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        // [1 2 3]   [7  8 ]   [1*7+2*9+3*11  1*8+2*10+3*12]   [58  64]
        // [4 5 6] @ [9  10] = [4*7+5*9+6*11  4*8+5*10+6*12] = [139 154]
        //           [11 12]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        approx_eq(&c.data, &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    #[test]
    fn test_matmul_identity() {
        // Multiplying by identity matrix should return the original.
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let eye = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = a.matmul(&eye);
        approx_eq(&c.data, &a.data, 1e-5);
    }

    #[test]
    fn test_matvec() {
        // [1 2 3]   [1]   [1+4+9]   [14]
        // [4 5 6] @ [2] = [4+10+18] = [32]
        //           [3]
        let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = Tensor::from_vec(&[1.0, 2.0, 3.0]);
        let r = m.matvec(&v);
        assert_eq!(r.shape, vec![2]);
        approx_eq(&r.data, &[14.0, 32.0], 1e-5);
    }

    #[test]
    fn test_softmax_basic() {
        // softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
        let t = Tensor::from_vec(&[0.0, 0.0, 0.0]);
        let s = t.softmax();
        approx_eq(&s.data, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-5);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let t = Tensor::from_vec(&[1.0, 2.0, 3.0, 4.0]);
        let s = t.softmax();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {}", sum);
    }

    #[test]
    fn test_softmax_large_values() {
        // This would overflow without the subtract-max trick.
        let t = Tensor::from_vec(&[1000.0, 1001.0, 1002.0]);
        let s = t.softmax();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should handle large values");
        // The largest input should get the highest probability
        assert!(s.data[2] > s.data[1] && s.data[1] > s.data[0]);
    }

    #[test]
    fn test_rmsnorm() {
        let x = Tensor::from_vec(&[1.0, 2.0, 3.0, 4.0]);
        let w = Tensor::from_vec(&[1.0, 1.0, 1.0, 1.0]); // identity weights

        let result = x.rmsnorm(&w);

        // rms = sqrt(mean([1, 4, 9, 16]) + 1e-5) = sqrt(7.5 + 1e-5) ≈ 2.7386
        // normalized: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
        let rms = (7.5f32 + 1e-5).sqrt();
        let expected: Vec<f32> = vec![1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
        approx_eq(&result.data, &expected, 1e-4);
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        let t = Tensor::from_vec(&[0.0]);
        let s = t.silu();
        approx_eq(&s.data, &[0.0], 1e-5);

        // silu(x) for large positive x ≈ x (sigmoid → 1)
        let t = Tensor::from_vec(&[10.0]);
        let s = t.silu();
        assert!((s.data[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_element_wise_add() {
        let a = Tensor::from_vec(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_vec(&[4.0, 5.0, 6.0]);
        let c = a.add(&b);
        approx_eq(&c.data, &[5.0, 7.0, 9.0], 1e-5);
    }

    #[test]
    fn test_element_wise_mul() {
        let a = Tensor::from_vec(&[2.0, 3.0, 4.0]);
        let b = Tensor::from_vec(&[5.0, 6.0, 7.0]);
        let c = a.mul(&b);
        approx_eq(&c.data, &[10.0, 18.0, 28.0], 1e-5);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_vec(&[0.1, 0.3, 0.9, 0.2]);
        assert_eq!(t.argmax(), 2);
    }

    #[test]
    fn test_row() {
        // 2x3 matrix, extract rows
        let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r0 = m.row(0);
        let r1 = m.row(1);
        approx_eq(&r0.data, &[1.0, 2.0, 3.0], 1e-5);
        approx_eq(&r1.data, &[4.0, 5.0, 6.0], 1e-5);
    }

    // REVIEW: Test coverage for the new cosine_similarity method added for
    // the deep-embeddings binary. Verifies identity, orthogonality, and
    // zero-vector edge cases to ensure correctness of the fused single-pass
    // implementation.
    #[test]
    fn test_cosine_similarity_identical() {
        let a = Tensor::from_vec(&[1.0, 2.0, 3.0]);
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-5,
            "identical vectors should have cosine similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Tensor::from_vec(&[1.0, 0.0]);
        let b = Tensor::from_vec(&[0.0, 1.0]);
        assert!(a.cosine_similarity(&b).abs() < 1e-5,
            "orthogonal vectors should have cosine similarity 0.0");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = Tensor::from_vec(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_vec(&[-1.0, -2.0, -3.0]);
        assert!((a.cosine_similarity(&b) - (-1.0)).abs() < 1e-5,
            "opposite vectors should have cosine similarity -1.0");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = Tensor::from_vec(&[1.0, 2.0, 3.0]);
        let zero = Tensor::from_vec(&[0.0, 0.0, 0.0]);
        assert_eq!(a.cosine_similarity(&zero), 0.0,
            "zero vector should return 0.0 (not NaN)");
    }
}
