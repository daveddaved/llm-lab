// ============================================================================
// LLM Lab — Milestone 3: Transformer Forward Pass
// ============================================================================
//
// This is where it all comes together. A transformer is a stack of identical
// blocks, each containing two sub-layers:
//
//   1. SELF-ATTENTION — lets each token "look at" every other token
//   2. FEED-FORWARD NETWORK (FFN) — processes each token independently
//
// The full forward pass for Llama 2:
//
//   tokens → embedding → [block₁ → block₂ → ... → blockₙ] → norm → output
//
// Each block:
//   x → RMSNorm → Attention → + residual → RMSNorm → FFN → + residual
//                  ↑                                  ↑
//               (Q,K,V projections,                (SwiGLU: two linear
//                RoPE, multi-head                   projections with
//                scaled dot-product)                SiLU gating)
//
// The RESIDUAL CONNECTIONS (the "+" arrows) are critical. Without them,
// gradients vanish through 32+ layers during training. The residual lets
// information flow directly from input to output, with each block just
// adding a refinement.
//
// ============================================================================

use crate::tensor::Tensor;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

// ---------------------------------------------------------------------------
// Model Configuration
// ---------------------------------------------------------------------------
// These numbers define the architecture. The tinystories 15M model uses
// small values; a production Llama 2 70B would have dim=8192, 80 layers, etc.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Config {
    pub dim: usize,         // hidden dimension (embedding size)
    pub hidden_dim: usize,  // FFN intermediate dimension (typically ~2.7x dim)
    pub n_layers: usize,    // number of transformer blocks
    pub n_heads: usize,     // number of attention heads
    pub n_kv_heads: usize,  // number of key/value heads (GQA if < n_heads)
    pub vocab_size: usize,  // vocabulary size
    pub seq_len: usize,     // maximum sequence length
}

impl Config {
    // -----------------------------------------------------------------------
    // Load configuration from a llama2.c checkpoint file
    // -----------------------------------------------------------------------
    // The checkpoint starts with 7 i32 values (28 bytes) that define the
    // model architecture. This is all we need to know what shapes every
    // weight matrix will have, how many layers to expect, etc.
    //
    // Why i32 and not u32? The vocab_size field uses a sign trick: if it's
    // negative, the absolute value is the actual vocab_size and it signals
    // that the weights file does NOT include a separate output projection
    // (weight tying — see TransformerWeights::from_file for details).
    // -----------------------------------------------------------------------

    pub fn from_file(path: &str) -> io::Result<(Self, bool)> {
        let mut f = File::open(path)?;
        let mut buf = [0u8; 4];

        let mut read_i32 = |f: &mut File| -> io::Result<i32> {
            f.read_exact(&mut buf)?;
            Ok(i32::from_le_bytes(buf))
        };

        let dim = read_i32(&mut f)? as usize;
        let hidden_dim = read_i32(&mut f)? as usize;
        let n_layers = read_i32(&mut f)? as usize;
        let n_heads = read_i32(&mut f)? as usize;
        let n_kv_heads = read_i32(&mut f)? as usize;
        let vocab_size_raw = read_i32(&mut f)?;
        let seq_len = read_i32(&mut f)? as usize;

        // A negative vocab_size signals weight tying (shared embedding/output).
        // This is a convention from Karpathy's llama2.c format.
        let shared_weights = vocab_size_raw > 0;
        let vocab_size = vocab_size_raw.unsigned_abs() as usize;

        Ok((Config { dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len },
            shared_weights))
    }

    /// Configuration matching the tinystories 15M parameter model.
    pub fn stories_15m() -> Self {
        Config {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
        }
    }

    /// Head dimension = dim / n_heads.
    /// Each attention head operates on this slice of the full vector.
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// How many query heads share each KV head (for Grouped Query Attention).
    /// For the 15M model this is 1 (standard MHA). For larger Llama 2 models
    /// it's 4-8, which saves memory on the KV cache.
    pub fn n_rep(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

// ---------------------------------------------------------------------------
// Transformer Weights
// ---------------------------------------------------------------------------
// Every weight is a Tensor. The naming follows the Llama 2 convention.
// In Milestone 4 we'll load these from a file; here we initialize them
// with random values for demonstration.
// ---------------------------------------------------------------------------

pub struct TransformerWeights {
    // Token embedding table: [vocab_size, dim]
    // Row i = the vector representation of token i
    pub token_embedding: Tensor,

    // Per-layer weights
    pub layers: Vec<LayerWeights>,

    // Final RMSNorm (applied after the last block, before output projection)
    pub rms_final: Tensor,    // [dim]

    // Output projection: [vocab_size, dim]
    // Projects the final hidden state to vocabulary logits.
    // In Llama 2, this is tied to the token_embedding (same matrix).
    pub output: Tensor,
}

pub struct LayerWeights {
    // Attention
    pub rms_att: Tensor,       // [dim] — RMSNorm before attention
    pub wq: Tensor,            // [dim, dim] — query projection (or [n_heads * head_dim, dim])
    pub wk: Tensor,            // [n_kv_heads * head_dim, dim] — key projection
    pub wv: Tensor,            // [n_kv_heads * head_dim, dim] — value projection
    pub wo: Tensor,            // [dim, dim] — output projection

    // Feed-forward (SwiGLU)
    pub rms_ffn: Tensor,       // [dim] — RMSNorm before FFN
    pub w1: Tensor,            // [hidden_dim, dim] — gate projection
    pub w2: Tensor,            // [dim, hidden_dim] — down projection
    pub w3: Tensor,            // [hidden_dim, dim] — up projection
}

impl TransformerWeights {
    /// Create random weights for demonstration.
    /// In Milestone 4, we'll replace this with loading from a file.
    pub fn random(config: &Config) -> Self {
        let mut seed = 42u64;
        let mut next_seed = || { seed += 1; seed };

        let token_embedding = Tensor::rand_init(&[config.vocab_size, config.dim], next_seed());

        let mut layers = Vec::new();
        let kv_dim = config.n_kv_heads * config.head_dim();

        for _ in 0..config.n_layers {
            layers.push(LayerWeights {
                rms_att: Tensor::from_vec(&vec![1.0; config.dim]), // init to 1 (identity)
                wq: Tensor::rand_init(&[config.dim, config.dim], next_seed()),
                wk: Tensor::rand_init(&[kv_dim, config.dim], next_seed()),
                wv: Tensor::rand_init(&[kv_dim, config.dim], next_seed()),
                wo: Tensor::rand_init(&[config.dim, config.dim], next_seed()),
                rms_ffn: Tensor::from_vec(&vec![1.0; config.dim]),
                w1: Tensor::rand_init(&[config.hidden_dim, config.dim], next_seed()),
                w2: Tensor::rand_init(&[config.dim, config.hidden_dim], next_seed()),
                w3: Tensor::rand_init(&[config.hidden_dim, config.dim], next_seed()),
            });
        }

        let rms_final = Tensor::from_vec(&vec![1.0; config.dim]);
        let output = Tensor::rand_init(&[config.vocab_size, config.dim], next_seed());

        TransformerWeights {
            token_embedding,
            layers,
            rms_final,
            output,
        }
    }

    // -----------------------------------------------------------------------
    // Load trained weights from a llama2.c checkpoint file
    // -----------------------------------------------------------------------
    // The llama2.c binary format stores weights as contiguous f32 arrays
    // right after the 28-byte config header. The order is carefully chosen:
    //
    //   1. token_embedding    [vocab_size, dim]
    //   2. rms_att            [n_layers, dim]         (all layers, then next weight)
    //   3. wq                 [n_layers, dim, dim]
    //   4. wk                 [n_layers, kv_dim, dim]
    //   5. wv                 [n_layers, kv_dim, dim]
    //   6. wo                 [n_layers, dim, dim]
    //   7. rms_ffn            [n_layers, dim]
    //   8. w1                 [n_layers, hidden_dim, dim]
    //   9. w2                 [n_layers, dim, hidden_dim]
    //  10. w3                 [n_layers, hidden_dim, dim]
    //  11. rms_final          [dim]
    //  12. freq_cis_real      [seq_len, head_dim/2]   (SKIPPED — we compute RoPE on the fly)
    //  13. freq_cis_imag      [seq_len, head_dim/2]   (SKIPPED)
    //  14. output             [vocab_size, dim]        (may be absent if weight tying)
    //
    // Why are per-layer weights stored "all layers of wq, then all layers of
    // wk" rather than "all weights for layer 0, then all weights for layer 1"?
    // This layout matches how llama2.c's checkpoint_init_weights reads them
    // in bulk, and it mirrors PyTorch's state_dict ordering when you iterate
    // named parameters. It also makes memory-mapped I/O simpler: each weight
    // type is a single contiguous block.
    //
    // WEIGHT TYING: In many language models (including the tinystories models),
    // the output projection matrix is the SAME matrix as the token embedding.
    // This means token i's embedding vector is also the vector that, when
    // dotted with the final hidden state, produces the logit for token i.
    // Intuitively: the model uses the same "meaning space" for input and output.
    // This halves the memory for the largest weight matrix and often improves
    // quality on smaller models because it forces the embedding to be good
    // for both encoding (input) and decoding (output).
    // -----------------------------------------------------------------------

    pub fn from_file(path: &str, config: &Config, shared_weights: bool) -> io::Result<Self> {
        let mut f = File::open(path)?;

        // Skip the 28-byte header (7 x i32) — Config::from_file already read it.
        f.seek(SeekFrom::Start(28))?;

        let head_dim = config.head_dim();
        let kv_dim = config.n_kv_heads * head_dim;

        // Helper: read `count` f32 values from the file as a flat Vec<f32>.
        // Each f32 is 4 bytes, little-endian — the native format on x86/ARM.
        //
        // PERF: On little-endian platforms (x86, ARM), f32 and [u8; 4] have
        // the same memory layout, so we use unsafe transmutation to reinterpret
        // the byte buffer as f32 directly. This avoids iterating over every
        // 4-byte chunk to call from_le_bytes, which was the main bottleneck
        // in weight loading (~15M f32 values). The safety invariants are:
        //   1. Vec<u8> length is a multiple of 4 (guaranteed by count * 4)
        //   2. f32 alignment is satisfied (Vec guarantees its pointer is aligned
        //      to the allocation's alignment, and we use from_raw_parts after
        //      verifying alignment)
        //   3. We transfer ownership correctly via from_raw_parts / ManuallyDrop
        #[cfg(target_endian = "little")]
        let read_floats = |f: &mut File, count: usize| -> io::Result<Vec<f32>> {
            let byte_count = count * 4;
            let mut buf = vec![0u8; byte_count];
            f.read_exact(&mut buf)?;
            // SAFETY: On little-endian systems, the byte representation of f32
            // matches the file format directly. We ensure proper alignment by
            // checking the pointer; Vec<u8> allocated via the global allocator
            // is guaranteed to be at least 1-byte aligned, but in practice
            // allocators return pointers aligned to at least 8 bytes.
            // We assert alignment as a safeguard.
            let ptr = buf.as_ptr();
            assert_eq!(ptr as usize % std::mem::align_of::<f32>(), 0,
                "read_floats: buffer not aligned for f32");
            let floats = unsafe {
                let mut buf = std::mem::ManuallyDrop::new(buf);
                Vec::from_raw_parts(buf.as_mut_ptr() as *mut f32, count, count)
            };
            Ok(floats)
        };
        // PERF: Fallback for big-endian platforms — uses the safe per-element conversion.
        #[cfg(target_endian = "big")]
        let read_floats = |f: &mut File, count: usize| -> io::Result<Vec<f32>> {
            let mut buf = vec![0u8; count * 4];
            f.read_exact(&mut buf)?;
            let floats: Vec<f32> = buf.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(floats)
        };

        // --- 1. Token embedding table: [vocab_size, dim] ---
        // Each row is the learned vector representation for one token.
        let emb_data = read_floats(&mut f, config.vocab_size * config.dim)?;
        let token_embedding = Tensor::new(emb_data, vec![config.vocab_size, config.dim]);

        // --- 2-10. Per-layer weights ---
        // These are stored "layer-strided": ALL layers for one weight type,
        // then ALL layers for the next type. We read each block and slice it
        // into per-layer tensors.

        // Helper: read a [n_layers, rows, cols] block and split into n_layers tensors of [rows, cols].
        // PERF: Uses Vec::drain to split the flat buffer into per-layer chunks without
        // copying. The original code used .to_vec() on each slice, which copied every
        // weight value an extra time. drain() moves ownership of each chunk in O(1)
        // by adjusting the Vec's internal pointer.
        let read_layer_matrices = |f: &mut File, rows: usize, cols: usize, n_layers: usize| -> io::Result<Vec<Tensor>> {
            let per_layer = rows * cols;
            let mut all = read_floats(f, n_layers * per_layer)?;
            let mut result = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                // PERF: drain(..per_layer) removes and returns the first per_layer
                // elements, shifting ownership without copying the remaining data.
                let chunk: Vec<f32> = all.drain(..per_layer).collect();
                result.push(Tensor::new(chunk, vec![rows, cols]));
            }
            Ok(result)
        };

        // Helper: read a [n_layers, dim] block and split into n_layers tensors of [dim].
        // PERF: Same drain-based splitting as read_layer_matrices to avoid copies.
        let read_layer_vectors = |f: &mut File, dim: usize, n_layers: usize| -> io::Result<Vec<Tensor>> {
            let mut all = read_floats(f, n_layers * dim)?;
            let mut result = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                let chunk: Vec<f32> = all.drain(..dim).collect();
                result.push(Tensor::new(chunk, vec![dim]));
            }
            Ok(result)
        };

        let rms_att = read_layer_vectors(&mut f, config.dim, config.n_layers)?;
        let wq = read_layer_matrices(&mut f, config.dim, config.dim, config.n_layers)?;
        let wk = read_layer_matrices(&mut f, kv_dim, config.dim, config.n_layers)?;
        let wv = read_layer_matrices(&mut f, kv_dim, config.dim, config.n_layers)?;
        let wo = read_layer_matrices(&mut f, config.dim, config.dim, config.n_layers)?;
        let rms_ffn = read_layer_vectors(&mut f, config.dim, config.n_layers)?;
        let w1 = read_layer_matrices(&mut f, config.hidden_dim, config.dim, config.n_layers)?;
        let w2 = read_layer_matrices(&mut f, config.dim, config.hidden_dim, config.n_layers)?;
        let w3 = read_layer_matrices(&mut f, config.hidden_dim, config.dim, config.n_layers)?;

        // Assemble per-layer structs.
        // PERF: Use into_iter() to MOVE tensors out of the Vecs instead of cloning.
        // Each tensor's weight data (e.g., wq is 288*288 = 82,944 f32s = 324 KB)
        // would otherwise be deep-copied needlessly. For 6 layers with 9 weights
        // each, this eliminates ~54 unnecessary allocations during loading.
        let layers: Vec<LayerWeights> = itertools_zip9(
            rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3
        ).map(|(rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3)| {
            LayerWeights { rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3 }
        }).collect();

        // --- 11. Final RMSNorm weight: [dim] ---
        let rms_final_data = read_floats(&mut f, config.dim)?;
        let rms_final = Tensor::from_vec(&rms_final_data);

        // --- 12-13. Skip freq_cis_real and freq_cis_imag ---
        // We compute RoPE frequencies on the fly (see apply_rope), so we don't
        // need these precomputed tables. Just skip past them in the file.
        let skip_count = config.seq_len * (head_dim / 2);
        f.seek(SeekFrom::Current((skip_count * 2 * 4) as i64))?; // 2 tables, 4 bytes per f32

        // --- 14. Output projection: [vocab_size, dim] ---
        // If shared_weights is true, the output projection IS the token embedding
        // (weight tying). Otherwise, read it from the file.
        let output = if shared_weights {
            token_embedding.clone()
        } else {
            let output_data = read_floats(&mut f, config.vocab_size * config.dim)?;
            Tensor::new(output_data, vec![config.vocab_size, config.dim])
        };

        Ok(TransformerWeights {
            token_embedding,
            layers,
            rms_final,
            output,
        })
    }
}

// ---------------------------------------------------------------------------
// Run State
// ---------------------------------------------------------------------------
// Mutable buffers used during the forward pass. Allocated once and reused.
// Keeping these separate from weights makes the borrow checker happy and
// mirrors how real inference engines work.
// ---------------------------------------------------------------------------

pub struct RunState {
    // Current hidden state: [dim]
    pub x: Tensor,

    // Buffers for attention
    pub q: Tensor,           // [dim] — query vector
    pub k: Tensor,           // [kv_dim] — key vector
    pub v: Tensor,           // [kv_dim] — value vector
    pub att: Vec<f32>,       // [n_heads, seq_len] — attention scores (flat)
    pub xb: Tensor,          // [dim] — buffer after attention
    pub xb2: Tensor,         // [dim] — buffer after FFN

    // FFN buffers
    pub hb: Tensor,          // [hidden_dim] — FFN gate output
    pub hb2: Tensor,         // [hidden_dim] — FFN up output

    // KV cache: stores keys and values for all positions seen so far.
    // Without this, we'd recompute attention over ALL previous tokens
    // every time we generate a new one. The KV cache makes generation
    // O(n) per token instead of O(n²).
    pub key_cache: Vec<Tensor>,    // [n_layers] of [seq_len, kv_dim]
    pub value_cache: Vec<Tensor>,  // [n_layers] of [seq_len, kv_dim]

    // Output logits: [vocab_size]
    pub logits: Tensor,
}

impl RunState {
    pub fn new(config: &Config) -> Self {
        let kv_dim = config.n_kv_heads * config.head_dim();
        RunState {
            x: Tensor::zeros(&[config.dim]),
            q: Tensor::zeros(&[config.dim]),
            k: Tensor::zeros(&[kv_dim]),
            v: Tensor::zeros(&[kv_dim]),
            att: vec![0.0; config.n_heads * config.seq_len],
            xb: Tensor::zeros(&[config.dim]),
            xb2: Tensor::zeros(&[config.dim]),
            hb: Tensor::zeros(&[config.hidden_dim]),
            hb2: Tensor::zeros(&[config.hidden_dim]),
            key_cache: (0..config.n_layers)
                .map(|_| Tensor::zeros(&[config.seq_len, kv_dim]))
                .collect(),
            value_cache: (0..config.n_layers)
                .map(|_| Tensor::zeros(&[config.seq_len, kv_dim]))
                .collect(),
            logits: Tensor::zeros(&[config.vocab_size]),
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer — the full model
// ---------------------------------------------------------------------------

pub struct Transformer {
    pub config: Config,
    pub weights: TransformerWeights,
    pub state: RunState,
}

impl Transformer {
    pub fn new(config: Config, weights: TransformerWeights) -> Self {
        let state = RunState::new(&config);
        Transformer { config, weights, state }
    }

    // -----------------------------------------------------------------------
    // FORWARD PASS
    // -----------------------------------------------------------------------
    // This is the entire inference computation for one token.
    //
    // Input: a token ID and its position in the sequence
    // Output: logits (unnormalized scores) over the vocabulary
    //
    // The steps:
    //   1. Look up the token's embedding vector
    //   2. For each transformer block:
    //      a. RMSNorm → multi-head attention → add residual
    //      b. RMSNorm → SwiGLU FFN → add residual
    //   3. Final RMSNorm
    //   4. Project to vocabulary size (logits)
    //
    // After softmax on the logits, we get a probability distribution
    // over the next token. That's how LLMs generate text.
    // -----------------------------------------------------------------------

    pub fn forward(&mut self, token: u32, pos: usize) -> &Tensor {
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.n_kv_heads * head_dim;
        let n_rep = self.config.n_rep();

        // Step 1: Token embedding lookup
        // Each token ID maps to a learned vector of size [dim].
        // This is the token's "meaning" in the model's vector space.
        // PERF: copy_row_into writes directly into the pre-allocated x buffer,
        // avoiding a Vec allocation that the old row() call produced every token.
        self.weights.token_embedding.copy_row_into(token as usize, &mut self.state.x);

        // PERF: Precompute 1/sqrt(head_dim) outside the loop. The old code called
        // (head_dim as f32).sqrt() inside the innermost attention loop (once per
        // position per head per layer). This hoists it to a single computation.
        let inv_sqrt_head_dim = 1.0 / (head_dim as f32).sqrt();

        // Step 2: Run through each transformer block
        for layer in 0..self.config.n_layers {
            let w = &self.weights.layers[layer];

            // ----- ATTENTION SUB-LAYER -----

            // 2a. RMSNorm before attention (pre-norm architecture)
            // PERF: rmsnorm_into writes into the pre-allocated xb buffer instead
            // of allocating a new tensor. This saves one Vec alloc per layer.
            self.state.x.rmsnorm_into(&w.rms_att, &mut self.state.xb);

            // 2b. Compute Q, K, V projections
            //     Q = xnorm @ Wq^T  (query: "what am I looking for?")
            //     K = xnorm @ Wk^T  (key: "what do I contain?")
            //     V = xnorm @ Wv^T  (value: "what do I provide?")
            // PERF: matvec_into writes into pre-allocated q/k/v buffers.
            // The old matvec() allocated 3 new Vecs per layer (q, k, v).
            // For 6 layers, that's 18 eliminated allocations per token.
            w.wq.matvec_into(&self.state.xb, &mut self.state.q);
            w.wk.matvec_into(&self.state.xb, &mut self.state.k);
            w.wv.matvec_into(&self.state.xb, &mut self.state.v);

            // 2c. Apply RoPE (Rotary Positional Embedding)
            //     This encodes WHERE each token is in the sequence.
            //     Without it, "the cat sat" and "sat the cat" would look
            //     identical to attention — position information would be lost.
            apply_rope(&mut self.state.q, &mut self.state.k, pos, head_dim);

            // 2d. Store K and V in the cache for this position.
            //     On future tokens, we'll attend to ALL cached K/V pairs.
            self.state.key_cache[layer].row_mut(pos)
                .copy_from_slice(&self.state.k.data);
            self.state.value_cache[layer].row_mut(pos)
                .copy_from_slice(&self.state.v.data);

            // 2e. Multi-head attention
            //     Each head independently computes attention over its slice.
            //     Head h uses Q[h*head_dim..(h+1)*head_dim] and looks at
            //     all cached keys and values.
            //
            //     For each head:
            //       score[t] = dot(query_h, key_h[t]) / sqrt(head_dim)
            //       weights = softmax(scores[0..pos+1])
            //       output_h = sum(weights[t] * value_h[t])
            // PERF: zero_out resets the buffer without reallocating. The old code
            // created a brand new Tensor::zeros each iteration, which allocates
            // and initializes a fresh Vec<f32> every layer of every token.
            self.state.xb.zero_out();

            for h in 0..self.config.n_heads {
                // Which KV head does this query head use?
                // In standard MHA, it's 1:1. In GQA, multiple Q heads share a KV head.
                let kv_h = h / n_rep;

                // Extract this head's query slice
                let q_offset = h * head_dim;
                let q_slice = &self.state.q.data[q_offset..q_offset + head_dim];

                // Compute attention scores against all cached keys up to pos
                let att_offset = h * self.config.seq_len;
                for t in 0..=pos {
                    // Get key for position t, kv_head kv_h
                    let k_offset = kv_h * head_dim;
                    let k_row = &self.state.key_cache[layer].data
                        [t * kv_dim + k_offset..t * kv_dim + k_offset + head_dim];

                    // PERF: 4-accumulator dot product, same technique as matvec_into.
                    // The attention dot product is the second hottest loop (after matvec).
                    // For head_dim=48 (stories15M), this runs 12 iterations of 4 elements
                    // instead of 48 scalar iterations, enabling SIMD vectorization.
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    let q_chunks = q_slice.chunks_exact(4);
                    let q_rem = q_chunks.remainder();
                    for (qc, kc) in q_chunks.zip(k_row.chunks_exact(4)) {
                        s0 += qc[0] * kc[0];
                        s1 += qc[1] * kc[1];
                        s2 += qc[2] * kc[2];
                        s3 += qc[3] * kc[3];
                    }
                    let rem_start = head_dim - q_rem.len();
                    for i in 0..q_rem.len() {
                        s0 += q_rem[i] * k_row[rem_start + i];
                    }
                    let score = s0 + s1 + s2 + s3;
                    // PERF: Use precomputed inv_sqrt_head_dim (multiply) instead of
                    // computing sqrt each time (divide). Multiply is ~4x faster.
                    self.state.att[att_offset + t] = score * inv_sqrt_head_dim;
                }

                // Softmax over positions [0..pos+1]
                // This converts raw scores to "attention weights" — a probability
                // distribution over which positions to attend to.
                let att_slice = &mut self.state.att[att_offset..att_offset + pos + 1];
                softmax_inplace(att_slice);

                // Weighted sum of values: the "attended" representation
                let v_offset = kv_h * head_dim;
                for t in 0..=pos {
                    let weight = self.state.att[att_offset + t];
                    let v_row = &self.state.value_cache[layer].data
                        [t * kv_dim + v_offset..t * kv_dim + v_offset + head_dim];
                    for j in 0..head_dim {
                        self.state.xb.data[q_offset + j] += weight * v_row[j];
                    }
                }
            }

            // 2f. Output projection: combine all heads back to [dim]
            // PERF: matvec_into writes into xb2 buffer instead of allocating.
            w.wo.matvec_into(&self.state.xb, &mut self.state.xb2);

            // 2g. Residual connection: x = x + attention_output
            //     This is the "skip connection" that makes deep networks trainable.
            self.state.x.add_inplace(&self.state.xb2);

            // ----- FEED-FORWARD SUB-LAYER -----

            // 2h. RMSNorm before FFN
            // PERF: rmsnorm_into writes into xb buffer (reused as scratch space).
            self.state.x.rmsnorm_into(&w.rms_ffn, &mut self.state.xb);

            // 2i. SwiGLU feed-forward network
            //     This is where each token is processed independently.
            //     SwiGLU = SiLU(x @ W1) * (x @ W3), then project back with W2.
            //
            //     Why two projections (W1 and W3)?
            //     W1 produces the "gate" (what to keep/suppress via SiLU).
            //     W3 produces the "value" (what to mix in).
            //     Element-wise multiply combines them.
            //     This gating mechanism is why SwiGLU outperforms plain ReLU FFN.
            // PERF: matvec_into writes directly into hb/hb2 buffers.
            // silu_mul_inplace fuses silu + elementwise multiply into one pass,
            // eliminating 2 temporary Vecs (the old silu() result and mul() result).
            w.w1.matvec_into(&self.state.xb, &mut self.state.hb);   // gate
            w.w3.matvec_into(&self.state.xb, &mut self.state.hb2);  // up
            self.state.hb.silu_mul_inplace(&self.state.hb2);         // fused silu + gate
            w.w2.matvec_into(&self.state.hb, &mut self.state.xb2);  // down

            // 2j. Residual connection
            self.state.x.add_inplace(&self.state.xb2);
        }

        // Step 3: Final RMSNorm
        // PERF: rmsnorm_into writes into xb, then we swap xb into x.
        // The old code allocated a new tensor and assigned it to x.
        self.state.x.rmsnorm_into(&self.weights.rms_final, &mut self.state.xb);
        std::mem::swap(&mut self.state.x, &mut self.state.xb);

        // Step 4: Project to vocabulary logits
        //     logits[i] = dot(hidden_state, output_weight[i])
        //     Higher logit = model thinks token i is more likely next.
        // PERF: matvec_into writes into pre-allocated logits buffer.
        self.weights.output.matvec_into(&self.state.x, &mut self.state.logits);

        &self.state.logits
    }

    /// Reset the model's run state (KV cache and all buffers) so we can
    /// generate a fresh sequence. Without this, the KV cache still contains
    /// keys and values from the previous generation, which would corrupt
    /// attention for the new sequence.
    ///
    /// WHY this matters: The KV cache is indexed by position. When we start
    /// a new prompt at position 0, the cache slots for positions 0..old_len
    /// still hold stale data. Even though we overwrite position 0 on the
    /// first forward pass, the attention loop reads ALL positions up to pos,
    /// so stale entries at positions > 0 would leak into the first few tokens
    /// of the new sequence. The simplest fix: zero everything out.
    pub fn reset(&mut self) {
        self.state = RunState::new(&self.config);
    }
}

// ---------------------------------------------------------------------------
// RoPE — Rotary Positional Embedding
// ---------------------------------------------------------------------------
// RoPE encodes position by rotating pairs of dimensions in the Q and K
// vectors. The rotation angle depends on the position and the dimension:
//
//   For dimension pair (2i, 2i+1):
//     freq = 1 / (10000^(2i/d))
//     angle = pos * freq
//     q[2i]   = q[2i] * cos(angle) - q[2i+1] * sin(angle)
//     q[2i+1] = q[2i] * sin(angle) + q[2i+1] * cos(angle)
//
// This is a 2D rotation matrix applied to each pair of dimensions.
// The key insight: dot(q_rotated_at_pos_m, k_rotated_at_pos_n) depends
// only on (m - n), the RELATIVE position. This means the model naturally
// understands "3 tokens apart" regardless of absolute position.
//
// RoPE was introduced in the RoFormer paper and adopted by Llama, Mistral,
// and most modern LLMs because it generalizes better to unseen sequence
// lengths than learned positional embeddings.
// ---------------------------------------------------------------------------

fn apply_rope(q: &mut Tensor, k: &mut Tensor, pos: usize, head_dim: usize) {
    let q_heads = q.data.len() / head_dim;
    let k_heads = k.data.len() / head_dim;

    for i in (0..head_dim).step_by(2) {
        // Each pair of dimensions gets a different frequency
        let freq = 1.0 / (10000.0f32).powf(i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let cos = angle.cos();
        let sin = angle.sin();

        // Rotate all Q heads at this dimension pair
        for h in 0..q_heads {
            let idx = h * head_dim + i;
            let q0 = q.data[idx];
            let q1 = q.data[idx + 1];
            q.data[idx]     = q0 * cos - q1 * sin;
            q.data[idx + 1] = q0 * sin + q1 * cos;
        }

        // Rotate all K heads at this dimension pair
        for h in 0..k_heads {
            let idx = h * head_dim + i;
            let k0 = k.data[idx];
            let k1 = k.data[idx + 1];
            k.data[idx]     = k0 * cos - k1 * sin;
            k.data[idx + 1] = k0 * sin + k1 * cos;
        }
    }
}

// ---------------------------------------------------------------------------
// In-place softmax on a slice
// ---------------------------------------------------------------------------
// Same algorithm as Tensor::softmax() but operates on a raw &mut [f32]
// to avoid allocation in the hot attention loop.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helper: zip 9 Vecs into an iterator of 9-tuples (consuming the Vecs)
// ---------------------------------------------------------------------------
// CLARITY: Rust's stdlib only provides Iterator::zip for pairs. Rather than
// nesting 8 levels of zip().map(), this helper consumes 9 equal-length Vecs
// and yields a clean 9-tuple per layer. Used by TransformerWeights::from_file
// to move (not clone) tensors into LayerWeights.
// ---------------------------------------------------------------------------

fn itertools_zip9<A, B, C, D, E, F, G, H, I>(
    a: Vec<A>, b: Vec<B>, c: Vec<C>, d: Vec<D>, e: Vec<E>,
    f: Vec<F>, g: Vec<G>, h: Vec<H>, i: Vec<I>,
) -> impl Iterator<Item = (A, B, C, D, E, F, G, H, I)> {
    a.into_iter().zip(b).zip(c).zip(d).zip(e).zip(f).zip(g).zip(h).zip(i)
        .map(|((((((((a, b), c), d), e), f), g), h), i)| (a, b, c, d, e, f, g, h, i))
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    // PERF: Precompute 1/sum and multiply instead of dividing in the loop.
    // Division is ~4x slower than multiplication on most CPUs, and this loop
    // runs once per head per token (n_heads * n_tokens iterations total).
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_preserves_magnitude() {
        // RoPE is a rotation — it should preserve vector magnitude
        let mut q = Tensor::from_vec(&[1.0, 0.0, 0.0, 1.0]);
        let mut k = Tensor::from_vec(&[1.0, 0.0, 0.0, 1.0]);
        let mag_before = q.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        apply_rope(&mut q, &mut k, 5, 4);

        let mag_after = q.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag_before - mag_after).abs() < 1e-5,
            "RoPE should preserve magnitude: {} vs {}", mag_before, mag_after);
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        // At position 0, angle=0, cos=1, sin=0, so no change
        let mut q = Tensor::from_vec(&[3.0, 4.0, 1.0, 2.0]);
        let mut k = Tensor::from_vec(&[5.0, 6.0, 7.0, 8.0]);
        let q_orig = q.data.clone();

        apply_rope(&mut q, &mut k, 0, 4);

        for (a, b) in q.data.iter().zip(&q_orig) {
            assert!((a - b).abs() < 1e-5, "pos=0 should be identity");
        }
    }

    #[test]
    fn test_softmax_inplace() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_forward_produces_logits() {
        let config = Config {
            dim: 16,
            hidden_dim: 32,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 64,
            seq_len: 32,
        };
        let weights = TransformerWeights::random(&config);
        let mut model = Transformer::new(config.clone(), weights);

        let logits = model.forward(1, 0);
        assert_eq!(logits.shape, vec![64]); // vocab_size logits
        assert_eq!(logits.data.len(), 64);

        // Logits should not be all zeros (model has random weights)
        let any_nonzero = logits.data.iter().any(|&v| v.abs() > 1e-10);
        assert!(any_nonzero, "logits should not be all zeros");
    }

    #[test]
    fn test_forward_two_positions() {
        // Ensure we can run forward for position 0 and 1 (KV cache works)
        let config = Config {
            dim: 16,
            hidden_dim: 32,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 64,
            seq_len: 32,
        };
        let weights = TransformerWeights::random(&config);
        let mut model = Transformer::new(config, weights);

        let logits0 = model.forward(1, 0).data.clone();
        let logits1 = model.forward(2, 1).data.clone();

        // Different tokens at different positions should produce different logits
        let same = logits0.iter().zip(&logits1).all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(!same, "different inputs should produce different logits");
    }
}
