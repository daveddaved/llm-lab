# LLM Lab

A from-scratch LLM inference engine in Rust, built as a series of progressive milestones. Each milestone adds a new layer of the inference stack, teaching both Rust and transformer internals simultaneously.

By the end, you have a working inference engine that loads real trained weights and generates coherent English text at ~580 tokens/sec — with zero external dependencies.

## Quick Start

```bash
# Download model and tokenizer
./download-models.sh

# Run any milestone
cargo run --bin milestone1          # tensors demo
cargo run --bin milestone4          # first real text generation
cargo run --release --bin milestone6  # performance benchmark (~580 tok/s)

# Run tests
cargo test
```

## Milestones

### Milestone 1: Tensors & Linear Algebra
`cargo run --bin milestone1`

Builds the mathematical foundation. Implements a `Tensor` struct with matmul, softmax, RMSNorm, and SiLU — the primitives every transformer layer is built from. Demos show how these ops combine into a mini attention calculation.

**Files:** `src/tensor.rs`, `src/main.rs`

### Milestone 2: Tokenizer (BPE)
`cargo run --bin milestone2`

Implements Byte Pair Encoding from scratch. Loads a real 32K-token vocabulary from the tinystories model and demonstrates encoding/decoding. Shows why tokenization matters for context windows and compute costs.

**Files:** `src/tokenizer.rs`, `src/milestone2.rs`
**Requires:** `data/tokenizer.bin`

### Milestone 3: Transformer Forward Pass
`cargo run --bin milestone3`

The core architecture. Implements the full Llama 2 transformer: token embeddings, RoPE positional encoding, multi-head self-attention, SwiGLU feed-forward network, RMSNorm, residual connections, and KV cache. Runs with random weights to demonstrate the data flow.

**Files:** `src/model.rs`, `src/milestone3.rs`

### Milestone 4: Weight Loading & Real Inference
`cargo run --bin milestone4`

Loads trained weights from the tinystories 15M parameter model. The same forward pass from Milestone 3 now generates coherent children's stories. Handles the llama2.c binary format and weight tying.

**Files:** `src/model.rs` (extended), `src/milestone4.rs`
**Requires:** `data/stories15M.bin`, `data/tokenizer.bin`

### Milestone 5: Sampling Strategies
`cargo run --bin milestone5`

Replaces greedy decoding with configurable sampling. Implements temperature scaling, top-k filtering, top-p (nucleus) sampling, and repetition penalty. Demos compare how each strategy affects output quality and creativity.

**Files:** `src/sampler.rs`, `src/generate.rs`, `src/milestone5.rs`
**Requires:** `data/stories15M.bin`, `data/tokenizer.bin`

### Milestone 6: Performance
`cargo run --release --bin milestone6`

Optimizes the inference engine for speed. Adds 4-accumulator matvec for instruction-level parallelism, in-place operations to eliminate allocations, and release profile tuning (LTO, codegen-units=1). Includes memory bandwidth analysis and cache hierarchy discussion.

**Files:** `src/tensor.rs` (optimized), `src/milestone6.rs`
**Requires:** `data/stories15M.bin`, `data/tokenizer.bin`

## Architecture

```
src/
├── tensor.rs       # Tensor struct, matmul, softmax, RMSNorm, SiLU
├── tokenizer.rs    # BPE tokenizer (encode/decode)
├── model.rs        # Llama 2 transformer (config, weights, forward pass, RoPE)
├── sampler.rs      # Temperature, top-k, top-p, repetition penalty
├── generate.rs     # Autoregressive generation loop
├── lib.rs          # Library root
├── main.rs         # Milestone 1 binary
├── milestone2.rs   # Milestone 2 binary
├── milestone3.rs   # Milestone 3 binary
├── milestone4.rs   # Milestone 4 binary
├── milestone5.rs   # Milestone 5 binary
└── milestone6.rs   # Milestone 6 binary
```

Each milestone binary is self-contained with educational output explaining what's happening at each step.

## Model

Uses the [tinystories 15M](https://huggingface.co/karpathy/tinyllamas) model from Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project:

- 6 transformer layers, 6 attention heads
- 288-dimensional hidden state, 768-dimensional FFN
- 32K vocabulary, 256 max sequence length
- ~24M parameters (58 MB in f32)

## Requirements

- Rust (edition 2024)
- No external dependencies
- ~60 MB disk space for model files
