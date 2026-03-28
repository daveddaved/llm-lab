// ============================================================================
// LLM Lab — Milestone 3: Transformer Forward Pass
// ============================================================================
//
// Run: cargo run --bin milestone3
//
// This milestone demonstrates the full transformer architecture using
// random weights. The model won't produce meaningful text (that requires
// trained weights from Milestone 4), but it shows:
//
//   - The complete data flow through the transformer
//   - How token embeddings, attention, FFN, and output projection work
//   - How the KV cache enables efficient autoregressive generation
//   - The shape of data at every stage
//
// ============================================================================

use llm_lab::model::{Config, Transformer, TransformerWeights};


fn main() {
    println!("+----------------------------------------------------------+");
    println!("|        LLM LAB — Milestone 3                             |");
    println!("|        Transformer Forward Pass                          |");
    println!("+----------------------------------------------------------+\n");

    // -----------------------------------------------------------------------
    // Demo 1: Architecture overview
    // -----------------------------------------------------------------------
    println!("--- Demo 1: Model architecture ---\n");

    let config = Config::stories_15m();
    println!("  Tinystories 15M configuration:");
    println!("    dim (hidden size):    {}", config.dim);
    println!("    hidden_dim (FFN):     {}", config.hidden_dim);
    println!("    n_layers:             {}", config.n_layers);
    println!("    n_heads:              {}", config.n_heads);
    println!("    n_kv_heads:           {}", config.n_kv_heads);
    println!("    head_dim:             {}", config.head_dim());
    println!("    vocab_size:           {}", config.vocab_size);
    println!("    seq_len:              {}", config.seq_len);
    println!();

    // Count parameters
    let emb_params = config.vocab_size * config.dim;
    let kv_dim = config.n_kv_heads * config.head_dim();
    let att_params_per_layer =
        config.dim * config.dim   // wq
        + kv_dim * config.dim     // wk
        + kv_dim * config.dim     // wv
        + config.dim * config.dim // wo
        + config.dim;             // rms_att
    let ffn_params_per_layer =
        config.hidden_dim * config.dim   // w1
        + config.dim * config.hidden_dim // w2
        + config.hidden_dim * config.dim // w3
        + config.dim;                    // rms_ffn
    let layer_params = att_params_per_layer + ffn_params_per_layer;
    let output_params = config.vocab_size * config.dim + config.dim; // output + rms_final
    let total = emb_params + config.n_layers * layer_params + output_params;

    println!("  Parameter count:");
    println!("    Token embeddings:     {:>10} ({:.1}M)", emb_params, emb_params as f64 / 1e6);
    println!("    Per layer (attention): {:>9}", att_params_per_layer);
    println!("    Per layer (FFN):       {:>9}", ffn_params_per_layer);
    println!("    Per layer total:       {:>9}", layer_params);
    println!("    All {} layers:        {:>10} ({:.1}M)",
        config.n_layers, config.n_layers * layer_params,
        (config.n_layers * layer_params) as f64 / 1e6);
    println!("    Output projection:    {:>10} ({:.1}M)", output_params, output_params as f64 / 1e6);
    println!("    TOTAL:                {:>10} ({:.1}M)", total, total as f64 / 1e6);
    println!();

    // -----------------------------------------------------------------------
    // Demo 2: Forward pass with a small model
    // -----------------------------------------------------------------------
    println!("--- Demo 2: Forward pass (random weights) ---\n");
    println!("  Using a tiny model to show the data flow clearly.");
    println!("  With random weights, the output is noise — but the");
    println!("  computation is identical to a trained model.\n");

    // Use a tiny config so the demo output is readable
    let tiny = Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 32000,
        seq_len: 64,
    };
    let weights = TransformerWeights::random(&tiny);
    let mut model = Transformer::new(tiny.clone(), weights);

    println!("  Tiny model: dim={}, {} layers, {} heads, head_dim={}\n",
        tiny.dim, tiny.n_layers, tiny.n_heads, tiny.head_dim());

    // Process a few tokens
    let tokens = [1u32, 10994, 3186]; // <s>, "Hello", " world"
    let token_names = ["<s>", "Hello", " world"];

    for (pos, (&tok, &name)) in tokens.iter().zip(&token_names).enumerate() {
        println!("  Position {}: token {} ({:?})", pos, tok, name);

        let logits = model.forward(tok, pos);

        // Show logit statistics
        let min = logits.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = logits.data.iter().sum::<f32>() / logits.data.len() as f32;
        let top_token = logits.argmax();

        println!("    Output: {} logits (one per vocab token)", logits.data.len());
        println!("    Logit stats: min={:.3}, max={:.3}, mean={:.3}", min, max, mean);
        println!("    Top prediction: token {} (with random weights, this is meaningless)", top_token);
        println!();
    }

    // -----------------------------------------------------------------------
    // Demo 3: Data flow through one transformer block
    // -----------------------------------------------------------------------
    println!("--- Demo 3: Inside a transformer block ---\n");
    println!("  Here's what happens to a single token's vector as it");
    println!("  flows through one transformer block:\n");

    println!("  ┌─────────────────────────────────────────────────┐");
    println!("  │  input x: [dim={}]                              │", tiny.dim);
    println!("  │    │                                            │");
    println!("  │    ├──► RMSNorm ──► Q,K,V projections           │");
    println!("  │    │      Q: [dim={}]  (={} heads x {} dims)   │",
        tiny.dim, tiny.n_heads, tiny.head_dim());
    println!("  │    │      K: [kv_dim={}]                        │",
        tiny.n_kv_heads * tiny.head_dim());
    println!("  │    │      V: [kv_dim={}]                        │",
        tiny.n_kv_heads * tiny.head_dim());
    println!("  │    │                                            │");
    println!("  │    │      Apply RoPE to Q,K (position encoding) │");
    println!("  │    │      Store K,V in cache                    │");
    println!("  │    │                                            │");
    println!("  │    │      Per head: score = Q·K / sqrt({})     │", tiny.head_dim());
    println!("  │    │                weights = softmax(scores)   │");
    println!("  │    │                out_h = weights · V         │");
    println!("  │    │                                            │");
    println!("  │    │      Concat heads ──► output proj ──► [{}]│", tiny.dim);
    println!("  │    │                                     │      │");
    println!("  │    └──────── + (residual) ◄──────────────┘      │");
    println!("  │    │                                            │");
    println!("  │    ├──► RMSNorm ──► SwiGLU FFN                  │");
    println!("  │    │      gate = SiLU(x @ W1): [{}]           │", tiny.hidden_dim);
    println!("  │    │      up   = x @ W3:       [{}]           │", tiny.hidden_dim);
    println!("  │    │      down = (gate * up) @ W2: [{}]        │", tiny.dim);
    println!("  │    │                              │             │");
    println!("  │    └──────── + (residual) ◄───────┘             │");
    println!("  │    │                                            │");
    println!("  │  output x: [dim={}]                             │", tiny.dim);
    println!("  └─────────────────────────────────────────────────┘");
    println!();

    // -----------------------------------------------------------------------
    // Demo 4: KV cache in action
    // -----------------------------------------------------------------------
    println!("--- Demo 4: KV cache (why it matters) ---\n");

    println!("  Without KV cache (naive approach):");
    println!("    Token 1: compute attention over [1]         = 1 dot product");
    println!("    Token 2: RECOMPUTE over [1,2]               = 2 dot products");
    println!("    Token 3: RECOMPUTE over [1,2,3]             = 3 dot products");
    println!("    Token N: RECOMPUTE over [1..N]              = N dot products");
    println!("    Total for N tokens: N*(N+1)/2               = O(N^2) dot products");
    println!();
    println!("  With KV cache (what we do):");
    println!("    Token 1: compute Q,K,V. Store K,V. Attend over [1]  = 1 dot product");
    println!("    Token 2: compute Q,K,V. Store K,V. Attend over [1,2] = 2 dot products");
    println!("    Token N: compute Q,K,V. Store K,V. Attend over [1..N] = N dot products");
    println!("    Total for N tokens: N*(N+1)/2               = O(N^2) dot products");
    println!();
    println!("  Wait, same complexity? Yes, but the KEY difference:");
    println!("    - Without cache: recompute K,V for ALL past tokens each step");
    println!("      (N matvecs per token per layer — very expensive)");
    println!("    - With cache: only compute K,V for the NEW token");
    println!("      (1 matvec per token per layer — much cheaper)");
    println!();

    let kv_cache_bytes = 2 * tiny.n_layers * tiny.seq_len
        * tiny.n_kv_heads * tiny.head_dim() * 4; // f32 = 4 bytes
    println!("  KV cache size for our tiny model: {} bytes ({:.1} KB)",
        kv_cache_bytes, kv_cache_bytes as f64 / 1024.0);

    let real_config = Config::stories_15m();
    let real_kv_bytes = 2 * real_config.n_layers * real_config.seq_len
        * real_config.n_kv_heads * real_config.head_dim() * 4;
    println!("  KV cache for stories-15M:         {} bytes ({:.1} KB)",
        real_kv_bytes, real_kv_bytes as f64 / 1024.0);

    // For reference, a Llama 2 70B model
    let llama70b_kv = 2u64 * 80 * 4096 * 8 * 128 * 4; // 80 layers, 8 KV heads, dim 128
    println!("  KV cache for Llama 2 70B (4K ctx): {} bytes ({:.1} GB)",
        llama70b_kv, llama70b_kv as f64 / 1e9);

    // -----------------------------------------------------------------------
    // Demo 5: Greedy generation loop (with random weights)
    // -----------------------------------------------------------------------
    println!("\n--- Demo 5: Generation loop (random weights) ---\n");
    println!("  This is the autoregressive loop: feed the predicted token");
    println!("  back as input for the next position. With random weights");
    println!("  the output is garbage, but the MECHANISM is correct.\n");

    // Reset model
    let weights2 = TransformerWeights::random(&tiny);
    let mut model2 = Transformer::new(tiny.clone(), weights2);

    let prompt_token = 1u32; // BOS token
    let mut next_token = prompt_token;
    let mut generated = vec![next_token];

    print!("  Generated token IDs: [{}",  next_token);
    for pos in 0..10 {
        let logits = model2.forward(next_token, pos);
        next_token = logits.argmax() as u32;
        generated.push(next_token);
        print!(", {}", next_token);
    }
    println!("]");
    println!("  ({} tokens generated via greedy argmax)\n", generated.len() - 1);
    println!("  With trained weights (Milestone 4), this would produce");
    println!("  coherent English text like children's stories.\n");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!("  MILESTONE 3 COMPLETE -- KEY CONCEPTS:");
    println!();
    println!("  1. A TRANSFORMER BLOCK has two sub-layers:");
    println!("     - Self-attention (tokens communicate with each other)");
    println!("     - Feed-forward network (each token processed alone)");
    println!("     Each sub-layer has a RMSNorm and residual connection.");
    println!();
    println!("  2. SELF-ATTENTION computes Q, K, V projections, then:");
    println!("     score = Q·K/sqrt(d), weights = softmax(scores),");
    println!("     output = weights·V. This is how tokens 'look at'");
    println!("     each other to build context-aware representations.");
    println!();
    println!("  3. MULTI-HEAD attention splits the vector into heads,");
    println!("     each attending independently. This lets the model");
    println!("     attend to different patterns simultaneously.");
    println!();
    println!("  4. RoPE encodes position by rotating Q,K vectors.");
    println!("     The rotation angle depends on position, so the model");
    println!("     learns relative position from the dot products.");
    println!();
    println!("  5. SwiGLU FFN: gate = SiLU(x@W1), up = x@W3,");
    println!("     output = (gate * up) @ W2. The gating mechanism");
    println!("     lets the model selectively activate dimensions.");
    println!();
    println!("  6. KV CACHE stores keys and values from previous tokens");
    println!("     so we don't recompute them. Essential for efficient");
    println!("     autoregressive generation.");
    println!();
    println!("  NEXT: Milestone 4 -- Load real trained weights");
    println!("------------------------------------------------------------");
}
