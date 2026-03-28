// ============================================================================
// LLM Lab — Milestone 4: Weight Loading & Real Inference
// ============================================================================
//
// Run: cargo run --bin milestone4
//
// This is the milestone where our transformer goes from "correct but useless"
// to "actually generating coherent English." The difference? TRAINED WEIGHTS.
//
// A neural network's architecture (the code in model.rs) defines what
// computations are POSSIBLE. The weights determine what the network actually
// DOES. Training adjusts millions of weights over billions of examples until
// the model learns to predict the next token well. We're loading 15 million
// parameters that were trained on children's stories (the "TinyStories" dataset).
//
// The weight file format comes from Karpathy's llama2.c project — a simple
// binary layout designed for easy loading in C. No protobuf, no HDF5, no
// safetensors — just raw f32 values in a known order.
//
// ============================================================================

use llm_lab::model::{Config, Transformer, TransformerWeights};
use llm_lab::tokenizer::Tokenizer;

use std::time::Instant;

fn main() {
    println!("+----------------------------------------------------------+");
    println!("|        LLM LAB — Milestone 4                             |");
    println!("|        Weight Loading & Real Inference                    |");
    println!("+----------------------------------------------------------+\n");

    let checkpoint_path = "data/stories15M.bin";
    let tokenizer_path = "data/tokenizer.bin";

    // -----------------------------------------------------------------------
    // Step 1: Load the model configuration from the checkpoint header
    // -----------------------------------------------------------------------
    // The first 28 bytes of the checkpoint are 7 i32 values describing the
    // architecture. This is how we know the dimensions of every weight
    // matrix before we start reading them.
    // -----------------------------------------------------------------------

    println!("--- Step 1: Loading config from checkpoint ---\n");

    let (config, shared_weights) = Config::from_file(checkpoint_path)
        .expect("Failed to read config from checkpoint");

    println!("  Model configuration (from file header):");
    println!("    dim:          {}", config.dim);
    println!("    hidden_dim:   {}", config.hidden_dim);
    println!("    n_layers:     {}", config.n_layers);
    println!("    n_heads:      {}", config.n_heads);
    println!("    n_kv_heads:   {}", config.n_kv_heads);
    println!("    vocab_size:   {}", config.vocab_size);
    println!("    seq_len:      {}", config.seq_len);
    println!("    head_dim:     {}", config.head_dim());
    println!("    weight tying: {} (embedding == output projection)", shared_weights);
    println!();

    // -----------------------------------------------------------------------
    // Step 2: Load the tokenizer
    // -----------------------------------------------------------------------
    // The tokenizer maps between human-readable text and the integer token
    // IDs that the model operates on. We need it to encode our prompt and
    // decode the model's output.
    // -----------------------------------------------------------------------

    println!("--- Step 2: Loading tokenizer ---\n");

    let tokenizer = Tokenizer::from_file(tokenizer_path, config.vocab_size)
        .expect("Failed to load tokenizer");
    println!("  Loaded {} tokens from {}", tokenizer.vocab_size(), tokenizer_path);
    println!("  Max token length: {} bytes", tokenizer.max_token_length);
    println!();

    // -----------------------------------------------------------------------
    // Step 3: Load the trained weights
    // -----------------------------------------------------------------------
    // This is the big one — ~58 MB of f32 values that encode everything the
    // model learned during training. Each weight was adjusted thousands of
    // times during training to minimize prediction error on the training data.
    //
    // Loading takes a moment because we're reading ~15 million f32 values
    // from disk and organizing them into the right tensor shapes.
    // -----------------------------------------------------------------------

    println!("--- Step 3: Loading weights ---\n");

    let t0 = Instant::now();
    let weights = TransformerWeights::from_file(checkpoint_path, &config, shared_weights)
        .expect("Failed to load weights");
    let load_time = t0.elapsed();

    // Report what we loaded
    let total_params: usize = weights.token_embedding.numel()
        + weights.layers.iter().map(|l| {
            l.rms_att.numel() + l.wq.numel() + l.wk.numel() + l.wv.numel()
            + l.wo.numel() + l.rms_ffn.numel() + l.w1.numel() + l.w2.numel()
            + l.w3.numel()
        }).sum::<usize>()
        + weights.rms_final.numel()
        + weights.output.numel();

    let total_bytes = total_params * 4; // f32 = 4 bytes each
    println!("  Loaded {} parameters ({:.1} MB) in {:.1}ms",
        total_params, total_bytes as f64 / 1e6, load_time.as_secs_f64() * 1000.0);
    println!("  Token embedding: {:?}", weights.token_embedding.shape);
    println!("  Layer 0 wq:      {:?}", weights.layers[0].wq.shape);
    println!("  Layer 0 wk:      {:?}", weights.layers[0].wk.shape);
    println!("  Layer 0 w1:      {:?}", weights.layers[0].w1.shape);
    println!("  Layer 0 w2:      {:?}", weights.layers[0].w2.shape);
    println!("  rms_final:       {:?}", weights.rms_final.shape);
    println!("  output:          {:?}", weights.output.shape);

    // Quick sanity check: trained weights should NOT be all zeros or all identical
    let emb_sample: Vec<f32> = weights.token_embedding.data[..8].to_vec();
    println!("\n  First 8 embedding values: {:?}", emb_sample);
    println!("  (non-zero, varied values confirm real trained weights)\n");

    // -----------------------------------------------------------------------
    // Step 4: Generate text!
    // -----------------------------------------------------------------------
    // This is the autoregressive generation loop:
    //
    //   1. Encode the prompt into token IDs
    //   2. Feed each prompt token through the model (building up the KV cache)
    //   3. Take the model's output logits and pick the most likely token (argmax)
    //   4. Feed that token back as input and repeat
    //
    // With greedy decoding (always picking the argmax), the output is
    // deterministic — the same prompt always produces the same story.
    // Temperature sampling or top-p would add variety, but argmax is
    // simplest and still produces coherent text.
    // -----------------------------------------------------------------------

    println!("--- Step 4: Generating text (greedy decoding) ---\n");

    let prompt = "Once upon a time";
    let num_tokens_to_generate = 100;

    println!("  Prompt: {:?}", prompt);
    println!("  Generating {} tokens with greedy (argmax) decoding...\n", num_tokens_to_generate);

    let mut model = Transformer::new(config.clone(), weights);

    // Encode the prompt
    let prompt_tokens = tokenizer.encode(prompt);
    println!("  Prompt tokens: {:?}", prompt_tokens);
    println!("  Decoded back:  {:?}\n", tokenizer.decode(&prompt_tokens));

    // The generation loop collects all output token IDs
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut next_token = prompt_tokens[0];
    all_tokens.push(next_token);

    let gen_start = Instant::now();

    // First, process all prompt tokens (the "prefill" phase).
    // During prefill we feed the known prompt tokens to build up the KV cache.
    // We don't emit output until after the prompt is fully processed.
    for pos in 0..prompt_tokens.len() {
        let logits = model.forward(prompt_tokens[pos], pos);
        next_token = logits.argmax() as u32;
        if pos + 1 < prompt_tokens.len() {
            // Still in the prompt — the next token is already known
            all_tokens.push(prompt_tokens[pos + 1]);
        } else {
            // End of prompt — now we use the model's prediction
            all_tokens.push(next_token);
        }
    }

    // Now the autoregressive generation phase: feed the model's own
    // predictions back as input, one token at a time.
    for i in 0..num_tokens_to_generate {
        let pos = prompt_tokens.len() + i;
        if pos >= config.seq_len {
            println!("  (reached max sequence length {})", config.seq_len);
            break;
        }
        let logits = model.forward(next_token, pos);
        next_token = logits.argmax() as u32;
        all_tokens.push(next_token);
    }

    let gen_time = gen_start.elapsed();

    // Decode and display the generated text
    let output_text = tokenizer.decode(&all_tokens);
    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │ GENERATED TEXT                                      │");
    println!("  ├─────────────────────────────────────────────────────┤");
    for line in output_text.lines() {
        println!("  │ {:<52}│", line);
    }
    println!("  └─────────────────────────────────────────────────────┘");
    println!();

    let total_tokens = all_tokens.len();
    let tokens_per_sec = total_tokens as f64 / gen_time.as_secs_f64();
    println!("  Generated {} tokens in {:.1}ms ({:.1} tokens/sec)",
        total_tokens, gen_time.as_secs_f64() * 1000.0, tokens_per_sec);
    println!();

    // -----------------------------------------------------------------------
    // Step 5: Try another prompt
    // -----------------------------------------------------------------------
    println!("--- Step 5: Another prompt ---\n");

    let prompt2 = "A little cat";
    generate_and_print(&config, checkpoint_path, shared_weights, &tokenizer, prompt2, 80);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!("  MILESTONE 4 COMPLETE -- KEY CONCEPTS:");
    println!();
    println!("  1. WEIGHT FORMAT: The llama2.c checkpoint is a simple binary");
    println!("     file. Header (7 x i32) defines the architecture, then");
    println!("     contiguous f32 arrays for each weight in a fixed order.");
    println!();
    println!("  2. WEIGHT TYING: The token embedding and output projection");
    println!("     can share the same matrix. This saves ~9.2M parameters");
    println!("     (vocab_size x dim = 32000 x 288) and often helps quality");
    println!("     on smaller models.");
    println!();
    println!("  3. LAYER-STRIDED LAYOUT: Per-layer weights are stored as");
    println!("     'all layers of wq, then all layers of wk, ...' rather");
    println!("     than 'all weights for layer 0, then layer 1, ...'");
    println!("     This matches PyTorch's state_dict iteration order.");
    println!();
    println!("  4. GREEDY DECODING: We always pick the highest-probability");
    println!("     token (argmax). This is deterministic but can be");
    println!("     repetitive. Temperature/top-p sampling adds variety.");
    println!();
    println!("  5. THIS IS REAL INFERENCE: The same math that runs in");
    println!("     ChatGPT, Claude, and Llama — just at a smaller scale.");
    println!("     The 15M model generates simple children's stories.");
    println!("------------------------------------------------------------");
}

/// Helper: create a fresh model, generate text from a prompt, and print it.
fn generate_and_print(
    config: &Config,
    checkpoint_path: &str,
    shared_weights: bool,
    tokenizer: &Tokenizer,
    prompt: &str,
    num_tokens: usize,
) {
    let weights = TransformerWeights::from_file(checkpoint_path, config, shared_weights)
        .expect("Failed to load weights");
    let mut model = Transformer::new(config.clone(), weights);

    let prompt_tokens = tokenizer.encode(prompt);
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut next_token = prompt_tokens[0];
    all_tokens.push(next_token);

    for pos in 0..prompt_tokens.len() {
        let logits = model.forward(prompt_tokens[pos], pos);
        next_token = logits.argmax() as u32;
        if pos + 1 < prompt_tokens.len() {
            all_tokens.push(prompt_tokens[pos + 1]);
        } else {
            all_tokens.push(next_token);
        }
    }

    for i in 0..num_tokens {
        let pos = prompt_tokens.len() + i;
        if pos >= config.seq_len { break; }
        let logits = model.forward(next_token, pos);
        next_token = logits.argmax() as u32;
        all_tokens.push(next_token);
    }

    let output_text = tokenizer.decode(&all_tokens);
    println!("  Prompt: {:?}", prompt);
    println!();
    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │ GENERATED TEXT                                      │");
    println!("  ├─────────────────────────────────────────────────────┤");
    for line in output_text.lines() {
        println!("  │ {:<52}│", line);
    }
    println!("  └─────────────────────────────────────────────────────┘");
    println!();
}
