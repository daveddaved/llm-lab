// ============================================================================
// LLM Lab — Milestone 1: Tensors & Linear Algebra
// ============================================================================
//
// This milestone builds the mathematical foundation for everything else.
// Every transformer operation — attention, projections, feed-forward layers —
// is built from these primitives: matmul, softmax, normalization.
//
// Run: cargo run
// Test: cargo test
//
// ============================================================================

use llm_lab::tensor::Tensor;

fn main() {
    println!("+----------------------------------------------------------+");
    println!("|        LLM LAB — Milestone 1                             |");
    println!("|        Tensors & Linear Algebra                          |");
    println!("+----------------------------------------------------------+\n");

    // -----------------------------------------------------------------------
    // Demo 1: Matrix Multiply — the core of everything
    // -----------------------------------------------------------------------
    println!("--- Demo 1: Matrix Multiply ---");
    println!("Why it matters: EVERY layer in a transformer is a matmul.");
    println!("Attention, projections, feed-forward — all matmuls.\n");

    // Imagine a tiny "model" with 3-dimensional hidden states and 2 tokens.
    // The weight matrix projects from 3 dims to 2 dims.
    //
    //   input [2 tokens, 3 dims] @ weights [3 dims, 2 dims] = output [2 tokens, 2 dims]
    //
    // In a real transformer:
    //   input is [seq_len, d_model] e.g. [512, 768]
    //   W_q is [d_model, d_head] e.g. [768, 64]
    //   output is [seq_len, d_head] e.g. [512, 64]

    let input = Tensor::new(
        vec![1.0, 2.0, 3.0,   // token 0's hidden state
             4.0, 5.0, 6.0],  // token 1's hidden state
        vec![2, 3],
    );
    let weights = Tensor::new(
        vec![0.1, 0.2,
             0.3, 0.4,
             0.5, 0.6],
        vec![3, 2],
    );
    let output = input.matmul(&weights);

    println!("  Input (2 tokens, 3 dims):");
    println!("    Token 0: [1.0, 2.0, 3.0]");
    println!("    Token 1: [4.0, 5.0, 6.0]");
    println!();
    println!("  Weights (3 dims -> 2 dims):");
    println!("    {}", weights);
    println!("  Output (2 tokens, 2 dims):");
    println!("    {}", output);
    println!("  Token 0: 1*0.1+2*0.3+3*0.5 = {:.1}, 1*0.2+2*0.4+3*0.6 = {:.1}",
        1.0*0.1 + 2.0*0.3 + 3.0*0.5,
        1.0*0.2 + 2.0*0.4 + 3.0*0.6);
    println!();

    // -----------------------------------------------------------------------
    // Demo 2: Softmax — turning scores into probabilities
    // -----------------------------------------------------------------------
    println!("--- Demo 2: Softmax ---");
    println!("Why it matters: attention weights must be probabilities.");
    println!("Softmax converts raw scores into 'how much to attend to each token.'\n");

    // Imagine 4 tokens. The current token computed similarity scores
    // (via Q @ K^T) against all 4 tokens. These raw scores need to
    // become a probability distribution.

    let attention_scores = Tensor::from_vec(&[2.0, 1.0, 0.1, -1.0]);
    let attention_weights = attention_scores.softmax();

    println!("  Raw attention scores:  {}", attention_scores);
    println!("  After softmax:         {}", attention_weights);
    println!("  Sum = {:.6} (should be 1.0)", attention_weights.data.iter().sum::<f32>());
    println!();
    println!("  Token 0 gets {:.1}% of attention (highest score = most relevant)",
        attention_weights.data[0] * 100.0);
    println!("  Token 3 gets {:.1}% of attention (negative score = least relevant)",
        attention_weights.data[3] * 100.0);
    println!();

    // Show what happens with extreme values
    println!("  With larger scores (sharper distribution):");
    let sharp = Tensor::from_vec(&[10.0, 1.0, 0.1, -1.0]).softmax();
    println!("    softmax([10, 1, 0.1, -1]) = {}", sharp);
    println!("    Token 0 gets {:.1}% — almost all attention on one token", sharp.data[0] * 100.0);
    println!("    This is why temperature scaling matters for generation.\n");

    // -----------------------------------------------------------------------
    // Demo 3: RMSNorm — keeping values stable
    // -----------------------------------------------------------------------
    println!("--- Demo 3: RMSNorm ---");
    println!("Why it matters: without normalization, values explode or vanish");
    println!("after flowing through 32+ transformer layers.\n");

    let hidden = Tensor::from_vec(&[3.0, -1.0, 4.0, 1.0, 5.0]);
    let norm_weights = Tensor::from_vec(&[1.0, 1.0, 1.0, 1.0, 1.0]); // identity
    let normalized = hidden.rmsnorm(&norm_weights);

    println!("  Before normalization: {}", hidden);
    println!("  After RMSNorm:        {}", normalized);

    let rms_before: f32 = (hidden.data.iter().map(|x| x * x).sum::<f32>() / 5.0).sqrt();
    let rms_after: f32 = (normalized.data.iter().map(|x| x * x).sum::<f32>() / 5.0).sqrt();
    println!();
    println!("  RMS before: {:.4}", rms_before);
    println!("  RMS after:  {:.4} (approx 1.0 -- that's the point!)", rms_after);
    println!();

    // Show what happens with very large values
    let big = Tensor::from_vec(&[100.0, 200.0, 300.0, 400.0, 500.0]);
    let big_normed = big.rmsnorm(&norm_weights);
    println!("  Big values:    {}", big);
    println!("  After RMSNorm: {}", big_normed);
    println!("  Values brought to unit scale regardless of input magnitude.\n");

    // -----------------------------------------------------------------------
    // Demo 4: SiLU activation
    // -----------------------------------------------------------------------
    println!("--- Demo 4: SiLU activation ---");
    println!("Why it matters: the feed-forward network in Llama 2 uses SwiGLU,");
    println!("which is based on SiLU. It adds nonlinearity -- without it,");
    println!("stacking layers would be pointless (matmul of matmul = matmul).\n");

    let x = Tensor::from_vec(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let activated = x.silu();
    println!("  Input:      {}", x);
    println!("  SiLU(x):    {}", activated);
    println!();
    println!("  Notice: silu(0) = 0, silu(large positive) approx x");
    println!("  Unlike ReLU, negative values aren't zeroed -- just suppressed.\n");

    // -----------------------------------------------------------------------
    // Demo 5: Putting it together — a tiny "attention" calculation
    // -----------------------------------------------------------------------
    println!("--- Demo 5: Mini attention (all pieces together) ---");
    println!("This combines matmul + scale + softmax to show how attention works.\n");

    // 3 tokens, each with a 4-dimensional hidden state
    let q = Tensor::new(
        vec![1.0, 0.0, 1.0, 0.0,   // token 0's query
             0.0, 1.0, 0.0, 1.0,   // token 1's query
             1.0, 1.0, 0.0, 0.0],  // token 2's query
        vec![3, 4],
    );
    // K^T (transposed) so we can do Q @ K^T
    let k_t = Tensor::new(
        vec![1.0, 0.0, 1.0,     // key 0 (transposed)
             0.0, 1.0, 1.0,     // key 1
             1.0, 0.0, 0.0,     // key 2
             0.0, 1.0, 0.0],    // key 3
        vec![4, 3],
    );

    // Step 1: Q @ K^T — raw attention scores
    let scores = q.matmul(&k_t);
    println!("  Step 1: Q @ K^T (raw scores, 3 tokens x 3 tokens):");
    println!("  {}", scores);

    // Step 2: Scale by 1/sqrt(d_k) — prevents scores from getting too large
    let d_k = 4.0f32;
    let scaled = scores.scale(1.0 / d_k.sqrt());
    println!("  Step 2: Scale by 1/sqrt({}) = 1/{:.2}:", d_k, d_k.sqrt());
    println!("  {}", scaled);

    // Step 3: Softmax each row — convert to probabilities
    println!("  Step 3: Softmax each row (attention weights):");
    for i in 0..3 {
        let row = scaled.row(i);
        let weights = row.softmax();
        println!("    Token {} attends to others: {}", i, weights);
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!();
    println!("------------------------------------------------------------");
    println!("  MILESTONE 1 COMPLETE -- KEY CONCEPTS:");
    println!();
    println!("  1. TENSORS are multi-dimensional arrays of floats.");
    println!("     Everything in a neural network is a tensor operation.");
    println!();
    println!("  2. MATMUL is the fundamental operation. Linear projections,");
    println!("     attention scores, feed-forward layers -- all matmuls.");
    println!("     A transformer is basically matmuls + softmax + normalization.");
    println!();
    println!("  3. SOFTMAX converts raw scores to probabilities.");
    println!("     The subtract-max trick prevents numerical overflow.");
    println!();
    println!("  4. RMSNORM keeps activations at unit scale across layers.");
    println!("     Without it, deep networks are untrainable.");
    println!();
    println!("  5. SiLU adds nonlinearity. Without activation functions,");
    println!("     multiple linear layers collapse into a single linear layer.");
    println!();
    println!("  NEXT: Milestone 2 -- Tokenizer (text <-> numbers)");
    println!("------------------------------------------------------------");
}
