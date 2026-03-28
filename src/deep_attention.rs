// ============================================================================
// Deep Dive: Attention Under the Microscope
// ============================================================================
//
// Attention is the mechanism that lets each token "look at" every other token
// in the sequence to decide what context matters. It's the single most
// important innovation in the Transformer architecture.
//
// But HOW does attention actually work in a trained model? Which tokens
// attend to which? Do different heads learn different patterns?
//
// This lesson visualizes real attention patterns from a trained model:
//   1. Attention weight matrices — who attends to whom
//   2. Causal masking — why future tokens are invisible
//   3. Multi-head diversity — heads specialize in different patterns
//   4. Attention sinks — why BOS gets disproportionate attention
//
// ============================================================================

use llm_lab::model::{Config, Transformer, TransformerWeights};
use llm_lab::tokenizer::Tokenizer;

fn main() {
    println!("================================================================");
    println!("   Deep Dive: Attention Under the Microscope");
    println!("   Visualizing Real Attention Patterns from a Trained LLM");
    println!("================================================================\n");

    // -----------------------------------------------------------------------
    // Load model and tokenizer
    // -----------------------------------------------------------------------
    let model_path = "data/stories15M.bin";
    let tokenizer_path = "data/tokenizer.bin";

    let (config, shared_weights) = Config::from_file(model_path)
        .expect("Failed to load model config");
    let weights = TransformerWeights::from_file(model_path, &config, shared_weights)
        .expect("Failed to load model weights");
    let tokenizer = Tokenizer::from_file(tokenizer_path, config.vocab_size)
        .expect("Failed to load tokenizer");

    println!("Model: dim={}, n_layers={}, n_heads={}, head_dim={}\n",
        config.dim, config.n_layers, config.n_heads, config.head_dim());

    // Build the transformer and enable attention capture
    let mut model = Transformer::new(config.clone(), weights);
    model.state.capture_attention = true;
    // Pre-allocate the attention_weights structure: [n_layers][n_heads]
    model.state.attention_weights = vec![vec![vec![]; config.n_heads]; config.n_layers];

    // ===================================================================
    // SECTION 1: Running a Forward Pass and Capturing Attention
    // ===================================================================
    //
    // We feed a short sentence through the model, processing each token
    // sequentially (just like during generation). At each position, the
    // model computes attention weights: a probability distribution over
    // all previous positions (including the current one).
    //
    // The attention weights tell us: "For this token at this position,
    // how much should I attend to each previous token?"
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 1: Attention Patterns for a Real Sentence");
    println!("================================================================\n");

    let prompt = "Once upon a time there was a little cat";
    let bos_token: u32 = 1;
    let prompt_tokens = tokenizer.encode(prompt);
    let mut all_tokens: Vec<u32> = vec![bos_token];
    all_tokens.extend_from_slice(&prompt_tokens);

    // Create labels for display
    let token_labels: Vec<String> = all_tokens.iter().enumerate().map(|(i, &tok)| {
        let s = tokenizer.token_str(tok);
        let s = s.trim();
        if i == 0 { "[BOS]".to_string() } else { s.to_string() }
    }).collect();

    println!("Input: \"{}\"\n", prompt);
    println!("Tokens ({} total, including BOS):", all_tokens.len());
    for (i, label) in token_labels.iter().enumerate() {
        println!("  pos {}: token {:>5} = '{}'", i, all_tokens[i], label);
    }
    println!();

    // Run the forward pass for all tokens to build up the KV cache
    // and capture attention weights at EACH position
    let n_tokens = all_tokens.len();

    // We want to capture the attention pattern at the LAST position
    // (when the model has seen the full sentence), so we need to run
    // the entire prefill.
    for pos in 0..n_tokens {
        model.forward(all_tokens[pos], pos);
    }

    // Now model.state.attention_weights contains the attention from the
    // LAST forward pass (position n_tokens-1, the word "cat").
    // attention_weights[layer][head] = vec of length n_tokens

    println!("Attention from the LAST token ('{}') attending to all previous tokens:",
        token_labels.last().unwrap());
    println!("(Higher value = more attention paid to that position)\n");

    // -----------------------------------------------------------------------
    // ASCII visualization of attention for layer 0
    // -----------------------------------------------------------------------
    let display_layer = 0;
    println!("--- Layer {} ---\n", display_layer);

    // Column headers (target positions)
    print!("         ");
    for label in &token_labels {
        print!("{:>7}", &label[..label.len().min(6)]);
    }
    println!("  <- attending TO these positions");
    println!("  {}", "-".repeat(9 + n_tokens * 7));

    for h in 0..config.n_heads {
        let att = &model.state.attention_weights[display_layer][h];
        print!("  Head {} [", h);
        for (t, &weight) in att.iter().enumerate() {
            // Use different characters for different attention levels
            let ch = if weight > 0.5 { '#' }
                else if weight > 0.2 { '*' }
                else if weight > 0.1 { '+' }
                else if weight > 0.05 { '.' }
                else { ' ' };
            // Also show the numeric value
            let _ = t;
            print!("{:>6}{}", format!("{:.2}", weight), ch);
        }
        println!("]");
    }

    println!("\n  Legend: '#' > 0.50, '*' > 0.20, '+' > 0.10, '.' > 0.05\n");

    // ===================================================================
    // SECTION 2: Causal Masking — Why You Can't Peek Ahead
    // ===================================================================
    //
    // In autoregressive generation, when predicting token at position t,
    // the model can ONLY attend to positions 0..t (past and current).
    // It CANNOT look at positions t+1, t+2, ... because those tokens
    // haven't been generated yet!
    //
    // This is called the CAUSAL MASK (or autoregressive mask). It's
    // implemented by simply not computing attention scores for future
    // positions — the attention loop runs from 0 to pos, not 0 to seq_len.
    //
    // Without causal masking, the model would "cheat" during training
    // by looking at future tokens, and at inference time it would be
    // confused because there ARE no future tokens to look at.
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 2: Causal Masking — The Autoregressive Constraint");
    println!("================================================================\n");

    println!("The causal mask ensures each position can only attend to earlier");
    println!("positions (and itself). Here's what the mask looks like:\n");

    let display_n = n_tokens.min(10);
    print!("          ");
    for (_i, label) in token_labels.iter().enumerate().take(display_n) {
        print!("{:>7}", &label[..label.len().min(6)]);
    }
    println!("   (Key positions)");
    println!("  {}", "-".repeat(10 + display_n * 7));

    for q_pos in 0..display_n {
        let label = &token_labels[q_pos];
        print!("  {:>6} [", &label[..label.len().min(6)]);
        for k_pos in 0..display_n {
            if k_pos <= q_pos {
                print!("   CAN ");
            } else {
                print!("   --- ");
            }
        }
        println!("]  (query pos {})", q_pos);
    }

    println!("\n  'CAN' = this position CAN attend here");
    println!("  '---' = MASKED (future position, not yet generated)\n");

    println!("  WHY THIS MATTERS:");
    println!("  - The first token (BOS) can only attend to itself (1 option)");
    println!("  - The last token can attend to ALL positions ({} options)", n_tokens);
    println!("  - This is why generation is sequential: token N needs tokens 0..N-1");
    println!("  - Bidirectional models (BERT) remove this mask for understanding tasks,");
    println!("    but they can't generate text autoregressively.");

    // ===================================================================
    // SECTION 3: Multi-Head Diversity
    // ===================================================================
    //
    // The model has multiple attention heads per layer. WHY?
    //
    // Each head independently learns what patterns to attend to:
    //   - Some heads attend to the immediately previous token (local context)
    //   - Some heads attend to the first token (BOS = global info)
    //   - Some heads attend to syntactically related tokens
    //   - Some heads form "induction heads" that copy patterns
    //
    // By having multiple heads, the model can simultaneously capture
    // multiple types of relationships between tokens.
    //
    // If we only had one head, it would have to somehow encode ALL
    // relationships into a single attention pattern — an impossible
    // compression for complex language.
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 3: Multi-Head Diversity — Different Heads, Different Jobs");
    println!("================================================================\n");

    // Run forward pass at each position and capture attention, then analyze patterns
    // Reset model to get clean state
    model.reset();
    model.state.capture_attention = true;
    model.state.attention_weights = vec![vec![vec![]; config.n_heads]; config.n_layers];

    // Process all tokens
    for pos in 0..n_tokens {
        model.forward(all_tokens[pos], pos);
    }

    println!("Analyzing attention patterns across all layers and heads...\n");
    println!("For the last token ('{}'), where does each head focus?\n",
        token_labels.last().unwrap());

    for layer in 0..config.n_layers {
        println!("  Layer {}:", layer);
        for h in 0..config.n_heads {
            let att = &model.state.attention_weights[layer][h];

            // Find the position with the most attention
            let (max_pos, max_weight) = att.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap_or((0, &0.0));

            // Compute entropy of the attention distribution
            // High entropy = spread out (attending to many tokens)
            // Low entropy = focused (attending to few tokens)
            let entropy: f32 = att.iter()
                .filter(|&&w| w > 0.0)
                .map(|&w| -w * w.ln())
                .sum();
            let max_entropy = (att.len() as f32).ln();
            // REVIEW: Guard against division by zero when att has only 1 element
            // (max_entropy=ln(1)=0). In that case focus is trivially 1.0 (fully focused).
            let focus = if max_entropy > 0.0 { 1.0 - entropy / max_entropy } else { 1.0 };

            // Check if this head attends mostly to BOS
            let bos_weight = att[0];

            // Check if this head attends to the previous token
            let prev_weight = if att.len() > 1 { att[att.len() - 2] } else { 0.0 };

            let pattern = if bos_weight > 0.4 {
                "BOS-focused (global)"
            } else if prev_weight > 0.4 {
                "Previous-token (local)"
            } else if focus > 0.5 {
                "Position-specific"
            } else {
                "Distributed"
            };

            println!("    Head {}: max_attn={:.2} at pos {} ('{}'), BOS={:.2}, prev={:.2}, focus={:.2} => {}",
                h, max_weight, max_pos,
                &token_labels[max_pos][..token_labels[max_pos].len().min(6)],
                bos_weight, prev_weight, focus, pattern);
        }
        println!();
    }

    // ===================================================================
    // SECTION 4: The Attention Sink Phenomenon
    // ===================================================================
    //
    // A surprising finding from the "Efficient Streaming Language Models
    // with Attention Sinks" paper (Xiao et al., 2023):
    //
    // The first few tokens (especially BOS) receive disproportionately
    // high attention weights, REGARDLESS of their semantic content.
    // This happens because:
    //
    //   1. During training, softmax must distribute 100% of attention
    //      somewhere. When no other position is particularly relevant,
    //      the model learns to "dump" attention on BOS as a default.
    //
    //   2. BOS appears in EVERY training example, so it becomes a
    //      "universal context token" — a safe default target.
    //
    //   3. The attention mechanism has no "attend to nothing" option.
    //      Softmax always outputs a valid probability distribution.
    //      BOS serves as a "trash can" for unused attention mass.
    //
    // This has practical implications: if you remove BOS from the KV
    // cache during streaming (to save memory), the model breaks. You
    // must always keep the first few "sink" tokens.
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 4: The Attention Sink — BOS Gets All the Attention");
    println!("================================================================\n");

    // Compute average attention to BOS across all layers and heads
    let mut total_bos_attention = 0.0f32;
    let mut total_heads = 0;

    println!("Attention weight assigned to BOS (position 0) by each head:\n");
    println!("  (Values > 0.3 are highlighted — BOS is getting lots of attention)\n");

    print!("  {:>12}", "");
    for h in 0..config.n_heads {
        print!("  Head {}", h);
    }
    println!("   Avg");
    println!("  {}", "-".repeat(12 + config.n_heads * 8 + 6));

    for layer in 0..config.n_layers {
        print!("  Layer {:>4} [", layer);
        let mut layer_sum = 0.0f32;
        for h in 0..config.n_heads {
            let bos_weight = model.state.attention_weights[layer][h][0];
            total_bos_attention += bos_weight;
            total_heads += 1;
            layer_sum += bos_weight;

            let marker = if bos_weight > 0.3 { "!" } else { " " };
            print!(" {:.3}{}", bos_weight, marker);
        }
        println!("]  {:.3}", layer_sum / config.n_heads as f32);
    }

    let avg_bos = total_bos_attention / total_heads as f32;
    println!("\n  Average BOS attention across all heads: {:.4}", avg_bos);

    // What would uniform attention be?
    let uniform = 1.0 / n_tokens as f32;
    println!("  Uniform attention (if no preference): {:.4}", uniform);
    println!("  BOS gets {:.1}x more attention than uniform!", avg_bos / uniform);

    println!("\n  ATTENTION SINK EXPLAINED:");
    println!("  ========================");
    println!("  The BOS token acts as an 'attention sink' — a dumping ground for");
    println!("  attention probability mass when no other token is relevant.");
    println!("  ");
    println!("  This happens because softmax MUST produce a valid probability");
    println!("  distribution (sums to 1.0). When the model has 'nothing useful'");
    println!("  to attend to, it concentrates weight on BOS rather than spreading");
    println!("  it thinly over irrelevant tokens.");
    println!("  ");
    println!("  Practical consequence: in streaming LLM inference, you cannot");
    println!("  evict the first few tokens from the KV cache. Even if the");
    println!("  conversation is 100k tokens long, you must keep the 'sink tokens'");
    println!("  or the model's attention distribution collapses.");

    // ===================================================================
    // SECTION 5: Attention Across Layers — How Understanding Deepens
    // ===================================================================
    //
    // Early layers tend to have more "mechanical" attention patterns:
    //   - Attending to adjacent tokens
    //   - Attending to punctuation / structure
    //
    // Later layers tend to have more "semantic" patterns:
    //   - Attending to meaning-related tokens
    //   - More distributed attention
    //
    // This reflects the model building representations bottom-up:
    //   Early layers: "what tokens are nearby?"
    //   Middle layers: "what syntactic structure is here?"
    //   Late layers: "what is the meaning and what should come next?"
    // ===================================================================

    println!("\n================================================================");
    println!("  SECTION 5: How Attention Evolves Across Layers");
    println!("================================================================\n");

    println!("Average attention entropy per layer (higher = more spread out):\n");

    for layer in 0..config.n_layers {
        let mut layer_entropy_sum = 0.0f32;
        for h in 0..config.n_heads {
            let att = &model.state.attention_weights[layer][h];
            let entropy: f32 = att.iter()
                .filter(|&&w| w > 0.0)
                .map(|&w| -w * w.ln())
                .sum();
            layer_entropy_sum += entropy;
        }
        let avg_entropy = layer_entropy_sum / config.n_heads as f32;
        let max_possible = (n_tokens as f32).ln();
        let bar_len = (avg_entropy / max_possible * 40.0) as usize;

        println!("  Layer {}: [{:40}] entropy={:.3} / {:.3}",
            layer,
            "#".repeat(bar_len.min(40)),
            avg_entropy,
            max_possible);
    }

    println!("\n  INTERPRETATION:");
    println!("  - Low entropy = focused attention (attending to specific positions)");
    println!("  - High entropy = distributed attention (attending broadly)");
    println!("  - Early layers often have lower entropy (local/structural patterns)");
    println!("  - Later layers may have higher entropy (semantic aggregation)");
    println!("  - But patterns vary — some heads in any layer may be very focused");

    println!("\n================================================================");
    println!("  END: Attention Under the Microscope");
    println!("================================================================");
}
