// ============================================================================
// Deep Dive: How Text Really Gets Generated
// ============================================================================
//
// When an LLM generates text, it's doing something deceptively simple:
// at each step, it predicts a probability distribution over the ENTIRE
// vocabulary, then picks one token from that distribution.
//
// But the details matter enormously:
//   - How confident is the model at each step?
//   - What tokens are "almost" chosen? (The road not taken)
//   - Can we do better than greedy, one-token-at-a-time generation?
//   - How can we measure whether text is "surprising" to the model?
//
// This lesson explores:
//   1. Token probability distributions — seeing the model's confidence
//   2. Beam search — keeping multiple candidates alive
//   3. Speculative decoding — the concept (with math, not implementation)
//   4. Perplexity — measuring how predictable text is
//
// ============================================================================

use llm_lab::model::{Config, Transformer, TransformerWeights};
use llm_lab::tokenizer::Tokenizer;
use llm_lab::sampler::SamplerConfig;

use std::io::{self, Write};

fn main() {
    println!("================================================================");
    println!("   Deep Dive: How Text Really Gets Generated");
    println!("   Probability Distributions, Beam Search, and Perplexity");
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

    println!("Model loaded: dim={}, vocab_size={}\n", config.dim, config.vocab_size);

    // ===================================================================
    // SECTION 1: Token Probability Distributions
    // ===================================================================
    //
    // At each generation step, the model outputs a vector of LOGITS:
    // one number per vocabulary token (32,000 numbers!). After softmax,
    // these become probabilities.
    //
    // The shape of this distribution tells us a lot:
    //   - PEAKED distribution: model is confident (one token dominates)
    //   - FLAT distribution: model is uncertain (many tokens are plausible)
    //
    // Let's see the top-20 candidates at each step of generating from
    // a prompt. This reveals the "decision landscape" the model navigates.
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 1: What the Model Sees at Each Generation Step");
    println!("================================================================\n");

    let prompt = "Once upon a time";
    let bos_token: u32 = 1;
    let prompt_tokens = tokenizer.encode(prompt);
    let mut all_tokens: Vec<u32> = vec![bos_token];
    all_tokens.extend_from_slice(&prompt_tokens);

    println!("Prompt: \"{}\"\n", prompt);
    println!("Generating 8 tokens with TOP-20 candidates at each step:\n");

    let mut model = Transformer::new(config.clone(), weights);

    // REVIEW: Prefill prompt tokens and capture the logits from the LAST prefill
    // step. The original code discarded these logits and redundantly re-ran
    // forward() for the same token/position in the generation loop, wasting
    // one full forward pass (~120 matvec calls for 6 layers).
    let mut last_logits_data: Vec<f32> = Vec::new();
    for pos in 0..all_tokens.len() {
        let logits = model.forward(all_tokens[pos], pos);
        if pos == all_tokens.len() - 1 {
            last_logits_data = logits.data.clone();
        }
    }

    // Generate tokens and show the probability distribution at each step
    let n_gen = 8;
    let mut generated_text = String::new();
    // REVIEW: Track whether this is the first generation step so we can reuse
    // the prefill logits instead of running an extra forward pass.
    let mut first_step = true;

    for step in 0..n_gen {
        let pos = all_tokens.len() - 1;
        if pos >= config.seq_len { break; }

        // REVIEW: On the first step, reuse the logits already computed during
        // prefill. On subsequent steps, run a new forward pass.
        let logits = if first_step {
            first_step = false;
            llm_lab::tensor::Tensor::new(last_logits_data.clone(), vec![config.vocab_size])
        } else {
            let l = model.forward(*all_tokens.last().unwrap(), pos);
            llm_lab::tensor::Tensor::new(l.data.clone(), vec![config.vocab_size])
        };

        // Apply softmax to get probabilities
        let probs = logits.softmax();

        // Sort by probability (descending) and take top 20
        let mut indexed: Vec<(usize, f32)> = probs.data.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // The greedy choice (argmax)
        let chosen_id = indexed[0].0 as u32;
        let chosen_str = tokenizer.token_str(chosen_id);

        println!("--- Step {} (generating after: \"{}{}\") ---",
            step + 1, prompt, generated_text);
        println!("    Top-20 candidates:\n");

        // Compute cumulative probability to show how much mass the top-k cover
        let mut cumulative = 0.0f32;
        for (rank, &(tok_id, prob)) in indexed.iter().take(20).enumerate() {
            cumulative += prob;
            let tok_str = tokenizer.token_str(tok_id as u32);
            let tok_display = tok_str.replace('\n', "\\n");

            // Visual bar proportional to probability
            let bar_len = (prob * 100.0) as usize;
            let bar: String = "#".repeat(bar_len.min(40));

            let marker = if rank == 0 { " <- CHOSEN (greedy)" } else { "" };
            println!("    {:>2}. {:>6.2}% [{:40}] '{}' (token {}){}",
                rank + 1, prob * 100.0, bar, tok_display.trim(), tok_id, marker);
        }

        println!("\n    Top-20 covers {:.1}% of total probability mass.", cumulative * 100.0);

        // Compute entropy of the full distribution
        let entropy: f32 = probs.data.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let max_entropy = (config.vocab_size as f32).ln();
        println!("    Entropy: {:.3} nats (max possible: {:.3})",
            entropy, max_entropy);
        println!("    Confidence: {:.1}% (1 - entropy/max_entropy)\n",
            (1.0 - entropy / max_entropy) * 100.0);

        generated_text.push_str(chosen_str);
        all_tokens.push(chosen_id);
    }

    println!("  GENERATED TEXT (greedy): \"{}{}\"\n", prompt, generated_text);

    // ===================================================================
    // SECTION 2: Beam Search — Exploring Multiple Paths
    // ===================================================================
    //
    // Greedy decoding picks the single best token at each step. But the
    // locally best choice isn't always globally best!
    //
    // Example: "The cat sat on the ___"
    //   Greedy step 1: "mat" (prob 0.3)
    //   But "warm" (prob 0.25) might lead to "warm blanket" (total prob 0.15)
    //   while "mat" leads to "mat." (total prob 0.09)
    //
    // BEAM SEARCH keeps the top-B candidates ("beams") at each step,
    // expanding all of them and keeping the best B overall. This explores
    // more of the search space without the exponential cost of exhaustive
    // search.
    //
    // beam_width=1 is greedy. beam_width=5 is typical for machine translation.
    // Very large beams (>20) often HURT quality due to the "length penalty"
    // problem: shorter sequences tend to have higher total probability.
    //
    // NOTE: LLM chatbots typically DON'T use beam search — they use
    // sampling (temperature + top-p) instead, because beam search produces
    // repetitive, "safe" text. Beam search is more common in translation
    // and summarization where there's one "right" answer.
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 2: Beam Search — Keeping Multiple Candidates Alive");
    println!("================================================================\n");

    let beam_prompt = "Once upon a time";
    let beam_widths = [1, 2, 3, 4];
    let beam_length = 12;

    for &beam_width in &beam_widths {
        // Reset model for each beam width
        let (config2, shared2) = Config::from_file(model_path).unwrap();
        let weights2 = TransformerWeights::from_file(model_path, &config2, shared2).unwrap();

        let result = beam_search(
            &config2, weights2, &tokenizer,
            beam_prompt, beam_width, beam_length,
        );

        println!("  beam_width={}: \"{}\"", beam_width, result.0);
        println!("              log_prob={:.4}, avg_log_prob={:.4}\n",
            result.1, result.1 / result.2 as f32);
    }

    println!("  KEY INSIGHT: With beam_width=1 (greedy), you get whatever token");
    println!("  is most likely at each step. With larger beams, the model can");
    println!("  'reconsider' earlier choices if a different path turns out better.");
    println!("  But notice: more beams don't always mean 'better' text — they");
    println!("  mean higher-probability text, which can be more generic/boring.");

    // ===================================================================
    // SECTION 3: Speculative Decoding — Making Generation Faster
    // ===================================================================
    //
    // LLM inference is SLOW because it's sequential: you must generate
    // token N before you can generate token N+1. Each token requires a
    // full forward pass through the model.
    //
    // SPECULATIVE DECODING is a clever trick to speed this up:
    //
    //   1. Use a SMALL, FAST "draft" model to generate K tokens quickly
    //   2. Feed ALL K tokens through the LARGE, ACCURATE model in one pass
    //   3. The large model verifies each draft token:
    //      - If p_large(token) >= p_draft(token): ACCEPT (free speedup!)
    //      - Otherwise: accept with probability p_large/p_draft (rejection sampling)
    //   4. Generation continues from the first rejected position
    //
    // WHY THIS WORKS:
    //   - The draft model is often RIGHT (especially for common continuations)
    //   - When it's right, we skip K-1 forward passes of the large model
    //   - When it's wrong, we're no worse off than normal generation
    //   - The math guarantees we get EXACTLY the same distribution as the
    //     large model alone (rejection sampling preserves the distribution)
    //
    // SPEED:
    //   - If the draft model accepts 70% of tokens and K=5:
    //     Expected tokens per large-model forward pass ≈ 1/(1-0.7) ≈ 3.3
    //     That's a 3.3x speedup with IDENTICAL quality!
    //
    // This technique is used in production at Google (PaLM), Meta (Llama),
    // and others. It's one of the most impactful inference optimizations.
    //
    // We won't implement it here (we'd need two models), but let's
    // demonstrate the MATH of why it works.
    // ===================================================================

    println!("\n================================================================");
    println!("  SECTION 3: Speculative Decoding — The Math of Free Speedup");
    println!("================================================================\n");

    println!("  CONCEPT: Use a small 'draft' model to guess multiple tokens,");
    println!("  then verify them all at once with the large model.\n");

    println!("  Suppose the large model's next-token distribution is:");
    println!("    p_large = {{ 'the': 0.40, 'a': 0.25, 'one': 0.15, ... }}\n");
    println!("  And the draft model guesses:");
    println!("    p_draft = {{ 'the': 0.35, 'a': 0.30, 'one': 0.10, ... }}\n");

    println!("  ACCEPTANCE RULE (rejection sampling):");
    println!("  For each draft token t:");
    println!("    accept_prob = min(1, p_large(t) / p_draft(t))\n");

    // Concrete numerical example
    let examples = [
        ("the",   0.40f32, 0.35f32),
        ("a",     0.25,    0.30),
        ("one",   0.15,    0.10),
        ("some",  0.08,    0.12),
    ];

    println!("  {:>8} {:>10} {:>10} {:>12} {:>10}",
        "Token", "p_large", "p_draft", "accept_prob", "Verdict");
    println!("  {}", "-".repeat(54));

    for (token, p_large, p_draft) in &examples {
        let accept_prob = (p_large / p_draft).min(1.0);
        let verdict = if accept_prob >= 1.0 { "ALWAYS accept" } else { "Maybe reject" };
        println!("  {:>8} {:>10.3} {:>10.3} {:>12.3} {:>14}",
            token, p_large, p_draft, accept_prob, verdict);
    }

    println!("\n  Notice: 'the' has higher p_large than p_draft, so it's ALWAYS");
    println!("  accepted. 'a' has lower p_large than p_draft, so it's accepted");
    println!("  with probability 0.25/0.30 = 0.833 — rejected 17% of the time.\n");

    println!("  THE GUARANTEE: After rejection sampling, the accepted tokens");
    println!("  follow EXACTLY the large model's distribution. No quality loss!");
    println!("  This is a theorem from probability theory, not a heuristic.\n");

    println!("  EXPECTED SPEEDUP:");
    println!("  If the draft model has acceptance rate alpha and generates K tokens:");
    println!("    E[accepted tokens] = (1 - alpha^(K+1)) / (1 - alpha)");
    println!("    For alpha=0.7, K=5: E = {:.2} tokens per large-model pass",
        (1.0 - 0.7f32.powi(6)) / (1.0 - 0.7));
    println!("    That's a {:.1}x speedup over sequential generation!",
        (1.0 - 0.7f32.powi(6)) / (1.0 - 0.7));

    // ===================================================================
    // SECTION 4: Perplexity — How Surprised Is the Model?
    // ===================================================================
    //
    // Perplexity measures how "surprised" a model is by a given text.
    //
    //   perplexity = exp( -1/N * sum(log(p(token_i | context))) )
    //
    // Intuitively:
    //   - If the model assigns probability 1.0 to every token: perplexity = 1 (perfect)
    //   - If it assigns probability 0.5: perplexity = 2 ("confused between 2 choices")
    //   - If it assigns probability 0.01: perplexity = 100 ("confused between 100 choices")
    //
    // A perplexity of P means "the model is as confused as if it were
    // randomly choosing between P equally likely tokens at each step."
    //
    // Lower perplexity = the model finds the text more predictable.
    //
    // USES:
    //   - Evaluating model quality (lower perplexity = better model)
    //   - Detecting AI-generated text (often has lower perplexity)
    //   - Measuring text difficulty/surprisingness
    //   - Comparing prompts (which prompt is more "natural"?)
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 4: Perplexity — Measuring Predictability");
    println!("================================================================\n");

    let test_texts = [
        ("Natural order", "Once upon a time there was a little cat"),
        ("Also natural", "The boy went to the park and played with his friends"),
        ("Slightly odd", "The happy dog very much enjoyed his bone"),
        ("Scrambled", "cat time upon once there was a little"),
        ("Nonsense order", "was the a upon once little time cat there"),
        ("Repetitive", "the the the the the the the the"),
    ];

    println!("Computing perplexity for various texts:\n");
    println!("  {:>20} {:>12} {:>12} {:>10}", "Text Type", "Perplexity", "Avg LogProb", "Tokens");
    println!("  {}", "-".repeat(58));

    for (label, text) in &test_texts {
        let (perplexity, avg_log_prob, n_tokens) = compute_perplexity(
            model_path, &config, shared_weights, &tokenizer, text
        );
        println!("  {:>20} {:>12.2} {:>12.4} {:>10}",
            label, perplexity, avg_log_prob, n_tokens);
    }

    println!("\n  The full texts:");
    for (label, text) in &test_texts {
        println!("    {}: \"{}\"", label, text);
    }

    println!("\n  INTERPRETATION:");
    println!("  - Natural sentences have LOW perplexity (model expects them)");
    println!("  - Scrambled sentences have HIGH perplexity (model is surprised)");
    println!("  - Repetitive text can have low perplexity (easy to predict)");
    println!("  - A perplexity of N means the model is as uncertain as choosing");
    println!("    randomly from N equally likely options at each token position");

    // -----------------------------------------------------------------------
    // Show per-token surprisal for one sentence
    // -----------------------------------------------------------------------
    println!("\n  Per-token surprisal for \"Once upon a time there was a little cat\":");
    println!("  (Surprisal = -log2(p), measured in bits. Higher = more surprising)\n");

    let detail_text = "Once upon a time there was a little cat";
    show_per_token_surprisal(model_path, &config, shared_weights, &tokenizer, detail_text);

    // ===================================================================
    // SECTION 5: Sampling vs Greedy — Side by Side
    // ===================================================================

    println!("\n================================================================");
    println!("  SECTION 5: Greedy vs Sampling — Same Prompt, Different Strategies");
    println!("================================================================\n");

    let gen_prompt = "The little cat";
    let gen_tokens = 40;

    println!("Prompt: \"{}\"\n", gen_prompt);

    // Greedy
    {
        let (c, sw) = Config::from_file(model_path).unwrap();
        let w = TransformerWeights::from_file(model_path, &c, sw).unwrap();
        let mut m = Transformer::new(c, w);
        let config_greedy = SamplerConfig::greedy();
        let result = llm_lab::generate::generate(&mut m, &tokenizer, gen_prompt, gen_tokens, &config_greedy);
        println!("  Greedy (T=0.0):         \"{}\"", result.trim());
    }

    // Low temperature
    {
        let (c, sw) = Config::from_file(model_path).unwrap();
        let w = TransformerWeights::from_file(model_path, &c, sw).unwrap();
        let mut m = Transformer::new(c, w);
        let config_low = SamplerConfig {
            temperature: 0.5, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 42, stream: false,
        };
        let result = llm_lab::generate::generate(&mut m, &tokenizer, gen_prompt, gen_tokens, &config_low);
        println!("  T=0.5 (focused):        \"{}\"", result.trim());
    }

    // Medium temperature
    {
        let (c, sw) = Config::from_file(model_path).unwrap();
        let w = TransformerWeights::from_file(model_path, &c, sw).unwrap();
        let mut m = Transformer::new(c, w);
        let config_med = SamplerConfig {
            temperature: 1.0, top_k: 0, top_p: 0.9,
            repetition_penalty: 1.0, seed: 42, stream: false,
        };
        let result = llm_lab::generate::generate(&mut m, &tokenizer, gen_prompt, gen_tokens, &config_med);
        println!("  T=1.0, top_p=0.9:       \"{}\"", result.trim());
    }

    // High temperature
    {
        let (c, sw) = Config::from_file(model_path).unwrap();
        let w = TransformerWeights::from_file(model_path, &c, sw).unwrap();
        let mut m = Transformer::new(c, w);
        let config_high = SamplerConfig {
            temperature: 1.5, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 42, stream: false,
        };
        let result = llm_lab::generate::generate(&mut m, &tokenizer, gen_prompt, gen_tokens, &config_high);
        println!("  T=1.5 (creative):       \"{}\"", result.trim());
    }

    println!("\n  Notice how temperature affects the output:");
    println!("  - T=0 (greedy): deterministic, safe, potentially repetitive");
    println!("  - T=0.5: mostly follows the greedy path with slight variation");
    println!("  - T=1.0 + top_p: the 'natural' distribution with tail trimming");
    println!("  - T=1.5: creative/chaotic, sometimes incoherent");

    println!("\n================================================================");
    println!("  END: How Text Really Gets Generated");
    println!("================================================================");
}

// ===========================================================================
// Beam Search Implementation
// ===========================================================================
//
// Beam search maintains B "beams" (candidate sequences). At each step:
//   1. For each beam, compute the next-token probability distribution
//   2. For each beam, consider the top-B extensions
//   3. From all B*B candidates, keep the top B by total log-probability
//
// This is a simple implementation — production beam search also has:
//   - Length normalization (longer sequences have lower total prob)
//   - Diverse beam search (penalize beams that are too similar)
//   - Early stopping (if all beams have generated EOS)
// ===========================================================================

fn beam_search(
    config: &Config,
    weights: TransformerWeights,
    tokenizer: &Tokenizer,
    prompt: &str,
    beam_width: usize,
    max_tokens: usize,
) -> (String, f32, usize) {
    let bos_token: u32 = 1;
    let eos_token: u32 = 2;
    let prompt_tokens = tokenizer.encode(prompt);
    let mut init_tokens: Vec<u32> = vec![bos_token];
    init_tokens.extend_from_slice(&prompt_tokens);

    // Each beam: (tokens, log_probability, model)
    // We need separate model instances because each beam has its own KV cache
    let mut beams: Vec<(Vec<u32>, f32, Transformer)> = Vec::new();

    // Initialize the first beam with the prompt
    let mut init_model = Transformer::new(config.clone(), weights);

    // Prefill the prompt
    for pos in 0..init_tokens.len() {
        init_model.forward(init_tokens[pos], pos);
    }

    beams.push((init_tokens, 0.0, init_model));

    // Generate tokens
    for _ in 0..max_tokens {
        let mut candidates: Vec<(Vec<u32>, f32, Transformer)> = Vec::new();

        for (tokens, log_prob, mut model) in beams.drain(..) {
            let pos = tokens.len() - 1;
            if pos >= config.seq_len { continue; }

            let logits = model.forward(*tokens.last().unwrap(), pos);
            let probs = logits.softmax();

            // Find top-beam_width tokens
            let mut indexed: Vec<(usize, f32)> = probs.data.iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for &(tok_id, prob) in indexed.iter().take(beam_width) {
                if prob <= 0.0 { continue; }
                let mut new_tokens = tokens.clone();
                new_tokens.push(tok_id as u32);
                let new_log_prob = log_prob + prob.ln();

                // REVIEW: Clone existing weights instead of reloading from disk.
                // The original called TransformerWeights::from_file() here, which
                // performed ~60MB of disk I/O per beam candidate per step. With
                // beam_width=4 and max_tokens=12, that was ~48 unnecessary file reads.
                // Since we replay all tokens below to rebuild the KV cache anyway,
                // we just need fresh weights — cloning from the first model is O(memory)
                // with no disk I/O.
                let new_model = Transformer::new(config.clone(), model.weights.clone());

                candidates.push((new_tokens, new_log_prob, new_model));
            }
        }

        // Keep top beam_width candidates
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(beam_width);

        // Rebuild KV caches for each candidate by replaying their tokens
        for (tokens, _, model) in &mut candidates {
            model.reset();
            for pos in 0..tokens.len() {
                model.forward(tokens[pos], pos);
            }
        }

        // REVIEW: Use iterator any() instead of manual loop + mutable bool for clarity.
        let finished = candidates.iter()
            .any(|(tokens, _, _)| *tokens.last().unwrap() == eos_token);

        beams = candidates;

        if finished || beams.is_empty() { break; }
    }

    // Return the best beam
    if let Some((tokens, log_prob, _)) = beams.into_iter().next() {
        let n_gen = tokens.len() - 1; // subtract BOS
        let text = tokenizer.decode(&tokens[1..]); // skip BOS
        (text, log_prob, n_gen)
    } else {
        (String::new(), 0.0, 0)
    }
}

// ===========================================================================
// Perplexity Computation
// ===========================================================================
//
// Perplexity is the exponential of the average negative log-likelihood:
//
//   PPL = exp( -1/N * sum_{i=1}^{N} log(p(token_i | token_1..token_{i-1})) )
//
// The log is natural log (base e). We sum the log-probability of each
// token given all previous tokens, divide by the number of tokens,
// negate, and exponentiate.
//
// Lower perplexity = the model assigns higher probability to the text
// = the text is more predictable/expected.
// ===========================================================================

fn compute_perplexity(
    model_path: &str,
    config: &Config,
    shared_weights: bool,
    tokenizer: &Tokenizer,
    text: &str,
) -> (f32, f32, usize) {
    let weights = TransformerWeights::from_file(model_path, config, shared_weights).unwrap();
    let mut model = Transformer::new(config.clone(), weights);

    let bos_token: u32 = 1;
    let text_tokens = tokenizer.encode(text);
    let mut all_tokens: Vec<u32> = vec![bos_token];
    all_tokens.extend_from_slice(&text_tokens);

    let mut total_log_prob = 0.0f32;
    let mut n_scored = 0;

    // Process each token and measure how well the model predicted it
    for pos in 0..all_tokens.len() - 1 {
        let logits = model.forward(all_tokens[pos], pos);
        let probs = logits.softmax();

        // The next token is what we're trying to predict
        let next_token = all_tokens[pos + 1] as usize;
        let prob = probs.data[next_token];

        if prob > 0.0 {
            total_log_prob += prob.ln();
            n_scored += 1;
        }
    }

    let avg_log_prob = total_log_prob / n_scored as f32;
    let perplexity = (-avg_log_prob).exp();

    (perplexity, avg_log_prob, n_scored)
}

// ===========================================================================
// Per-Token Surprisal Visualization
// ===========================================================================

fn show_per_token_surprisal(
    model_path: &str,
    config: &Config,
    shared_weights: bool,
    tokenizer: &Tokenizer,
    text: &str,
) {
    let weights = TransformerWeights::from_file(model_path, config, shared_weights).unwrap();
    let mut model = Transformer::new(config.clone(), weights);

    let bos_token: u32 = 1;
    let text_tokens = tokenizer.encode(text);
    let mut all_tokens: Vec<u32> = vec![bos_token];
    all_tokens.extend_from_slice(&text_tokens);

    println!("    {:>12} {:>10} {:>10} {:>10} {}", "Token", "Prob", "Surprisal", "Bar", "");
    println!("    {}", "-".repeat(60));

    for pos in 0..all_tokens.len() - 1 {
        let logits = model.forward(all_tokens[pos], pos);
        let probs = logits.softmax();

        let next_token = all_tokens[pos + 1];
        let prob = probs.data[next_token as usize];
        let surprisal = if prob > 0.0 { -prob.log2() } else { f32::INFINITY };

        let tok_str = tokenizer.token_str(next_token);
        let bar_len = (surprisal * 3.0) as usize;
        let bar: String = "#".repeat(bar_len.min(30));

        println!("    {:>12} {:>10.4} {:>8.2} b  [{:30}]",
            format!("'{}'", tok_str.trim()), prob, surprisal, bar);
    }

    println!("\n    Low surprisal = model expected this token (common/predictable)");
    println!("    High surprisal = model was surprised (rare/unexpected)");
    let _ = io::stdout().flush();
}
