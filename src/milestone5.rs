// ============================================================================
// LLM Lab — Milestone 5: Text Generation (Sampling Strategies)
// ============================================================================
//
// Run: cargo run --bin milestone5
//
// In Milestone 4, we used greedy decoding (argmax) — always picking the
// single most likely token. This produces coherent but predictable,
// repetitive text. Real LLM applications use SAMPLING: treating the
// model's output as a probability distribution and randomly drawing from
// it, with various strategies to control quality and diversity.
//
// This milestone demonstrates four key sampling strategies:
//   1. Temperature — controls overall randomness
//   2. Top-k — limits candidates to the k most likely tokens
//   3. Top-p (nucleus) — dynamically sizes the candidate set
//   4. Repetition penalty — prevents the model from looping
//
// Each demo shows the same prompt with different settings so you can
// see exactly how each parameter affects the output.
//
// ============================================================================

use llm_lab::model::{Config, Transformer, TransformerWeights};
use llm_lab::tokenizer::Tokenizer;
use llm_lab::sampler::SamplerConfig;
use llm_lab::generate::generate;

fn main() {
    println!("+----------------------------------------------------------+");
    println!("|        LLM LAB — Milestone 5                             |");
    println!("|        Sampling Strategies                               |");
    println!("+----------------------------------------------------------+\n");

    let checkpoint_path = "data/stories15M.bin";
    let tokenizer_path = "data/tokenizer.bin";

    // --- Load model and tokenizer ---
    println!("--- Loading model and tokenizer ---\n");

    let (config, shared_weights) = Config::from_file(checkpoint_path)
        .expect("Failed to read config from checkpoint");

    let tokenizer = Tokenizer::from_file(tokenizer_path, config.vocab_size)
        .expect("Failed to load tokenizer");

    let weights = TransformerWeights::from_file(checkpoint_path, &config, shared_weights)
        .expect("Failed to load weights");

    let mut model = Transformer::new(config.clone(), weights);

    println!("  Model: {} layers, dim={}, vocab={}", config.n_layers, config.dim, config.vocab_size);
    println!("  Tokenizer: {} tokens loaded\n", tokenizer.vocab_size());

    let prompt = "Once upon a time";
    let gen_tokens = 80;

    // ===================================================================
    // Demo 1: Temperature Comparison
    // ===================================================================
    // Temperature controls the "sharpness" of the probability distribution.
    // Lower temperature → more deterministic (sharper peaks)
    // Higher temperature → more random (flatter distribution)
    //
    // Mathematically: softmax(logits / T)
    //   T → 0: all probability concentrates on the argmax token
    //   T = 1: the model's natural distribution
    //   T → ∞: uniform distribution (pure random)
    // ===================================================================

    println!("===========================================================");
    println!("  DEMO 1: Temperature Comparison");
    println!("===========================================================");
    println!();
    println!("  Temperature controls how 'creative' vs 'focused' the");
    println!("  model is. Watch how the output changes as T increases:\n");

    for &temp in &[0.0, 0.5, 1.0, 1.5] {
        let label = match temp {
            t if t == 0.0 => "T=0.0 (greedy/argmax)",
            t if t == 0.5 => "T=0.5 (focused)",
            t if t == 1.0 => "T=1.0 (neutral)",
            _ => "T=1.5 (creative)",
        };
        println!("  --- {} ---", label);

        model.reset();
        let cfg = SamplerConfig {
            temperature: temp,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 42,
            stream: false,
        };
        let output = generate(&mut model, &tokenizer, prompt, gen_tokens, &cfg);
        print_boxed(&output);
        println!();
    }

    println!("  KEY INSIGHT: T=0 is deterministic (always the same output).");
    println!("  As T increases, the model takes more risks — sometimes");
    println!("  brilliant, sometimes nonsensical. T=0.8 is a common sweet spot.\n");

    // ===================================================================
    // Demo 2: Top-k Comparison
    // ===================================================================
    // Top-k limits the candidate set to the k most probable tokens.
    // This prevents sampling from the "long tail" of unlikely tokens
    // that can produce garbage.
    // ===================================================================

    println!("===========================================================");
    println!("  DEMO 2: Top-k Comparison");
    println!("===========================================================");
    println!();
    println!("  Top-k limits how many tokens the model can choose from.");
    println!("  k=1 is greedy, k=50 allows moderate diversity.\n");

    for &k in &[1, 10, 50] {
        let label = match k {
            1 => "k=1 (greedy — only the top token)",
            10 => "k=10 (conservative — top 10 candidates)",
            _ => "k=50 (moderate diversity)",
        };
        println!("  --- {} ---", label);

        model.reset();
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: k,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 42,
            stream: false,
        };
        let output = generate(&mut model, &tokenizer, prompt, gen_tokens, &cfg);
        print_boxed(&output);
        println!();
    }

    println!("  KEY INSIGHT: k=1 at temperature=1.0 is equivalent to greedy.");
    println!("  k=10 keeps the output sensible while allowing some variety.");
    println!("  The LIMITATION of top-k is that it's a fixed number regardless");
    println!("  of how confident the model is. Top-p fixes this.\n");

    // ===================================================================
    // Demo 3: Top-p (Nucleus) Comparison
    // ===================================================================
    // Top-p dynamically sizes the candidate set based on cumulative
    // probability. It adapts to the model's confidence level.
    // ===================================================================

    println!("===========================================================");
    println!("  DEMO 3: Top-p (Nucleus) Comparison");
    println!("===========================================================");
    println!();
    println!("  Top-p keeps tokens until cumulative probability exceeds p.");
    println!("  It ADAPTS: confident predictions keep fewer candidates,");
    println!("  uncertain predictions keep more.\n");

    for &p in &[0.5, 0.9] {
        let label = match p {
            x if x == 0.5 => "p=0.5 (conservative — top 50% probability mass)",
            _ => "p=0.9 (standard — top 90% probability mass)",
        };
        println!("  --- {} ---", label);

        model.reset();
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: p,
            repetition_penalty: 1.0,
            seed: 42,
            stream: false,
        };
        let output = generate(&mut model, &tokenizer, prompt, gen_tokens, &cfg);
        print_boxed(&output);
        println!();
    }

    println!("  KEY INSIGHT: p=0.9 is the most popular setting in production.");
    println!("  It keeps roughly the 'nucleus' of the distribution — the");
    println!("  smallest set of tokens that accounts for 90% of probability.\n");

    // ===================================================================
    // Demo 4: Repetition Penalty
    // ===================================================================
    // Without repetition penalty, greedy or low-temperature generation
    // often falls into loops. The penalty reduces the probability of
    // tokens that appeared recently.
    // ===================================================================

    println!("===========================================================");
    println!("  DEMO 4: Repetition Penalty");
    println!("===========================================================");
    println!();
    println!("  Using greedy decoding (T=0) to make loops visible.");
    println!("  Watch how repetition penalty breaks the cycle.\n");

    let rep_prompt = "The little dog";

    println!("  --- No penalty (rep=1.0) ---");
    model.reset();
    let cfg_no_rep = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 42,
        stream: false,
    };
    let output = generate(&mut model, &tokenizer, rep_prompt, 100, &cfg_no_rep);
    print_boxed(&output);
    println!();

    println!("  --- With penalty (rep=1.3) ---");
    model.reset();
    let cfg_rep = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.3,
        seed: 42,
        stream: false,
    };
    let output = generate(&mut model, &tokenizer, rep_prompt, 100, &cfg_rep);
    print_boxed(&output);
    println!();

    println!("  KEY INSIGHT: Greedy decoding often loops because generating");
    println!("  a token makes similar tokens more likely in the future (the");
    println!("  model is trained on natural text which has local patterns).");
    println!("  Repetition penalty breaks this feedback loop by reducing");
    println!("  the probability of recently-seen tokens.\n");

    // ===================================================================
    // Demo 5: "Production" Settings — Longer Story
    // ===================================================================
    // Real applications combine multiple strategies. A typical config:
    //   temperature=0.8, top_p=0.9, repetition_penalty=1.1
    // ===================================================================

    println!("===========================================================");
    println!("  DEMO 5: Production Settings (Longer Story)");
    println!("===========================================================");
    println!();
    println!("  Combining temperature=0.8, top_p=0.9, rep_penalty=1.1");
    println!("  This is close to what production LLM systems use.\n");

    model.reset();
    let cfg_prod = SamplerConfig {
        temperature: 0.8,
        top_k: 0,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: 42,
        stream: false,
    };
    let output = generate(&mut model, &tokenizer, prompt, 200, &cfg_prod);
    print_boxed(&output);
    println!();

    // ===================================================================
    // Summary
    // ===================================================================

    println!("===========================================================");
    println!("  MILESTONE 5 COMPLETE — KEY CONCEPTS:");
    println!();
    println!("  1. TEMPERATURE: Divides logits by T before softmax.");
    println!("     T<1 → sharper (more deterministic)");
    println!("     T>1 → flatter (more creative/random)");
    println!("     T=0 → greedy (argmax, no randomness)");
    println!();
    println!("  2. TOP-K: Keep only the k most likely tokens.");
    println!("     Prevents sampling garbage from the long tail.");
    println!("     Fixed k regardless of model confidence.");
    println!();
    println!("  3. TOP-P (NUCLEUS): Keep tokens covering top p%% of");
    println!("     probability mass. ADAPTS to confidence: keeps fewer");
    println!("     tokens when confident, more when uncertain.");
    println!();
    println!("  4. REPETITION PENALTY: Reduces logits for recent tokens.");
    println!("     Breaks the feedback loop that causes repetition,");
    println!("     especially visible with greedy decoding.");
    println!();
    println!("  5. PIPELINE ORDER MATTERS: penalty → temperature →");
    println!("     softmax → top-k → top-p → sample. Each step shapes");
    println!("     the distribution for the next.");
    println!();
    println!("  6. THE ART OF SAMPLING: There's no single 'best' config.");
    println!("     Factual Q&A wants low temperature (accurate).");
    println!("     Creative writing wants higher temperature (varied).");
    println!("     Tuning these knobs is part of prompt engineering.");
    println!("===========================================================");
}

/// Print text in a simple box for readability.
fn print_boxed(text: &str) {
    println!("  ┌─────────────────────────────────────────────────────┐");
    for line in text.lines() {
        // Truncate long lines to fit in the box
        let display = if line.len() > 51 { &line[..51] } else { line };
        println!("  │ {:<52}│", display);
    }
    println!("  └─────────────────────────────────────────────────────┘");
}
