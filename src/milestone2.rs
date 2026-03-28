// ============================================================================
// LLM Lab — Milestone 2: Tokenizer
// ============================================================================
//
// Run: cargo run --bin milestone2
//
// Requires: data/tokenizer.bin (from Karpathy's llama2.c project)
//   curl -L -o data/tokenizer.bin \
//     https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
//
// ============================================================================

use llm_lab::tokenizer::Tokenizer;

const VOCAB_SIZE: usize = 32000;

fn main() {
    println!("+----------------------------------------------------------+");
    println!("|        LLM LAB — Milestone 2                             |");
    println!("|        Tokenizer (Byte Pair Encoding)                    |");
    println!("+----------------------------------------------------------+\n");

    // Load the tokenizer
    let tok = match Tokenizer::from_file("data/tokenizer.bin", VOCAB_SIZE) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {}", e);
            eprintln!("Download it first:");
            eprintln!("  mkdir -p data");
            eprintln!("  curl -L -o data/tokenizer.bin \\");
            eprintln!("    https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin");
            std::process::exit(1);
        }
    };

    println!("  Loaded tokenizer: {} tokens, max token length = {} bytes\n",
        tok.vocab_size(), tok.max_token_length);

    // -----------------------------------------------------------------------
    // Demo 1: Peek at the vocabulary
    // -----------------------------------------------------------------------
    println!("--- Demo 1: What's in the vocabulary? ---");
    println!("The first tokens are special, then byte-level, then learned merges.\n");

    println!("  Special tokens:");
    for i in 0..3 {
        println!("    token {:>5} = {:?}", i, tok.token_str(i));
    }

    println!("\n  Byte tokens (raw bytes as fallback):");
    for i in 3..8 {
        println!("    token {:>5} = {:?}", i, tok.token_str(i));
    }

    println!("\n  Some learned tokens (common subwords and words):");
    // Show a range in the middle of the vocab -- these are common words
    for &i in &[259, 260, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000] {
        if i < tok.vocab_size() as u32 {
            println!("    token {:>5} = {:?}", i, tok.token_str(i));
        }
    }

    // -----------------------------------------------------------------------
    // Demo 2: Encoding — watch BPE merge characters into tokens
    // -----------------------------------------------------------------------
    println!("\n--- Demo 2: Encoding (text -> tokens) ---");
    println!("BPE starts with individual characters and merges common pairs.\n");

    let examples = [
        "Hello",
        "Hello world",
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog.",
        "unhappiness",
        "antidisestablishmentarianism",
    ];

    for text in &examples {
        let tokens = tok.encode(text);
        let pieces: Vec<String> = tokens.iter()
            .map(|&id| format!("{:?}", tok.token_str(id)))
            .collect();
        println!("  {:?}", text);
        println!("    {} chars -> {} tokens (compression: {:.1}x)",
            text.len(), tokens.len(), text.len() as f64 / tokens.len() as f64);
        println!("    tokens: {:?}", tokens);
        println!("    pieces: [{}]", pieces.join(", "));
        println!();
    }

    // -----------------------------------------------------------------------
    // Demo 3: Decoding — tokens back to text
    // -----------------------------------------------------------------------
    println!("--- Demo 3: Decoding (tokens -> text) ---");
    println!("Decoding is simple: concatenate the token strings.\n");

    let text = "Once upon a time, there was a little cat named Luna.";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens);

    println!("  Original:  {:?}", text);
    println!("  Tokens:    {:?}", tokens);
    println!("  Decoded:   {:?}", decoded);
    println!("  Match:     {}", if text == decoded { "PERFECT" } else { "MISMATCH!" });
    println!();

    // -----------------------------------------------------------------------
    // Demo 4: Why tokenization matters for LLMs
    // -----------------------------------------------------------------------
    println!("--- Demo 4: Why tokenization matters ---\n");

    // Common words = 1 token, rare words = many tokens
    let common = "the";
    let rare = "cryptocurrency";
    let common_tok = tok.encode(common);
    let rare_tok = tok.encode(rare);

    println!("  Common word: {:?} -> {} token(s): {:?}",
        common, common_tok.len(),
        common_tok.iter().map(|&id| tok.token_str(id)).collect::<Vec<_>>());
    println!("  Rare word:   {:?} -> {} token(s): {:?}",
        rare, rare_tok.len(),
        rare_tok.iter().map(|&id| tok.token_str(id)).collect::<Vec<_>>());
    println!();
    println!("  This matters because:");
    println!("    - Each token costs the same compute in the model");
    println!("    - Common words are 'cheap' (1 token), rare words are 'expensive'");
    println!("    - Context window is measured in TOKENS, not characters");
    println!();

    // Show context window implications
    let long_text = "In a galaxy far far away, there lived a brave little robot who dreamed of exploring the stars. Every night, the robot would look up at the sky and wonder what adventures awaited beyond the clouds.";
    let long_tokens = tok.encode(long_text);
    println!("  A paragraph ({} chars) = {} tokens", long_text.len(), long_tokens.len());
    println!("  At 4K context window: room for ~{} paragraphs like this",
        4096 / long_tokens.len());
    println!("  At 128K context window: room for ~{} paragraphs",
        131072 / long_tokens.len());

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!();
    println!("------------------------------------------------------------");
    println!("  MILESTONE 2 COMPLETE -- KEY CONCEPTS:");
    println!();
    println!("  1. TOKENIZATION converts text to numbers and back.");
    println!("     LLMs never see raw text -- only token IDs.");
    println!();
    println!("  2. BPE starts with bytes and greedily merges common pairs.");
    println!("     The vocabulary is a compression scheme: common patterns");
    println!("     get their own token, rare text decomposes into pieces.");
    println!();
    println!("  3. VOCAB SIZE (32K here) is a tradeoff:");
    println!("     Larger vocab = shorter sequences but bigger embedding table.");
    println!("     Smaller vocab = longer sequences but more attention cost.");
    println!();
    println!("  4. CONTEXT WINDOW is measured in tokens, not characters.");
    println!("     Efficient tokenization = more text fits in the window.");
    println!();
    println!("  5. The tokenizer is SEPARATE from the model. Same model can");
    println!("     use different tokenizers (GPT uses tiktoken, Llama uses");
    println!("     sentencepiece). But the vocab size must match.");
    println!();
    println!("  NEXT: Milestone 3 -- Transformer forward pass");
    println!("------------------------------------------------------------");
}
