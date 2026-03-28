// ============================================================================
// LLM Lab — Milestone 5: Text Generation
// ============================================================================
//
// This module ties together the transformer, tokenizer, and sampler into a
// single `generate()` function — the top-level API for text generation.
//
// The generation loop has two phases:
//
//   1. PREFILL: Feed the prompt tokens through the model one by one.
//      This builds up the KV cache so that when we start generating,
//      the model "remembers" the entire prompt. During prefill we don't
//      sample — the next token is already known (it's the next prompt token).
//
//   2. AUTOREGRESSIVE GENERATION: Repeatedly:
//      a. Run the forward pass to get logits
//      b. Sample from the logits using our sampling strategy
//      c. Feed the sampled token back as input
//      d. Repeat until we hit max_tokens or the EOS token
//
// The key insight: the model generates ONE token at a time, and each token
// depends on ALL previous tokens (via the KV cache). This is why LLM
// inference is sequential and hard to parallelize — you can't generate
// token 10 until you've generated tokens 1-9.
//
// ============================================================================

use crate::model::Transformer;
use crate::tokenizer::Tokenizer;
use crate::sampler::{Sampler, SamplerConfig};

use std::io::{self, Write};

/// Generate text from a prompt using the given model, tokenizer, and sampling config.
///
/// This is the main entry point for text generation. It handles:
///   - Encoding the prompt to token IDs
///   - Prepending the BOS (Beginning of Sentence) token
///   - Running the prefill phase (processing prompt tokens)
///   - Running the autoregressive generation loop
///   - Decoding the output tokens back to text
///   - Optional streaming (printing tokens as they're generated)
///
/// The model is passed as `&mut` because the forward pass mutates the
/// KV cache and run state. The caller should call `model.reset()` between
/// generations to clear the KV cache.
///
/// Returns the complete generated text (prompt + generated continuation).
pub fn generate(
    model: &mut Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    config: &SamplerConfig,
) -> String {
    let mut sampler = Sampler::new(config.seed);

    // --- Encode the prompt ---
    // The tokenizer converts text into token IDs that the model understands.
    let prompt_tokens = tokenizer.encode(prompt);

    // --- Prepend BOS (Beginning of Sentence) token ---
    // Token ID 1 is the BOS token in the Llama/sentencepiece convention.
    // It signals to the model that this is the start of a new sequence.
    // Without BOS, the model would behave as if we're continuing a previous
    // sentence, which can produce odd results.
    let bos_token: u32 = 1;
    let mut all_tokens: Vec<u32> = vec![bos_token];
    all_tokens.extend_from_slice(&prompt_tokens);

    // --- Prefill phase ---
    // Feed all prompt tokens (including BOS) through the model to build
    // up the KV cache. We don't sample during this phase because the next
    // token is already known — it's the next token in the prompt.
    //
    // The last forward pass of the prefill produces logits that predict
    // the FIRST generated token (the one after the prompt).
    let mut next_token = all_tokens[0];
    for pos in 0..all_tokens.len() {
        let logits = model.forward(all_tokens[pos], pos);

        if pos == all_tokens.len() - 1 {
            // Last prompt token: sample the first generated token
            next_token = sampler.sample(logits, &all_tokens, config);
        }
        // For earlier positions, next_token is just the next prompt token
        // (we don't need to set it since we use all_tokens[pos] directly)
    }

    // Add the first generated token
    all_tokens.push(next_token);

    // If streaming, print the prompt first, then each generated token
    if config.stream {
        // Print BOS-decoded prompt (skip BOS in output)
        print!("{}", tokenizer.decode(&all_tokens[..all_tokens.len() - 1]));
        // Print the first generated token
        print!("{}", tokenizer.token_str(next_token));
        io::stdout().flush().ok();
    }

    // --- Autoregressive generation phase ---
    // Now we generate tokens one at a time. Each generated token becomes
    // the input for the next forward pass. This continues until we hit
    // max_tokens or the EOS (End of Sentence) token.
    let eos_token: u32 = 2; // </s> in sentencepiece

    for _ in 0..max_tokens {
        let pos = all_tokens.len() - 1;
        if pos >= model.config.seq_len {
            break; // Reached max sequence length
        }

        let logits = model.forward(next_token, pos);
        next_token = sampler.sample(logits, &all_tokens, config);

        if next_token == eos_token {
            break; // Model decided to stop
        }

        all_tokens.push(next_token);

        if config.stream {
            print!("{}", tokenizer.token_str(next_token));
            io::stdout().flush().ok();
        }
    }

    if config.stream {
        println!(); // Final newline after streaming
    }

    // Decode all tokens (skip BOS) back to text
    tokenizer.decode(&all_tokens[1..])
}
