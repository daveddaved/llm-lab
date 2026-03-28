// ============================================================================
// LLM Lab — Milestone 2: Tokenizer (BPE)
// ============================================================================
//
// LLMs don't see text. They see NUMBERS. The tokenizer is the bridge:
//
//   "Hello world" → [15043, 3186] → model → [271, 306, 626] → "I am fine"
//
// But why not just use ASCII codes? Two reasons:
//
//   1. VOCABULARY SIZE vs SEQUENCE LENGTH tradeoff.
//      - Character-level: vocab=256, but "Hello" = 5 tokens. Long sequences
//        mean more attention computation (quadratic in length!).
//      - Word-level: "Hello"=1 token, but vocab=100k+ words. The embedding
//        table (vocab_size × d_model) becomes enormous, and rare words
//        get poor representations.
//      - SUBWORD (BPE): vocab=32k, "Hello"=1 token, "unhappiness"=2-3 tokens.
//        Best of both worlds.
//
//   2. BYTE PAIR ENCODING (BPE) handles ANY text, even unseen words.
//      It starts with individual bytes (256 base tokens) and merges the
//      most frequent pairs into new tokens. "th" + "e" → "the".
//      After training, common words are single tokens; rare words decompose
//      into known subwords.
//
// How BPE encoding works (at inference time):
//   1. Convert the input string to a sequence of byte tokens
//   2. Find the pair of adjacent tokens with the highest merge score
//   3. Merge that pair into a single token
//   4. Repeat until no more merges are possible
//
// The merge scores come from training (we load them from the model file).
// Higher score = this pair was merged earlier during training = more common.
//
// ============================================================================

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
// Stores the vocabulary (token_id → string) and merge scores.
// The merge scores tell us the priority of each possible merge.
// ---------------------------------------------------------------------------

pub struct Tokenizer {
    /// token_id → token string (e.g., 15043 → "Hello")
    vocab: Vec<String>,

    /// token_id → BPE merge score (higher = merge earlier = more common)
    scores: Vec<f32>,

    /// Reverse lookup: token string → token_id (for encoding)
    token_to_id: HashMap<String, u32>,

    /// Maximum token length in bytes (from the model file header)
    pub max_token_length: u32,
}

impl Tokenizer {
    // -----------------------------------------------------------------------
    // Load from llama2.c tokenizer.bin format
    // -----------------------------------------------------------------------
    // Binary format (all little-endian):
    //   max_token_length: u32
    //   for each token (vocab_size times):
    //     score: f32
    //     len: u32
    //     bytes: [u8; len]
    //
    // This is a simple format designed by Karpathy for llama2.c.
    // Production models use sentencepiece (.model) or tiktoken formats,
    // but the concept is identical.
    // -----------------------------------------------------------------------

    pub fn from_file(path: &str, vocab_size: usize) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let mut buf4 = [0u8; 4];

        // Read max_token_length
        f.read_exact(&mut buf4)?;
        let max_token_length = u32::from_le_bytes(buf4);

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut token_to_id = HashMap::new();

        for i in 0..vocab_size {
            // Read score (f32)
            f.read_exact(&mut buf4)?;
            let score = f32::from_le_bytes(buf4);

            // Read string length (u32)
            f.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4) as usize;

            // Read string bytes
            let mut str_buf = vec![0u8; len];
            f.read_exact(&mut str_buf)?;
            let token_str = String::from_utf8_lossy(&str_buf).to_string();

            token_to_id.insert(token_str.clone(), i as u32);
            vocab.push(token_str);
            scores.push(score);
        }

        Ok(Tokenizer {
            vocab,
            scores,
            token_to_id,
            max_token_length,
        })
    }

    /// Look up the string for a token ID.
    pub fn token_str(&self, id: u32) -> &str {
        &self.vocab[id as usize]
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    // -----------------------------------------------------------------------
    // DECODE: token IDs → string
    // -----------------------------------------------------------------------
    // Simple: just concatenate the token strings.
    // Special handling for:
    //   - Byte tokens like <0x0A> → the actual byte (0x0A = newline)
    //   - BOS/EOS tokens (<s>, </s>) → skip
    //
    // In a real system you'd also handle sentencepiece's "▁" (U+2581)
    // which represents a leading space. The tinystories tokenizer uses
    // a simpler scheme where spaces are literal in the token strings.
    // -----------------------------------------------------------------------

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        for &id in tokens {
            let s = self.token_str(id);

            // Skip BOS/EOS markers
            if s == "<s>" || s == "</s>" {
                continue;
            }

            // Handle byte tokens: <0xHH> → actual byte
            if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
                    result.push(byte as char);
                    continue;
                }
            }

            result.push_str(s);
        }
        result
    }

    // -----------------------------------------------------------------------
    // ENCODE: string → token IDs (BPE algorithm)
    // -----------------------------------------------------------------------
    // This is the core BPE algorithm at inference time:
    //
    //   1. Start with each byte of the input as a separate token
    //   2. Find the adjacent pair with the HIGHEST merge score
    //   3. Merge that pair into a single token (look up the combined string)
    //   4. Repeat until no more merges are possible
    //
    // Example with a toy vocabulary:
    //   Input: "low"
    //   Step 0: ['l', 'o', 'w']           (3 tokens)
    //   Step 1: merge 'l'+'o' → "lo"      (score 5.0)
    //   Step 2: merge "lo"+'w' → "low"    (score 8.0)
    //   Result: ["low"]                    (1 token)
    //
    // The greedy "highest score first" strategy is how sentencepiece works.
    // It's not globally optimal, but it's fast and works well in practice.
    // -----------------------------------------------------------------------

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Step 1: convert each byte to its token ID.
        // The first 256 tokens in the vocab are single bytes.
        // For the tinystories tokenizer, bytes are encoded as <0xHH> tokens
        // at positions 3..259 (after <unk>, <s>, </s>).
        let mut tokens: Vec<u32> = Vec::new();
        for ch in text.chars() {
            let s = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&s) {
                tokens.push(id);
            } else {
                // Fall back to byte-level tokens
                for byte in ch.to_string().as_bytes() {
                    let hex = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&hex) {
                        tokens.push(id);
                    }
                }
            }
        }

        // Step 2: repeatedly merge the highest-scoring adjacent pair.
        // This loop runs until no more merges are possible.
        //
        // Each iteration scans all adjacent pairs, finds the one with
        // the highest score that exists in our vocabulary, and merges it.
        //
        // Time complexity: O(n^2) in the worst case where n is the number
        // of initial tokens. For typical text this is fast because the
        // sequence shrinks quickly.
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            // Scan all adjacent pairs
            for i in 0..tokens.len().saturating_sub(1) {
                // Build the merged string
                let merged = format!(
                    "{}{}",
                    self.vocab[tokens[i] as usize],
                    self.vocab[tokens[i + 1] as usize],
                );

                // Is this merged string in our vocabulary?
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = id;
                    }
                }
            }

            // No more merges possible — we're done
            if best_idx == usize::MAX {
                break;
            }

            // Merge: replace tokens[best_idx] and tokens[best_idx+1]
            // with the merged token
            tokens[best_idx] = best_id;
            tokens.remove(best_idx + 1);
        }

        tokens
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_test_tokenizer() -> Option<Tokenizer> {
        let path = "data/tokenizer.bin";
        if !Path::new(path).exists() {
            eprintln!("Skipping test: {} not found", path);
            return None;
        }
        Some(Tokenizer::from_file(path, 32000).expect("failed to load tokenizer"))
    }

    #[test]
    fn test_load_tokenizer() {
        let Some(tok) = load_test_tokenizer() else { return };
        assert_eq!(tok.vocab_size(), 32000);
        assert_eq!(tok.token_str(0), "<unk>");
        assert!(tok.max_token_length > 0);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let Some(tok) = load_test_tokenizer() else { return };

        let text = "Hello world";
        let tokens = tok.encode(text);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, text, "roundtrip failed: {:?} -> {:?} -> {:?}", text, tokens, decoded);
    }

    #[test]
    fn test_encode_decode_longer() {
        let Some(tok) = load_test_tokenizer() else { return };

        let text = "Once upon a time, there was a little cat.";
        let tokens = tok.encode(text);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, text);

        // BPE should compress: fewer tokens than characters
        assert!(
            tokens.len() < text.len(),
            "BPE should compress: {} tokens vs {} chars",
            tokens.len(),
            text.len(),
        );
    }

    #[test]
    fn test_encode_empty() {
        let Some(tok) = load_test_tokenizer() else { return };
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn test_single_char() {
        let Some(tok) = load_test_tokenizer() else { return };

        let tokens = tok.encode("a");
        assert_eq!(tokens.len(), 1);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "a");
    }
}
