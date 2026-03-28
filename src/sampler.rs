// ============================================================================
// LLM Lab — Milestone 5: Sampling Strategies
// ============================================================================
//
// Greedy decoding (argmax) always picks the single most likely token. This
// is deterministic and often produces fluent text, but it has problems:
//
//   1. REPETITION: The model gets stuck in loops ("the cat sat on the mat
//      on the mat on the mat...") because once a pattern starts, the same
//      tokens keep having the highest probability.
//
//   2. BLANDNESS: Always picking the top token produces "safe" but boring
//      text. Creative writing, storytelling, and dialogue benefit from
//      occasionally choosing less-likely-but-still-reasonable tokens.
//
//   3. LACK OF DIVERSITY: The same prompt always produces the same output.
//      For applications like chatbots or story generators, variety matters.
//
// SAMPLING solves these problems by treating the model's output as a
// probability distribution and RANDOMLY drawing from it, with various
// strategies to control the quality/diversity tradeoff.
//
// The sampling pipeline (applied in order):
//   1. Repetition penalty — reduce logits for recently-seen tokens
//   2. Temperature scaling — sharpen or flatten the distribution
//   3. Softmax — convert logits to probabilities
//   4. Top-k filtering — keep only the k most likely tokens
//   5. Top-p (nucleus) filtering — keep tokens until cumulative prob > p
//   6. Weighted random selection — sample from the filtered distribution
//
// Each strategy addresses a different aspect of generation quality.
// In practice, you rarely use ALL of them — a typical "production" config
// might use temperature=0.8 + top_p=0.9 + repetition_penalty=1.1.
//
// ============================================================================

use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// SamplerConfig — bundles all sampling hyperparameters
// ---------------------------------------------------------------------------
// These are the knobs you turn to control generation behavior. Think of them
// as a mixing board: temperature controls overall randomness, top_k/top_p
// control the candidate set, and repetition_penalty fights loops.
// ---------------------------------------------------------------------------

pub struct SamplerConfig {
    /// Temperature scaling factor. Controls the entropy (randomness) of
    /// the output distribution.
    ///
    /// HOW IT WORKS: Logits are divided by temperature before softmax.
    ///   softmax(logits / T)
    ///
    /// WHY: Softmax converts logits to probabilities via exp(). When we
    /// divide by T < 1, the logits are amplified (spread further apart),
    /// making exp() differences larger → sharper distribution → more
    /// deterministic. When T > 1, logits are compressed → flatter
    /// distribution → more random.
    ///
    ///   T = 0.0 → equivalent to greedy (argmax)
    ///   T = 0.5 → fairly focused, but allows some variety
    ///   T = 1.0 → neutral (the model's natural distribution)
    ///   T = 1.5 → creative/chaotic, more unexpected tokens
    ///   T → ∞   → uniform distribution (pure random)
    pub temperature: f32,

    /// Top-k: only consider the k most probable tokens, zero out the rest.
    ///
    /// WHY: The vocabulary is 32,000 tokens, but at any given position most
    /// tokens are nonsensical. Even with temperature, sampling from all 32k
    /// tokens risks picking garbage from the "long tail" of near-zero
    /// probabilities. Top-k says: "only consider the best k candidates."
    ///
    ///   k = 1   → greedy decoding
    ///   k = 10  → conservative: only very plausible continuations
    ///   k = 50  → moderate diversity
    ///   k = 0   → disabled (consider all tokens)
    ///
    /// LIMITATION: k is fixed regardless of context. When the model is very
    /// confident (one token has 95% probability), k=50 still keeps 50 tokens.
    /// When the model is uncertain, k=50 might cut off reasonable options.
    /// Top-p addresses this by adapting to the distribution shape.
    pub top_k: usize,

    /// Top-p (nucleus sampling): keep tokens whose cumulative probability
    /// sums to at most p, then renormalize.
    ///
    /// WHY: Unlike top-k (fixed number of candidates), top-p ADAPTS to the
    /// model's confidence:
    ///   - If the model is 90% sure of one token, p=0.9 keeps ~1 token
    ///   - If the model is uncertain (10% each for 10 tokens), p=0.9 keeps ~9
    ///
    /// This is the key insight of nucleus sampling (Holtzman et al., 2019):
    /// the "nucleus" of the distribution varies in size depending on context.
    ///
    ///   p = 0.5 → conservative (keep tokens covering 50% of probability mass)
    ///   p = 0.9 → standard production setting
    ///   p = 1.0 → disabled (keep all tokens)
    pub top_p: f32,

    /// Repetition penalty: multiply logits of recently-seen tokens by this
    /// factor (applied as division for values > 0, multiplication for < 0).
    ///
    /// WHY: Language models, especially with low temperature or greedy
    /// decoding, tend to repeat themselves. The model learns that "the"
    /// often follows certain contexts, and once generated, "the" appears
    /// in the context making it even more likely → feedback loop.
    ///
    /// The penalty reduces the logit for any token that appeared in the
    /// recent context window:
    ///   - If logit > 0: logit /= penalty (makes it smaller, less likely)
    ///   - If logit < 0: logit *= penalty (makes it more negative, less likely)
    ///
    ///   penalty = 1.0 → disabled (no penalty)
    ///   penalty = 1.1 → mild penalty (standard production setting)
    ///   penalty = 1.5 → strong penalty (aggressively avoids repetition)
    pub repetition_penalty: f32,

    /// Random seed for reproducible sampling. Same seed → same output.
    /// This is critical for debugging and for educational demos where we
    /// want to show specific behaviors reliably.
    pub seed: u64,

    /// If true, print each token to stdout as it's generated (streaming).
    /// This mimics how ChatGPT displays text token-by-token.
    pub stream: bool,
}

impl SamplerConfig {
    /// Greedy decoding: always pick the most likely token.
    /// Equivalent to argmax — temperature 0 is handled as a special case.
    pub fn greedy() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 42,
            stream: false,
        }
    }

    /// A reasonable "production" config for story generation.
    pub fn default_sampling() -> Self {
        SamplerConfig {
            temperature: 0.8,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: 42,
            stream: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Sampler — applies the full sampling pipeline
// ---------------------------------------------------------------------------
// The sampler owns a PRNG state so that sampling is reproducible given the
// same seed. It applies each strategy in order, then draws a weighted
// random sample from the resulting distribution.
// ---------------------------------------------------------------------------

pub struct Sampler {
    /// PRNG state — a simple Linear Congruential Generator (LCG).
    /// The same algorithm used in tensor.rs for rand_init.
    ///
    /// LCG formula: state = state * a + c (mod 2^64)
    ///
    /// This is NOT cryptographically secure, but it's fast, reproducible,
    /// and perfectly fine for sampling from a probability distribution.
    /// Real inference engines use xoshiro256** or similar, but LCG is
    /// simplest to understand and implement.
    rng_state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Sampler { rng_state: seed }
    }

    /// Generate a random f32 in [0, 1).
    /// Uses the LCG pattern from tensor.rs: multiply by a large prime,
    /// add an offset, take the upper bits for quality.
    fn random_f32(&mut self) -> f32 {
        self.rng_state = self.rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use bits 33..63 (the upper bits are higher quality in an LCG)
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    // -----------------------------------------------------------------------
    // THE SAMPLING PIPELINE
    // -----------------------------------------------------------------------
    // This method applies all strategies in order and returns a token ID.
    //
    // Input:
    //   logits — raw output from the transformer's final layer [vocab_size]
    //   recent_tokens — tokens generated so far (for repetition penalty)
    //   config — all the knobs (temperature, top_k, top_p, etc.)
    //
    // The order matters:
    //   1. Repetition penalty modifies raw logits (before softmax)
    //   2. Temperature scales logits (before softmax)
    //   3. Softmax converts to probabilities
    //   4. Top-k zeros out unlikely tokens (after softmax)
    //   5. Top-p zeros out tokens beyond the cumulative threshold
    //   6. Renormalize and sample
    // -----------------------------------------------------------------------

    pub fn sample(&mut self, logits: &Tensor, recent_tokens: &[u32], config: &SamplerConfig) -> u32 {
        // Special case: temperature 0 means greedy (argmax).
        // No randomness at all — just pick the highest logit.
        if config.temperature == 0.0 {
            return logits.argmax() as u32;
        }

        // Work on a copy so we don't mutate the model's logit buffer.
        let mut probs = logits.data.clone();

        // --- Step 1: Repetition penalty ---
        // For each token that appeared recently, reduce its logit to make
        // the model less likely to repeat it.
        if config.repetition_penalty != 1.0 {
            apply_repetition_penalty(&mut probs, recent_tokens, config.repetition_penalty);
        }

        // --- Step 2: Temperature scaling ---
        // Divide all logits by temperature. This happens BEFORE softmax
        // because softmax(x/T) has a different shape than softmax(x).
        //
        // Math insight: softmax(x/T)_i = exp(x_i/T) / sum(exp(x_j/T))
        // When T < 1, the exponents are magnified → winner-take-all
        // When T > 1, the exponents are compressed → more uniform
        apply_temperature(&mut probs, config.temperature);

        // --- Step 3: Softmax ---
        // Convert scaled logits to a proper probability distribution.
        softmax_vec(&mut probs);

        // --- Step 4: Top-k filtering ---
        // Zero out all but the k highest-probability tokens.
        if config.top_k > 0 {
            apply_top_k(&mut probs, config.top_k);
        }

        // --- Step 5: Top-p (nucleus) filtering ---
        // Zero out tokens beyond the cumulative probability threshold.
        if config.top_p < 1.0 {
            apply_top_p(&mut probs, config.top_p);
        }

        // --- Step 6: Renormalize and sample ---
        // After filtering, probabilities may not sum to 1. Renormalize,
        // then draw a weighted random sample.
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        self.weighted_random_sample(&probs)
    }

    /// Draw a random token ID from a probability distribution.
    ///
    /// HOW: Generate a random number r in [0, 1), then walk through the
    /// distribution accumulating probability. When the cumulative sum
    /// exceeds r, return that token. Tokens with higher probability
    /// occupy more of the [0, 1) interval, so they're more likely to
    /// be selected.
    ///
    /// This is the "inverse CDF" sampling method — simple and correct,
    /// though O(vocab_size). Real engines use more sophisticated methods
    /// for huge vocabularies, but for 32k tokens this is fine.
    fn weighted_random_sample(&mut self, probs: &[f32]) -> u32 {
        let r = self.random_f32();
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative > r {
                return i as u32;
            }
        }
        // Fallback: return last token (can happen due to float rounding)
        (probs.len() - 1) as u32
    }
}

// ---------------------------------------------------------------------------
// Individual sampling strategies (free functions for clarity)
// ---------------------------------------------------------------------------

/// Apply repetition penalty to logits for tokens that appeared recently.
///
/// The penalty is asymmetric:
///   - Positive logits are DIVIDED by the penalty (making them smaller)
///   - Negative logits are MULTIPLIED by the penalty (making them more negative)
///
/// In both cases, the token becomes LESS likely after softmax.
///
/// WHY asymmetric? If we just divided all logits by the penalty, negative
/// logits would move TOWARD zero (becoming MORE likely), which is the
/// opposite of what we want. The sign-aware approach ensures the penalty
/// always pushes the token's probability DOWN.
fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
    // PERF: Use a HashSet for O(1) dedup lookup instead of potentially penalizing
    // the same token index multiple times when it appears repeatedly in recent_tokens.
    // For large context windows this also avoids redundant penalty applications.
    let penalized: std::collections::HashSet<usize> = recent_tokens
        .iter()
        .map(|&t| t as usize)
        .filter(|&idx| idx < logits.len())
        .collect();
    for idx in penalized {
        if logits[idx] > 0.0 {
            logits[idx] /= penalty;
        } else {
            logits[idx] *= penalty;
        }
    }
}

/// Divide all logits by temperature.
///
/// This is the simplest but most impactful sampling strategy. It directly
/// controls the entropy (information content / uncertainty) of the output
/// distribution:
///
///   H(softmax(x/T)) increases monotonically with T
///
/// Low temperature → low entropy → deterministic
/// High temperature → high entropy → random
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }
}

/// In-place softmax on a Vec<f32>.
/// Same subtract-max trick as Tensor::softmax() for numerical stability.
fn softmax_vec(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    // PERF: Precompute 1/sum and multiply instead of dividing in the loop.
    // Division has ~4x higher latency than multiplication on modern CPUs.
    let inv_sum = 1.0 / sum;
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }
}

/// Top-k filtering: keep only the k highest-probability tokens.
///
/// Algorithm:
///   1. Find the k-th highest probability (the threshold)
///   2. Zero out everything below the threshold
///
/// PERF: Uses select_nth_unstable (introselect) to find the k-th largest
/// element in O(n) average time, instead of a full O(n log n) sort. For
/// a vocab of 32k tokens, this reduces work by ~15x (log2(32k) ~ 15).
/// select_nth_unstable partially reorders the slice so that the element
/// at index k is in its final sorted position, elements before it are
/// all >= it, and elements after are all <= it. We only need the value
/// at position k as a threshold, not a fully sorted array.
fn apply_top_k(probs: &mut [f32], k: usize) {
    if k >= probs.len() {
        return; // k larger than vocab — no filtering needed
    }

    // PERF: Clone into a working buffer and use select_nth_unstable to find
    // the k-th largest value in O(n) average time. We sort descending, so
    // index k holds the (k+1)-th largest value, which is our cutoff threshold.
    let mut buf: Vec<f32> = probs.iter().copied().collect();
    // select_nth_unstable_by partitions so that buf[k] is the element that
    // would be at index k in a fully descending sort.
    buf.select_nth_unstable_by(k, |a, b| b.partial_cmp(a).unwrap());
    let threshold = buf[k];

    // Zero out everything below the threshold
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

/// Top-p (nucleus) sampling: keep tokens until cumulative probability > p.
///
/// Algorithm:
///   1. Create (index, probability) pairs and sort by probability descending
///   2. Walk through sorted list, accumulating probability
///   3. Once cumulative prob exceeds p, zero out all remaining tokens
///
/// The "nucleus" is the smallest set of tokens whose combined probability
/// mass exceeds p. This adapts to the model's confidence:
///   - Confident predictions (one token dominates) → small nucleus
///   - Uncertain predictions (flat distribution) → large nucleus
fn apply_top_p(probs: &mut [f32], p: f32) {
    // Create sorted (index, prob) pairs
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find where cumulative probability exceeds p
    let mut cumulative = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (rank, &(_, prob)) in indexed.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            cutoff_idx = rank + 1; // keep this one (it pushed us over), cut the rest
            break;
        }
    }

    // Zero out tokens beyond the nucleus
    for &(idx, _) in &indexed[cutoff_idx..] {
        probs[idx] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = Tensor::from_vec(&[1.0, 5.0, 3.0, 2.0]);
        let config = SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 42, stream: false };
        let mut sampler = Sampler::new(config.seed);
        let token = sampler.sample(&logits, &[], &config);
        assert_eq!(token, 1, "temperature 0 should pick argmax (index 1)");
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let logits = Tensor::from_vec(&[1.0, 2.0, 3.0, 2.0, 1.0]);
        let config = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 123, stream: false };

        let mut sampler1 = Sampler::new(config.seed);
        let mut sampler2 = Sampler::new(config.seed);

        let t1 = sampler1.sample(&logits, &[], &config);
        let t2 = sampler2.sample(&logits, &[], &config);
        assert_eq!(t1, t2, "same seed should produce same result");
    }

    #[test]
    fn test_top_k_filters() {
        let mut probs = vec![0.1, 0.5, 0.2, 0.15, 0.05];
        apply_top_k(&mut probs, 2);
        // Only the top 2 (0.5 and 0.2) should remain
        assert_eq!(probs[4], 0.0); // 0.05 zeroed
        assert_eq!(probs[0], 0.0); // 0.1 zeroed
        assert!(probs[1] > 0.0);   // 0.5 kept
        assert!(probs[2] > 0.0);   // 0.2 kept
    }

    #[test]
    fn test_top_p_filters() {
        let mut probs = vec![0.05, 0.7, 0.15, 0.08, 0.02];
        apply_top_p(&mut probs, 0.9);
        // Sorted: 0.7, 0.15, 0.08, 0.05, 0.02
        // Cumulative: 0.7, 0.85, 0.93 → keep first 3 (0.7 + 0.15 + 0.08 = 0.93 > 0.9)
        assert!(probs[1] > 0.0);  // 0.7 kept
        assert!(probs[2] > 0.0);  // 0.15 kept
        assert!(probs[3] > 0.0);  // 0.08 kept
        assert_eq!(probs[0], 0.0); // 0.05 zeroed
        assert_eq!(probs[4], 0.0); // 0.02 zeroed
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0, 3.0, -1.0, 1.0];
        apply_repetition_penalty(&mut logits, &[1, 2], 2.0);
        assert!((logits[0] - 2.0).abs() < 1e-6); // not penalized
        assert!((logits[1] - 1.5).abs() < 1e-6); // 3.0 / 2.0
        assert!((logits[2] - (-2.0)).abs() < 1e-6); // -1.0 * 2.0
        assert!((logits[3] - 1.0).abs() < 1e-6); // not penalized
    }
}
