// ============================================================================
// Deep Dive: The Geometry of Meaning
// ============================================================================
//
// This lesson explores TOKEN EMBEDDINGS — the foundation of how LLMs
// understand language. Every token in the vocabulary has a learned vector
// (an embedding) that represents its "meaning" in a high-dimensional space.
//
// The key insight: meaning is encoded as GEOMETRY. Words with similar
// meanings cluster together. Relationships between words become directions
// in the space. The famous example: king - man + woman ≈ queen.
//
// We'll explore:
//   1. Cosine similarity between embeddings (which words are "close"?)
//   2. What individual dimensions mean (spoiler: not much individually)
//   3. How RoPE (Rotary Positional Embedding) changes vectors at different positions
//
// ============================================================================

use llm_lab::model::{Config, TransformerWeights};
use llm_lab::tokenizer::Tokenizer;
use llm_lab::tensor::Tensor;

fn main() {
    println!("================================================================");
    println!("   Deep Dive: The Geometry of Meaning");
    println!("   Exploring Token Embeddings in a Trained LLM");
    println!("================================================================\n");

    // -----------------------------------------------------------------------
    // Load the model and tokenizer
    // -----------------------------------------------------------------------
    let model_path = "data/stories15M.bin";
    let tokenizer_path = "data/tokenizer.bin";

    let (config, shared_weights) = Config::from_file(model_path)
        .expect("Failed to load model config");
    let weights = TransformerWeights::from_file(model_path, &config, shared_weights)
        .expect("Failed to load model weights");
    let tokenizer = Tokenizer::from_file(tokenizer_path, config.vocab_size)
        .expect("Failed to load tokenizer");

    println!("Model loaded: dim={}, vocab_size={}, n_layers={}\n",
        config.dim, config.vocab_size, config.n_layers);

    // ===================================================================
    // SECTION 1: Cosine Similarity — The Metric of Meaning
    // ===================================================================
    //
    // Cosine similarity measures the angle between two vectors:
    //   cos(a, b) = dot(a, b) / (||a|| * ||b||)
    //
    // Value ranges from -1 to +1:
    //   +1 = identical direction (same meaning)
    //    0 = orthogonal (unrelated)
    //   -1 = opposite direction
    //
    // In practice, trained embeddings rarely go below 0 for unrelated
    // words. Values above 0.5 indicate strong similarity.
    //
    // WHY COSINE AND NOT EUCLIDEAN DISTANCE?
    // Embeddings can have wildly different magnitudes. A common word
    // like "the" might have a large vector simply because it was updated
    // more during training. Cosine similarity ignores magnitude and
    // focuses on DIRECTION — which is where meaning lives.
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 1: Cosine Similarity Between Token Embeddings");
    println!("================================================================\n");

    // Helper: look up a word's embedding and token ID
    let find_token = |word: &str| -> Option<(u32, String)> {
        let tokens = tokenizer.encode(word);
        if tokens.len() == 1 {
            Some((tokens[0], tokenizer.token_str(tokens[0]).to_string()))
        } else {
            // Try with leading space (sentencepiece convention)
            let spaced = format!(" {}", word);  // leading space
            let tokens2 = tokenizer.encode(&spaced);
            // The first token might be the space-prefixed version
            if tokens2.len() == 1 {
                Some((tokens2[0], tokenizer.token_str(tokens2[0]).to_string()))
            } else if !tokens.is_empty() {
                // Fall back to first subword token
                Some((tokens[0], tokenizer.token_str(tokens[0]).to_string()))
            } else {
                None
            }
        }
    };

    let get_embedding = |token_id: u32| -> Tensor {
        weights.token_embedding.row(token_id as usize)
    };

    // Compare pairs of words — semantically similar vs dissimilar
    let compare_words = |w1: &str, w2: &str| {
        if let (Some((id1, tok1)), Some((id2, tok2))) = (find_token(w1), find_token(w2)) {
            let emb1 = get_embedding(id1);
            let emb2 = get_embedding(id2);
            let sim = emb1.cosine_similarity(&emb2);
            println!("  {:>12} (token {:>5} '{}') vs {:>12} (token {:>5} '{}')  =>  cosine = {:.4}",
                w1, id1, tok1.trim(), w2, id2, tok2.trim(), sim);
        } else {
            println!("  Could not find tokens for '{}' / '{}'", w1, w2);
        }
    };

    println!("Comparing semantically SIMILAR words:");
    println!("  (These should have higher cosine similarity)\n");
    compare_words("cat", "dog");
    compare_words("boy", "girl");
    compare_words("happy", "glad");
    compare_words("big", "large");
    compare_words("run", "walk");
    compare_words("mother", "father");

    println!("\nComparing semantically DIFFERENT words:");
    println!("  (These should have lower cosine similarity)\n");
    compare_words("cat", "mountain");
    compare_words("happy", "table");
    compare_words("run", "blue");
    compare_words("mother", "stone");

    println!("\nComparing RELATED but different part-of-speech words:");
    println!("  (These often have moderate similarity)\n");
    compare_words("happy", "happiness");
    compare_words("run", "running");
    compare_words("big", "bigger");

    // -----------------------------------------------------------------------
    // Self-similarity sanity check
    // -----------------------------------------------------------------------
    println!("\nSanity check — a word compared to itself should be exactly 1.0:");
    if let Some((id, _)) = find_token("cat") {
        let emb = get_embedding(id);
        println!("  cosine(cat, cat) = {:.6}", emb.cosine_similarity(&emb));
    }

    // ===================================================================
    // SECTION 2: What Do Individual Dimensions Mean?
    // ===================================================================
    //
    // Short answer: individual dimensions are NOT interpretable.
    //
    // Unlike a hand-crafted feature vector where dim 0 might be "is_animal"
    // and dim 1 might be "is_positive", neural network embeddings spread
    // meaning across ALL dimensions simultaneously. This is called a
    // DISTRIBUTED REPRESENTATION.
    //
    // The meaning is in the DIRECTION (combination of dimensions), not in
    // any single dimension. This is why cosine similarity works — it
    // measures the overall directional alignment.
    //
    // Let's look at raw dimension values to see this in action.
    // ===================================================================

    println!("\n================================================================");
    println!("  SECTION 2: What Do Individual Dimensions Mean?");
    println!("================================================================\n");

    let words_to_inspect = ["cat", "dog", "the", "happy", "big"];
    let n_dims_to_show = 10;

    println!("First {} dimensions of several word embeddings:", n_dims_to_show);
    println!("  (Notice: no single dimension has an obvious 'meaning')\n");

    // Print header
    print!("  {:>10}", "dim ->");
    for d in 0..n_dims_to_show {
        print!("  {:>7}", d);
    }
    println!();
    println!("  {}", "-".repeat(10 + n_dims_to_show * 9));

    for word in &words_to_inspect {
        if let Some((id, _)) = find_token(word) {
            let emb = get_embedding(id);
            print!("  {:>10}", word);
            for d in 0..n_dims_to_show {
                print!("  {:>7.3}", emb.data[d]);
            }
            println!();
        }
    }

    println!("\n  KEY INSIGHT: There is no dimension you can point to and say");
    println!("  'this is the animal dimension' or 'this is the size dimension'.");
    println!("  Meaning is distributed across ALL {} dimensions.", config.dim);
    println!("  It is the DIRECTION of the vector (the combination of dimensions)");
    println!("  that encodes meaning, not any individual dimension.");

    // -----------------------------------------------------------------------
    // Show embedding statistics
    // -----------------------------------------------------------------------
    println!("\nEmbedding statistics (distribution of values):\n");

    for word in &["cat", "the", "happy"] {
        if let Some((id, _)) = find_token(word) {
            let emb = get_embedding(id);
            let mean: f32 = emb.data.iter().sum::<f32>() / emb.data.len() as f32;
            let variance: f32 = emb.data.iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f32>() / emb.data.len() as f32;
            let magnitude: f32 = emb.data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let min = emb.data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = emb.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            println!("  '{}': mean={:.4}, std={:.4}, magnitude={:.4}, range=[{:.4}, {:.4}]",
                word, mean, variance.sqrt(), magnitude, min, max);
        }
    }

    // ===================================================================
    // SECTION 3: Nearest Neighbors in Embedding Space
    // ===================================================================
    //
    // For a given word, which other words are closest in the embedding
    // space? This reveals what the model has learned about word relationships.
    //
    // Note: This is O(vocab_size) per query — we compute cosine similarity
    // against every token. Real systems use approximate nearest neighbor
    // (ANN) algorithms like HNSW or IVF for speed.
    // ===================================================================

    println!("\n================================================================");
    println!("  SECTION 3: Nearest Neighbors in Embedding Space");
    println!("================================================================\n");

    // REVIEW: Compute cosine similarity directly from the embedding table's raw
    // data using slice math, avoiding ~32k Tensor allocations per query word.
    // The original called get_embedding() (which allocates via row()) for every
    // vocab token. Here we access the flat embedding data with slice indexing.
    let dim = config.dim;
    let emb_data = &weights.token_embedding.data;

    let cosine_sim_raw = |a_offset: usize, b_offset: usize| -> f32 {
        let (dot, sq_a, sq_b) = (0..dim).fold(
            (0.0f32, 0.0f32, 0.0f32),
            |(d, sa, sb), j| {
                let a = emb_data[a_offset + j];
                let b = emb_data[b_offset + j];
                (d + a * b, sa + a * a, sb + b * b)
            },
        );
        let denom = sq_a.sqrt() * sq_b.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    };

    let find_nearest = |word: &str, top_n: usize| {
        if let Some((id, _)) = find_token(word) {
            let query_offset = id as usize * dim;

            // Compute cosine similarity against all tokens
            let mut similarities: Vec<(u32, f32)> = (0..config.vocab_size as u32)
                .filter(|&i| i != id)           // exclude the word itself
                .filter(|&i| {
                    // Skip byte tokens, special tokens, and single-char tokens
                    // to get more meaningful results
                    let s = tokenizer.token_str(i);
                    s.len() > 1 && !s.starts_with('<') && !s.starts_with('\0')
                })
                .map(|i| {
                    let other_offset = i as usize * dim;
                    (i, cosine_sim_raw(query_offset, other_offset))
                })
                .collect();

            // Sort by similarity (descending)
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("  Nearest neighbors to '{}' (token {}):", word, id);
            for (i, (tok_id, sim)) in similarities.iter().take(top_n).enumerate() {
                let tok_str = tokenizer.token_str(*tok_id);
                println!("    {}. '{}' (token {}) — cosine = {:.4}",
                    i + 1, tok_str.trim(), tok_id, sim);
            }

            // Also show the FARTHEST neighbors (most dissimilar)
            println!("  Farthest neighbors (most dissimilar):");
            for (i, (tok_id, sim)) in similarities.iter().rev().take(3).enumerate() {
                let tok_str = tokenizer.token_str(*tok_id);
                println!("    {}. '{}' (token {}) — cosine = {:.4}",
                    i + 1, tok_str.trim(), tok_id, sim);
            }
            println!();
        }
    };

    find_nearest("cat", 10);
    find_nearest("king", 10);
    find_nearest("happy", 10);

    // ===================================================================
    // SECTION 4: RoPE — How Position Changes the Embedding
    // ===================================================================
    //
    // RoPE (Rotary Positional Embedding) modifies the query and key vectors
    // by ROTATING pairs of dimensions based on position. This is how the
    // model knows WHERE a token is in the sequence.
    //
    // Key properties of RoPE:
    //   1. It's a ROTATION — the vector magnitude stays the same
    //   2. The rotation angle depends on both position and dimension index
    //   3. Low-frequency dimensions encode long-range position info
    //   4. High-frequency dimensions encode fine-grained position info
    //   5. The dot product Q.K depends only on RELATIVE position (m-n)
    //
    // Let's visualize how RoPE transforms the same vector at different
    // positions.
    // ===================================================================

    println!("================================================================");
    println!("  SECTION 4: RoPE — How Position Changes the Embedding");
    println!("================================================================\n");

    let head_dim = config.head_dim();
    println!("Head dimension: {} (we rotate pairs of dimensions, so {} rotation planes)\n",
        head_dim, head_dim / 2);

    // Create a simple test vector and rotate it at different positions
    let base_vec: Vec<f32> = (0..head_dim).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();

    println!("Starting vector (alternating 1, 0 pairs):");
    print!("  [");
    for (i, v) in base_vec.iter().enumerate().take(12) {
        if i > 0 { print!(", "); }
        print!("{:.1}", v);
    }
    println!(", ...]\n");

    println!("After RoPE rotation at different positions (first 6 dims):");
    println!("  (Notice: larger positions => more rotation)\n");

    let positions = [0, 1, 5, 10, 50, 100];
    print!("  {:>8}", "pos");
    for d in 0..6 {
        print!("  {:>8}", format!("dim{}", d));
    }
    print!("  magnitude");
    println!();
    println!("  {}", "-".repeat(8 + 6 * 10 + 12));

    for &pos in &positions {
        // Apply RoPE manually (same formula as model.rs apply_rope)
        let mut rotated = base_vec.clone();
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / (10000.0f32).powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let v0 = rotated[i];
            let v1 = rotated[i + 1];
            rotated[i]     = v0 * cos - v1 * sin;
            rotated[i + 1] = v0 * sin + v1 * cos;
        }

        let magnitude: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        print!("  {:>8}", pos);
        for d in 0..6 {
            print!("  {:>8.4}", rotated[d]);
        }
        print!("  {:>8.4}", magnitude);
        println!();
    }

    println!("\n  KEY INSIGHT: The magnitude is PRESERVED (it's always {:.1}).", (head_dim as f32 / 2.0).sqrt());
    println!("  Only the direction changes. Low-indexed dimension pairs rotate");
    println!("  faster (high frequency) while high-indexed pairs rotate slowly");
    println!("  (low frequency). This multi-frequency encoding lets the model");
    println!("  distinguish positions at multiple scales simultaneously.");

    // -----------------------------------------------------------------------
    // Show how RoPE affects the dot product at different relative positions
    // -----------------------------------------------------------------------
    println!("\n  RoPE Relative Position Encoding:");
    println!("  The dot product of two RoPE'd vectors depends only on their");
    println!("  RELATIVE position (m - n), not absolute positions.\n");

    let compute_rope = |v: &[f32], pos: usize| -> Vec<f32> {
        let mut rotated = v.to_vec();
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / (10000.0f32).powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let v0 = rotated[i];
            let v1 = rotated[i + 1];
            rotated[i]     = v0 * cos - v1 * sin;
            rotated[i + 1] = v0 * sin + v1 * cos;
        }
        rotated
    };

    let q_base: Vec<f32> = (0..head_dim).map(|i| ((i + 1) as f32) * 0.1).collect();
    let k_base: Vec<f32> = (0..head_dim).map(|i| ((i + 2) as f32) * 0.1).collect();

    println!("  dot(Q@pos_m, K@pos_n) for various (m, n) with same relative distance:\n");
    print!("  {:>10} {:>10} {:>10} {:>12}", "q_pos (m)", "k_pos (n)", "m - n", "dot product");
    println!();
    println!("  {}", "-".repeat(46));

    let pairs = [(5, 3), (10, 8), (50, 48), (100, 98)];  // all have m-n = 2
    for &(m, n) in &pairs {
        let q_rotated = compute_rope(&q_base, m);
        let k_rotated = compute_rope(&k_base, n);
        let dot: f32 = q_rotated.iter().zip(&k_rotated).map(|(a, b)| a * b).sum();
        println!("  {:>10} {:>10} {:>10} {:>12.4}", m, n, m as i32 - n as i32, dot);
    }
    println!("\n  Notice: all pairs with the same relative distance (m-n=2) give");
    println!("  approximately the same dot product. This is the key property of");
    println!("  RoPE — it encodes RELATIVE positions, which is what attention cares about.");

    println!("\n================================================================");
    println!("  END: The Geometry of Meaning");
    println!("================================================================");
}
