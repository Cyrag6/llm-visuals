use std::collections::{HashMap, VecDeque};

/// Collects generated token text for display
#[derive(Debug, Default)]
pub struct GeneratedText {
    pub tokens: Vec<String>,
}

impl GeneratedText {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a token's text, keeping the last 200 chars max
    pub fn push(&mut self, index: usize, text: String) {
        if index < self.tokens.len() {
            self.tokens[index] = text;
        } else {
            self.tokens.push(text);
        }
    }

    /// Last 200 Unicode scalars of generated text (full string if a single status blob).
    pub fn to_string(&self) -> String {
        let full: String = self.tokens.iter().map(|t| t.as_str()).collect();
        if self.tokens.len() == 1 && self.tokens[0].contains('\n') {
            return full;
        }
        let mut char_count = 0;
        for (i, _) in full.char_indices().rev() {
            char_count += 1;
            if char_count == 200 {
                return full[i..].to_string();
            }
        }
        full
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Represents a single attention weight from one head at one query position
#[derive(Debug, Clone)]
pub struct AttentionWeight {
    pub layer: usize,
    pub head: usize,
    pub query_pos: usize,
    #[allow(dead_code)]
    pub key_pos: usize,
    /// Normalized weight value [0.0, 1.0]
    pub weight: f32,
}

/// Aggregated activity for one head at one token position
#[derive(Debug, Clone, Copy)]
pub struct HeadActivity {
    pub layer: usize,
    pub head: usize,
    pub intensity: f32,
}

/// A column of head activities for one token position across all layers/heads
#[derive(Debug, Clone)]
pub struct TokenColumn {
    #[allow(dead_code)]
    pub token_index: usize,
    pub activities: Vec<HeadActivity>,
}

/// Circular buffer that stores recent token columns for rendering
#[derive(Debug)]
pub struct TokenBuffer {
    capacity: usize,
    data: VecDeque<TokenColumn>,
}

impl TokenBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, column: TokenColumn) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(column);
    }

    pub fn iter(&self) -> impl Iterator<Item = &TokenColumn> {
        self.data.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Collects attention weights per query position and builds TokenColumns
#[derive(Debug, Default)]
pub struct ActivityAggregator {
    /// query_pos -> (layer, head) -> max weight
    pending: HashMap<usize, HashMap<(usize, usize), f32>>,
}

impl ActivityAggregator {
    pub fn new(_num_layers: usize, _num_heads: usize) -> Self {
        Self::default()
    }

    pub fn process(&mut self, weight: AttentionWeight) {
        let entry = self
            .pending
            .entry(weight.query_pos)
            .or_default()
            .entry((weight.layer, weight.head))
            .or_insert(0.0);
        if weight.weight > *entry {
            *entry = weight.weight;
        }
    }

    /// Finalize a query position into a TokenColumn (removes it from pending)
    pub fn finalize(&mut self, query_pos: usize) -> Option<TokenColumn> {
        let map = self.pending.remove(&query_pos)?;
        if map.is_empty() {
            return None;
        }
        let mut activities: Vec<HeadActivity> = map
            .into_iter()
            .map(|((layer, head), intensity)| HeadActivity {
                layer,
                head,
                intensity,
            })
            .collect();
        activities.sort_by_key(|a| (a.layer, a.head));
        Some(TokenColumn {
            token_index: query_pos,
            activities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_text_truncates_on_char_boundary() {
        let mut t = GeneratedText::new();
        t.push(0, "é".repeat(250));
        let s = t.to_string();
        assert_eq!(s.chars().count(), 200);
        assert!(s.is_char_boundary(0));
        assert!(s.is_char_boundary(s.len()));
    }

    #[test]
    fn aggregator_finalize_builds_dense_column() {
        let mut agg = ActivityAggregator::new(2, 2);
        for layer in 0..2 {
            for head in 0..2 {
                agg.process(AttentionWeight {
                    layer,
                    head,
                    query_pos: 3,
                    key_pos: 0,
                    weight: 0.4 + head as f32 * 0.1,
                });
            }
        }
        let col = agg.finalize(3).expect("column");
        assert_eq!(col.activities.len(), 4);
        assert!(col.activities.iter().all(|a| a.intensity >= 0.4));
    }

    #[test]
    fn token_buffer_is_bounded() {
        let mut b = TokenBuffer::new(2);
        for i in 0..5 {
            b.push(TokenColumn {
                token_index: i,
                activities: vec![],
            });
        }
        assert_eq!(b.iter().count(), 2);
        assert_eq!(b.iter().next().unwrap().token_index, 3);
    }
}
