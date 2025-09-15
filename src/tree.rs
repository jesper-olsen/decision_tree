use crate::data::{Sample, SampleValue, Vocabulary};
use crate::node::{Counter, Node, Summary};
use std::collections::HashMap;

pub struct DecisionTree<'a> {
    pub root: Node,
    pub header: Vec<String>,
    pub vocab: &'a Vocabulary, // TODO: owned?
}

impl<'a> DecisionTree<'a> {
    pub fn new(root: Node, header: Vec<String>, vocab: &'a Vocabulary) -> Self {
        DecisionTree {
            root,
            header,
            vocab,
        }
    }

    pub fn size(&self) -> usize {
        self.root.size()
    }

    fn eval_fn(criterion: &str) -> Box<dyn Fn(&Counter) -> f64> {
        match criterion {
            "entropy" => Box::new(entropy),
            "gini" => Box::new(gini),
            _ => panic!("Unknown criterion: {criterion}"),
        }
    }

    pub fn train(
        data: Vec<Sample>,
        header: Vec<String>,
        vocab: &'a Vocabulary,
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
    ) -> Self {
        let eval_fn = Self::eval_fn(criterion);
        let row_indices: Vec<usize> = (0..data.len()).collect();
        let root = Self::grow_tree(
            &data,
            &row_indices,
            min_samples_split,
            max_depth,
            eval_fn.as_ref(),
            0,
        );
        DecisionTree::new(root, header, vocab)
    }

    fn grow_tree(
        all_data: &[Sample],
        row_indices: &[usize],
        min_samples_split: usize,
        max_depth: Option<usize>,
        criterion: &dyn Fn(&Counter) -> f64,
        depth: usize,
    ) -> Node {
        if row_indices.is_empty() {
            // The only way this can happen is if the initial dataset was empty.
            panic!("grow_tree cannot be called with an empty set of row indices.");
        }

        let current_counts = count_classes(all_data, row_indices);
        let current_score = criterion(&current_counts);

        // Pre-pruning
        let summary = Summary {
            impurity: current_score,
            samples: row_indices.len(),
        };

        if (max_depth.is_some() && depth >= max_depth.unwrap())
            || (row_indices.len() < min_samples_split)
        {
            return Node::leaf(current_counts, summary);
        }

        // find best gain
        let mut best_gain = 0.0;
        let mut best_rule: Option<(usize, SampleValue)> = None;
        let mut best_sets: Option<(Vec<usize>, Vec<usize>)> = None;

        let column_count = all_data[0].len() - 1;
        for col in 0..column_count {
            let mut column_values = std::collections::HashSet::new();
            for &row_idx in row_indices {
                column_values.insert(all_data[row_idx][col].clone());
            }

            for value in column_values {
                if matches!(value, SampleValue::None) {
                    continue; // don't split on a missing value 
                }

                let (set1_indices, set2_indices) = split_set(all_data, row_indices, col, &value);

                if set1_indices.is_empty() || set2_indices.is_empty() {
                    continue;
                }

                let p = set1_indices.len() as f64 / row_indices.len() as f64;

                let gain = current_score
                    - p * criterion(&count_classes(all_data, &set1_indices))
                    - (1.0 - p) * criterion(&count_classes(all_data, &set2_indices));

                if gain > best_gain {
                    best_gain = gain;
                    best_rule = Some((col, value));
                    best_sets = Some((set1_indices, set2_indices));
                }
            }
        }

        if best_gain > 0.0 {
            let (col, value) = best_rule.unwrap();
            let (set1_indices, set2_indices) = best_sets.unwrap();

            let true_branch = Box::new(Self::grow_tree(
                all_data,
                &set1_indices,
                min_samples_split,
                max_depth,
                criterion,
                depth + 1,
            ));

            let false_branch = Box::new(Self::grow_tree(
                all_data,
                &set2_indices,
                min_samples_split,
                max_depth,
                criterion,
                depth + 1,
            ));

            Node::internal(col, value, true_branch, false_branch, summary)
        } else {
            Node::leaf(current_counts, summary)
        }
    }

    // Simple prediction - returns the class label
    pub fn predict(&self, sample: &Sample) -> usize {
        self.root.predict(sample)
    }

    // Get class distribution without cloning
    pub fn predict_proba(&self, sample: &Sample) -> &Counter {
        self.root.get_leaf_counts(sample)
    }

    /// Classifies a sample, returning a map of class IDs to their scores/counts.
    /// Handles missing data by calculating weighted scores and rounding them to the
    /// nearest integer count.
    pub fn classify(&self, sample: &Sample, handle_missing: bool) -> Counter {
        if handle_missing {
            let weighted_counts = self.root.classify_with_missing_data(sample);

            weighted_counts
                .into_iter()
                .map(|(k, v)| (k, v.round() as usize))
                .collect()
        } else {
            self.root.get_leaf_counts(sample).clone()
        }
    }

    pub fn prune(&mut self, min_gain: f64, criterion: &str, notify: bool) {
        let eval_fn = Self::eval_fn(criterion);
        self.root.prune(min_gain, eval_fn.as_ref(), notify);
    }
}

impl<'a> std::fmt::Display for DecisionTree<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.root.to_string(Some(&self.header), "", self.vocab)
        )
    }
}

// Helper functions

fn split_set(
    all_data: &[Sample],
    row_indices: &[usize],
    column: usize,
    value: &SampleValue,
) -> (Vec<usize>, Vec<usize>) {
    let mut set1_indices = Vec::new();
    let mut set2_indices = Vec::new();

    for &row_idx in row_indices {
        let row = &all_data[row_idx];
        let v = &row[column];

        if matches!(v, SampleValue::None) {
            // missing value - add row to both sets
            set1_indices.push(row_idx);
            set2_indices.push(row_idx);
        } else if matches!(value, SampleValue::Numeric(_)) {
            if v.ge(value) {
                set1_indices.push(row_idx);
            } else {
                set2_indices.push(row_idx);
            }
        } else if v.eq(value) {
            set1_indices.push(row_idx);
        } else {
            set2_indices.push(row_idx);
        }
    }
    (set1_indices, set2_indices)
}

fn count_classes(all_data: &[Sample], row_indices: &[usize]) -> Counter {
    let mut counts = HashMap::new();
    for &row_idx in row_indices {
        let row = &all_data[row_idx];
        if let Some(SampleValue::String(id)) = row.last() {
            *counts.entry(*id).or_insert(0) += 1;
        }
    }
    counts
}

fn entropy(counts: &Counter) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f64 = total as f64;
    -counts
        .values()
        .map(|&c| {
            let p = c as f64 / total_f64;
            p * p.log2()
        })
        .sum::<f64>()
}

fn gini(counts: &Counter) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f64 = total as f64;
    1.0 - counts
        .values()
        .map(|&c| {
            let p = c as f64 / total_f64;
            p * p
        })
        .sum::<f64>()
}
