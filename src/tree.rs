use crate::data::{ColumnType, DatasetMetadata, Sample, SampleValue};
use crate::error::TreeError;
use crate::node::{Counter, Node, Summary};
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Clone, Copy, Debug)]
pub enum Criterion {
    Entropy,
    Gini,
}

impl FromStr for Criterion {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "entropy" => Ok(Criterion::Entropy),
            "gini" => Ok(Criterion::Gini),
            _ => Err(format!(
                "Unknown criterion: '{s}'. Valid options: entropy, gini"
            )),
        }
    }
}

pub struct DecisionTree<'a> {
    pub root: Node,
    pub meta: &'a DatasetMetadata,
}

impl<'a> DecisionTree<'a> {
    pub fn new(root: Node, meta: &'a DatasetMetadata) -> Self {
        DecisionTree { root, meta }
    }

    pub fn size(&self) -> usize {
        self.root.size()
    }

    fn eval_fn(criterion: Criterion) -> Box<dyn Fn(&Counter) -> f64> {
        match criterion {
            Criterion::Entropy => Box::new(entropy),
            Criterion::Gini => Box::new(gini),
        }
    }

    pub fn train(
        data: &'a [Sample],
        meta: &'a DatasetMetadata,
        criterion: Criterion,
        max_depth: Option<usize>,
        min_samples_split: usize,
    ) -> Result<Self, TreeError> {
        let eval_fn = Self::eval_fn(criterion);
        let row_indices: Vec<usize> = (0..data.len()).collect();

        let root = Self::grow_tree(
            data,
            meta,
            &row_indices,
            min_samples_split,
            max_depth,
            eval_fn.as_ref(),
            0,
        )?;

        Ok(DecisionTree::new(root, meta))
    }

    // specialisation of find_best_split - avoids 'redundant' splits
    fn find_best_numeric_split(
        current_score: f64,
        all_data: &[Sample],
        row_indices: &[usize],
        col: usize,
        criterion: &dyn Fn(&Counter) -> f64,
        target_idx: usize,
    ) -> Option<(f64, SampleValue, Vec<usize>, Vec<usize>)> {
        // Collect non-missing numeric values with their class labels and row indices
        let mut value_label_idx: Vec<(f64, usize, usize)> = Vec::new();
        let mut missing_indices: Vec<usize> = Vec::new();

        for &row_idx in row_indices {
            let label_value = all_data[row_idx].get(target_idx);
            match (&all_data[row_idx][col], label_value) {
                (SampleValue::Numeric(val), Some(SampleValue::String(label))) => {
                    value_label_idx.push((*val, *label, row_idx));
                }
                (SampleValue::None, _) => {
                    missing_indices.push(row_idx);
                }
                _ => {} // Skip non-numeric or rows without proper labels
            }
        }

        if value_label_idx.len() < 2 {
            return None;
        }

        // Sort by numeric value ascending
        value_label_idx
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Initialize class counts for the single-pass algorithm
        let mut left_counts: Counter = HashMap::new();
        let mut right_counts: Counter = HashMap::new();

        // Start with all non-missing samples in the right partition
        for &(_, label, _) in &value_label_idx {
            *right_counts.entry(label).or_insert(0) += 1;
        }

        let mut best_gain = 0.0;
        let mut best_split_value = 0.0;

        // Single pass through sorted data
        for i in 0..(value_label_idx.len() - 1) {
            let (val, label, _) = value_label_idx[i];
            let (next_val, _, _) = value_label_idx[i + 1];

            // Move current sample from right to left
            *left_counts.entry(label).or_insert(0) += 1;
            if let Some(right_count) = right_counts.get_mut(&label) {
                *right_count -= 1;
                if *right_count == 0 {
                    right_counts.remove(&label);
                }
            }

            // Only consider a split point between two different consecutive values.
            if val < next_val {
                let midpoint = (val + next_val) / 2.0;

                let gain = if missing_indices.is_empty() {
                    let left_size = i + 1;
                    let total_non_missing = value_label_idx.len();
                    let p = left_size as f64 / total_non_missing as f64;

                    // Ensure partitions are not empty
                    if left_counts.is_empty() || right_counts.is_empty() {
                        0.0
                    } else {
                        current_score
                            - p * criterion(&left_counts)
                            - (1.0 - p) * criterion(&right_counts)
                    }
                } else {
                    // If missing values exist, we must use the slower but correct method
                    // of splitting the full set to account for missing values being
                    // sent to both children.
                    let (set1, set2) =
                        split_set(all_data, row_indices, col, &SampleValue::Numeric(midpoint));
                    if set1.is_empty() || set2.is_empty() {
                        0.0
                    } else {
                        let p_actual = set1.len() as f64 / row_indices.len() as f64;
                        current_score
                            - p_actual * criterion(&count_classes(all_data, &set1, target_idx))
                            - (1.0 - p_actual) * criterion(&count_classes(all_data, &set2, target_idx))
                    }
                };

                if gain > best_gain {
                    best_gain = gain;
                    best_split_value = midpoint;
                }
            }
        }

        if best_gain > 0.0 {
            let mut set1: Vec<usize> = Vec::new(); // true branch: values >= midpoint
            let mut set2: Vec<usize> = Vec::new(); // false branch: values < midpoint

            for &(val, _label, row_idx) in &value_label_idx {
                if val >= best_split_value {
                    set1.push(row_idx);
                } else {
                    set2.push(row_idx);
                }
            }

            // Add missing indices to both sets
            set1.extend_from_slice(&missing_indices);
            set2.extend_from_slice(&missing_indices);

            Some((
                best_gain,
                SampleValue::Numeric(best_split_value),
                set1,
                set2,
            ))
        } else {
            None
        }
    }

    fn find_best_split(
        current_score: f64,
        all_data: &[Sample],
        row_indices: &[usize],
        col: usize,
        criterion: &dyn Fn(&Counter) -> f64,
        target_idx: usize,
    ) -> Option<(f64, SampleValue, Vec<usize>, Vec<usize>)> {
        let mut best_gain = 0.0;
        let mut best: Option<(SampleValue, Vec<usize>, Vec<usize>)> = None;

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
                - p * criterion(&count_classes(all_data, &set1_indices, target_idx))
                - (1.0 - p) * criterion(&count_classes(all_data, &set2_indices, target_idx));

            if gain > best_gain {
                best_gain = gain;
                best = Some((value, set1_indices, set2_indices));
            }
        }
        if let Some((value, set1, set2)) = best {
            Some((best_gain, value, set1, set2))
        } else {
            None
        }
    }

    fn grow_tree(
        all_data: &[Sample],
        meta: &'a DatasetMetadata,
        row_indices: &[usize],
        min_samples_split: usize,
        max_depth: Option<usize>,
        criterion: &dyn Fn(&Counter) -> f64,
        depth: usize,
    ) -> Result<Node, TreeError> {
        if row_indices.is_empty() {
            // This case can happen if a split results in one of the branches
            // having no data (e.g., only missing values that go both ways,
            // but no actual values for one side).
            return Err(TreeError::EmptySplit);
        }

        let target_idx = meta.target_column_index;
        let current_counts = count_classes(all_data, row_indices, target_idx);
        let current_score = criterion(&current_counts);

        // Pre-pruning
        let summary = Summary {
            impurity: current_score,
            samples: row_indices.len(),
        };

        let is_max_depth_reached = max_depth.is_some_and(|max| depth >= max);
        if is_max_depth_reached || row_indices.len() < min_samples_split {
            return Ok(Node::leaf(current_counts, summary));
        }

        // find best gain
        let mut best_gain = 0.0;
        let mut best_rule: Option<(usize, SampleValue)> = None;
        let mut best_sets: Option<(Vec<usize>, Vec<usize>)> = None;

        let total_columns = meta.header.len();
        for col in (0..total_columns).filter(|&i| i!=target_idx) {
            let split = match meta.column_types[col] {
                ColumnType::Numeric => Self::find_best_numeric_split(
                    current_score,
                    all_data,
                    row_indices,
                    col,
                    criterion,
                    target_idx,
                ),
                ColumnType::Categorical => {
                    Self::find_best_split(current_score, all_data, row_indices, col, criterion, target_idx)
                }
                ColumnType::Mixed => return Err(TreeError::MixedTypesInColumn),
            };
            if let Some((gain, value, set1, set2)) = split {
                if gain > best_gain {
                    best_gain = gain;
                    best_rule = Some((col, value));
                    best_sets = Some((set1, set2));
                }
            }
        }

        if best_gain > 0.0 {
            let (col, value) = best_rule.expect("best_gain > 0.0 but best_rule is None");
            let (set1_indices, set2_indices) =
                best_sets.expect("best_gain > 0.0 but best_sets is None");

            let true_branch = Box::new(Self::grow_tree(
                all_data,
                meta,
                &set1_indices,
                min_samples_split,
                max_depth,
                criterion,
                depth + 1,
            )?);

            let false_branch = Box::new(Self::grow_tree(
                all_data,
                meta,
                &set2_indices,
                min_samples_split,
                max_depth,
                criterion,
                depth + 1,
            )?);

            Ok(Node::internal(
                col,
                value,
                true_branch,
                false_branch,
                summary,
            ))
        } else {
            Ok(Node::leaf(current_counts, summary))
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
    /// TODO: return HashMap<String,f64> instead of rounding?
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

    pub fn prune(&mut self, min_gain: f64, criterion: Criterion, notify: bool) {
        let eval_fn = Self::eval_fn(criterion);
        self.root.prune(min_gain, eval_fn.as_ref(), notify);
    }
}

impl<'a> std::fmt::Display for DecisionTree<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.root
                .to_string(Some(&self.meta.header), "", &self.meta.vocabulary)
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
            // Missing value - distribute sample to both child nodes to ensures all data is used
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

fn count_classes(all_data: &[Sample], row_indices: &[usize], target_idx: usize) -> Counter {
    let mut counts = HashMap::new();
    for &row_idx in row_indices {
        let row = &all_data[row_idx];
        if let Some(SampleValue::String(id)) = row.get(target_idx) {
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
        .filter(|&&c| c > 0)
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

pub struct DecisionTreeBuilder {
    verbose: u8,
    criterion: Criterion,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_gain_prune: Option<f64>,
}

impl Default for DecisionTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeBuilder {
    pub fn new() -> Self {
        // Sensible defaults
        Self {
            verbose: 0,
            criterion: Criterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_gain_prune: None,
        }
    }

    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    pub fn verbose(mut self, verbose: u8) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn min_samples_split(mut self, samples: usize) -> Self {
        self.min_samples_split = samples;
        self
    }

    pub fn min_gain_prune(mut self, min_gain: f64) -> Self {
        self.min_gain_prune = Some(min_gain);
        self
    }

    pub fn build<'a>(
        self,
        data: &'a [Sample],
        meta: &'a DatasetMetadata,
    ) -> Result<DecisionTree<'a>, TreeError> {
        if data.is_empty() {
            return Err(TreeError::EmptyDataset);
        }

        let mut tree = DecisionTree::train(
            data,
            meta,
            self.criterion,
            self.max_depth,
            self.min_samples_split,
        )?;

        if self.verbose > 0 {
            println!("Trained a model with {} nodes", tree.size());
        }

        // Apply post-pruning if specified
        if let Some(min_gain) = self.min_gain_prune {
            tree.prune(min_gain, self.criterion, self.verbose > 1);
            if self.verbose > 0 {
                println!("Pruned model down to {} nodes", tree.size());
            }
        }
        Ok(tree)
    }
}
