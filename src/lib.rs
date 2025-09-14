use crate::data::{Sample, SampleValue, Vocabulary};
use std::collections::HashMap;
pub mod data;

pub type Counter = HashMap<usize, usize>;

#[derive(Debug, Clone)]
pub struct Summary {
    pub impurity: f64,
    pub samples: usize,
}

// common data for all tree nodes
#[derive(Debug)]
pub struct Node {
    pub summary: Summary,
    pub kind: NodeKind,
}

#[derive(Debug)]
pub enum NodeKind {
    // A leaf holds the final prediction counts.
    Leaf {
        class_counts: Counter,
    },
    // An internal node holds the split condition and children.
    Internal {
        col: usize,
        value: SampleValue,
        true_branch: Box<Node>, // Note: This now refers to the new `Node` type
        false_branch: Box<Node>, // And so does this
    },
}

impl Node {
    fn leaf(class_counts: Counter, summary: Summary) -> Self {
        Node {
            summary,
            kind: NodeKind::Leaf { class_counts },
        }
    }

    fn internal(
        col: usize,
        value: SampleValue,
        true_branch: Box<Node>,
        false_branch: Box<Node>,
        summary: Summary,
    ) -> Self {
        Node {
            summary,
            kind: NodeKind::Internal {
                col,
                value,
                true_branch,
                false_branch,
            },
        }
    }

    pub fn size(&self) -> usize {
        match &self.kind {
            NodeKind::Leaf { .. } => 1,
            NodeKind::Internal {
                true_branch,
                false_branch,
                ..
            } => 1 + true_branch.size() + false_branch.size(),
        }
    }

    // Return the predicted class (most common label)
    pub fn predict(&self, sample: &Sample) -> usize {
        match &self.kind {
            NodeKind::Leaf { class_counts } => {
                *class_counts.iter().max_by_key(|&(_, &v)| v).unwrap().0
            }

            NodeKind::Internal {
                col,
                value,
                true_branch,
                false_branch,
                ..
            } => {
                let v = &sample[*col];
                let cond = match (v, value) {
                    (SampleValue::Numeric(_), _) => v.ge(value),
                    _ => v.eq(value),
                };
                let branch = if cond { true_branch } else { false_branch };
                branch.predict(sample)
            }
        }
    }

    // Return full distribution - a reference to a leafs's counter
    pub fn get_leaf_counts(&self, sample: &Sample) -> &Counter {
        let mut current_node = self;
        loop {
            match &current_node.kind {
                NodeKind::Leaf { class_counts } => return class_counts,
                NodeKind::Internal {
                    col,
                    value,
                    true_branch,
                    false_branch,
                    ..
                } => {
                    let v = &sample[*col];
                    let cond = match (v, value) {
                        (SampleValue::Numeric(_), _) => v.ge(value),
                        _ => v.eq(value),
                    };
                    current_node = if cond { true_branch } else { false_branch };
                }
            }
        }
    }

    pub fn classify_with_missing_data(&self, sample: &Sample) -> Counter {
        let mut current_node = self;
        loop {
            match &current_node.kind {
                NodeKind::Leaf { class_counts } => return class_counts.clone(),
                NodeKind::Internal {
                    col,
                    value,
                    true_branch,
                    false_branch,
                    ..
                } => {
                    let v = &sample[*col];
                    if *v == SampleValue::None {
                        // Handle missing feature
                        let tr = true_branch.classify_with_missing_data(sample);
                        let fr = false_branch.classify_with_missing_data(sample);

                        let tcount: usize = tr.values().sum();
                        let fcount: usize = fr.values().sum();

                        if tcount + fcount == 0 {
                            return Counter::new();
                        }

                        let mut result = Counter::new();
                        for k in tr.keys().chain(fr.keys()) {
                            let t_val = tr.get(k).unwrap_or(&0);
                            let f_val = fr.get(k).unwrap_or(&0);
                            result.insert(*k, t_val * tcount + f_val * fcount);
                        }

                        return result;
                    }
                    let cond = match (v, value) {
                        (SampleValue::Numeric(_), _) => v.ge(value),
                        _ => v.eq(value),
                    };
                    current_node = if cond { true_branch } else { false_branch };
                }
            }
        }
    }

    pub fn prune(&mut self, min_gain: f64, criterion: &dyn Fn(&Counter) -> f64, notify: bool) {
        if let NodeKind::Internal {
            true_branch,
            false_branch,
            ..
        } = &mut self.kind
        {
            // Recursive calls
            true_branch.prune(min_gain, criterion, notify);
            false_branch.prune(min_gain, criterion, notify);

            // Check if both children are now leaves
            if let (
                NodeKind::Leaf {
                    class_counts: true_counts,
                },
                NodeKind::Leaf {
                    class_counts: false_counts,
                },
            ) = (&true_branch.kind, &false_branch.kind)
            {
                let mut merged_counts = true_counts.clone();
                for (k, v) in false_counts {
                    *merged_counts.entry(k.clone()).or_insert(0) += v;
                }

                let merged_impurity = criterion(&merged_counts);
                let total_samples: usize = merged_counts.values().sum();
                let true_samples: usize = true_counts.values().sum();
                let p_true = true_samples as f64 / total_samples as f64;

                let child_impurity =
                    p_true * criterion(true_counts) + (1.0 - p_true) * criterion(false_counts);

                let gain = merged_impurity - child_impurity;

                if gain < min_gain {
                    if notify {
                        println!("A branch was pruned: gain = {gain:.4}");
                    }
                    self.kind = NodeKind::Leaf {
                        class_counts: merged_counts,
                    };
                }
            }
        }
    }

    pub fn to_string(
        &self,
        headers: Option<&[String]>,
        indent: &str,
        vocab: &Vocabulary,
    ) -> String {
        match &self.kind {
            NodeKind::Leaf { class_counts } => {
                let mut sorted: Vec<_> = class_counts.iter().collect();
                sorted.sort_by_key(|&(k, _)| k);
                sorted
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
            NodeKind::Internal {
                col,
                value,
                true_branch,
                false_branch,
                ..
            } => {
                let column_name = headers
                    .map(|h| h[*col].as_str())
                    .unwrap_or_else(|| "Column");

                let decision = match value {
                    SampleValue::Numeric(n) => format!("{column_name} >= {n}?"),
                    // Look up the string from the ID
                    SampleValue::String(id) => {
                        format!("{column_name} == {}?", vocab.get_str(*id).unwrap())
                    }
                    SampleValue::None => format!("{column_name} == None?"),
                };

                let true_branch_str = format!(
                    "{indent}yes -> {}",
                    true_branch.to_string(headers, &format!("{indent}    "), vocab)
                );

                let false_branch_str = format!(
                    "{indent}no  -> {}",
                    false_branch.to_string(headers, &format!("{indent}    "), vocab)
                );

                format!("{decision}\n{true_branch_str}\n{false_branch_str}")
            }
        }
    }
}

pub struct DecisionTree<'a> {
    pub root: Node,
    pub header: Vec<String>,
    pub vocab: &'a Vocabulary,
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
                    continue;
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

    // Full classification with missing data handling (creates new Counter)
    pub fn classify(&self, sample: &Sample, handle_missing: bool) -> Counter {
        if handle_missing {
            self.root.classify_with_missing_data(sample)
        } else {
            self.root.get_leaf_counts(sample).clone()
        }
    }
    pub fn prune(&mut self, min_gain: f64, criterion: &str, notify: bool) {
        let eval_fn = Self::eval_fn(criterion);
        self.root.prune(min_gain, eval_fn.as_ref(), notify);
    }

    pub fn export_graph(&self, filename: &str) {
        // Stub implementation as requested
        println!("export_graph not implemented - would export to {filename}");
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
