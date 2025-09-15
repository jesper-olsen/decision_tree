use crate::data::{Sample, SampleValue, Vocabulary};
use std::collections::HashMap;

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
        true_branch: Box<Node>,
        false_branch: Box<Node>,
    },
}

impl Node {
    pub fn leaf(class_counts: Counter, summary: Summary) -> Self {
        Node {
            summary,
            kind: NodeKind::Leaf { class_counts },
        }
    }

    pub fn internal(
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

    /// Classifies a sample, handling missing data by exploring all possible paths
    /// and returning a weighted sum of the leaf node counts.
    pub fn classify_with_missing_data(&self, sample: &Sample) -> HashMap<usize, f64> {
        let mut weighted_counts: HashMap<usize, f64> = HashMap::new();
        let mut stack: Vec<(&Node, f64)> = vec![(self, 1.0)];

        while let Some((current_node, weight)) = stack.pop() {
            match &current_node.kind {
                NodeKind::Leaf { class_counts } => {
                    for (&class, &count) in class_counts {
                        *weighted_counts.entry(class).or_insert(0.0) += count as f64 * weight;
                    }
                }
                NodeKind::Internal {
                    col,
                    value,
                    true_branch,
                    false_branch,
                } => {
                    let v = &sample[*col];
                    if matches!(v, SampleValue::None) {
                        let total_samples = current_node.summary.samples as f64;
                        if total_samples > 0.0 {
                            let true_samples = true_branch.summary.samples as f64;
                            let p_true = true_samples / total_samples;
                            stack.push((true_branch, weight * p_true));
                            stack.push((false_branch, weight * (1.0 - p_true)));
                        }
                    } else {
                        let cond = match (v, value) {
                            (SampleValue::Numeric(_), _) => v.ge(value),
                            _ => v.eq(value),
                        };
                        let branch = if cond { true_branch } else { false_branch };
                        stack.push((branch, weight));
                    }
                }
            }
        }
        weighted_counts
    }

    pub fn prune(&mut self, min_gain: f64, criterion: &dyn Fn(&Counter) -> f64, notify: bool) {
        if let NodeKind::Internal {
            true_branch,
            false_branch,
            ..
        } = &mut self.kind
        {
            // Recursive calls (post-order traversal, so we prune children first)
            if let NodeKind::Internal { .. } = true_branch.kind {
                true_branch.prune(min_gain, criterion, notify);
            }
            if let NodeKind::Internal { .. } = false_branch.kind {
                false_branch.prune(min_gain, criterion, notify);
            }

            // Check if both children are now leaves, making this node a candidate for merging
            if let (
                NodeKind::Leaf {
                    class_counts: true_counts,
                },
                NodeKind::Leaf {
                    class_counts: false_counts,
                },
            ) = (&true_branch.kind, &false_branch.kind)
            {
                let true_samples: usize = true_counts.values().sum();
                let false_samples: usize = false_counts.values().sum();
                let total_samples: usize = true_samples + false_samples;
                if total_samples == 0 {
                    return;
                };
                let p_true: f64 = true_samples as f64 / total_samples as f64;

                let mut merged_counts = true_counts.clone();
                for (&k, &v) in false_counts {
                    *merged_counts.entry(k).or_insert(0) += v;
                }
                let merged_impurity = criterion(&merged_counts);
                let child_impurity =
                    p_true * criterion(&true_counts) + (1.0 - p_true) * criterion(&false_counts);
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
