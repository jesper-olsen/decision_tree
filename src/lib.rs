use crate::data::{Sample, SampleValue, Vocabulary};
use std::collections::HashMap;
pub mod data;

pub type Counter = HashMap<usize, usize>;

#[derive(Debug, Clone)]
pub struct Summary {
    pub impurity: f64,
    pub samples: usize,
}

#[derive(Debug)]
pub struct DecisionNode {
    pub col: Option<usize>,
    pub value: Option<SampleValue>,
    pub class_counts: Option<Counter>,
    pub true_branch: Option<Box<DecisionNode>>,
    pub false_branch: Option<Box<DecisionNode>>,
    pub summary: Option<Summary>,
}

impl DecisionNode {
    fn new() -> Self {
        DecisionNode {
            col: None,
            value: None,
            class_counts: None,
            true_branch: None,
            false_branch: None,
            summary: None,
        }
    }

    fn leaf(class_counts: Counter, summary: Summary) -> Self {
        DecisionNode {
            col: None,
            value: None,
            class_counts: Some(class_counts),
            true_branch: None,
            false_branch: None,
            summary: Some(summary),
        }
    }

    fn internal(
        col: usize,
        value: SampleValue,
        true_branch: Box<DecisionNode>,
        false_branch: Box<DecisionNode>,
        summary: Summary,
    ) -> Self {
        DecisionNode {
            col: Some(col),
            value: Some(value),
            class_counts: None,
            true_branch: Some(true_branch),
            false_branch: Some(false_branch),
            summary: Some(summary),
        }
    }

    pub fn size(&self) -> usize {
        if self.class_counts.is_some() {
            1
        } else {
            let true_size = self.true_branch.as_ref().map(|b| b.size()).unwrap_or(0);
            let false_size = self.false_branch.as_ref().map(|b| b.size()).unwrap_or(0);
            true_size + false_size + 1
        }
    }

    fn pick_branch(&self, v: &SampleValue) -> &DecisionNode {
        let value = self.value.as_ref().unwrap();
        let cond = match (v, value) {
            (SampleValue::Numeric(_), _) => v.ge(value),
            _ => v.eq(value),
        };

        if cond {
            self.true_branch.as_ref().unwrap()
        } else {
            self.false_branch.as_ref().unwrap()
        }
    }

    // Return the predicted class (most common label)
    pub fn predict(&self, sample: &Sample) -> usize {
        if let Some(ref counts) = self.class_counts {
            return *counts.iter().max_by_key(|&(_, &v)| v).unwrap().0;
        }

        let col = self.col.unwrap();
        let v = &sample[col];
        let branch = self.pick_branch(v);
        branch.predict(sample)
    }

    // Return full distribution - a reference to a leafs's counter
    pub fn get_leaf_counts(&self, sample: &Sample) -> &Counter {
        if let Some(ref counts) = self.class_counts {
            return counts;
        }

        let col = self.col.unwrap();
        let v = &sample[col];
        let branch = self.pick_branch(v);
        branch.get_leaf_counts(sample)
    }

    pub fn classify_with_missing_data(&self, sample: &Sample) -> Counter {
        if let Some(ref counts) = self.class_counts {
            return counts.iter().map(|(&k, &v)| (k, v)).collect();
        }

        let col = self.col.unwrap();
        let v = &sample[col];

        if !matches!(v, SampleValue::None) {
            let branch = self.pick_branch(v);
            return branch.classify_with_missing_data(sample);
        }

        // Handle missing feature
        let tr = self
            .true_branch
            .as_ref()
            .unwrap()
            .classify_with_missing_data(sample);
        let fr = self
            .false_branch
            .as_ref()
            .unwrap()
            .classify_with_missing_data(sample);

        let tcount: usize = tr.values().sum();
        let fcount: usize = fr.values().sum();

        if tcount + fcount == 0 {
            return Counter::new();
        }

        let mut result = HashMap::new();

        for k in tr.keys().chain(fr.keys()) {
            let t_val = tr.get(k).unwrap_or(&0);
            let f_val = fr.get(k).unwrap_or(&0);
            result.insert(*k, t_val * tcount + f_val * fcount);
        }

        result
    }

    pub fn prune(&mut self, min_gain: f64, criterion: &dyn Fn(&Counter) -> f64, notify: bool) {
        // Recursive call for each branch
        if let Some(ref mut tb) = self.true_branch {
            if tb.class_counts.is_none() {
                tb.prune(min_gain, criterion, notify);
            }
        }

        if let Some(ref mut fb) = self.false_branch {
            if fb.class_counts.is_none() {
                fb.prune(min_gain, criterion, notify);
            }
        }

        // Merge leaves potentially
        if let (Some(tb), Some(fb)) = (&self.true_branch, &self.false_branch) {
            if let (Some(true_counts), Some(false_counts)) = (&tb.class_counts, &fb.class_counts) {
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
                    self.true_branch = None;
                    self.false_branch = None;
                    self.class_counts = Some(merged_counts);
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
        if let Some(ref counts) = self.class_counts {
            let mut sorted: Vec<_> = counts.iter().collect();
            sorted.sort_by_key(|&(k, _)| k);
            return sorted
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ");
        }

        let col = self.col.unwrap();
        let value = self.value.as_ref().unwrap();
        let column_name = headers.map(|h| h[col].as_str()).unwrap_or_else(|| "Column");

        let decision = match value {
            SampleValue::Numeric(n) => format!("{column_name} >= {n}?"),
            // Look up the string from the ID
            SampleValue::String(id) => format!("{column_name} == {}?", vocab.get_str(*id).unwrap()),
            SampleValue::None => format!("{column_name} == None?"),
        };

        let true_branch_str = format!(
            "{indent}yes -> {}",
            self.true_branch
                .as_ref()
                .unwrap()
                .to_string(headers, &format!("{indent}    "), vocab)
        );

        let false_branch_str = format!(
            "{indent}no  -> {}",
            self.false_branch
                .as_ref()
                .unwrap()
                .to_string(headers, &format!("{indent}    "), vocab)
        );

        format!("{decision}\n{true_branch_str}\n{false_branch_str}")
    }
}

pub struct DecisionTree<'a> {
    pub root_node: DecisionNode,
    pub header: Vec<String>,
    pub vocab: &'a Vocabulary,
}

impl<'a> DecisionTree<'a> {
    pub fn new(root_node: DecisionNode, header: Vec<String>, vocab: &'a Vocabulary) -> Self {
        DecisionTree {
            root_node,
            header,
            vocab,
        }
    }

    pub fn size(&self) -> usize {
        self.root_node.size()
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
        let root = Self::grow_tree(&data, min_samples_split, max_depth, eval_fn.as_ref(), 0);
        DecisionTree::new(root, header, vocab)
    }

    fn grow_tree(
        rows: &[Sample],
        min_samples_split: usize,
        max_depth: Option<usize>,
        criterion: &dyn Fn(&Counter) -> f64,
        depth: usize,
    ) -> DecisionNode {
        if rows.is_empty() {
            return DecisionNode::new();
        }

        let current_score = criterion(&count_classes(rows));

        // Pre-pruning
        let summary = Summary {
            impurity: current_score,
            samples: rows.len(),
        };

        if (max_depth.is_some() && depth >= max_depth.unwrap()) || (rows.len() < min_samples_split)
        {
            return DecisionNode::leaf(count_classes(rows), summary);
        }

        let mut best_gain = 0.0;
        let mut best_rule: Option<(usize, SampleValue)> = None;
        let mut best_sets: Option<(Vec<Sample>, Vec<Sample>)> = None;

        let column_count = rows[0].len() - 1;
        for col in 0..column_count {
            let mut column_values = std::collections::HashSet::new();
            for row in rows {
                column_values.insert(row[col].clone());
            }

            for value in column_values {
                if matches!(value, SampleValue::None) {
                    continue;
                }

                let (set1, set2) = split_set(rows, col, &value);
                if set1.is_empty() || set2.is_empty() {
                    continue;
                }

                let p = set1.len() as f64 / rows.len() as f64;
                let gain = current_score
                    - p * criterion(&count_classes(&set1))
                    - (1.0 - p) * criterion(&count_classes(&set2));

                if gain > best_gain {
                    best_gain = gain;
                    best_rule = Some((col, value));
                    best_sets = Some((set1, set2));
                }
            }
        }

        if best_gain > 0.0 {
            let (col, value) = best_rule.unwrap();
            let (set1, set2) = best_sets.unwrap();

            let true_branch = Box::new(Self::grow_tree(
                &set1,
                min_samples_split,
                max_depth,
                criterion,
                depth + 1,
            ));

            let false_branch = Box::new(Self::grow_tree(
                &set2,
                min_samples_split,
                max_depth,
                criterion,
                depth + 1,
            ));

            DecisionNode::internal(col, value, true_branch, false_branch, summary)
        } else {
            DecisionNode::leaf(count_classes(rows), summary)
        }
    }

    // Simple prediction - returns the class label
    pub fn predict(&self, sample: &Sample) -> usize {
        self.root_node.predict(sample)
    }

    // Get class distribution without cloning
    pub fn predict_proba(&self, sample: &Sample) -> &Counter {
        self.root_node.get_leaf_counts(sample)
    }

    // Full classification with missing data handling (creates new Counter)
    pub fn classify(&self, sample: &Sample, handle_missing: bool) -> Counter {
        if handle_missing {
            self.root_node.classify_with_missing_data(sample)
        } else {
            self.root_node.get_leaf_counts(sample).clone()
        }
    }
    pub fn prune(&mut self, min_gain: f64, criterion: &str, notify: bool) {
        let eval_fn = Self::eval_fn(criterion);
        self.root_node.prune(min_gain, eval_fn.as_ref(), notify);
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
            self.root_node.to_string(Some(&self.header), "", self.vocab)
        )
    }
}

// Helper functions

fn split_set(rows: &[Sample], column: usize, value: &SampleValue) -> (Vec<Sample>, Vec<Sample>) {
    let mut set1 = Vec::new();
    let mut set2 = Vec::new();

    for row in rows {
        let v = &row[column];
        if matches!(v, SampleValue::None) {
            // Missing value - duplicate row into both sets
            set1.push(row.clone());
            set2.push(row.clone());
        } else if matches!(value, SampleValue::Numeric(_)) {
            if v.ge(value) {
                set1.push(row.clone());
            } else {
                set2.push(row.clone());
            }
        } else if v.eq(value) {
            set1.push(row.clone());
        } else {
            set2.push(row.clone());
        }
    }

    (set1, set2)
}

fn count_classes(rows: &[Sample]) -> Counter {
    let mut counts = HashMap::new();
    for row in rows {
        if let Some(last) = row.last() {
            // The class label is the last value in the row
            let SampleValue::String(id) = last else {
                unimplemented!() // class labels are always "strings"
            };
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
