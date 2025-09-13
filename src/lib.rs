use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};

#[derive(Debug, Default)]
pub struct Vocabulary {
    map: HashMap<String, usize>,
    pub vec: Vec<String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns a string, returning its unique ID.
    pub fn get_or_intern(&mut self, s: &str) -> usize {
        if let Some(id) = self.map.get(s) {
            *id
        } else {
            let id = self.vec.len();
            let owned_s = s.to_string();
            self.vec.push(owned_s.clone());
            self.map.insert(owned_s, id);
            id
        }
    }

    /// Retrieves the string slice for a given ID.
    pub fn get_str(&self, id: usize) -> Option<&str> {
        self.vec.get(id).map(|s| s.as_str())
    }

    /// Looks up the ID for a given string slice.
    pub fn get_id(&self, s: &str) -> Option<usize> {
        self.map.get(s).copied() // .copied() converts Option<&usize> to Option<usize>
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SampleValue {
    Numeric(f64),
    String(usize),
    None,
}

impl Eq for SampleValue {} // Manually implement Eq - treating NaN values as equal to themselves for this use case

impl Hash for SampleValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            SampleValue::Numeric(f) => {
                0u8.hash(state); // discriminant
                // For floats, we'll use the bit representation
                // This treats -0.0 and 0.0 as different, but that's okay for our use case
                f.to_bits().hash(state);
            }
            SampleValue::String(s) => {
                1u8.hash(state); // discriminant
                s.hash(state);
            }
            SampleValue::None => {
                2u8.hash(state); // discriminant
            }
        }
    }
}

impl SampleValue {
    fn ge(&self, other: &SampleValue) -> bool {
        match (self, other) {
            (SampleValue::Numeric(a), SampleValue::Numeric(b)) => a >= b,
            _ => false,
        }
    }

    fn eq(&self, other: &SampleValue) -> bool {
        match (self, other) {
            (SampleValue::Numeric(a), SampleValue::Numeric(b)) => a == b,
            (SampleValue::String(a), SampleValue::String(b)) => a == b,
            (SampleValue::None, SampleValue::None) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for SampleValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleValue::Numeric(fl) => write!(f, "Numeric {fl}"),
            SampleValue::String(s) => write!(f, "String {s}"),
            SampleValue::None => write!(f, "None"),
        }
    }
}

pub type Sample = Vec<SampleValue>;
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

    pub fn classify(&self, sample: &Sample) -> Counter {
        if let Some(ref counts) = self.class_counts {
            return counts.clone();
        }

        let col = self.col.unwrap();
        let v = &sample[col];
        let branch = self.pick_branch(v);
        branch.classify(sample)
    }

    pub fn classify_with_missing_data(&self, sample: &Sample) -> HashMap<usize, f64> {
        if let Some(ref counts) = self.class_counts {
            return counts.iter().map(|(k, &v)| (k.clone(), v as f64)).collect();
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

        let tcount: f64 = tr.values().sum();
        let fcount: f64 = fr.values().sum();
        let total_count = tcount + fcount;

        if total_count == 0.0 {
            return HashMap::new();
        }

        let tw = tcount / total_count;
        let fw = fcount / total_count;

        let mut result = HashMap::new();
        let all_keys: std::collections::HashSet<_> = tr.keys().chain(fr.keys()).collect();

        for k in all_keys {
            let t_val = tr.get(k).unwrap_or(&0.0);
            let f_val = fr.get(k).unwrap_or(&0.0);
            result.insert(k.clone(), t_val * tw + f_val * fw);
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
        let root = Self::grow_tree(
            &data,
            min_samples_split,
            max_depth,
            eval_fn.as_ref(),
            0,
        );
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

    pub fn classify(&self, sample: &Sample, handle_missing: bool) -> HashMap<usize, f64> {
        if handle_missing {
            self.root_node.classify_with_missing_data(sample)
        } else {
            self.root_node
                .classify(sample)
                .into_iter()
                .map(|(k, v)| (k, v as f64))
                .collect()
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
                unimplemented!()  // class labels are always "strings"
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

// Utility functions


/// Scan csv file, validate number of columns and add target column labels to vocab
/// Return: header
/// Scan csv file, validate number of columns and add target column labels to vocab
/// Return: (header, target_column_index, num_rows)
pub fn scan_csv(
    fname: &str,
    vocab: &mut Vocabulary,
    target_column: Option<usize>, // None means last column
    verbose: bool,
) -> Result<(Vec<String>, usize, usize), Box<dyn std::error::Error>> {
    let file = File::open(fname)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let header_line = lines.next().ok_or("Empty file")??;
    let header: Vec<String> = header_line.split(',').map(|s| s.trim().to_string()).collect();
    let num_columns = header.len();
    
    // Determine target column index
    let target_idx = target_column.unwrap_or(num_columns.saturating_sub(1));
    if target_idx >= num_columns {
        return Err(format!("Target column index {target_idx} is out of bounds (file has {num_columns} columns)").into());
    }

    let mut line_number = 1; // Header was line 0
    let mut target_labels = std::collections::HashSet::new();

    for line_result in lines {
        line_number += 1;
        let line = line_result?;
        
        let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        
        // Validate row length
        if values.len() != num_columns {
            return Err(format!(
                "Line {} has {} columns, expected {} (header has {} columns)",
                line_number, values.len(), num_columns, num_columns
            ).into());
        }

        if values[target_idx] != "?" {
            target_labels.insert(values[target_idx].to_string());
        }
    }

    // Sort and intern target labels to ensure consistent ordering
    let mut sorted_labels: Vec<String> = target_labels.into_iter().collect();
    sorted_labels.sort();
    
    for label in &sorted_labels {
        vocab.get_or_intern(label);
    }

    if verbose {
        println!("{fname}: {} data rows", line_number - 1);
        println!("header: {header:?}");
        println!("target column: {} ('{}')", target_idx, header[target_idx]);
        println!("class labels: {sorted_labels:?}");
    }

    Ok((header, target_idx, line_number - 1))
}

/// Read csv file data using pre-built vocabulary
pub fn read_csv(
    fname: &str,
    vocab: &mut Vocabulary,
    target_idx: usize,
    expected_columns: usize,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let file = File::open(fname)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Skip header
    let _ = lines.next().ok_or("Empty file")??;
    
    let mut data = Vec::new();
    let mut line_number = 1;

    for line_result in lines {
        line_number += 1;
        let line = line_result?;
        let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        
        if values.len() != expected_columns {
            return Err(format!(
                "Line {} has {} columns, expected {}",
                line_number, values.len(), expected_columns
            ).into());
        }
        
        let row: Sample = values
            .into_iter()
            .enumerate()
            .map(|(idx, s)| {
                if s == "?" {
                    return SampleValue::None;
                }
                
                // Target column - must already be in vocabulary
                if idx == target_idx {
                    if let Some(id) = vocab.get_id(s) {
                        return SampleValue::String(id);
                    } else {
                        // This shouldn't happen if scan_csv was run first
                        panic!("Unknown target label '{}' - was scan_csv run?", s);
                    }
                }
                
                // For non-target columns
                if let Ok(f) = s.parse::<f64>() {
                    return SampleValue::Numeric(f);
                }
                
                // String value - intern it
                let id = vocab.get_or_intern(s);
                SampleValue::String(id)
            })
            .collect();
            
        data.push(row);
    }

    Ok(data)
}

/// Load a single CSV file 
pub fn load_single_csv(
    fname: &str,
    target_column: Option<usize>,
    verbose: bool,
) -> Result<(Vec<String>, Vec<Sample>, Vocabulary, usize), Box<dyn std::error::Error>> {
    let mut vocab = Vocabulary::new();
    
    // First pass: scan for validation and target labels
    let (header, target_idx, _num_rows) = scan_csv(fname, &mut vocab, target_column, verbose)?;
    let num_classes = vocab.len(); // Number of unique target labels
    
    // Second pass: read the data
    let data = read_csv(fname, &mut vocab, target_idx, header.len())?;
    
    Ok((header, data, vocab, num_classes))
}

/// Load separate train and test CSV files
/// Returns: header, training_samples, test samples, vocabulary, num_classes
pub fn load_train_test_csv(
    train_fname: &str,
    test_fname: &str,
    target_column: Option<usize>,
    verbose: bool,
) -> Result<(Vec<String>, Vec<Sample>, Vec<Sample>, Vocabulary, usize), Box<dyn std::error::Error>> {
    let mut vocab = Vocabulary::new();
    
    // Scan both files to build complete vocabulary
    let (train_header, train_target_idx, _) = scan_csv(train_fname, &mut vocab, target_column, verbose)?;
    let num_classes = vocab.len(); // Number of unique target labels from training
    
    let (test_header, test_target_idx, _) = scan_csv(test_fname, &mut vocab, target_column, verbose)?;
    
    // Validate headers match
    if train_header != test_header {
        return Err(format!(
            "Train and test files have different headers:\nTrain: {:?}\nTest: {:?}", 
            train_header, test_header
        ).into());
    }
    
    if train_target_idx != test_target_idx {
        return Err("Train and test files have different target column indices".into());
    }
    
    // Check if test set introduced new labels
    if vocab.len() > num_classes && verbose {
        println!("Warning: Test set contains {} new class labels not seen in training", 
                 vocab.len() - num_classes);
    }
    
    // Read the actual data
    let train_data = read_csv(train_fname, &mut vocab, train_target_idx, train_header.len())?;
    let test_data = read_csv(test_fname, &mut vocab, test_target_idx, test_header.len())?;
    
    Ok((train_header, train_data, test_data, vocab, num_classes))
}
