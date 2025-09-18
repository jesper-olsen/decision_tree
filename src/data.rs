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
    pub fn ge(&self, other: &SampleValue) -> bool {
        match (self, other) {
            (SampleValue::Numeric(a), SampleValue::Numeric(b)) => a >= b,
            _ => false,
        }
    }

    pub fn eq(&self, other: &SampleValue) -> bool {
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

#[derive(Debug, Clone)]
pub enum ColumnType {
    Numeric,
    Categorical,
    Mixed, // Contains both numeric and categorical values
}

#[derive(Debug)]
pub struct DatasetMetadata {
    pub header: Vec<String>,
    pub column_types: Vec<ColumnType>,
    pub target_column_index: usize,
    pub num_classes: usize,
    pub vocabulary: Vocabulary,
}

pub struct LoadedDataset {
    pub metadata: DatasetMetadata,
    pub data: Vec<Sample>,
}

pub struct LoadedSplitDataset {
    pub metadata: DatasetMetadata,
    pub train_data: Vec<Sample>,
    pub test_data: Vec<Sample>,
}

/// Scan csv file, validate number of columns and add target column labels to vocab
/// Return: (header, target_column_index, num_rows)
fn scan_csv(
    fname: &str,
    vocab: &mut Vocabulary,
    target_column: Option<usize>, // None interpreted as last column
    verbose: bool,
) -> Result<(Vec<String>, usize, usize), Box<dyn std::error::Error>> {
    let file = File::open(fname)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let header_line = lines.next().ok_or("Empty file")??;
    let header: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let num_columns = header.len();

    // Determine target column index
    let target_idx = match target_column {
        Some(idx) if idx < num_columns => idx,
        Some(idx) => {
            return Err(format!(
                "Target column index {idx} is out of bounds (file has {num_columns} columns)"
            )
            .into());
        }
        None => num_columns.saturating_sub(1),
    };

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
                line_number,
                values.len(),
                num_columns,
                num_columns
            )
            .into());
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
fn read_csv(
    fname: &str,
    vocab: &mut Vocabulary,
    target_idx: usize,
    expected_columns: usize,
) -> Result<(Vec<Sample>, Vec<ColumnType>), Box<dyn std::error::Error>> {
    let file = File::open(fname)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    let _ = lines.next().ok_or("Empty file")??;

    let mut data = Vec::new();
    let mut line_number = 1;

    // Track what we've seen in each column - has_numeric, has_string, has_missing
    let mut column_info: Vec<(bool, bool, bool)> = vec![(false, false, false); expected_columns];

    for line_result in lines {
        line_number += 1;
        let line = line_result?;
        let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        if values.len() != expected_columns {
            return Err(format!(
                "Line {line_number} has {} columns, expected {expected_columns}",
                values.len(),
            )
            .into());
        }

        let row: Sample = values
            .into_iter()
            .enumerate()
            .map(|(idx, s)| {
                if s == "?" {
                    column_info[idx].2 = true; // has_missing
                    return SampleValue::None;
                }

                // Target column - must already be in vocabulary
                if idx == target_idx {
                    column_info[idx].1 = true; // has_string
                    if let Some(id) = vocab.get_id(s) {
                        return SampleValue::String(id);
                    } else {
                        // This shouldn't happen if scan_csv was run first
                        panic!("Unknown target label '{s}' - was scan_csv run?");
                    }
                }

                // For non-target columns
                if let Ok(f) = s.parse::<f64>() {
                    column_info[idx].0 = true; // has_numeric
                    return SampleValue::Numeric(f);
                }

                // String value - intern it
                column_info[idx].1 = true; // has_string
                let id = vocab.get_or_intern(s);
                SampleValue::String(id)
            })
            .collect();

        data.push(row);
    }

    // Convert column info to ColumnType
    let column_types = column_info
        .into_iter()
        .map(|(has_num, has_str, _)| {
            match (has_num, has_str) {
                (true, false) => ColumnType::Numeric,
                (false, true) => ColumnType::Categorical,
                (true, true) => ColumnType::Mixed,
                (false, false) => ColumnType::Categorical, // All missing - treat as categorical
            }
        })
        .collect();

    Ok((data, column_types))
}

/// Load a single CSV file
pub fn load_single_csv(
    fname: &str,
    target_column: Option<usize>,
    verbose: bool,
) -> Result<LoadedDataset, Box<dyn std::error::Error>> {
    let mut vocab = Vocabulary::new();

    // First pass: scan for validation and target labels
    let (header, target_idx, _num_rows) = scan_csv(fname, &mut vocab, target_column, verbose)?;
    let num_classes = vocab.len(); // Number of unique target labels

    // Second pass: read the data
    let (data, column_types) = read_csv(fname, &mut vocab, target_idx, header.len())?;

    if verbose {
        println!("Column types: {column_types:?}");
    }

    let metadata = DatasetMetadata {
        header,
        column_types,
        target_column_index: target_idx,
        num_classes,
        vocabulary: vocab,
    };

    Ok(LoadedDataset { metadata, data })
}

/// Load separate train and test CSV files
pub fn load_train_test_csv(
    train_fname: &str,
    test_fname: &str,
    target_column: Option<usize>,
    verbose: bool,
) -> Result<LoadedSplitDataset, Box<dyn std::error::Error>> {
    let mut vocab = Vocabulary::new();

    // Scan both files to build complete vocabulary
    let (train_header, train_target_idx, _num_rows1) =
        scan_csv(train_fname, &mut vocab, target_column, verbose)?;
    let num_classes = vocab.len(); // Number of unique target labels from training

    let (test_header, test_target_idx, _num_rows2) =
        scan_csv(test_fname, &mut vocab, target_column, verbose)?;

    // Validate headers match
    if train_header != test_header {
        return Err(format!(
            "Train and test files have different headers:\nTrain: {train_header:?}\nTest: {test_header:?}")
        .into());
    }

    if train_target_idx != test_target_idx {
        return Err("Train and test files have different target column indices".into());
    }

    // Check if test set introduced new labels
    if vocab.len() > num_classes && verbose {
        println!(
            "Warning: Test set contains {} new class labels not seen in training",
            vocab.len() - num_classes
        );
    }

    // Read the actual data
    let (train_data, column_types) = read_csv(
        train_fname,
        &mut vocab,
        train_target_idx,
        train_header.len(),
    )?;
    let (test_data, _) = read_csv(test_fname, &mut vocab, test_target_idx, test_header.len())?;

    if verbose {
        println!("Column types: {column_types:?}");
    }

    let metadata = DatasetMetadata {
        header: train_header,
        column_types,
        target_column_index: train_target_idx,
        num_classes,
        vocabulary: vocab,
    };

    Ok(LoadedSplitDataset {
        metadata,
        train_data,
        test_data,
    })
}
