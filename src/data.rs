use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use std::hash::{Hash,Hasher};

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

/// Scan csv file, validate number of columns and add target column labels to vocab
/// Return: header
/// Scan csv file, validate number of columns and add target column labels to vocab
/// Return: (header, target_column_index, num_rows)
fn scan_csv(
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
    let header: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let num_columns = header.len();

    // Determine target column index
    let target_idx = target_column.unwrap_or(num_columns.saturating_sub(1));
    if target_idx >= num_columns {
        return Err(format!(
            "Target column index {target_idx} is out of bounds (file has {num_columns} columns)"
        )
        .into());
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
                line_number,
                values.len(),
                expected_columns
            )
            .into());
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
) -> Result<(Vec<String>, Vec<Sample>, Vec<Sample>, Vocabulary, usize), Box<dyn std::error::Error>>
{
    let mut vocab = Vocabulary::new();

    // Scan both files to build complete vocabulary
    let (train_header, train_target_idx, _) =
        scan_csv(train_fname, &mut vocab, target_column, verbose)?;
    let num_classes = vocab.len(); // Number of unique target labels from training

    let (test_header, test_target_idx, _) =
        scan_csv(test_fname, &mut vocab, target_column, verbose)?;

    // Validate headers match
    if train_header != test_header {
        return Err(format!(
            "Train and test files have different headers:\nTrain: {:?}\nTest: {:?}",
            train_header, test_header
        )
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
    let train_data = read_csv(
        train_fname,
        &mut vocab,
        train_target_idx,
        train_header.len(),
    )?;
    let test_data = read_csv(test_fname, &mut vocab, test_target_idx, test_header.len())?;

    Ok((train_header, train_data, test_data, vocab, num_classes))
}
