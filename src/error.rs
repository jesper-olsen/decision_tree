use std::error::Error;
use std::fmt;
use std::io;

#[derive(Debug)]
pub enum TreeError {
    // Configuration errors
    InvalidConfiguration(String),
    MissingVocabulary,

    // Data errors
    EmptyDataset,
    MixedTypesInColumn,
    InvalidDataFormat(String),
    InconsistentFeatures { expected: usize, found: usize },

    // Training errors
    NoValidSplit,
    EmptyLeaf,
    EmptySplit,

    // IO errors
    IoError(io::Error),

    // Export errors
    GraphvizNotFound,
    ExportFailed(String),
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::MixedTypesInColumn => {
                write!(
                    f,
                    "Columns can not contain both categorical and numeric values"
                )
            }
            TreeError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {msg}")
            }
            TreeError::MissingVocabulary => {
                write!(f, "Vocabulary must be provided before building tree")
            }
            TreeError::EmptyDataset => {
                write!(f, "Cannot build tree from empty dataset")
            }
            TreeError::EmptySplit => {
                write!(f, "Cannot build tree from empty set of row indices.")
            }
            TreeError::InvalidDataFormat(msg) => {
                write!(f, "Invalid data format: {msg}")
            }
            TreeError::InconsistentFeatures { expected, found } => {
                write!(
                    f,
                    "Inconsistent feature count: expected {expected}, found {found}"
                )
            }
            TreeError::NoValidSplit => {
                write!(f, "No valid split found for current node")
            }
            TreeError::EmptyLeaf => {
                write!(f, "Attempted to create empty leaf node")
            }
            TreeError::IoError(err) => {
                write!(f, "IO error: {err}")
            }
            TreeError::GraphvizNotFound => {
                write!(
                    f,
                    "Graphviz 'dot' command not found. Please install Graphviz."
                )
            }
            TreeError::ExportFailed(msg) => {
                write!(f, "Export failed: {msg}")
            }
        }
    }
}

impl Error for TreeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TreeError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

// Implement From for automatic conversion
impl From<io::Error> for TreeError {
    fn from(err: io::Error) -> Self {
        TreeError::IoError(err)
    }
}

pub type TreeResult<T> = Result<T, TreeError>;
