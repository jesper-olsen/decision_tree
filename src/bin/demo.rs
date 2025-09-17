use clap::Parser;
use decision_tree::data::{LoadedDataset, Sample, SampleValue, Vocabulary, load_single_csv};
use decision_tree::export::export_graph;
use decision_tree::node::Counter;
use decision_tree::tree::{Criterion, DecisionTreeBuilder};
use std::str::FromStr;

pub fn print_classification_result(sample: &Sample, result: &Counter, vocab: &Vocabulary) {
    if result.is_empty() {
        println!("Could not classify sample: {sample:?}");

        return;
    }

    // Determine the final prediction
    let prediction = result
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(k, _)| vocab.get_str(*k).unwrap())
        .unwrap();

    // Sort results by score (descending)
    let mut sorted_results: Vec<_> = result.iter().collect();
    sorted_results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("{}", "-".repeat(40));
    println!("Input Sample: {sample:?}");
    println!("--> Predicted Class: '{prediction}'");
    println!("\nDetailed Scores (Leaf Node Counts / Weights):");

    for (class_id, score) in sorted_results {
        let class_name = vocab.get_str(*class_id).unwrap();
        println!("    - {class_name:<12}: {score}");
    }
    println!("{}", "-".repeat(40));
}

pub fn small_example(
    criterion: Criterion,
    plot: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dataset: LoadedDataset = load_single_csv("data/tbc.csv", None, true)?;

    let dt = DecisionTreeBuilder::new()
        .criterion(criterion)
        .min_samples_split(2)
        .build(
            dataset.data,
            &dataset.metadata
        )?;

    println!("{dt}");

    println!("\n--- Classification Examples ---");

    // Example 1: A sample with complete data
    let complete_sample: Sample = vec![
        SampleValue::String(dataset.metadata.vocabulary.get_id("ohne").unwrap()),
        SampleValue::String(dataset.metadata.vocabulary.get_id("leicht").unwrap()),
        SampleValue::String(dataset.metadata.vocabulary.get_id("Streifen").unwrap()),
        SampleValue::String(dataset.metadata.vocabulary.get_id("normal").unwrap()),
        SampleValue::String(dataset.metadata.vocabulary.get_id("normal").unwrap()),
    ];
    let result1 = dt.classify(&complete_sample, false);
    print_classification_result(&complete_sample, &result1, &dataset.metadata.vocabulary);

    // Example 2: A sample with missing data
    let missing_sample: Sample = vec![
        SampleValue::None,
        SampleValue::String(dataset.metadata.vocabulary.get_id("leicht").unwrap()),
        SampleValue::None,
        SampleValue::String(dataset.metadata.vocabulary.get_id("Flocken").unwrap()),
        SampleValue::String(dataset.metadata.vocabulary.get_id("fiepend").unwrap()),
    ];
    let result2 = dt.classify(&missing_sample, true);
    print_classification_result(&missing_sample, &result2, &dataset.metadata.vocabulary);

    if let Some(filename) = plot {
        export_graph(&dt, filename)?;
    }

    Ok(())
}

pub fn bigger_example(
    criterion: Criterion,
    plot: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dataset: LoadedDataset = load_single_csv("data/iris.csv", None, true)?;

    let mut dt = DecisionTreeBuilder::new()
        .criterion(criterion)
        .min_samples_split(2)
        .build(
            dataset.data,
            &dataset.metadata
        )?;

    println!("{dt}");

    // Prune the tree
    dt.prune(0.5, criterion, true);
    println!("{dt}");

    println!("\n--- Classification Examples ---");

    // Example 1: A sample with complete data
    let complete_sample: Sample = vec![
        SampleValue::Numeric(6.0),
        SampleValue::Numeric(2.2),
        SampleValue::Numeric(5.0),
        SampleValue::Numeric(1.5),
    ];
    let result1 = dt.classify(&complete_sample, false);
    print_classification_result(&complete_sample, &result1, &dataset.metadata.vocabulary);

    // Example 2: A sample with missing data
    let missing_sample: Sample = vec![
        SampleValue::None,
        SampleValue::None,
        SampleValue::None,
        SampleValue::Numeric(1.5),
    ];
    let result2 = dt.classify(&missing_sample, true);
    print_classification_result(&missing_sample, &result2, &dataset.metadata.vocabulary);

    if let Some(filename) = plot {
        export_graph(&dt, filename)?;
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Which example to run: 1=small dataset (tbc), 2=larger dataset (iris)
    #[arg(value_parser = clap::value_parser!(u8).range(1..=2))]
    example: u8,

    /// Criterion to use for splits
    #[arg(long, default_value = "entropy", value_parser = ["entropy", "gini"])]
    criterion: String,

    /// Export the decision tree as a Graphviz image (e.g. tree.png or tree.svg)
    #[arg(long)]
    plot: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let criterion = Criterion::from_str(&args.criterion)?;

    match args.example {
        1 => small_example(criterion, args.plot.as_deref()),
        2 => bigger_example(criterion, args.plot.as_deref()),
        _ => {
            println!("Invalid example - should be 1 or 2");
            Ok(())
        }
    }
}
