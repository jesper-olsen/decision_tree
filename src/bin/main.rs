use clap::Parser;
use decision_tree::data::{Sample, SampleValue, load_single_csv, Vocabulary};
use decision_tree::DecisionTree;
use std::collections::HashMap;

pub fn print_classification_result(sample: &Sample, result: &HashMap<usize, f64>, vocab: &Vocabulary) {
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
        let class_name=vocab.get_str(*class_id).unwrap();
        if score.fract() != 0.0 {
            println!("    - {class_name:<12}: {score:.4}");
        } else {
            println!("    - {class_name:<12}: {}", *score as usize);
        }
    }
    println!("{}", "-".repeat(40));
}

pub fn small_example(
    criterion: &str,
    plot: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (header, training_data, vocab, _num_classes) = load_single_csv("data/tbc.csv", None, true)?;
    let dt = DecisionTree::train(training_data, header, &vocab, criterion, None, 2);
    println!("{dt}");

    println!("\n--- Classification Examples ---");

    // Example 1: A sample with complete data
    let complete_sample: Sample = vec![
        SampleValue::String(vocab.get_id("ohne").unwrap()),
        SampleValue::String(vocab.get_id("leicht").unwrap()),
        SampleValue::String(vocab.get_id("Streifen").unwrap()),
        SampleValue::String(vocab.get_id("normal").unwrap()),
        SampleValue::String(vocab.get_id("normal").unwrap()),
    ];
    let result1 = dt.classify(&complete_sample, false);
    print_classification_result(&complete_sample, &result1, &vocab);

    // Example 2: A sample with missing data
    let missing_sample: Sample = vec![
        SampleValue::None,
        SampleValue::String(vocab.get_id("leicht").unwrap()),
        SampleValue::None,
        SampleValue::String(vocab.get_id("Flocken").unwrap()),
        SampleValue::String(vocab.get_id("fiepend").unwrap()),
    ];
    let result2 = dt.classify(&missing_sample, true);
    print_classification_result(&missing_sample, &result2, &vocab);

    if let Some(filename) = plot {
        dt.export_graph(filename);
    }

    Ok(())
}

pub fn bigger_example(
    criterion: &str,
    plot: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (header, training_data, vocab, _num_classes) = load_single_csv("data/iris.csv", None, true)?;
    let mut dt = DecisionTree::train(training_data, header, &vocab, criterion, None, 2);
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
    print_classification_result(&complete_sample, &result1, &vocab);

    // Example 2: A sample with missing data
    let missing_sample: Sample = vec![
        SampleValue::None,
        SampleValue::None,
        SampleValue::None,
        SampleValue::Numeric(1.5),
    ];
    let result2 = dt.classify(&missing_sample, true);
    print_classification_result(&missing_sample, &result2, &vocab);

    if let Some(filename) = plot {
        dt.export_graph(filename);
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

    match args.example {
        1 => small_example(&args.criterion, args.plot.as_deref()),
        2 => bigger_example(&args.criterion, args.plot.as_deref()),
        _ => {
            println!("Invalid example - should be 1 or 2");
            Ok(())
        }
    }
}
