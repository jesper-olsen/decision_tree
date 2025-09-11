use clap::Parser;
use decision_tree::{DecisionTree, Sample, SampleValue, load_csv, print_classification_result};

pub fn small_example(
    criterion: &str,
    plot: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (header, training_data) = load_csv("data/tbc.csv")?;
    let dt = DecisionTree::train(training_data, header, criterion, None, 2);
    println!("{}", dt);

    println!("\n--- Classification Examples ---");

    // Example 1: A sample with complete data
    let complete_sample: Sample = vec![
        SampleValue::String("ohne".to_string()),
        SampleValue::String("leicht".to_string()),
        SampleValue::String("Streifen".to_string()),
        SampleValue::String("normal".to_string()),
        SampleValue::String("normal".to_string()),
    ];
    let result1 = dt.classify(&complete_sample, false);
    print_classification_result(&complete_sample, &result1);

    // Example 2: A sample with missing data
    let missing_sample: Sample = vec![
        SampleValue::None,
        SampleValue::String("leicht".to_string()),
        SampleValue::None,
        SampleValue::String("Flocken".to_string()),
        SampleValue::String("fiepend".to_string()),
    ];
    let result2 = dt.classify(&missing_sample, true);
    print_classification_result(&missing_sample, &result2);

    if let Some(filename) = plot {
        dt.export_graph(filename);
    }

    Ok(())
}

pub fn bigger_example(
    criterion: &str,
    plot: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (header, training_data) = load_csv("data/fishiris.csv")?;
    let mut dt = DecisionTree::train(training_data, header, criterion, None, 2);
    println!("{}", dt);

    // Prune the tree
    dt.prune(0.5, criterion, true);
    println!("{}", dt);

    println!("\n--- Classification Examples ---");

    // Example 1: A sample with complete data
    let complete_sample: Sample = vec![
        SampleValue::Float(6.0),
        SampleValue::Float(2.2),
        SampleValue::Float(5.0),
        SampleValue::Float(1.5),
    ];
    let result1 = dt.classify(&complete_sample, false);
    print_classification_result(&complete_sample, &result1);

    // Example 2: A sample with missing data
    let missing_sample: Sample = vec![
        SampleValue::None,
        SampleValue::None,
        SampleValue::None,
        SampleValue::Float(1.5),
    ];
    let result2 = dt.classify(&missing_sample, true);
    print_classification_result(&missing_sample, &result2);

    if let Some(filename) = plot {
        dt.export_graph(filename);
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Which example to run: 1=small dataset (tbc), 2=larger dataset (fishiris)
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
