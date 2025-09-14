use clap::Parser;
use decision_tree::DecisionTree;
use decision_tree::data::{Sample, SampleValue, Vocabulary, load_single_csv, load_train_test_csv};
use decision_tree::export::export_graph;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train and evaluate a decision tree on a given dataset", long_about = None)]
struct Args {
    /// Path to the training CSV dataset file
    file_path: String,

    /// Optional path to a separate CSV test dataset file. If provided, overrides split/k-fold
    #[arg(long)]
    test_file: Option<String>,

    /// Proportion for training in a simple split
    #[arg(short, long, default_value = "0.8")]
    split_ratio: f64,

    /// The splitting criterion to use
    #[arg(short, long, default_value = "gini", value_parser = ["entropy", "gini"])]
    criterion: String,

    /// Minimum gain to keep a branch (0.0 = no pruning)
    #[arg(short, long, default_value = "0.0")]
    prune: f64,

    /// Export the decision tree as a Graphviz image
    #[arg(long)]
    plot: Option<String>,

    /// Number of folds for k-fold cross-validation. If > 1, this overrides --split_ratio
    #[arg(short, long, default_value = "0")]
    k_folds: usize,

    /// Maximum depth of the tree (None = unlimited)
    #[arg(long)]
    max_depth: Option<usize>,

    /// Minimum number of samples required to split a node
    #[arg(long, default_value = "2")]
    min_samples_split: usize,
}

fn create_folds(data: &[Sample], k: usize) -> Vec<Vec<Sample>> {
    // Clone and shuffle the data
    let mut shuffled_data = data.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(42); // Use a seed for reproducible results
    shuffled_data.shuffle(&mut rng);

    let mut folds = Vec::new();
    let fold_size = shuffled_data.len() / k;
    let mut start_index = 0;

    for i in 0..k {
        let end_index = if i < k - 1 {
            start_index + fold_size
        } else {
            shuffled_data.len()
        };

        folds.push(shuffled_data[start_index..end_index].to_vec());
        start_index = end_index;
    }

    folds
}

fn split_data(data: &[Sample], split_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    let mut shuffled_data = data.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    shuffled_data.shuffle(&mut rng);

    let split_index = (shuffled_data.len() as f64 * split_ratio) as usize;
    let train_data = shuffled_data[..split_index].to_vec();
    let test_data = shuffled_data[split_index..].to_vec();

    (train_data, test_data)
}

fn calculate_accuracy(model: &DecisionTree, test_data: &[Sample]) -> f64 {
    if test_data.is_empty() {
        return 0.0;
    }

    let mut correct_predictions = 0;
    let total_predictions = test_data.len();

    for row in test_data {
        let features = &row[..row.len() - 1];
        let true_label = &row[row.len() - 1];

        let predicted_label = model.predict(&features.to_vec());
        if SampleValue::String(predicted_label) == *true_label {
            correct_predictions += 1;
        }
    }

    (correct_predictions as f64 / total_predictions as f64) * 100.0
}

fn kfold_eval(data: Vec<Sample>, header: Vec<String>, vocab: Vocabulary, args: &Args) {
    println!("\nPerforming {}-fold cross-validation...", args.k_folds);
    let folds = create_folds(&data, args.k_folds);
    let mut fold_accuracies = Vec::new();

    for i in 0..args.k_folds {
        // The i-th fold is the test set
        let test_data = &folds[i];

        // All other folds combined are the training set
        let mut train_data = Vec::new();
        for (j, fold) in folds.iter().enumerate() {
            if i != j {
                train_data.extend_from_slice(fold);
            }
        }

        println!("\n--- Fold {}/{} ---", i + 1, args.k_folds);
        println!(
            "Training on {} samples, testing on {} samples.",
            train_data.len(),
            test_data.len()
        );

        // Train the model for this fold
        let mut model = DecisionTree::train(
            train_data,
            header.clone(),
            &vocab,
            &args.criterion,
            args.max_depth,
            args.min_samples_split,
        );
        println!("Trained a model with {} nodes", model.size());

        // Prune if requested
        if args.prune > 0.0 {
            model.prune(args.prune, &args.criterion, false);
            println!("Pruned model down to {} nodes", model.size());
        }

        // Evaluate and store accuracy
        let accuracy = calculate_accuracy(&model, test_data);
        fold_accuracies.push(accuracy);
        println!("Fold {} Accuracy: {:.2}%", i + 1, accuracy);
    }

    // Calculate statistics
    println!("\n{}", "=".repeat(30));
    println!("Cross-Validation Summary");
    println!("{}", "=".repeat(30));

    let avg_accuracy: f64 = fold_accuracies.iter().sum::<f64>() / fold_accuracies.len() as f64;

    let std_dev = if fold_accuracies.len() > 1 {
        let variance: f64 = fold_accuracies
            .iter()
            .map(|&x| (x - avg_accuracy).powi(2))
            .sum::<f64>()
            / (fold_accuracies.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    println!("Average Accuracy: {avg_accuracy:.2}%");
    println!("Standard Deviation: {std_dev:.2}%");
    println!("{}", "=".repeat(30));
}

fn split_eval(
    data: Vec<Sample>,
    header: Vec<String>,
    vocab: Vocabulary,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nPerforming a simple train/test split...");
    let (train_data, test_data) = split_data(&data, args.split_ratio);
    println!(
        "Data split into {} training samples and {} test samples.",
        train_data.len(),
        test_data.len()
    );
    println!("{}", "-".repeat(30));

    println!(
        "Training the decision tree model (criterion: {})...",
        args.criterion
    );
    let mut model = DecisionTree::train(
        train_data,
        header,
        &vocab,
        &args.criterion,
        args.max_depth,
        args.min_samples_split,
    );
    println!("Trained a model with {} nodes", model.size());

    if args.prune > 0.0 {
        println!("Pruning the tree with min_gain = {}...", args.prune);
        model.prune(args.prune, &args.criterion, true);
        println!("Pruned down to {} nodes", model.size());
    } else {
        println!("No pruning requested.");
    }

    if let Some(ref plot_file) = args.plot {
        export_graph(&model, plot_file)?;
    }

    println!("\nEvaluating model accuracy on the test set...");
    let accuracy = calculate_accuracy(&model, &test_data);

    println!("\n--- Evaluation Result ---");
    println!("Model Accuracy: {accuracy:.2}%");
    println!("{}", "-".repeat(30));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if let Some(test_file) = args.test_file {
        // Case 1: External test file provided
        let (header, train_data, test_data, vocab, _num_classes) =
            load_train_test_csv(&args.file_path, &test_file, None, true)?;

        // Train
        println!(
            "\nTraining decision tree (criterion: {})...",
            args.criterion
        );
        let mut model = DecisionTree::train(
            train_data,
            header,
            &vocab,
            &args.criterion,
            args.max_depth,
            args.min_samples_split,
        );
        println!("Trained a model with {} nodes", model.size());

        if args.prune > 0.0 {
            println!("Pruning the tree with min_gain = {}...", args.prune);
            model.prune(args.prune, &args.criterion, true);
            println!("Pruned model down to {} nodes", model.size());
        }

        if let Some(ref plot_file) = args.plot {
            export_graph(&model, plot_file)?;
        }

        println!("\nEvaluating model accuracy on the external test set...");
        let accuracy = calculate_accuracy(&model, &test_data);
        println!("\n--- Evaluation Result ---");
        println!("Model Accuracy: {accuracy:.2}%");
        println!("{}", "-".repeat(30));
    } else {
        // Case 2: Single file, split or k-fold
        let (header, data, vocab, _num_classes) = load_single_csv(&args.file_path, None, true)?;
        println!("Dataset loaded successfully with {} rows.", data.len());

        // Choose evaluation method
        if args.k_folds > 1 {
            kfold_eval(data, header, vocab, &args);
        } else {
            split_eval(data, header, vocab, &args)?;
        }
    }

    Ok(())
}
