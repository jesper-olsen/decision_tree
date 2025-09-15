use crate::data::{SampleValue, Vocabulary};
use crate::node::{Node, NodeKind};
use crate::tree::DecisionTree;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

/// Exports the decision tree to an image file by generating a .dot file
/// and calling the `dot` command-line tool.
///
/// This function requires Graphviz to be installed and in the system's PATH.
pub fn export_graph(tree: &DecisionTree, filename: &str) -> io::Result<()> {
    let dot_content = generate_dot(tree);

    // Write the content to a temporary .dot file
    let output_path = Path::new(filename);
    let dot_filename = output_path.with_extension("dot");
    let mut file = File::create(&dot_filename)?;
    file.write_all(dot_content.as_bytes())?;
    println!("Generated temporary file: {}", dot_filename.display());

    // Call the `dot` command-line tool to generate the image
    let ext = output_path
        .extension()
        .unwrap_or_else(|| "png".as_ref())
        .to_str()
        .unwrap();
    let status = Command::new("dot")
        .arg(format!("-T{ext}"))
        .arg("-o")
        .arg(output_path)
        .arg(&dot_filename)
        .status()?;

    if status.success() {
        println!("Decision tree exported to {}", output_path.display());
        // 4. Clean up the temporary .dot file
        std::fs::remove_file(&dot_filename)?;
    } else {
        eprintln!("Error: Graphviz `dot` command failed. Is Graphviz installed and in your PATH?");
        eprintln!(
            "The DOT source file was saved at: {}",
            dot_filename.display()
        );
    }

    Ok(())
}

/// Generates DOT format representation of the tree
fn generate_dot(tree: &DecisionTree) -> String {
    let mut dot_content = String::new();
    dot_content.push_str("digraph Tree {\n");
    dot_content.push_str("    node [shape=box, style=\"filled, rounded\"];\n\n");
    let mut node_counter = 0;
    export_node_recursive(
        &tree.root,
        tree.header,
        tree.vocab,
        &mut dot_content,
        &mut node_counter,
    );
    dot_content.push_str("}\n");

    dot_content
}

/// Recursive helper to generate the .dot format string for a node.
/// Returns the unique ID of the node it just processed.
fn export_node_recursive(
    node: &Node,
    header: &[String],
    vocab: &Vocabulary,
    dot_content: &mut String,
    counter: &mut usize,
) -> String {
    let node_id = format!("node{}", *counter);
    *counter += 1;

    // The logic inside this match block is the only part that changes.
    let (label, fillcolor) = match &node.kind {
        // --- LEAF NODE ---
        NodeKind::Leaf { class_counts } => {
            let mut sorted_counts: Vec<_> = class_counts.iter().collect();
            sorted_counts.sort_by_key(|&(k, _)| k);

            // Format each class count as its own table row for alignment
            let counts_rows = sorted_counts
                .iter()
                .map(|(k, v)| {
                    let class_name = vocab.get_str(**k).unwrap_or("?");
                    format!(r#"<TR><TD ALIGN="LEFT">{class_name}: {v}</TD></TR>"#)
                })
                .collect::<String>();

            // Use a borderless table to enforce left alignment for all content.
            let label_html = format!(
                r#"<
                    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                    <TR><TD><B>Leaf</B></TD></TR>
                    {}
                    <TR><TD ALIGN="LEFT">impurity = {:.3}</TD></TR>
                    <TR><TD ALIGN="LEFT">samples = {}</TD></TR>
                    </TABLE>
                    >"#,
                counts_rows, node.summary.impurity, node.summary.samples
            );
            (label_html, "#e58139aa")
        }
        // --- INTERNAL NODE ---
        NodeKind::Internal {
            col,
            value,
            true_branch,
            false_branch,
        } => {
            let column_name = &header[*col];
            let condition = match value {
                SampleValue::Numeric(n) => format!("{column_name} &ge; {n:.2}"),
                SampleValue::String(id) => {
                    format!("{} == {}", column_name, vocab.get_str(*id).unwrap_or("?"))
                }
                SampleValue::None => format!("{column_name} == None"),
            };

            // Also use a table here for alignment
            let label_html = format!(
                r#"<
                    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                    <TR><TD><B>{}</B></TD></TR>
                    <TR><TD ALIGN="LEFT">impurity = {:.3}</TD></TR>
                    <TR><TD ALIGN="LEFT">samples = {}</TD></TR>
                    </TABLE>
                    >"#,
                condition.replace('<', "&lt;").replace('>', "&gt;"),
                node.summary.impurity,
                node.summary.samples
            );

            let true_child_id =
                export_node_recursive(true_branch, header, vocab, dot_content, counter);
            let false_child_id =
                export_node_recursive(false_branch, header, vocab, dot_content, counter);

            dot_content.push_str(&format!(
                "    {node_id} -> {true_child_id} [label=\"True\"];\n",
            ));
            dot_content.push_str(&format!(
                "    {node_id} -> {false_child_id} [label=\"False\"];\n",
            ));

            (label_html, "#399de5aa")
        }
    };

    dot_content.push_str(&format!(
        "    {node_id} [label={label}, fillcolor=\"{fillcolor}\"];\n",
    ));

    node_id
}
