//! Helper functions for the Needleman-Wunsch algorithm.

use crate::{number::UInt, strings::{levenshtein, Penalties}};

/// The direction of best alignment at a given position in the DP table
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Diagonal (Up and Left) for a match.
    Diagonal,
    /// Up for a gap in the first sequence.
    Up,
    /// Left for a gap in the second sequence.
    Left,
}

/// The type of edit needed to turn one sequence into another.
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum Edit {
    /// Delete a character at the given index.
    Del(usize),
    /// Insert a character at the given index.
    Ins(usize, char),
    /// Substitute a character at the given index.
    Sub(usize, char),
}

/// Computes the Needleman-Wunsch dynamic programming table for two sequences.
///
/// Users can input custom penalities, but those penalties should satisfy the following:
/// * All penalties are non-negative
/// * The match "penalty" is 0
///
///  Our implementation *minimizes* the total penalty.
///
/// # Arguments
///
/// * `x`: The first sequence.
/// * `y`: The second sequence.
/// * `penalties`: The penalties for a match, mismatch, and gap.
///
/// # Returns
///
/// A nested vector of tuples of total-penalty and Direction, representing the
/// best alignment at each position.
pub fn compute_table<U: UInt>(
    x: &str,
    y: &str,
    penalties: Penalties<U>,
) -> Vec<Vec<(U, Direction)>> {
    // Initializing table; the inner vectors represent rows in the table.
    let mut table = vec![vec![(U::zero(), Direction::Diagonal); x.len() + 1]; y.len() + 1];

    // The top-left cell starts with a total penalty of zero and no direction.
    table[0][0] = (U::zero(), Direction::Diagonal);

    // Initialize left-most column of distance values.
    for (i, row) in table.iter_mut().enumerate().skip(1) {
        row[0] = (penalties.gap * U::from(i), Direction::Up);
    }

    // Initialize top row of distance values.
    for (j, cell) in table[0].iter_mut().enumerate().skip(1) {
        *cell = (penalties.gap * U::from(j), Direction::Left);
    }

    // Set values for the body of the table
    for (i, y_c) in y.chars().enumerate() {
        for (j, x_c) in x.chars().enumerate() {
            // Check if sequences match at position `i` in `x` and `j` in `y`.
            let mismatch_penalty = if x_c == y_c {
                penalties.match_
            } else {
                penalties.mismatch
            };

            // Compute the three possible penalties and use the minimum to set
            // the value for the next entry in the table.
            let d00 = (table[i][j].0 + mismatch_penalty, Direction::Diagonal);
            let d01 = (table[i][j + 1].0 + penalties.gap, Direction::Up);
            let d10 = (table[i + 1][j].0 + penalties.gap, Direction::Left);

            table[i + 1][j + 1] = min2(d00, min2(d01, d10));
        }
    }

    table
}

/// Returns the minimum of two penalties, defaulting to the first input.
fn min2<U: UInt>(a: (U, Direction), b: (U, Direction)) -> (U, Direction) {
    if a.0 <= b.0 {
        a
    } else {
        b
    }
}

/// Iteratively traces back through the Needleman-Wunsch table to get the alignment of two sequences.
///
/// For now, we ignore ties in the paths that can be followed to get the best alignments.
/// We break ties by always choosing the `Diagonal` path first, then the `Left`
/// path, then the `Up` path.
///
/// # Arguments
///
/// * `table`: The Needleman-Wunsch table.
/// * `[x, y]`: The two sequences to align.
///
/// # Returns
///
/// A tuple of the two aligned sequences.
#[must_use]
pub fn trace_back_iterative<U: UInt>(
    table: &[Vec<(U, Direction)>],
    [x, y]: [&str; 2],
) -> (String, String) {
    let (x, y) = (x.as_bytes(), y.as_bytes());

    let (mut row_i, mut col_i) = (y.len(), x.len());
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    while row_i > 0 || col_i > 0 {
        match table[row_i][col_i].1 {
            Direction::Diagonal => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
                col_i -= 1;
            }
            Direction::Left => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(b'-');
                col_i -= 1;
            }
            Direction::Up => {
                aligned_x.push(b'-');
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
            }
        }
    }

    aligned_x.reverse();
    aligned_y.reverse();

    let aligned_x = String::from_utf8(aligned_x)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));
    let aligned_y = String::from_utf8(aligned_y)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));

    (aligned_x, aligned_y)
}

/// Recursively traces back through the Needleman-Wunsch table to get the alignment of two sequences.
///
/// For now, we ignore ties in the paths that can be followed to get the best alignments.
/// We break ties by always choosing the `Diagonal` path first, then the `Left`
/// path, then the `Up` path.
///
/// # Arguments
///
/// * `table`: The Needleman-Wunsch table.
/// * `[x, y]`: The two sequences to align.
///
/// # Returns
///
/// A tuple of the two aligned sequences.
#[must_use]
pub fn trace_back_recursive<U: UInt>(
    table: &[Vec<(U, Direction)>],
    [x, y]: [&str; 2],
) -> (String, String) {
    let (mut aligned_x, mut aligned_y) = (Vec::new(), Vec::new());

    _trace_back_recursive(
        table,
        [y.len(), x.len()],
        [x.as_bytes(), y.as_bytes()],
        [&mut aligned_x, &mut aligned_y],
    );

    aligned_x.reverse();
    aligned_y.reverse();

    let aligned_x = String::from_utf8(aligned_x)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));
    let aligned_y = String::from_utf8(aligned_y)
        .unwrap_or_else(|_| unreachable!("We know we added valid characters."));

    (aligned_x, aligned_y)
}

/// Helper function for `trace_back_recursive`.
///
/// # Arguments
///
/// * `table`: The Needleman-Wunsch table.
/// * `[row_i, col_i]`: mutable indices into the table.
/// * `[x, y]`: The two sequences to align, passed as slices of bytes.
/// * `[aligned_x, aligned_y]`: mutable aligned sequences that will be built
/// up from initially empty vectors.
fn _trace_back_recursive<U: UInt>(
    table: &[Vec<(U, Direction)>],
    [mut row_i, mut col_i]: [usize; 2],
    [x, y]: [&[u8]; 2],
    [aligned_x, aligned_y]: [&mut Vec<u8>; 2],
) {
    if row_i > 0 || col_i > 0 {
        match table[row_i][col_i].1 {
            Direction::Diagonal => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
                col_i -= 1;
            }
            Direction::Left => {
                aligned_x.push(x[col_i - 1]);
                aligned_y.push(b'-');
                col_i -= 1;
            }
            Direction::Up => {
                aligned_x.push(b'-');
                aligned_y.push(y[row_i - 1]);
                row_i -= 1;
            }
        };
        _trace_back_recursive(table, [row_i, col_i], [x, y], [aligned_x, aligned_y]);
    }
}

/// Converts two aligned sequences into vectors of edits.
///
/// # Arguments
///
/// * `x`: The first aligned sequence.
/// * `y`: The second aligned sequence.
///
/// # Returns
///
/// A 2-slice of Vec<Edit>, each containing the edits needed to convert one aligned
/// sequence into the other.
/// Since both input sequences are aligned, all edits are substitutions in the returned vectors are Substitutions.
#[must_use]
pub fn compute_edits(x: &str, y: &str) -> [Vec<Edit>; 2] {
    [_x_to_y(x, y), _x_to_y(y, x)]
}

/// Helper for `compute_edits` to compute the edits for turning aligned `x` into aligned `y`.
///
/// Expects `x` and `y` to be aligned sequences generated by `trace_back_iterative` or `trace_back_recursive`.
/// Returns a vector of substitutions needed to turn `x` into `y`.
///
/// # Arguments
///
/// * `x`: The first aligned sequence.
/// * `y`: The second aligned sequence.
///
/// # Returns
///
/// A vector of edits needed to convert `x` into `y`.
#[must_use]
pub fn _x_to_y(x: &str, y: &str) -> Vec<Edit> {
    x.chars()
        .zip(y.chars())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, (_, y))| Edit::Sub(i, y))
        .collect()
}

/// Given two aligned strings, returns into a sequence of edits to transform the unaligned
/// version of one into the unaligned version of the other.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Expects to receive two strings which were aligned using either `trace_back_iterative` or `trace_back_recursive`.
///
/// # Arguments
///
/// * `x`: An aligned string.
/// * `y`: An aligned string.
///
/// # Returns
///
/// A vector of edits to transform the unaligned version of `x` into the unaligned version of `y`.
#[must_use]
pub fn unaligned_x_to_y(x: &str, y: &str) -> Vec<Edit> {
    let mut unaligned_x_to_y = Vec::new();
    let mut modifier = 0;

    x.chars()
        .zip(y.chars())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;
            if c_x == '-' {
                unaligned_x_to_y.push(Edit::Ins(i, c_y));
            } else if c_y == '-' {
                unaligned_x_to_y.push(Edit::Del(i));
                modifier += 1;
            } else {
                unaligned_x_to_y.push(Edit::Sub(i, c_y));
            }
        });
    unaligned_x_to_y
}

/// Given two unaligned strings, returns into a sequence of edits to transform the unaligned
/// version of one into the unaligned version of the other.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Expects to receive two strings which were aligned using either `trace_back_iterative` or `trace_back_recursive`.
///
/// # Arguments
///
/// * `x`: A unaligned string.
/// * `y`: A unaligned string.
///
/// # Returns
///
/// A vector of edits to transform thenaligned version of `x` into the aligned version of `y`.
pub fn aligned_x_to_y(x: &str, y: &str) -> Vec<Edit> {
    let table = compute_table::<u16>(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    let mut aligned_x_to_y: Vec<Edit> = Vec::new();
    let mut modifier = 0;

    aligned_x
        .chars()
        .zip(aligned_y.chars())
        .enumerate()
        .filter(|(_, (aligned_x, aligned_y))| aligned_x != aligned_y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;

            if c_x == '-' {
                aligned_x_to_y.push(Edit::Ins(i, c_y));
            } else if c_y == '-' {
                aligned_x_to_y.push(Edit::Del(i));
                modifier += 1;
            } else {
                aligned_x_to_y.push(Edit::Sub(i, c_y));
            }
        });
    aligned_x_to_y
}

/// Given two unaligned strings, returns the edits related to gaps to align the 2 strings.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Uses the `traceback_iterative` ftn to align the strings.
///
/// # Arguments
///
/// * `x`: A unaligned string.
/// * `y`: A unaligned string.
///
/// # Returns
///
/// A vector of edits to transform the aligned version of `x` into the aligned version of `y` excluding substitutions.
pub fn aligned_x_to_y_no_sub(x: &str, y: &str) -> Vec<Edit> {
    let table = compute_table::<u16>(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    let mut aligned_x_to_y: Vec<Edit> = Vec::new();
    let mut modifier = 0;
    aligned_x
        .chars()
        .zip(aligned_y.chars())
        .enumerate()
        .filter(|(_, (aligned_x, aligned_y))| aligned_x != aligned_y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;

            if c_x == '-' {
                aligned_x_to_y.push(Edit::Ins(i, c_y));
            } else if c_y == '-' {
                aligned_x_to_y.push(Edit::Del(i));
                modifier += 1;
            }
        });
    aligned_x_to_y
}

/// Given two unaligned strings, returns the location of the gaps needed to align the 2 strings.
///
/// Requires that gaps are never aligned with gaps (our NW implementation with the default penalties ensures this).
/// Uses the `traceback_iterative` ftn to align the strings.
///
/// # Arguments
///
/// * `x`: A unaligned string.
/// * `y`: A unaligned string.
///
/// # Returns
///
/// An array of 2 vectors of gaps to align `x` and `y`.
pub fn x_to_y_alignment(x: &str, y: &str) -> [Vec<usize>; 2] {
    let table = compute_table::<u16>(x, y, Penalties::default());
    let (aligned_x, aligned_y) = trace_back_iterative(&table, [x, y]);
    let mut gap_indices: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    let mut modifier: usize = 0;
    aligned_x
        .chars()
        .zip(aligned_y.chars())
        .enumerate()
        .filter(|(_, (aligned_x, aligned_y))| aligned_x != aligned_y)
        .for_each(|(index, (c_x, c_y))| {
            let i = index - modifier;
            if c_x == '-' {
                gap_indices[0].push(index);
            } else if c_y == '-' {
                gap_indices[1].push(i);
                modifier += 1;
            }
        });
    gap_indices
}

/// Helper function to get gap indices to transform cluster_center and aligned_x into the other
/// 
/// # Arguments:
/// * 'cluster_center': an aligned center of a MSA
/// * 'aligned_x': an aligned sequence
/// 
/// Returns:
/// 
/// An array of 2 vectors of gaps to transform `cluster_x` and 'align_x'.
fn get_center_gaps(cluster_center: &str, aligned_x: &str) -> (Vec<usize>, Vec<usize>) {
    let get_gaps = |s: &str| -> Vec<usize> {
        s.chars()
            .enumerate()
            .filter_map(|(j, c)| if c == '-' { Some(j) } else { None })
            .collect()
    };

    // Get and return gaps for the original center and new alignments
    (get_gaps(cluster_center), get_gaps(aligned_x))
}

/// Adjusts the indices of gaps by subtracting the index value.
/// 
/// # Arguments: 
/// * 'indices': an vector of indices of gaps
fn adjust_gaps(indices: &mut Vec<usize>) {
    indices.iter_mut().enumerate().for_each(|(i, x)| {
        if i > 0 {
            *x -= i;
        }
    });
}

// Inserts gaps into aligned strings based on gap indices.
/// # Arguments:
/// 
/// * aligned_x: an aligned sequence
/// * aligned_y: an aligned sequence
/// * gap: the number of gaps already in the sequence
/// * index: the index of the location of the gap in the non-aligned sequence
fn insert_gaps_into_strings(aligned_x: &mut String, aligned_y: &mut String, gap: usize, index: usize) {
    if gap == aligned_x.len() - index {
        aligned_x.push('-');
        aligned_y.push('-');
    } else {
        aligned_x.insert(gap + index, '-');
        aligned_y.insert(gap + index, '-');
    }
}

/// Adjusts gaps in the entire cluster based on the gap size.
/// # Arguments:
/// 
/// * aligned_cluster: an aligned cluster
/// * gap: the number of gaps already in the cluster
/// * index: the index of the location of the gap in the non-aligned sequence
fn adjust_cluster_gaps(aligned_cluster: &mut Vec<String>, gap: usize, index: usize) {
    if gap == aligned_cluster[0].len() - index {
        aligned_cluster.iter_mut().for_each(|s| s.push('-'));
    } else {
        aligned_cluster.iter_mut().for_each(|s| s.insert(gap + index, '-'));
    }
}

/// Takes in a cluster x and aligns it 
/// # Arguments: 
/// 
/// * `x`: a unaligned cluster.
/// 
/// # Returns:
/// 
/// * A vector representing the MSA of the inputted cluster.
fn align_cluster(x: Vec<String>) -> Vec<String> {
    // Initialize aligned_cluster with the first two strings' alignment
    let mut aligned_cluster: Vec<String> = Vec::new();
    if x.len() < 2 {
        return aligned_cluster;
    }

    for i in 1..x.len() {
        let table = compute_table::<u16>(&x[0], &x[i], Penalties::default());
        let (mut aligned_x, mut aligned_y) = trace_back_iterative(&table, [&x[0], &x[i]]);

        if i == 1 {
            aligned_cluster = vec![aligned_x, aligned_y];
        } else {
            let (mut original_center_gaps, mut new_center_gaps) = get_center_gaps(&aligned_cluster[0], &aligned_x);

            // Adjust the gap indices
            adjust_gaps(&mut original_center_gaps);
            adjust_gaps(&mut new_center_gaps);

            // Merge gaps and update alignments
            let mut k = 0;
            while k < std::cmp::max(original_center_gaps.len(), new_center_gaps.len()) {
                let (oc, nc) = (original_center_gaps.get(k), new_center_gaps.get(k));
                match (oc, nc) {
                    (Some(&oc_val), Some(&nc_val)) => {
                        if oc_val < nc_val {
                            new_center_gaps.insert(k, oc_val);
                            insert_gaps_into_strings(&mut aligned_x, &mut aligned_y, oc_val, k);
                        } else if oc_val > nc_val {
                            original_center_gaps.insert(k, nc_val);
                            adjust_cluster_gaps(&mut aligned_cluster, nc_val, k);
                        }
                    }
                    (Some(&oc_val), None) => {
                        new_center_gaps.push(oc_val);
                        insert_gaps_into_strings(&mut aligned_x, &mut aligned_y, oc_val, k);
                    }
                    (None, Some(&nc_val)) => {
                        original_center_gaps.push(nc_val);
                        adjust_cluster_gaps(&mut aligned_cluster, nc_val, k);
                    }
                    (None, None) => break,
                }
                k += 1;
            }
            aligned_cluster.push(aligned_y.clone());
        }
    }
    aligned_cluster
}

/// Takes in a cluster x, and its center that was aligned to data outside the cluster and combines the two alignments
/// # Arguments: 
/// 
/// * `x`: an aligned cluster.
/// * 'center': the center of the cluster that was aligned outside
/// 
/// # Returns:
/// 
/// * A vector representing the MSA of the inputted cluster that also includes the outside alignment.
fn add_gaps_to_cluster(mut x: Vec<String>, center: String, center_index: usize) -> Vec<String>{
    let mut index = 0;
    while  index < center.len(){
        // If the needed gap is not at the end, insert it in every sequence within x in the correct position.
        if index >= x[center_index].len() {
            for j in 0..x.len() {
                x[j].push('-')
            }
        }
        // If the needed gap is at the end, add it to the end of every sequence in x.
        else if center.chars().nth(index).unwrap() != x[center_index].chars().nth(index).unwrap() {
            for j in 0..x.len() {
                x[j].insert(index, '-')
            }
        }       
        index += 1;
    }
    return x;
}


fn find_gaps(x: &str, y: &str, index: usize) -> u8 {
    return match x.chars().into_iter().nth(index) {
        Some('-') => {match y.chars().into_iter().nth(index) {
            Some('-') => 3,
            Some(_) => 1,
            None => panic!("Error")
        }}
        Some(_) => {match y.chars().into_iter().nth(index) {
            Some('-') => 2,
            Some(_) => 0,
            None => panic!("Error")
        }}
        None => panic!("Error")
    };
}

/// Takes in two MSA clusters and combines them into one MSA
/// 
/// All Strings in the same Vec must have the same length
/// 
/// # Arguments: 
/// 
/// * `x`: an aligned cluster.
/// * `y`: an aligned cluster.
/// 
/// # Returns:
/// 
/// * A vector representing the combined MSA of the two inputted clusters.
#[must_use] 
fn align_clusters(mut aligned_x: Vec<String>, mut aligned_y: Vec<String>) -> Vec<String> {
    //Aligns the centers
    let table = compute_table::<u16>(&aligned_x[0], &aligned_y[0], Penalties::default());
    let (aligned_x_center, aligned_y_center) = trace_back_iterative(&table, [&aligned_x[0], &aligned_y[0]]);

    aligned_x = add_gaps_to_cluster(aligned_x, aligned_x_center, 0);
    aligned_y = add_gaps_to_cluster(aligned_y, aligned_y_center, 0);

    aligned_x.append(&mut aligned_y);
    aligned_x
}

/// Aligns two partial-MSAs together and returns the indices where gaps were
/// added in each partial-MSA.
/// 
/// This function is akin to the `align_clusters` function, but it modifies the
/// sequences in place rather than returning a new MSA.
/// The order of the sequences in `left` and `right` is also preserved.
/// 
/// # Arguments
/// 
/// * `left`: A partial MSA.
/// * `right`: A partial MSA.
/// * `left_center`: The index of the center of the left MSA.
/// * `right_center`: The index of the center of the right MSA.
/// 
/// # Returns
/// 
/// An array of two vectors:
/// * The indices where gaps were added in the left MSA.
/// * The indices where gaps were added in the right MSA.
pub fn align_in_place(
    left: &Vec<String>,
    right: &Vec<String>,
    left_center: usize,
    right_center: usize,
) -> [Vec<usize>; 2] {
    return x_to_y_alignment(&left[left_center], &right[right_center]);
}

/// Applies a set of edits to a reference (unaligned) string to get a target (unaligned) string.
///
/// # Arguments
///
/// * `x`: The unaligned reference string.
/// * `edits`: The edits to apply to the reference string.
///
/// # Returns
///
/// The unaligned target string.
#[must_use]
pub fn apply_edits(x: &str, edits: &[Edit]) -> String {
    let mut x: Vec<char> = x.chars().collect();

    for edit in edits {
        match edit {
            Edit::Sub(i, c) => {
                x[*i] = *c;
            }
            Edit::Ins(i, c) => {
                x.insert(*i, *c);
            }
            Edit::Del(i) => {
                x.remove(*i);
            }
        }
    }
    x.into_iter().collect()
}

/// Adds a unaligned string to a MSA cluster
/// # Arguments
/// 
/// * 'cluster': A full multiple sequence alignment
/// * 'x': A unaligned string
/// 
/// Returns
/// 
/// An updated MSA including all sequences in the cluster and 'x'
fn add_x_to_cluster(cluster: &Vec<String>, x: &str) -> Vec<String> {
    // Gets the string and its index that is closest to x
    let (index, cluster_seq) = cluster.iter()
    .enumerate()
    .map(|(i, s)| (i, levenshtein::<u64>(x, s)))
    .min_by_key(|&(_, dist)| dist)
    .map(|(i, _)| (i, &cluster[i]))
    .unwrap();

    let table = compute_table::<u16>(x, &cluster_seq, Penalties::default());
    let (_, aligned_cluster_seq) = trace_back_iterative(&table, [x, &cluster_seq]);

    return add_gaps_to_cluster(cluster.to_vec(), aligned_cluster_seq, index);
}

/// All the different metrics that can be used to measure the quality of an MSA.
///
/// This, combined with the `MsaPenalties` struct, can be used to score an MSA
/// in a generic way in CLAM.
pub enum MsaMetrics {
    UnweighedScoringColumns,
    UnweighedScoringPairwise,
    WeighedScoringColumns,
    WeighedScoringPairwise,
    AvgPDistance,
    MaxPDistance,
}

impl MsaMetrics {
    /// Scores a multiple sequence alignment based on the chosen metric.
    pub fn score(&self, msa: &Vec<String>, penalties: &MsaPenalties) -> f64 {
        match self {
            MsaMetrics::UnweighedScoringColumns => {
                unweighed_scoring_columns(msa, penalties.gap_pen, penalties.mismatch_pen) as f64
            },
            MsaMetrics::UnweighedScoringPairwise => {
                unweighed_scoring_pairwise(msa, penalties.gap_pen, penalties.mismatch_pen) as f64
            },
            MsaMetrics::WeighedScoringColumns => todo!(),
            MsaMetrics::WeighedScoringPairwise => todo!(),
            MsaMetrics::AvgPDistance => todo!(),
            MsaMetrics::MaxPDistance => todo!(),
        }
    }
}

/// A struct to hold the penalties for the different scoring metrics for MSA.
pub struct MsaPenalties {
    gap_pen: u64,
    mismatch_pen: u64,
    first_gap_pen: u64,
    reg_gap_pen: u64,
}

impl MsaPenalties {
    pub fn new(gap_pen: u64, mismatch_pen: u64, first_gap_pen: u64, reg_gap_pen: u64) -> Self {
        Self {
            gap_pen,
            mismatch_pen,
            first_gap_pen,
            reg_gap_pen,
        }
    }
}

/// One of the four scoring metrics for MSA. This one scores column by column and every gap character has the same penalty 
/// # Arguments
///
/// * `x': A full multiple sequence alignment
/// * 'gap_pen': The numerical value of the penalty of 1 gap
/// * 'mismatch_pen': The numerical value of the penalty of 1 mismatch
/// 
/// Returns
/// 
/// The score based on the unweighed gap penalties calculated by columns.
fn unweighed_scoring_columns(x: &Vec<String>, gap_pen: u64, mismatch_pen: u64) -> u64 {
    let mut score: u64 = 0;

    for i in 0..x[0].len() {
        for j in 0..x.len() {
            for k in (j+1)..x.len() {
                let char_j = x[j].chars().nth(i).unwrap();
                let char_k = x[k].chars().nth(i).unwrap();

                match (char_j, char_k) {
                    // Check for gaps
                    ('-', _) | (_, '-') => score += gap_pen,
                    // Check for mismatches
                    (ch1, ch2) if ch1 != ch2 => score += mismatch_pen,
                    _ => (),
                }
            }
        }
    }
    score
}



/// One of the four scoring metrics for MSA. This one scores each pairwise alignment as a whole and every gap character has the same penalty 
/// # Arguments
///
/// * `x': A full multiple sequence alignment
/// * 'gap_pen': The penalty of one gap
/// * 'mismatch': The penalty of one mismatch.
/// 
/// Returns
/// 
/// The score based on the unweighed gap penalties calculated by pairwise alignments.
fn unweighed_scoring_pairwise(x: &Vec<String>, gap_pen: u64, mismatch_pen: u64) -> u64{
    let mut score: u64 = 0;
    for (i, seq) in x.iter().enumerate() {
        for seq2 in x[i+1..].iter() {
            score = score + gap_pen * seq.chars()
                                .zip(seq2.chars())
                                .filter(|(char_a, char_b)| *char_a == '-' || *char_b == '-')
                                .count() as u64;
            score = score + mismatch_pen * seq.chars()
                                        .zip(seq2.chars())
                                        .filter(|(char_a, char_b)| *char_a != *char_b && *char_a != '-' && *char_b != '-')
                                        .count() as u64;
        }
    }
    return score;
}

/// One of the four scoring metrics for MSA. This one scores column by column and gaps that do not immediately follow another gap weighs more than gaps that do
/// # Arguments
///
/// * `x': A full multiple sequence alignment
/// * 'first_gap_pen': The penalty of a gap that does not directly follow another
/// * 'reg_gap_pen': The penalty of a gap that directly follows another
/// 
/// Returns
/// 
/// The score based on the weighed gap penalties calculated by columns.
fn weighed_scoring_columns(x: &Vec<String>, first_gap_pen: u64, reg_gap_pen: u64, mismatch_pen: u64) -> u64 {
    let mut score: u64 = 0;

    for i in 0..x[0].len() {
        for j in 0..x.len() {
            for k in (j + 1)..x.len() {
                let char_j = x[j].chars().nth(i).unwrap();
                let char_k = x[k].chars().nth(i).unwrap();

                // Handle mismatches
                if char_j != '-' && char_k != '-' && char_j != char_k {
                    score += mismatch_pen;
                }

                // Handle gaps in j
                if char_j == '-' {
                    score += if i == 0 {
                        first_gap_pen
                    } else {
                        match x[j].chars().nth(i - 1) {
                            Some('-') => {
                                if char_k == '-' && x[k].chars().nth(i - 1).unwrap() != '-' {
                                    first_gap_pen
                                } else {
                                    reg_gap_pen
                                }
                            }
                            _ => first_gap_pen,
                        }
                    };
                }
                // Handle gaps in k
                else if char_k == '-' {
                    score += if i == 0 {
                        first_gap_pen
                    } else {
                        match x[k].chars().nth(i - 1) {
                            Some('-') => reg_gap_pen,
                            _ => first_gap_pen,
                        }
                    };
                }
            }
        }
    }
    score
}


/// One of the four scoring metrics for MSA. This one scores each pairwise alignment as a whole and gaps that do not immediately follow another gap weighs more than gaps that do
/// 
/// 
/// # Arguments
/// 
/// * `x`: A full multiple sequence alignment
/// * `first_gap_pen`: The penalty of a gap that does not directly follow another
/// * `reg_gap_pen`: The penalty of a gap that directly follows another
/// * `mismatch_pen`: The penalty for mismatches
/// 
/// # Returns
/// 
/// The score based on the weighted gap penalties calculated by pairwise alignments.
fn weighed_scoring_pairwise(x: &Vec<String>, first_gap_pen: u64, reg_gap_pen: u64, mismatch_pen: u64) -> u64 {
    let mut score: u64 = 0;

    for (i, seq1) in x.iter().enumerate() {
        for seq2 in x[i + 1..].iter() {
            // Handle the first character separately
            if seq1.chars().nth(0).unwrap() == '-' || seq2.chars().nth(0).unwrap() == '-' {
                score += first_gap_pen;
            }
            else if seq1.chars().nth(0).unwrap() != seq2.chars().nth(0).unwrap() {
                score += mismatch_pen;
            }
            // Iterate through the remaining characters
            for j in 1..seq1.len() {
                let char1 = seq1.chars().nth(j).unwrap();
                let char2 = seq2.chars().nth(j).unwrap();

                // Handle first-gap penalties
                if (char1 == '-' && seq1.chars().nth(j - 1).unwrap() != '-') || (char2 == '-' && seq2.chars().nth(j - 1).unwrap() != '-') {
                    score += first_gap_pen;
                }
                // Handle regular gaps
                else if char1 == '-' || char2 == '-' {
                    score += reg_gap_pen;
                }
                // Handle mismatches
                else if char1 != char2 {
                    score += mismatch_pen;
                }
            }
        }
    }
    score
}

// Calculates the average 'p-distance' of an aligned multiple sequence alignment (used to measure Tandy Warnow's MAGUS + eHMMs method)
/// 
/// 
/// # Arguments
/// 
/// * `x`: A full multiple sequence alignment
/// 
/// Returns
/// The average percent of mismatched base pairs in the pairwise alignments contained in x
fn calculate_avg_p_distance(x: &Vec<String>) -> f64 {
    let mut sum_p_distance: f64 = 0.0;
    let mut pairwise_count: usize = 0;
    for (i, s_0) in x.iter().enumerate(){
        for s_1 in x[i+1..].iter() {
            let mut mismatch:u64 = 0;
            for j in 0..s_0.len() {
                if s_0.chars().nth(j).unwrap() != '-' && s_1.chars().nth(j).unwrap() != '-' && s_0.chars().nth(j).unwrap() != s_1.chars().nth(j).unwrap() {
                    mismatch += 1;
                }
            }
            sum_p_distance += mismatch as f64;
            pairwise_count += 1;
        }
    }
    let avg_p = sum_p_distance / ((x[0].len() * pairwise_count) as f64);
    return format!("{:.3}", avg_p).parse().expect("Err");
}

// Calculates the maximum 'p-distance' of an aligned multiple sequence alignment (used to measure Tandy Warnow's MAGUS + eHMMs method)
/// 
/// 
/// # Arguments
/// 
/// * `x`: A full multiple sequence alignment
/// 
/// Returns
/// The maximum mismatched base pairs in the pairwise alignments contained in x
fn calculate_max_p_distance(x: &Vec<String>) -> f64 {
    let mut max_p_distance: f64 = 0.0;
    for (i, s_0) in x.iter().enumerate(){
        for s_1 in x[i+1..].iter() {
            let mut mismatch:f64 = 0.0;
            for j in 0..s_0.len() {
                if s_0.chars().nth(j).unwrap() != '-' && s_1.chars().nth(j).unwrap() != '-' && s_0.chars().nth(j).unwrap() != s_1.chars().nth(j).unwrap() {
                    mismatch += 1.0;
                }
            }
            if mismatch / s_0.len() as f64 > max_p_distance {
                max_p_distance = mismatch / (s_0.len() as f64);
            }
        }
    }
    return ((max_p_distance * 1000.0).round() / 1000.0) as f64;
}

// Calculates the percent of the MSA that are gaps
/// 
/// 
/// # Arguments
/// 
/// * `x`: A full multiple sequence alignment
/// 
/// Returns
/// The percent of characters in the entire MSA that are '-'
fn calculate_percent_of_gaps(x: &Vec<String>) -> f64 {
    let mut gap_count = 0;
    for s in x {
        gap_count += s.chars().filter(|&c| c == '-').count();
    }
    return ((gap_count * 100) / (x.len() * x[0].len())) as f64;
}

/// Takes a group of unaligned sequences or a MSA and calculates the average length of a sequence
/// 
/// # Arguments
/// 
/// * `x`: A full multiple sequence alignment or a group of unaligned sequences
/// 
/// Returns 
/// The average length of a sequence in x
fn calculate_avg_length(x: &Vec<String>) -> usize {
    return (x.iter().map(|s| s.len()).sum::<usize>()) / x.len();
}


#[cfg(test)]
mod tests {
    use crate::strings::needleman_wunsch;

    use super::*;

    #[test]
    fn test_compute_table() {
        let x = "NAJIBPEPPERSEATS";
        let y = "NAJIBEATSPEPPERS";
        let table = compute_table::<u16>(x, y, Penalties::default());

        #[rustfmt::skip]
        let true_table: [[(u16, Direction); 17]; 17] = [
            [( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), ( 5, Direction::Left    ), ( 6, Direction::Left    ), (7, Direction::Left    ), (8, Direction::Left    ), (9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    ), (14, Direction::Left    ), (15, Direction::Left    ), (16, Direction::Left    )],
            [( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), ( 5, Direction::Left    ), (6, Direction::Left    ), (7, Direction::Left    ), (8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    ), (14, Direction::Left    ), (15, Direction::Left    )],
            [( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), ( 4, Direction::Left    ), (5, Direction::Left    ), (6, Direction::Left    ), (7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Diagonal), (13, Direction::Left    ), (14, Direction::Left    )],
            [( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), ( 3, Direction::Left    ), (4, Direction::Left    ), (5, Direction::Left    ), (6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    ), (13, Direction::Left    )],
            [( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), ( 2, Direction::Left    ), (3, Direction::Left    ), (4, Direction::Left    ), (5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    ), (12, Direction::Left    )],
            [( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 0, Direction::Diagonal), ( 1, Direction::Left    ), (2, Direction::Left    ), (3, Direction::Left    ), (4, Direction::Left    ), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    ), (11, Direction::Left    )],
            [( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 1, Direction::Up      ), ( 1, Direction::Diagonal), (1, Direction::Diagonal), (2, Direction::Left    ), (3, Direction::Left    ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Diagonal), ( 8, Direction::Left    ), ( 9, Direction::Left    ), (10, Direction::Left    )],
            [( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Diagonal), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 2, Direction::Up      ), ( 2, Direction::Diagonal), (2, Direction::Diagonal), (2, Direction::Diagonal), (3, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Left    ), ( 9, Direction::Left    )],
            [( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 3, Direction::Up      ), ( 3, Direction::Diagonal), (3, Direction::Diagonal), (3, Direction::Diagonal), (3, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Left    )],
            [( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Up      ), ( 4, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Up      ), ( 7, Direction::Diagonal)],
            [(10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), (5, Direction::Diagonal), (4, Direction::Diagonal), (4, Direction::Diagonal), ( 5, Direction::Diagonal), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 8, Direction::Up      )],
            [(11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), (4, Direction::Diagonal), (5, Direction::Up      ), (5, Direction::Diagonal), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Up      ), ( 6, Direction::Diagonal), (5, Direction::Up      ), (4, Direction::Diagonal), (5, Direction::Diagonal), ( 5, Direction::Up      ), ( 5, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), ( 7, Direction::Diagonal), (6, Direction::Up      ), (5, Direction::Diagonal), (4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Diagonal), ( 6, Direction::Diagonal), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), ( 8, Direction::Up      ), (7, Direction::Diagonal), (6, Direction::Up      ), (5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 6, Direction::Diagonal), ( 7, Direction::Left    ), ( 8, Direction::Left    ), ( 9, Direction::Diagonal)],
            [(15, Direction::Up      ), (14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), ( 9, Direction::Up      ), (8, Direction::Up      ), (7, Direction::Up      ), (6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Diagonal), ( 8, Direction::Diagonal), ( 9, Direction::Diagonal)],
            [(16, Direction::Up      ), (15, Direction::Up      ), (14, Direction::Up      ), (13, Direction::Up      ), (12, Direction::Up      ), (11, Direction::Up      ), (10, Direction::Up      ), (9, Direction::Up      ), (8, Direction::Up      ), (7, Direction::Up      ), ( 6, Direction::Up      ), ( 5, Direction::Up      ), ( 4, Direction::Diagonal), ( 5, Direction::Left    ), ( 6, Direction::Left    ), ( 7, Direction::Left    ), ( 8, Direction::Diagonal)]
        ];

        assert_eq!(table, true_table);
    }

    #[test]
    fn test_trace_back() {
        let peppers_x = "NAJIBPEPPERSEATS";
        let peppers_y = "NAJIBEATSPEPPERS";
        let peppers_table = compute_table::<u16>(peppers_x, peppers_y, Penalties::default());

        let (aligned_x, aligned_y) = trace_back_recursive(&peppers_table, [peppers_x, peppers_y]);
        assert_eq!(aligned_x, "NAJIB-PEPPERSEATS");
        assert_eq!(aligned_y, "NAJIBEATSPEPPE-RS");

        let (aligned_x, aligned_y) = trace_back_iterative(&peppers_table, [peppers_x, peppers_y]);
        assert_eq!(aligned_x, "NAJIB-PEPPERSEATS");
        assert_eq!(aligned_y, "NAJIBEATSPEPPE-RS");

        let guilty_x = "NOTGUILTY";
        let guilty_y = "NOTGUILTY";
        let guilty_table = compute_table::<u16>(guilty_x, guilty_y, Penalties::default());

        let (aligned_x, aligned_y) = trace_back_recursive(&guilty_table, [guilty_x, guilty_y]);
        assert_eq!(aligned_x, "NOTGUILTY");
        assert_eq!(aligned_y, "NOTGUILTY");

        let (aligned_x, aligned_y) = trace_back_iterative(&guilty_table, [guilty_x, guilty_y]);
        assert_eq!(aligned_x, "NOTGUILTY");
        assert_eq!(aligned_y, "NOTGUILTY");
    }

    #[test]
    fn cluster_align() {
        let sequence_group_0:Vec<String> = vec!["TCACTACATCCGTTGGATCG".to_string(), "GAAACATCTGTTCAGGAGCC".to_string(), "TAAATCCCTTGTTCGAGAGA".to_string(), "ATCCACCCATGCCCCGTAAG".to_string(), "TCACCAAAGCCACTGGAAGG".to_string()];
        let aligned_0 = align_cluster(sequence_group_0);
        let check_0: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(), "---GA-AACATCTGTTCAGGAGCC-".to_string(), "-T-AAATCCCT-TGTT--CGAGAGA".to_string(), "ATCCAC-CCATGCCCC--GTA-AG-".to_string(), "-T-CACCAAAGCCACT--GGAAGG-".to_string()];
        assert_eq!(aligned_0.len(), check_0.len());
        for i in 0..aligned_0.len() {
            assert_eq!(aligned_0[i], check_0[i]);
        }
        let sequence_group_1:Vec<String> = vec!["CCCGAAAGACTGTGGCGCTA".to_string(), "GACTCCCATAATTGACGCTA".to_string(), "CAAGACATAAAATAGGGTCG".to_string(), "CGGCCAATTGCGTCGCATCA".to_string(), "CTCAATATTAGACCGCGGGC".to_string()];
        let aligned_1 = align_cluster(sequence_group_1);
        let check_1: Vec<String> = vec!["--C-CCGA-A--AGACTGTGGCGC-T-A".to_string(), "GACTCCCA-T--A-A-T-TGACGC-T-A".to_string(), "--C-AAGA-C--ATAAAATAGGGT-C-G".to_string(), "----CGGC-C--A-ATTGCGTCGCATCA".to_string(), "--C-TCAATATTAGACCGCGG-GC----".to_string()];
        assert_eq!(aligned_1.len(), check_1.len());
        for i in 0..aligned_1.len() {
            assert_eq!(aligned_1[i], check_1[i]);
        }
        
        let sequence_group_2:Vec<String> = vec!["AGTTGTCTGGCCCCAGGCCA".to_string(), "TGTTACCGACCAGGCCGTAC".to_string(), "CGCAATGTGGATTCTCTGTG".to_string(), "GTGCATGTGCTTGAAACTCA".to_string(), "AAATTTTCAGATGGCGTTTA".to_string(), "ACGCCAGCATCAGGCCACGC".to_string()];
        let aligned_2 = align_cluster(sequence_group_2);
        let check_2: Vec<String> = vec!["-AGTTGTCTG-GCCCCAGGC-C--A---".to_string(), "-TGTT-AC-C-G-ACCAGGC-CGTAC--".to_string(), "-CGCAATGTG-GATTCTCTG-T--G---".to_string(), "--G-TGCATGTGCTTGAAACTC--A---".to_string(), "AAATTTTCAG-ATGGC-GTT-T--A---".to_string(), "-A--CG-CCA-GCATCAGGC-C--ACGC".to_string()];
        assert_eq!(aligned_2.len(), check_2.len());
        for i in 0..aligned_2.len() {
            assert_eq!(aligned_2[i], check_2[i]);
        }

        let sequence_group_3:Vec<String> = vec!["ACGCGGTACCGCTGATTCCT".to_string(), "TTGGGTAATTATAGGAAATC".to_string(), "CCTTGCGTTGCCGCGCTAGC".to_string(), "AAGAGAACGGGGGGATATCA".to_string()];
        let aligned_3 = align_cluster(sequence_group_3);
        let check_3: Vec<String> = vec!["AC--GCGGTAC-CGCT--GATTCCT".to_string(), "TT--G-GGTAA-TTATAGGAAATC-".to_string(), "CCTTGCGTTGC-CGC---GCTAGC-".to_string(), "AA--G-AGAACGGGGG--GATATCA".to_string()];
        assert_eq!(aligned_3.len(), check_3.len());
        for i in 0..aligned_3.len() {
            assert_eq!(aligned_3[i], check_3[i]);
        }
    }

    #[test]
    fn combine_2_clusters() {
        let sequence_group_0: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(), "---GA-AACATCTGTTCAGGAGCC-".to_string(), "-T-AAATCCCTT-GTT--CGAGAGA".to_string(), "ATCCAC-CCATGCCCC--GTA-AG-".to_string(), "-T-CACCAAAGCCACT--GGAAGG-".to_string()];
        let sequence_group_1: Vec<String> = vec!["--C-CCGA-A--AGACTGTGGCGC-T-A".to_string(), "GACTCCCA-T--A-A-T-TGACGC-T-A".to_string(), "--C-AAGA-C--ATAAAATAGGGT-C-G".to_string(), "----CGGC-C--A-ATTGCGTCGCATCA".to_string(), "--C-TCAATATTAGACCGCGG-GC----".to_string()];
        let aligned = align_clusters(sequence_group_0, sequence_group_1);
        let check: Vec<String> = vec!["-T-C-ACTACATCCG-TT--GGATC-G--".to_string(), "---G-A-AACATCTG-TTCAGGAGC-C--".to_string(), "-T-A-AATCCCTT-G-TT--CGAGA-GA-".to_string(), "ATCC-AC-CCATGCC-CC--GTA-A-G--".to_string(), "-T-C-ACCAAAGCCA-CT--GGAAG-G--".to_string(), 
        "---C-CCGA-A--AGACTGTGGCGC-T-A".to_string(), "GA-CTCCCA-T--A-A-T-TGACGC-T-A".to_string(), "---C-AAGA-C--ATAAAATAGGGT-C-G".to_string(), "-----CGGC-C--A-ATTGCGTCGCATCA".to_string(), "---C-TCAATATTAGACCGCGG-GC----".to_string()];
        for i in 0..aligned.len() {
            assert_eq!(aligned[i], check[i]);
        }
    }

    #[test]
    fn add_to_cluster() {
        let sequence_group_0: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(), "---GA-AACATCTGTTCAGGAGCC-".to_string(), "-T-AAATCCCTT-GTT--CGAGAGA".to_string(), "ATCCAC-CCATGCCCC--GTA-AG-".to_string(), "-T-CACCAAAGCCACT--GGAAGG-".to_string()];
        let sequence_group_1: Vec<String> = vec!["--C-CCGA-A--AGACTGTGGCGC-T-A".to_string(), "GACTCCCA-T--A-A-T-TGACGC-T-A".to_string(), "--C-AAGA-C--ATAAAATAGGGT-C-G".to_string(), "----CGGC-C--A-ATTGCGTCGCATCA".to_string(), "--C-TCAATATTAGACCGCGG-GC----".to_string()];
        let new_seq_0 = "TATGTCCCAGTCTCTGATAT".to_string();
        let new_seq_1 = "AGGCCAACGCTTAGTCACAA".to_string();

        let sequence_group_0_0: Vec<String> = add_x_to_cluster(&sequence_group_0, &new_seq_0);
        let check_0_0: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(), "---GA-AACATCTGTTCAGGAGCC-".to_string(), "-T-AAATCCCTT-GTT--CGAGAGA".to_string(), "ATCCAC-CCATGCCCC--GTA-AG-".to_string(), "-T-CACCAAAGCCACT--GGAAGG-".to_string(), "-T-ATGTCCC--AGTCTCTGATA-T".to_string()];
        for i in 0..sequence_group_0_0.len(){
            assert_eq!(sequence_group_0_0[i], check_0_0[i])
        }

        let sequence_group_0_1: Vec<String> = add_x_to_cluster(&sequence_group_0, &new_seq_1);
        let check_0_1: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(), "---GA-AACATCTGTTCAGGAGCC-".to_string(), "-T-AAATCCCTT-GTT--CGAGAGA".to_string(), "ATCCAC-CCATGCCCC--GTA-AG-".to_string(), "-T-CACCAAAGCCACT--GGAAGG-".to_string(), "AGGCCAACGCTTAG-T--C-ACA-A".to_string()];
        for i in 0..sequence_group_0_1.len(){
            assert_eq!(sequence_group_0_1[i], check_0_1[i])
        }

        let sequence_group_1_0: Vec<String> = add_x_to_cluster(&sequence_group_1, &new_seq_0);
        let check_1_0: Vec<String> = vec!["---C-CCGA-A--AGACTGTGGCGC-T-A".to_string(), "GA-CTCCCA-T--A-A-T-TGACGC-T-A".to_string(), "---C-AAGA-C--ATAAAATAGGGT-C-G".to_string(), "-----CGGC-C--A-ATTGCGTCGCATCA".to_string(), "---C-TCAATATTAGACCGCGG-GC----".to_string()];
        for i in 0..sequence_group_1_0.len(){
            assert_eq!(sequence_group_1_0[i], check_1_0[i])
        }

        let sequence_group_1_1: Vec<String> = add_x_to_cluster(&sequence_group_1, &new_seq_1);
        let check_1_1: Vec<String> = vec!["--C-CCGA-A--AGACTGTGGCG-C-T-A".to_string(), "GACTCCCA-T--A-A-T-TGACG-C-T-A".to_string(), "--C-AAGA-C--ATAAAATAGGG-T-C-G".to_string(), "----CGGC-C--A-ATTGCGTCG-CATCA".to_string(), "--C-TCAATATTAGACCGCGG-G-C----".to_string()];
        for i in 0..sequence_group_1_1.len(){
            assert_eq!(sequence_group_1_1[i], check_1_1[i])
        }
    }

    #[test]
    fn scoring() {
        let msa_0: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(), 
                                      "---GA-AACATCTGTTCAGGAGCC-".to_string(), 
                                      "-T-AAATCCCTT-GTT--CGAGAGA".to_string(), 
                                      "ATCCAC-CCATGCCCC--GTA-AG-".to_string(), 
                                      "-T-CACCAAAGCCACT--GGAAGG-".to_string(), 
                                      "-T-ATGTCCC--AGTCTCTGATA-T".to_string()];
        assert_eq!(unweighed_scoring_columns(&msa_0, 1, 1), 247);
        assert_eq!(unweighed_scoring_columns(&msa_0, 2, 1), 359);

        assert_eq!(unweighed_scoring_pairwise(&msa_0, 1, 1), 247);
        assert_eq!(unweighed_scoring_pairwise(&msa_0, 2, 1), 359);

        assert_eq!(weighed_scoring_columns(&msa_0, 2, 1, 1), 334);
        assert_eq!(weighed_scoring_columns(&msa_0, 3, 2, 1), 446);
        
        assert_eq!(weighed_scoring_pairwise(&msa_0, 2, 1, 1), 334);
        assert_eq!(weighed_scoring_pairwise(&msa_0, 3, 2, 1), 446);


        let msa_1: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(),
                                      "---GA-AACATCTGTTCAGGAGCC-".to_string(),
                                      "-T-AAATCCCTT-GTT--CGAGAGA".to_string(),
                                      "ATCCAC-CCATGCCCC--GTA-AG-".to_string(),
                                      "-T-CACCAAAGCCACT--GGAAGG-".to_string(),
                                      "AGGCCAACGCTTAG-T--C-ACA-A".to_string()];
        assert_eq!(unweighed_scoring_columns(&msa_1, 1, 1), 250);
        assert_eq!(unweighed_scoring_columns(&msa_1, 2, 1), 362);

        assert_eq!(unweighed_scoring_pairwise(&msa_1, 1, 1), 250);
        assert_eq!(unweighed_scoring_pairwise(&msa_1, 2, 1), 362);

        assert_eq!(weighed_scoring_columns(&msa_1, 2, 1, 1), 340);
        assert_eq!(weighed_scoring_columns(&msa_1, 3, 2, 1), 452);
        
        assert_eq!(weighed_scoring_pairwise(&msa_1, 2, 1, 1), 340);
        assert_eq!(weighed_scoring_pairwise(&msa_1, 3, 2, 1), 452);
        

        let msa_2: Vec<String> = vec!["---C-CCGA-A--AGACTGTGGCGC-T-A".to_string(), 
                                      "GA-CTCCCA-T--A-A-T-TGACGC-T-A".to_string(), 
                                      "---C-AAGA-C--ATAAAATAGGGT-C-G".to_string(), 
                                      "-----CGGC-C--A-ATTGCGTCGCATCA".to_string(), 
                                      "---C-TCAATATTAGACCGCGG-GC----".to_string(), 
                                      "TATGTCCCAGT-----CTCTGA----TAT".to_string()];
        assert_eq!(unweighed_scoring_columns(&msa_2, 1, 1), 316);
        assert_eq!(unweighed_scoring_columns(&msa_2, 2, 1), 512);

        assert_eq!(unweighed_scoring_pairwise(&msa_2, 1, 1), 316);
        assert_eq!(unweighed_scoring_pairwise(&msa_2, 2, 1), 512);

        assert_eq!(weighed_scoring_columns(&msa_2, 2, 1, 1), 430);
        assert_eq!(weighed_scoring_columns(&msa_2, 3, 2, 1), 626);
        
        assert_eq!(weighed_scoring_pairwise(&msa_2, 2, 1, 1), 430);
        assert_eq!(weighed_scoring_pairwise(&msa_2, 3, 2, 1), 626);


        let msa_3: Vec<String> = vec!["--C-CCGA-A--AGACTGTGGCG-C-T-A".to_string(), 
                                      "GACTCCCA-T--A-A-T-TGACG-C-T-A".to_string(), 
                                      "--C-AAGA-C--ATAAAATAGGG-T-C-G".to_string(), 
                                      "----CGGC-C--A-ATTGCGTCG-CATCA".to_string(),
                                      "--C-TCAATATTAGACCGCGG-G-C----".to_string(),
                                      "----AGGC-C--A-A-CGCTTAGTCACAA".to_string()];
        assert_eq!(unweighed_scoring_columns(&msa_3, 1, 1), 307);
        assert_eq!(unweighed_scoring_columns(&msa_3, 2, 1), 490);

        assert_eq!(unweighed_scoring_pairwise(&msa_3, 1, 1), 307);
        assert_eq!(unweighed_scoring_pairwise(&msa_3, 2, 1), 490);

        assert_eq!(weighed_scoring_columns(&msa_3, 2, 1, 1), 436);
        assert_eq!(weighed_scoring_columns(&msa_3, 3, 2, 1), 619);
        
        assert_eq!(weighed_scoring_pairwise(&msa_3, 2, 1, 1), 436);
        assert_eq!(weighed_scoring_pairwise(&msa_3, 3, 2, 1), 619);
    }

    #[test]
    fn data_collecting() {
        let msa_0: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(), 
                                      "---GA-AACATCTGTTCAGGAGCC-".to_string(), 
                                      "-T-AAATCCCTT-GTT--CGAGAGA".to_string(), 
                                      "ATCCAC-CCATGCCCC--GTA-AG-".to_string(), 
                                      "-T-CACCAAAGCCACT--GGAAGG-".to_string(), 
                                      "-T-ATGTCCC--AGTCTCTGATA-T".to_string()];

        let seq_0: Vec<String> = vec!["TCACTACATCCGTTGGATCG".to_string(), 
                                      "GAAACATCTGTTCAGGAGCC".to_string(), 
                                      "TAAATCCCTTGTTCGAGAGA".to_string(), 
                                      "ATCCACCCATGCCCCGTAAG".to_string(), 
                                      "TCACCAAAGCCACTGGAAGG".to_string(), 
                                      "TATGTCCCAGTCTCTGATAT".to_string()];
        
        assert_eq!(calculate_avg_p_distance(&msa_0), 0.36);
        assert_eq!(calculate_max_p_distance(&msa_0), 0.56);
        assert_eq!(calculate_percent_of_gaps(&msa_0), 20.0);
        assert_eq!(calculate_avg_length(&seq_0), 20);


        let msa_1: Vec<String> = vec!["-T-CACTACATCCGTT--GGATCG-".to_string(),
                                      "---GA-AACATCTGTTCAGGAGCC-".to_string(),
                                      "-T-AAATCCCTT-GTT--CGAGAGA".to_string(),
                                      "ATCCAC-CCATGCCCC--GTA-AG-".to_string(),
                                      "-T-CACCAAAGCCACT--GGAAGG-".to_string(),
                                      "AGGCCAACGCTTAG-T--C-ACA-A".to_string()];
        
        let seq_1: Vec<String> = vec!["TCACTACATCCGTTGGATCG".to_string(),
                                      "GAAACATCTGTTCAGGAGCC".to_string(),
                                      "TAAATCCCTTGTTCGAGAGA".to_string(),
                                      "ATCCACCCATGCCCCGTAAG".to_string(),
                                      "TCACCAAAGCCACTGGAAGG".to_string(),
                                      "AGGCCAACGCTTAGTCACAA".to_string()];
        
        assert_eq!(calculate_avg_p_distance(&msa_1), 0.368);
        assert_eq!(calculate_max_p_distance(&msa_1), 0.56);
        assert_eq!(calculate_percent_of_gaps(&msa_1), 20.0);
        assert_eq!(calculate_avg_length(&seq_1), 20);


        let msa_2: Vec<String> = vec!["---C-CCGA-A--AGACTGTGGCGC-T-A".to_string(), 
                                      "GA-CTCCCA-T--A-A-T-TGACGC-T-A".to_string(), 
                                      "---C-AAGA-C--ATAAAATAGGGT-C-G".to_string(), 
                                      "-----CGGC-C--A-ATTGCGTCGCATCA".to_string(), 
                                      "---C-TCAATATTAGACCGCGG-GC----".to_string(), 
                                      "TATGTCCCAGT-----CTCTGA----TAT".to_string()];
        
        let seq_2: Vec<String> = vec!["CCCGAAAGACTGTGGCGCTA".to_string(), 
                                      "GACTCCCATAATTGACGCTA".to_string(), 
                                      "CAAGACATAAAATAGGGTCG".to_string(), 
                                      "CGGCCAATTGCGTCGCATCA".to_string(), 
                                      "CTCAATATTAGACCGCGGGC".to_string(), 
                                      "TATGTCCCAGTCTCTGATAT".to_string()];
        
        assert_eq!(calculate_avg_p_distance(&msa_2), 0.276);
        assert_eq!(calculate_max_p_distance(&msa_2), 0.448);
        assert_eq!(calculate_percent_of_gaps(&msa_2), 31.0);
        assert_eq!(calculate_avg_length(&seq_2), 20);


        let msa_3: Vec<String> = vec!["--C-CCGA-A--AGACTGTGGCG-C-T-A".to_string(), 
                                      "GACTCCCA-T--A-A-T-TGACG-C-T-A".to_string(), 
                                      "--C-AAGA-C--ATAAAATAGGG-T-C-G".to_string(), 
                                      "----CGGC-C--A-ATTGCGTCG-CATCA".to_string(),
                                      "--C-TCAATATTAGACCGCGG-G-C----".to_string(),
                                      "----AGGC-C--A-A-CGCTTAGTCACAA".to_string()];

        let seq_3: Vec<String> = vec!["CCCGAAAGACTGTGGCGCTA".to_string(), 
                                      "GACTCCCATAATTGACGCTA".to_string(), 
                                      "CAAGACATAAAATAGGGTCG".to_string(), 
                                      "CGGCCAATTGCGTCGCATCA".to_string(),
                                      "CTCAATATTAGACCGCGGGC".to_string(),
                                      "AGGCCAACGCTTAGTCACAA".to_string()];

        assert_eq!(calculate_avg_p_distance(&msa_3), 0.285);
        assert_eq!(calculate_max_p_distance(&msa_3), 0.448);
        assert_eq!(calculate_percent_of_gaps(&msa_3), 31.0);
        assert_eq!(calculate_avg_length(&seq_3), 20);
    }
 
}
