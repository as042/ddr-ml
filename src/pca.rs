//! Principal component analysis of the model's input features.

use std::{error::Error, fs, path::Path};

use serde::{Deserialize, Serialize};

/// A PCA of standardized columns. Components are in order of decreasing variance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pca {
    pub feature_names: Vec<String>,
    /// Of the fitted rows, per column
    pub mean: Vec<f64>,
    /// Population standard deviation of the fitted rows, per column. 1 for a constant column.
    pub std: Vec<f64>,
    /// Unit vectors over the standardized columns, one per component. Each one's largest
    /// loading is positive.
    pub components: Vec<Vec<f64>>,
    /// Variance of the fitted rows along each component. Sums to the number of non-constant
    /// columns.
    pub variances: Vec<f64>,
}

impl Pca {
    /// Fit on the training split only if the components will feed a model.
    ///
    /// # Panics
    ///
    /// If there are fewer than 2 rows, or a row's length isn't `feature_names.len()`.
    pub fn fit(feature_names: &[&str], rows: &[Vec<f32>]) -> Self {
        let width = feature_names.len();
        assert!(rows.len() >= 2, "a PCA needs at least 2 rows, but got {}", rows.len());
        assert!(rows.iter().all(|row| row.len() == width), "every row must have {width} columns");

        let n = rows.len() as f64;
        let column = |j: usize| rows.iter().map(move |row| row[j] as f64);
        let mean: Vec<f64> = (0..width).map(|j| column(j).sum::<f64>() / n).collect();
        let std: Vec<f64> = (0..width)
            .map(|j| {
                // Compared exactly: a constant column's variance can round to just above 0
                let constant = column(j).all(|x| x == rows[0][j] as f64);
                let variance = column(j).map(|x| (x - mean[j]).powi(2)).sum::<f64>() / n;
                if constant { 1.0 } else { variance.sqrt() }
            })
            .collect();

        let mut covariance = vec![vec![0.0; width]; width];
        for row in rows {
            let z = standardize(row, &mean, &std);
            for i in 0..width {
                for j in i..width {
                    covariance[i][j] += z[i] * z[j] / n;
                }
            }
        }
        for i in 0..width {
            for j in 0..i {
                covariance[i][j] = covariance[j][i];
            }
        }

        let (variances, mut components) = symmetric_eigen(covariance);
        for component in &mut components {
            let largest = component.iter().copied().fold(0.0, |a: f64, b| if b.abs() > a.abs() { b } else { a });
            if largest < 0.0 {
                component.iter_mut().for_each(|loading| *loading = -*loading);
            }
        }
        Pca {
            feature_names: feature_names.iter().map(|name| name.to_string()).collect(),
            mean,
            std,
            components,
            // Rounding can leave a null component slightly negative
            variances: variances.into_iter().map(|variance| variance.max(0.0)).collect(),
        }
    }

    /// Each row's coordinates along the first `num_components` components.
    ///
    /// # Panics
    ///
    /// If there are fewer than `num_components` components, or a row has the wrong length.
    pub fn transform(&self, rows: &[Vec<f32>], num_components: usize) -> Vec<Vec<f32>> {
        assert!(
            num_components <= self.components.len(),
            "asked for {num_components} components, but there are {}",
            self.components.len()
        );
        rows.iter()
            .map(|row| {
                assert_eq!(row.len(), self.mean.len(), "a row has {} columns, not {}", row.len(), self.mean.len());
                let z = standardize(row, &self.mean, &self.std);
                self.components[..num_components]
                    .iter()
                    .map(|component| component.iter().zip(&z).map(|(loading, value)| loading * value).sum::<f64>() as f32)
                    .collect()
            })
            .collect()
    }

    /// Each component's share of the total variance.
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total: f64 = self.variances.iter().sum();
        self.variances.iter().map(|variance| variance / total).collect()
    }

    /// The fewest leading components whose shares of the variance add up to `fraction`.
    ///
    /// # Panics
    ///
    /// If `fraction` isn't above 0 and at most 1.
    pub fn components_for_variance(&self, fraction: f64) -> usize {
        assert!(fraction > 0.0 && fraction <= 1.0, "a share of the variance is above 0 and at most 1, not {fraction}");
        let mut total = 0.0;
        for (k, share) in self.explained_variance_ratio().into_iter().enumerate() {
            total += share;
            // The shares can add up to just under 1
            if total >= fraction - 1e-12 {
                return k + 1;
            }
        }
        self.components.len()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        if let Some(dir) = path.as_ref().parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    #[allow(dead_code)] // Only tests use it
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

fn standardize(row: &[f32], mean: &[f64], std: &[f64]) -> Vec<f64> {
    row.iter().zip(mean.iter().zip(std)).map(|(&value, (mean, std))| (value as f64 - mean) / std).collect()
}

/// Eigenvalues of a symmetric matrix, largest first, with their unit eigenvectors. Cyclic
/// Jacobi rotations.
fn symmetric_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    // Eigenvectors in columns
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let scale: f64 = a.iter().flatten().map(|x| x * x).sum();
    for _sweep in 0..100 {
        let off_diagonal: f64 = (0..n).flat_map(|p| (p + 1..n).map(move |q| (p, q))).map(|(p, q)| a[p][q] * a[p][q]).sum();
        if off_diagonal <= scale * 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                // The rotation that zeroes a[p][q]
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut().chain(v.iter_mut()) {
                    (row[p], row[q]) = (c * row[p] - s * row[q], s * row[p] + c * row[q]);
                }
                for k in 0..n {
                    (a[p][k], a[q][k]) = (c * a[p][k] - s * a[q][k], s * a[p][k] + c * a[q][k]);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let values = order.iter().map(|&i| a[i][i]).collect();
    let vectors = order.iter().map(|&i| (0..n).map(|k| v[k][i]).collect()).collect();
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, what: &str) {
        assert!((actual - expected).abs() < 1e-9, "{what}: {actual} isn't {expected}");
    }

    /// Seeded rows with correlated columns and one constant column (index 3).
    fn rows() -> Vec<Vec<f32>> {
        let mut state = 7u64;
        let mut next = || {
            // SplitMix64, to [-1, 1)
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        (0..200)
            .map(|_| {
                let (a, b, c) = (next(), next(), next());
                vec![(100.0 * a) as f32, (a + 0.1 * b) as f32, (5.0 * b - a) as f32, 0.1, (c * 0.01 + 3.0) as f32]
            })
            .collect()
    }

    const NAMES: [&str; 5] = ["a", "b", "c", "constant", "e"];

    #[test]
    fn finds_eigenpairs() {
        let (values, vectors) = symmetric_eigen(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        assert_close(values[0], 3.0, "largest eigenvalue");
        assert_close(values[1], 1.0, "smallest eigenvalue");
        let half = 0.5f64.sqrt();
        assert_close(vectors[0][0].abs(), half, "first eigenvector");
        assert_close(vectors[0][0] * vectors[0][1], 0.5, "first eigenvector is along (1, 1)");
        assert_close(vectors[1][0] * vectors[1][1], -0.5, "second eigenvector is along (1, -1)");

        // A v = lambda v for a larger matrix
        let a = vec![vec![4.0, 1.0, -2.0, 0.5], vec![1.0, 3.0, 0.0, 1.5], vec![-2.0, 0.0, 5.0, -1.0], vec![0.5, 1.5, -1.0, 2.0]];
        let (values, vectors) = symmetric_eigen(a.clone());
        assert!(values.windows(2).all(|pair| pair[0] >= pair[1]), "{values:?}");
        for (value, vector) in values.iter().zip(&vectors) {
            for (row, &x) in a.iter().zip(vector) {
                let product: f64 = row.iter().zip(vector).map(|(a, v)| a * v).sum();
                assert_close(product, value * x, "A v");
            }
        }
    }

    #[test]
    fn one_component_for_perfectly_correlated_columns() {
        let rows: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 2.0 * i as f32 + 1.0]).collect();
        let pca = Pca::fit(&["x", "y"], &rows);
        assert_close(pca.explained_variance_ratio()[0], 1.0, "first component's share");
        assert_close(pca.variances[1], 0.0, "second component's variance");
        assert_close(pca.components[0][0], 0.5f64.sqrt(), "loading of x");
        assert_close(pca.components[0][1], 0.5f64.sqrt(), "loading of y");
    }

    #[test]
    fn components_are_orthonormal_and_ordered() {
        let pca = Pca::fit(&NAMES, &rows());
        assert!(pca.variances.windows(2).all(|pair| pair[0] >= pair[1]), "{:?}", pca.variances);
        assert_close(pca.variances.iter().sum(), 4.0, "total variance is the number of non-constant columns");
        for (i, a) in pca.components.iter().enumerate() {
            for (j, b) in pca.components.iter().enumerate() {
                let dot: f64 = a.iter().zip(b).map(|(a, b)| a * b).sum();
                assert_close(dot, if i == j { 1.0 } else { 0.0 }, "component dot product");
            }
            let largest = a.iter().copied().fold(0.0, |x: f64, y| if y.abs() > x.abs() { y } else { x });
            assert!(largest > 0.0, "component {i}'s largest loading should be positive");
        }
    }

    #[test]
    fn constant_columns_dont_break_it() {
        let pca = Pca::fit(&NAMES, &rows());
        assert_eq!(pca.std[3], 1.0);
        assert!(pca.components.iter().flatten().chain(&pca.variances).all(|x| x.is_finite()));
        assert_close(*pca.variances.last().unwrap(), 0.0, "the null component's variance");
        // Only the null component loads on the constant column
        assert!(pca.components[..4].iter().all(|component| component[3].abs() < 1e-9));
    }

    #[test]
    fn transformed_rows_are_uncorrelated_with_the_fitted_variances() {
        let rows = rows();
        let pca = Pca::fit(&NAMES, &rows);
        let scores = pca.transform(&rows, 5);
        let n = rows.len() as f64;
        for i in 0..5 {
            for j in 0..5 {
                let covariance = scores.iter().map(|row| row[i] as f64 * row[j] as f64).sum::<f64>() / n;
                let expected = if i == j { pca.variances[i] } else { 0.0 };
                // f32 scores
                assert!((covariance - expected).abs() < 1e-5, "covariance of components {i} and {j}: {covariance}, not {expected}");
            }
        }
        // Fewer components are a prefix
        let two = pca.transform(&rows[..3], 2);
        assert_eq!(two, scores[..3].iter().map(|row| row[..2].to_vec()).collect::<Vec<_>>());
    }

    #[test]
    fn counts_components_for_a_share_of_the_variance() {
        let mut pca = Pca::fit(&NAMES, &rows());
        pca.variances = vec![5.0, 3.0, 1.5, 0.5, 0.0];
        // (share, components): running totals are 50%, 80%, 95%, 100%
        for (fraction, count) in [(0.01, 1), (0.5, 1), (0.51, 2), (0.8, 2), (0.9, 3), (0.95, 3), (0.96, 4), (1.0, 4)] {
            assert_eq!(pca.components_for_variance(fraction), count, "{fraction}");
        }
    }

    #[test]
    #[should_panic(expected = "a share of the variance")]
    fn rejects_shares_outside_0_to_1() {
        Pca::fit(&NAMES, &rows()).components_for_variance(1.5);
    }

    #[test]
    fn save_and_load_round_trip() {
        let pca = Pca::fit(&NAMES, &rows());
        let dir = std::env::temp_dir().join(format!("ddr-trainer-pca-{}", std::process::id()));
        let path = dir.join("nested").join("pca.json");
        pca.save(&path).unwrap();
        let loaded = Pca::load(&path).unwrap();
        fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded, pca);
    }

    #[test]
    #[should_panic(expected = "asked for 6 components")]
    fn rejects_too_many_components() {
        let rows = rows();
        Pca::fit(&NAMES, &rows).transform(&rows, 6);
    }
}
