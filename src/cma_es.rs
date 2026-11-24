//! CMA-ES (Hansen's (mu/mu_w, lambda) evolution strategy): minimises a function of a few numbers
//! that has no gradient to follow. Each generation samples points around a mean, moves the mean
//! toward the best of them, and reshapes the sampling to the directions that paid off. Only the
//! points' order matters, so flat stretches and jumps in the function are fine.

use crate::pca::symmetric_eigen;

pub struct Minimum {
    pub point: Vec<f64>,
    pub value: f64,
    pub evaluations: usize,
}

/// One generation, for progress reports.
pub struct Generation {
    pub evaluations: usize,
    /// Lowest value so far
    pub best: f64,
    /// Lowest of this generation
    pub generation_best: f64,
    /// Overall step size
    pub step: f64,
}

/// Standard normal numbers from SplitMix64, the same on every run and machine.
struct Normal {
    state: u64,
}

impl Normal {
    /// In (0, 1)
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 + f64::EPSILON / 2.0
    }

    /// Box-Muller
    fn sample(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// The lowest point found from `start` in about `evaluations` calls of `objective`. `step`: how
/// far from `start` to look at first, in every coordinate, so scale them alike.
pub fn minimize(
    start: &[f64],
    step: f64,
    evaluations: usize,
    seed: u64,
    mut objective: impl FnMut(&[f64]) -> f64,
    mut each_generation: impl FnMut(&Generation),
) -> Minimum {
    let n = start.len();
    let size = n as f64;
    let mut normal = Normal { state: seed };

    // Hansen's defaults
    let population = 4 + (3.0 * size.ln()).floor() as usize;
    let parents = population / 2;
    let mut weights: Vec<f64> = (0..parents).map(|i| (parents as f64 + 0.5).ln() - (i as f64 + 1.0).ln()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|weight| *weight /= total);
    let spread = 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    let path_rate = (4.0 + spread / size) / (size + 4.0 + 2.0 * spread / size);
    let step_rate = (spread + 2.0) / (size + spread + 5.0);
    let rank_one = 2.0 / ((size + 1.3).powi(2) + spread);
    let rank_many = (1.0 - rank_one).min(2.0 * (spread - 2.0 + 1.0 / spread) / ((size + 2.0).powi(2) + spread));
    let damping = 1.0 + 2.0 * (((spread - 1.0) / (size + 1.0)).sqrt() - 1.0).max(0.0) + step_rate;
    let expected_length = size.sqrt() * (1.0 - 1.0 / (4.0 * size) + 1.0 / (21.0 * size * size));

    let mut mean = start.to_vec();
    let mut step = step;
    let mut covariance: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| f64::from(i == j)).collect()).collect();
    // The covariance's axes (rows) and the spread along each
    let mut axes = covariance.clone();
    let mut lengths = vec![1.0; n];
    let (mut path, mut step_path) = (vec![0.0; n], vec![0.0; n]);
    let mut best = Minimum { point: start.to_vec(), value: f64::INFINITY, evaluations: 0 };

    let mut generation = 0;
    while best.evaluations + population <= evaluations.max(population) {
        generation += 1;
        // (value, direction from the mean in steps)
        let mut sampled: Vec<(f64, Vec<f64>)> = (0..population)
            .map(|_| {
                let along: Vec<f64> = lengths.iter().map(|length| length * normal.sample()).collect();
                let direction: Vec<f64> = (0..n).map(|k| (0..n).map(|axis| axes[axis][k] * along[axis]).sum()).collect();
                let point: Vec<f64> = (0..n).map(|k| mean[k] + step * direction[k]).collect();
                let value = objective(&point);
                best.evaluations += 1;
                if value < best.value {
                    (best.value, best.point) = (value, point);
                }
                (value, direction)
            })
            .collect();
        sampled.sort_by(|a, b| a.0.total_cmp(&b.0));

        let moved: Vec<f64> = (0..n).map(|k| (0..parents).map(|i| weights[i] * sampled[i].1[k]).sum()).collect();
        for k in 0..n {
            mean[k] += step * moved[k];
        }
        // `moved` with the covariance's shape taken out
        let whitened: Vec<f64> = {
            let along: Vec<f64> = (0..n).map(|axis| (0..n).map(|k| axes[axis][k] * moved[k]).sum::<f64>() / lengths[axis]).collect();
            (0..n).map(|k| (0..n).map(|axis| axes[axis][k] * along[axis]).sum()).collect()
        };
        let momentum = (step_rate * (2.0 - step_rate) * spread).sqrt();
        for k in 0..n {
            step_path[k] = (1.0 - step_rate) * step_path[k] + momentum * whitened[k];
        }
        let step_length = step_path.iter().map(|x| x * x).sum::<f64>().sqrt();
        let settled = (1.0 - (1.0 - step_rate).powi(2 * generation)).sqrt();
        let steady = step_length / settled / expected_length < 1.4 + 2.0 / (size + 1.0);
        let momentum = if steady { (path_rate * (2.0 - path_rate) * spread).sqrt() } else { 0.0 };
        for k in 0..n {
            path[k] = (1.0 - path_rate) * path[k] + momentum * moved[k];
        }
        let stalled = if steady { 0.0 } else { path_rate * (2.0 - path_rate) };
        for i in 0..n {
            for j in 0..n {
                let from_parents: f64 = (0..parents).map(|p| weights[p] * sampled[p].1[i] * sampled[p].1[j]).sum();
                covariance[i][j] = (1.0 - rank_one - rank_many) * covariance[i][j]
                    + rank_one * (path[i] * path[j] + stalled * covariance[i][j])
                    + rank_many * from_parents;
            }
        }
        step *= ((step_rate / damping) * (step_length / expected_length - 1.0)).exp();
        // On a flat stretch, look further
        if sampled[0].0 == sampled[(population * 7).div_ceil(10).min(population - 1)].0 {
            step *= (0.2 + step_rate / damping).exp();
        }

        let (values, vectors) = symmetric_eigen(covariance.clone());
        axes = vectors;
        lengths = values.into_iter().map(|value| value.max(1e-20).sqrt()).collect();
        each_generation(&Generation { evaluations: best.evaluations, best: best.value, generation_best: sampled[0].0, step });
        if step * lengths.iter().fold(0.0, |longest: f64, &length| longest.max(length)) < 1e-10 {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(_: &Generation) {}

    #[test]
    fn samples_standard_normals() {
        let mut normal = Normal { state: 7 };
        let samples: Vec<f64> = (0..20_000).map(|_| normal.sample()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.05, "variance {variance}");
        assert!(samples.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn finds_the_bottom_of_a_tilted_valley() {
        // Narrow along one diagonal, wide along the other: the minimum is at (3, -1, 2, 0, 5, 1)
        let target = [3.0, -1.0, 2.0, 0.0, 5.0, 1.0];
        let valley = |point: &[f64]| {
            let off: Vec<f64> = point.iter().zip(target).map(|(x, target)| x - target).collect();
            let across: f64 = off.iter().sum();
            off.iter().map(|x| x * x).sum::<f64>() + 100.0 * across * across
        };
        let found = minimize(&[0.0; 6], 1.0, 6000, 1, valley, quiet);
        assert!(found.value < 1e-10, "{}", found.value);
        for (found, target) in found.point.iter().zip(target) {
            assert!((found - target).abs() < 1e-4, "{found} for {target}");
        }
        assert!(found.evaluations <= 6000);
    }

    #[test]
    fn steps_down_a_staircase() {
        // Flat between whole numbers
        let stairs = |point: &[f64]| point.iter().map(|x| (x - 4.0).abs().floor()).sum::<f64>();
        let found = minimize(&[20.0, -15.0, 9.0], 1.0, 1500, 3, stairs, quiet);
        assert_eq!(found.value, 0.0);
    }

    #[test]
    fn is_the_same_every_run_and_keeps_to_its_budget() {
        let bowl = |point: &[f64]| point.iter().map(|x| x * x).sum::<f64>();
        let mut generations = 0;
        let first = minimize(&[2.0, 2.0, 2.0], 0.5, 100, 9, bowl, |_| generations += 1);
        let again = minimize(&[2.0, 2.0, 2.0], 0.5, 100, 9, bowl, quiet);
        assert_eq!((first.value, &first.point), (again.value, &again.point));
        // 7 a generation
        assert_eq!((first.evaluations, generations), (98, 14));
        assert_ne!(minimize(&[2.0, 2.0, 2.0], 0.5, 100, 10, bowl, quiet).point, first.point);
    }
}
