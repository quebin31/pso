pub mod fitness;
pub mod pso;

use fitness::Fitness;
use indicatif::ProgressIterator;
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use pso::{Options, Particles};

fn main() {
    let fitness = Fitness::new(
        |vec: &Array1<f64>| {
            let x = vec[0];
            let y = vec[1];

            x.powi(2) + y.powi(2)
        },
        true,
    );

    let size = 6;
    let dim = 2;
    let iters = 100;

    let value_range = Uniform::new(-5., 5.);
    let velocity_range = Uniform::new(-1., 1.);

    let options = Options {
        omega: None,
        phi_1: 2.0,
        phi_2: 2.0,
    };

    let mut particles = Particles::new(size, dim, value_range, velocity_range, fitness);

    for _ in (0..iters).progress() {
        particles.step(options);
        println!("Global best: {}", particles.best());
    }
}
