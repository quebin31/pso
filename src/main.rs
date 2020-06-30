pub mod fitness;
pub mod pso;

use anyhow::Error;
use fitness::Fitness;
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use plotters::prelude::*;
use pso::{Options, Particles};

fn f(vec: &Array1<f64>) -> f64 {
    let x = vec[0];
    let y = vec[1];

    (x + 2.0 * y - 7.0).powi(2) + (2.0 * x + y - 5.0).powi(2)
}

fn main() -> Result<(), Error> {
    // Define some parameters
    let size = 10;
    let dim = 2;
    let iters = 80;

    let value_distr = Uniform::new(-10., 10.);
    let velocity_range = (-1.0, 1.0);
    let velocity_distr = Uniform::new(velocity_range.0, velocity_range.1);

    let options = Options {
        omega: None,
        phi_1: 2.0,
        phi_2: 2.0,
    };

    // Show the parameters
    println!("Parámetros:");
    println!("- Tamaño de la población: {}", size);
    println!("- Velocidad inicial entre {:?}", velocity_range);
    println!("- Omega (ω): Aleatorio entre 0 y 1 para cada iteración");
    println!("- rand1, rand2: Aleatoria entre 0 y 1 para cada individuo");
    println!("- Phi_1 (φ_1): {}", options.phi_1);
    println!("- Phi_2 (φ_2): {}", options.phi_2);
    println!("- Cantidad de iteraciones: {}", iters);
    println!();

    // Generate initial particles
    let mut particles = Particles::new(
        size,
        dim,
        value_distr,
        velocity_distr,
        Fitness::new(f, true),
    );

    // Show initial particles, fitnesses and best locals
    println!("{}", particles.summary(true)?);

    let root = BitMapBackend::gif("animation.gif", (600, 600), 250)?.into_drawing_area();
    particles.plot(&root, 0)?;

    // Run a step 'iters' times
    for i in 0..iters {
        println!("\n>>>> Iteración {} <<<<", i + 1);
        particles.step(options);
        println!("{}", particles.summary(false)?);
        particles.plot(&root, i + 1)?;
    }

    // Show global best
    let best = particles.best();
    println!("\n>>> Mejor global: x: {}, fitness: {}", best, f(&best));

    Ok(())
}
