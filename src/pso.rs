use crate::fitness::Fitness;
use anyhow::Error;
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use plotters::coord::Shift;
use plotters::drawing::backend::DrawingBackend;
use plotters::prelude::*;
use rand::{thread_rng, Rng};
use std::fmt::Error as FmtError;

#[derive(Debug, Clone)]
pub struct Particle {
    curr_value: Array1<f64>,
    best_value: Array1<f64>,
    velocity: Array1<f64>,
}

impl Particle {
    fn new(dim: usize, value_distr: &Uniform<f64>, velocity_distr: &Uniform<f64>) -> Self {
        let value = Array1::random((dim,), value_distr);
        let velocity = Array1::random((dim,), velocity_distr);

        Self {
            curr_value: value.clone(),
            best_value: value,
            velocity,
        }
    }

    pub fn best(&self) -> &Array1<f64> {
        &self.best_value
    }

    pub fn value(&self) -> &Array1<f64> {
        &self.curr_value
    }

    pub fn velocity(&self) -> &Array1<f64> {
        &self.velocity
    }

    fn update_velocity(&mut self, global_best: &Array1<f64>, mut rng: impl Rng, options: &Options) {
        let omega = options.omega.expect("Omega was None");
        let phi_1 = options.phi_1;
        let phi_2 = options.phi_2;

        let fst_term = self.velocity.mapv(|v| v * omega);

        let rand_1 = rng.gen_range(0.0, 1.0);
        let snd_term = (&self.best_value - &self.curr_value).mapv(|v| v * phi_1 * rand_1);

        let rand_2 = rng.gen_range(0.0, 1.0);
        let trd_term = (global_best - &self.curr_value).mapv(|v| v * phi_2 * rand_2);

        println!("rand_1: {}", rand_1);
        println!("rand_2: {}", rand_2);

        self.velocity = fst_term + snd_term + trd_term;
    }

    fn update_value(&mut self) {
        self.curr_value = &self.curr_value + &self.velocity;
    }

    fn update_best(&mut self, fitness: &Fitness<Array1<f64>>) {
        let curr_fitness = fitness.calculate_for_maximization(&self.curr_value);
        let best_fitness = fitness.calculate_for_maximization(&self.best_value);

        if best_fitness < curr_fitness {
            self.best_value = self.curr_value.clone();
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Options {
    pub omega: Option<f64>,
    pub phi_1: f64,
    pub phi_2: f64,
}

pub struct Particles<'a> {
    particles: Vec<Particle>,
    fitness: Fitness<'a, Array1<f64>>,
    global_best: Array1<f64>,
}

impl<'a> Particles<'a> {
    pub fn new(
        size: usize,
        dim: usize,
        value_range: Uniform<f64>,
        velocity_range: Uniform<f64>,
        fitness: Fitness<'a, Array1<f64>>,
    ) -> Self {
        let particles: Vec<_> = (0..size)
            .map(|_| Particle::new(dim, &value_range, &velocity_range))
            .collect();

        let global_best = particles
            .iter()
            .max_by(|a, b| {
                let fa = fitness.calculate_for_maximization(&a.value());
                let fb = fitness.calculate_for_maximization(&b.value());

                fa.partial_cmp(&fb).expect("Received a NaN")
            })
            .expect("No particles were created")
            .curr_value
            .clone();

        Self {
            particles,
            fitness,
            global_best,
        }
    }

    pub fn step(&mut self, mut options: Options) {
        let mut rng = thread_rng();

        // If not provided with an omega, generate one for this iteration
        options.omega = if let Some(omega) = options.omega {
            Some(omega)
        } else {
            Some(rng.gen_range(0.0, 1.0))
        };

        println!("Omega (ω): {}", options.omega.unwrap());

        for (i, particle) in &mut self.particles.iter_mut().enumerate() {
            particle.update_velocity(&self.global_best, &mut rng, &options);
            particle.update_value();
            particle.update_best(&self.fitness);
            println!(
                "{}) x: {}, v: {}",
                i + 1,
                particle.value(),
                particle.velocity()
            );
        }

        let local_best = self
            .particles
            .iter()
            .max_by(|a, b| {
                let fa = self.fitness.calculate_for_maximization(&a.value());
                let fb = self.fitness.calculate_for_maximization(&b.value());

                fa.partial_cmp(&fb).expect("Received a NaN")
            })
            .expect("Population is empty")
            .curr_value
            .clone();

        let best_fitness = self.fitness.calculate_for_maximization(&self.global_best);
        let local_fitness = self.fitness.calculate_for_maximization(&local_best);
        println!(
            "Mejor en esta iteración: x: {}, fitness: {}",
            local_best,
            self.fitness.calculate(&local_best)
        );

        if best_fitness < local_fitness {
            println!("El mejor global cambió");
            self.global_best = local_best;
        }
    }

    pub fn best(&self) -> &Array1<f64> {
        &self.global_best
    }

    pub fn particles(&self) -> &Vec<Particle> {
        &self.particles
    }

    pub fn summary(&self, show_particles: bool) -> Result<String, FmtError> {
        use std::fmt::Write;

        let mut particles_out = String::new();
        let mut fitness_out = String::new();
        let mut blocals_out = String::new();

        if show_particles {
            writeln!(particles_out, ">>> Cúmulo de partículas <<<")?;
        }

        writeln!(fitness_out, ">>> Fitness <<<")?;
        writeln!(blocals_out, ">>> Mejores locales <<<")?;

        for (i, particle) in self.particles.iter().enumerate() {
            if show_particles {
                writeln!(
                    particles_out,
                    "{}) x: {},  v: {}",
                    i + 1,
                    particle.value(),
                    particle.velocity()
                )?;
            }

            let fitness = self.fitness.calculate(particle.value());
            writeln!(fitness_out, "{}) {}", i + 1, fitness)?;

            let best_fitness = self.fitness.calculate(particle.best());
            writeln!(
                blocals_out,
                "{}) x: {}, fitness: {}",
                i + 1,
                particle.best(),
                best_fitness
            )?;
        }

        let best_fitness = self.fitness.calculate(self.best());
        let best_global = format!(
            ">>> Mejor global: x: {}, fitness: {}",
            self.best(),
            best_fitness
        );

        let out = format!(
            "{}{}{}{}",
            particles_out, fitness_out, blocals_out, best_global
        );

        Ok(out)
    }

    pub fn plot<D>(&self, root: &DrawingArea<D, Shift>, i: usize) -> Result<(), Error>
    where
        D: DrawingBackend,
        <D as DrawingBackend>::ErrorType: 'static,
    {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(root)
            .set_label_area_size(LabelAreaPosition::Left, 50)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .set_label_area_size(LabelAreaPosition::Right, 50)
            .caption(format!("PSO (iter = {})", i), ("sans-serif", 50))
            .build_ranged(-5.0..5.0, -5.0..5.0)?;

        chart.configure_mesh().draw()?;
        chart.draw_series(self.particles.iter().map(|p| {
            let value = p.value();
            let center = (value[0], value[1]);
            Circle::new(center, 5, ShapeStyle::from(&BLUE).filled())
        }))?;

        Ok(root.present()?)
    }
}
