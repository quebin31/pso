use crate::fitness::Fitness;
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::{thread_rng, Rng};

#[derive(Debug, Clone)]
pub struct Particle {
    curr_value: Array1<f64>,
    best_value: Array1<f64>,
    velocity: Array1<f64>,
}

impl Particle {
    fn new(dim: usize, value_range: &Uniform<f64>, velocity_range: &Uniform<f64>) -> Self {
        let value = Array1::random((dim,), value_range);
        let velocity = Array1::random((dim,), velocity_range);

        Self {
            curr_value: value.clone(),
            best_value: value,
            velocity,
        }
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

        self.velocity = fst_term + snd_term + trd_term;
    }

    fn update_value(&mut self) {
        self.curr_value = &self.curr_value - &self.velocity;
    }

    fn update_best(&mut self, fitness: &Fitness<Array1<f64>>) {
        let curr_fitness = fitness.calculate_for_maximization(&self.curr_value);
        let best_fitness = fitness.calculate_for_maximization(&self.best_value);

        if curr_fitness < best_fitness {
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
                let fa = fitness.calculate_for_maximization(&a.curr_value);
                let fb = fitness.calculate_for_maximization(&b.curr_value);

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

        for particle in &mut self.particles {
            particle.update_velocity(&self.global_best, &mut rng, &options);
            particle.update_value();
            particle.update_best(&self.fitness);
        }

        let local_best = self
            .particles
            .iter()
            .max_by(|a, b| {
                let fa = self.fitness.calculate_for_maximization(&a.curr_value);
                let fb = self.fitness.calculate_for_maximization(&b.curr_value);

                fa.partial_cmp(&fb).expect("Received a NaN")
            })
            .expect("Population is empty")
            .curr_value
            .clone();

        let best_fitness = self.fitness.calculate_for_maximization(&self.global_best);
        let local_fitness = self.fitness.calculate_for_maximization(&local_best);

        if best_fitness < local_fitness {
            self.global_best = local_best;
        }
    }

    pub fn best(&self) -> &Array1<f64> {
        &self.global_best
    }
}
