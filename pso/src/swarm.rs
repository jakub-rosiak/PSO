use glam::Vec2;

use crate::{errors::SwarmError, functions::Function, particle::Particle};

pub struct Swarm {
    particles: Vec<Particle>,
    inertia_weight: f32,
    c_coeff: f32,
    s_coeff: f32,
    gbest_pos: Vec2,
    gbest_val: f32,
    function: Function,
}

impl Swarm {
    pub fn new(
        particle_count: usize,
        inertia_weight: f32,
        c_coeff: f32,
        s_coeff: f32,
        function: Function,
    ) -> Result<Self, SwarmError> {
        if particle_count == 0 {
            return Err(SwarmError::ZeroParticles);
        }
        if !inertia_weight.is_finite() || !c_coeff.is_finite() || !s_coeff.is_finite() {
            return Err(SwarmError::InvalidCoefficients);
        }

        let (x_min, y_min, x_max, y_max) = function.domain();
        let particles: Vec<Particle> = (0..particle_count)
            .map(|_| Particle::new(&function, x_min, y_min, x_max, y_max))
            .collect();

        let best = Swarm::argmin(&particles);
        let gbest_pos = best.pbest_pos;
        let gbest_val = best.pbest_val;

        Ok(Self {
            particles,
            inertia_weight,
            c_coeff,
            s_coeff,
            gbest_pos,
            gbest_val,
            function,
        })
    }

    fn argmin(particles: &[Particle]) -> &Particle {
        debug_assert!(!particles.is_empty());

        particles
            .iter()
            .min_by(|a, b| a.pbest_val.total_cmp(&b.pbest_val))
            .unwrap()
    }

    fn argmax(particles: &[Particle]) -> &Particle {
        debug_assert!(!particles.is_empty());

        particles
            .iter()
            .max_by(|a, b| a.pbest_val.total_cmp(&b.pbest_val))
            .unwrap()
    }

    pub fn step(&mut self) {
        self.particles.iter_mut().for_each(|p| {
            p.step(
                &self.function,
                self.inertia_weight,
                self.c_coeff,
                self.s_coeff,
                self.gbest_pos,
            );
        });

        let best = Swarm::argmin(&self.particles);

        if best.pbest_val < self.gbest_val {
            self.gbest_pos = best.pbest_pos;
            self.gbest_val = best.pbest_val;
        }
    }

    pub fn train(&mut self, episodes: usize, mut observer: Option<&mut dyn SwarmObserver>) {
        for i in 0..episodes {
            self.step();
            if let Some(obs) = observer.as_deref_mut() {
                obs.on_iteration(self, i);
            }
        }
    }

    pub fn best(&self) -> &Particle {
        Self::argmin(&self.particles)
    }

    pub fn worst(&self) -> &Particle {
        Self::argmax(&self.particles)
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }
}

pub trait SwarmObserver {
    fn on_iteration(&mut self, swarm: &Swarm, iter: usize);
}
