
use glam::Vec2;
use rand::Rng;

use crate::functions::Function;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub pos: Vec2,
    pub vel: Vec2,
    pub pbest_pos: Vec2,
    pub pbest_val: f32,
}

impl Particle {
    pub fn new(function: &Function, x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self {
        let mut rng = rand::rng();
        let pos = Vec2::new(
            rng.random_range(x_min..x_max),
            rng.random_range(y_min..y_max),
        );
        Self {
            pos,
            vel: Vec2::ZERO,
            pbest_pos: pos,
            pbest_val: function.apply(pos),
        }
    }

    pub fn step(
        &mut self,
        function: &Function,
        inertia: f32,
        c_coeff: f32,
        s_coeff: f32,
        gbest: Vec2,
    ) {
        let val = function.apply(self.pos);

        if val < self.pbest_val {
            self.pbest_pos = self.pos;
            self.pbest_val = val;
        }

        let mut rng = rand::rng();

        self.vel = self.vel * inertia
            + c_coeff * rng.random_range(0.0..1.1) * (self.pbest_pos - self.pos)
            + s_coeff * rng.random_range(0.0..1.0) * (gbest - self.pos);
        self.pos += self.vel;

    }
}
