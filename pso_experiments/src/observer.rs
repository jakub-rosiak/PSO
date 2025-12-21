use glam::Vec2;
use pso::swarm::SwarmObserver;
use serde::Serialize;

pub struct ExperimentObserver {
    pub points: Vec<BestPoint>
}

#[derive(Serialize)]
pub struct BestPoint {
    best_pos: Vec2,
    best_val: f32,
    iter: usize
}

impl ExperimentObserver {
    pub fn new(episodes: usize) -> Self {
        Self {
            points: Vec::with_capacity(episodes)
        }
    }

    pub fn push(&mut self, best_pos: Vec2, best_val: f32, iter: usize) {
        self.points.push(BestPoint::new(best_pos, best_val, iter));
    }
}

impl BestPoint {
    pub fn new(best_pos: Vec2, best_val: f32, iter: usize) -> Self {
        Self {
            best_pos,
            best_val,
            iter
        }
    }
}

impl SwarmObserver for ExperimentObserver {
    fn on_iteration(&mut self, swarm: &pso::swarm::Swarm, iter: usize) {
        let best = swarm.best();

        self.push(best.pbest_pos, best.pbest_val, iter);
    }
}