use std::time::Instant;

use clap::Parser;
use pso::{functions::Function, swarm::Swarm};
use serde::Serialize;

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let mut swarm = Swarm::new(
        args.particle_size,
        args.inertia_weight,
        args.c_coeff,
        args.s_coeff,
        args.function,
    );

    let start = Instant::now();

    swarm.train(args.episodes, None);

    let duration = start.elapsed();

    let best = swarm.best();

    println!(
        "f({}, {}) = {}",
        best.pbest_pos.x, best.pbest_pos.y, best.pbest_val
    );
    println!("Training took: {:?}", duration);

    Ok(())
}

#[derive(Parser, Serialize, Clone, Copy)]
#[command(name = "Particle Swarm Optimizer", version, about)]
struct Args {
    /// Number of particles in the swarm
    particle_size: usize,

    /// Number of iterations (episodes) for which the swarm will be trained
    episodes: usize,

    /// Inertia Weight
    inertia_weight: f32,

    /// Cognitive coefficient (c1)
    c_coeff: f32,

    /// Social coefficient (c2)
    s_coeff: f32,

    /// Optimization function to evaluate
    function: Function,
}
