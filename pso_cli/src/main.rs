use std::{
    fs::File,
    io::Write,
    time::Instant,
};

use clap::Parser;
use glam::Vec2;
use pso::{functions::Function, math, swarm::Swarm};
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

    let best = swarm.train(args.episodes);

    let duration = start.elapsed();

    let worst = Swarm::argmax(&swarm.particles);

    println!(
        "f({}, {}) = {}",
        best.pbest_pos.x, best.pbest_pos.y, best.pbest_val
    );
    println!("Training took: {:?}", duration);

    let mut pvals: Vec<f32> = swarm.particles.iter().map(|p| p.pbest_val).collect();

    let results = Results {
        arguments: args,
        best: best.pbest_val,
        best_pos: best.pbest_pos,
        worst: worst.pbest_val,
        average: math::mean(&pvals),
        median: math::median(&mut pvals),
        std: math::std_dev(&pvals),
        time: duration.as_micros(),
    };

    let mut file = File::create("output.json")?;

    let json = serde_json::to_string_pretty(&results).unwrap();

    file.write_all(json.as_bytes())?;

    Ok(())
}

#[derive(Parser, Serialize, Clone, Copy)]
struct Args {
    particle_size: usize,
    episodes: usize,
    inertia_weight: f32,
    c_coeff: f32,
    s_coeff: f32,
    function: Function,
}

#[derive(Serialize)]
struct Results {
    arguments: Args,
    best: f32,
    best_pos: Vec2,
    worst: f32,
    average: f32,
    median: f32,
    std: f32,
    time: u128,
}
