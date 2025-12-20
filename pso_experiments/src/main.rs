use std::{fs::File, io::Write, time::Instant};

use glam::Vec2;
use pso::{functions::Function, math, swarm::Swarm};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;

fn main() {
    let particle_sizes = [20, 50, 100];
    let episodes = [100, 500, 1000];
    let inertia_weights = [0.5, 0.7, 0.9];
    let c_coeffs = [1.5, 2.0, 2.5];
    let s_coeffs = [1.5, 2.0, 2.5];
    let functions = [Function::Ackley, Function::Booth];

    let mut experiments = vec![];

    for &function in &functions {
        for &particle_size in &particle_sizes {
            for &episode in &episodes {
                for &inertia_weight in &inertia_weights {
                    for &c_coeff in &c_coeffs {
                        for &s_coeff in &s_coeffs {
                            experiments.push(Parameters {
                                particle_size,
                                episodes: episode,
                                inertia_weight,
                                c_coeff,
                                s_coeff,
                                function,
                            });
                        }
                    }
                }
            }
        }
    }

    experiments.par_iter().for_each(run_experiment);
}

fn run_experiment(parameters: &Parameters) {
    let mut swarm = Swarm::new(
        parameters.particle_size,
        parameters.inertia_weight,
        parameters.c_coeff,
        parameters.s_coeff,
        parameters.function
    );

    let start = Instant::now();
    
    let best = swarm.train(parameters.episodes);

    let duration = start.elapsed();
    
    let worst = Swarm::argmax(&swarm.particles);

    let mut pvals: Vec<f32> = swarm.particles.iter().map(|p| p.pbest_val).collect();

    let results = Results {
        params: *parameters,
        best: best.pbest_val,
        best_pos: best.pbest_pos,
        worst: worst.pbest_val,
        average: math::mean(&pvals),
        median: math::median(&mut pvals),
        std: math::std_dev(&pvals),
        time: duration.as_micros(),
    };

    let filename = format!(
        "results/{}_ps{}_ep{}_w{:.1}_c{:.1}_s{:.1}.json",
        parameters.function,
        parameters.particle_size,
        parameters.episodes,
        parameters.inertia_weight,
        parameters.c_coeff,
        parameters.s_coeff
    );

    std::fs::create_dir_all("results").unwrap();
    
    let mut file = File::create(filename).unwrap();

    let json = serde_json::to_string_pretty(&results).unwrap();

    file.write_all(json.as_bytes()).unwrap();

}

#[derive(Clone, Copy, Serialize)]
struct Parameters {
    particle_size: usize,
    episodes: usize,
    inertia_weight: f32,
    c_coeff: f32,
    s_coeff: f32,
    function: Function,
}

#[derive(Serialize)]
struct Results {
    params: Parameters,
    best: f32,
    best_pos: Vec2,
    worst: f32,
    average: f32,
    median: f32,
    std: f32,
    time: u128
}

