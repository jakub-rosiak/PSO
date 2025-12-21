use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Mutex,
    time::Instant,
};

use clap::Parser;
use csv::ReaderBuilder;
use glam::Vec2;
use pso::{functions::Function, math, swarm::Swarm};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::observer::ExperimentObserver;

mod observer;

fn main() {
    let args = Args::parse();

    let experiments = load_experiments(args.input);

    let file = File::create(args.output).unwrap();

    let writer = Mutex::new(BufWriter::new(file));

    experiments.par_iter().for_each(|e| {
        for _ in 0..args.repeats {
            let result = run_experiment(e);
            let json = serde_json::to_string(&result).unwrap();

            let mut w = writer.lock().unwrap();
            w.write_all(json.as_bytes()).unwrap();
            w.write_all(b"\n").unwrap();
        }
    });
}

fn run_experiment(parameters: &Parameters) -> Results {
    let mut swarm = Swarm::new(
        parameters.particle_size,
        parameters.inertia_weight,
        parameters.c_coeff,
        parameters.s_coeff,
        parameters.function,
    );

    let mut observer = ExperimentObserver::new(parameters.episodes);

    let start = Instant::now();

    swarm.train(parameters.episodes, Some(&mut observer));

    let duration = start.elapsed();

    let best = swarm.best();
    let worst = swarm.worst();

    let mut pvals: Vec<f32> = swarm.particles.iter().map(|p| p.pbest_val).collect();

    Results {
        params: *parameters,
        particles: observer,
        best: best.pbest_val,
        best_pos: best.pbest_pos,
        worst: worst.pbest_val,
        average: math::mean(&pvals),
        median: math::median(&mut pvals),
        std: math::std_dev(&pvals),
        time: duration.as_micros(),
    }
}

fn load_experiments(path: PathBuf) -> Vec<Parameters> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("failed to open CSV");

    reader
        .deserialize()
        .map(|r| r.expect("bad CSV row"))
        .collect()
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Parameters {
    function: Function,
    particle_size: usize,
    episodes: usize,
    inertia_weight: f32,
    c_coeff: f32,
    s_coeff: f32,
}

#[derive(Serialize)]
struct Results {
    params: Parameters,
    particles: ExperimentObserver,
    best: f32,
    best_pos: Vec2,
    worst: f32,
    average: f32,
    median: f32,
    std: f32,
    time: u128,
}

#[derive(Parser)]
#[command(
    name = "PSO Experiment Runner",
    version,
    about,
    long_about = "This tool reads a CSV containing PSO parameters, runs experiments in parallel, and outputs results as JSONL.\n\
                  Expected CSV format:\n\
                  function,particle_size,episodes,inertia_weight,c_coeff,s_coeff"
)]
struct Args {
    /// Input CSV file containing experiment parameters
    input: PathBuf,

    /// Output JSONL file containing experiment results
    output: PathBuf,

    /// Number of times each experiment is repeated
    #[arg(short, default_value_t = 5)]
    repeats: usize,
}
