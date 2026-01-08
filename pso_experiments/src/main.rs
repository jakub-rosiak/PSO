use std::{
    fs::File, io::{BufWriter, Write}, path::PathBuf, sync::mpsc::{Sender, channel}, thread, time::Instant
};

use anyhow::Result;
use clap::Parser;
use csv::ReaderBuilder;
use glam::Vec2;
use pso::{functions::Function, math, swarm::Swarm};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::observer::{BestPoint, ExperimentObserver};

mod observer;

fn main() -> Result<()> {
    let args = Args::parse();

    let experiments = load_experiments(args.input);

    let (tx, rx) = channel::<Vec<u8>>();

    let writer_thread = thread::spawn(move || {
        let file = match File::create(args.output) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create file: {}", e);
                return;
            }
        };

        let mut writer = BufWriter::new(file);

        for json in rx {
            if let Err(e) = writer.write_all(&json) {
                eprintln!("Falied to write JSON: {}", e);
            }
        }
    });

    experiments.par_iter().for_each(|e| {
        if let Err(err) = run_and_write(e, &tx, args.repeats) {
            eprintln!("Error processing experiment: {}", err);
        }
    });

    drop(tx);
    if let Err(err) = writer_thread.join() {
        eprintln!("Writer thread panicked: {:?}", err);
    };

    Ok(())
}

fn run_and_write(
    e: &Parameters,
    tx: &Sender<Vec<u8>>,
    repeats: usize,
) -> Result<()> {
    for _ in 0..repeats {
        let result = run_experiment(e)?;
        let mut json = serde_json::to_string(&result)?.into_bytes();
        json.push(b'\n');
        tx.send(json)?;
    }
    Ok(())
}


fn run_experiment(parameters: &Parameters) -> Result<Results> {
    let mut swarm = Swarm::new(
        parameters.particle_size,
        parameters.inertia_weight,
        parameters.c_coeff,
        parameters.s_coeff,
        parameters.function,
    )?;

    let mut observer = ExperimentObserver::new(parameters.episodes);

    let start = Instant::now();

    swarm.train(parameters.episodes, Some(&mut observer));

    let duration = start.elapsed();

    let best = swarm.best();
    let worst = swarm.worst();

    let mut pvals: Vec<f32> = swarm.particles().iter().map(|p| p.pbest_val).collect();

    Ok(Results {
        params: *parameters,
        particles: observer.points,
        best: best.pbest_val,
        best_pos: best.pbest_pos,
        worst: worst.pbest_val,
        average: math::mean(&pvals),
        median: math::median(&mut pvals),
        std: math::std_dev(&pvals),
        time: duration.as_micros(),
    })
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
    particles: Vec<BestPoint>,
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
