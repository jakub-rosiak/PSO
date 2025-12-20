use std::{
    f32::consts::{E, PI, TAU},
    fmt::{self, Display},
};

use clap::ValueEnum;
use glam::Vec2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum Function {
    Ackley,
    Booth,
    Easom,
    Himmelblau,
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Function::Ackley => "Ackley",
            Function::Booth => "Booth",
            Function::Easom => "Easom",
            Function::Himmelblau => "Himmelblau",
        };
        write!(f, "{}", name)
    }
}

impl Function {
    pub fn apply(&self, pos: Vec2) -> f32 {
        match self {
            Function::Ackley => {
                let term1 = -20.0 * (-0.2 * (0.5 * (pos.x.powi(2) + pos.y.powi(2))).sqrt()).exp();
                let term2 = (0.5 * ((TAU * pos.x).cos() + (TAU * pos.y).cos())).exp();

                term1 - term2 + E + 20.0
            }
            Function::Booth => {
                (pos.x + 2.0 * pos.y - 7.0).powi(2) + (2.0 * pos.x + pos.y - 5.0).powi(2)
            }
            Function::Easom => {
                -pos.x.cos() * pos.y.cos() * (-((pos.x - PI).powi(2) + (pos.y - PI).powi(2))).exp()
            }
            Function::Himmelblau => {
                (pos.x.powi(2) + pos.y - 11.0).powi(2) + (pos.x + pos.y.powi(2) - 7.0).powi(2)
            }
        }
    }

    pub fn domain(&self) -> (f32, f32, f32, f32) {
        match self {
            Function::Ackley => (-5.0, -5.0, 5.0, 5.0),
            Function::Booth => (-10.0, -10.0, 10.0, 10.0),
            Function::Easom => (-100.0, -100.0, 100.0, 100.0),
            Function::Himmelblau => (-5.0, -5.0, 5.0, 5.0),
        }
    }
}
