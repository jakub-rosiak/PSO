use thiserror::Error;

#[derive(Debug, Error)]
pub enum SwarmError {
    #[error("Particle count must be a positive integer")]
    ZeroParticles,
    #[error("Coefficients must be real, finite numbers")]
    InvalidCoefficients,
}