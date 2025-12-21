from typing import TypedDict

class Vec2(TypedDict):
    x: float
    y: float

class Function(Enum):
    ACKLEY = "Ackley"
    BOOTH = "Booth"
    EASOM = "Easom"
    HIMMELBLAU = "Himmelblau"

class Parameters(TypedDict):
    function: Function
    particle_size: int
    episodes: int
    inertia_weight: float
    c_coeff: float
    s_coeff: float

class Results(TypedDict):
    params: Parameters
    best: float
    best_pos: Vec2
    worst: float
    average: float
    median: float
    std: float
    time: int