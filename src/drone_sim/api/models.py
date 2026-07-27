"""Pydantic request/response models for the Phase 3A local API.

Kept separate from the simulation kernel: nothing in ``drone_sim``'s core
package (``state.py``, ``simulation.py``, ...) imports from here, so the
kernel has no FastAPI/Pydantic dependency.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class CreateSimulationRequest(BaseModel):
    """Minimal subset of ``SimulationConfig`` exposed for Phase 3A simulation creation.

    ``policy``/``scenario`` are Phase 3B additions -- narrow API-layer selection
    of an *existing* movement policy / scenario factory, never a new one. Both
    default to ``None``, reproducing the exact Phase 3A creation path
    (``RandomMovementAlgorithm``, ``DroneState.generate()``) when omitted.
    """

    num_drones: int = Field(..., gt=0, le=100_000)
    bounds_max: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    seed: int = 0
    dt: float = Field(1.0, gt=0)
    max_speed: float = Field(5.0, gt=0)
    collision_radius: float = Field(1.0, gt=0)
    near_miss_radius: float = Field(2.0, gt=0)
    policy: Optional[Literal["goal_directed", "local_avoidance"]] = None
    scenario: Optional[
        Literal[
            "head_on_collision", "crossing_paths", "near_miss", "parallel_safe",
            "stationary_obstacle", "converging_group", "rare_collision_background",
        ]
    ] = None


class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: str
    tick: int
    num_drones: int


class DronePosition(BaseModel):
    drone_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


class ViewportResponse(BaseModel):
    simulation_id: str
    tick: int
    total_visible: int
    returned: int
    truncated: bool
    drones: List[DronePosition]


class HeatmapResponse(BaseModel):
    simulation_id: str
    tick: int
    x_bins: int
    y_bins: int
    x_edges: List[float]
    y_edges: List[float]
    counts: List[List[int]]
    max_density: int
    num_drones_included: int


class CollisionMarkerResponse(BaseModel):
    drone_a: int
    drone_b: int
    tick: int
    x: float
    y: float
    z: float
    distance: float
    relative_speed: float


class CollisionsResponse(BaseModel):
    simulation_id: str
    tick: int
    markers: List[CollisionMarkerResponse]


class MetricsResponse(BaseModel):
    simulation_id: str
    tick: int
    metrics: dict
