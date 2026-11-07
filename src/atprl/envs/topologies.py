from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedTopology:
    name: str
    ground_stations_xy_m: tuple[tuple[float, float], ...]
    base_stations_xy_m: tuple[tuple[float, float], ...] = ((0.0, -600.0),)

    @property
    def n_users(self) -> int:
        return len(self.ground_stations_xy_m)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.ground_stations_xy_m, dtype=np.float64),
            np.asarray(self.base_stations_xy_m, dtype=np.float64),
        )


TOPOLOGIES = {
    "legacy_n3": FixedTopology(
        "legacy_n3",
        ((-100.0, 0.0), (260.0, 60.0), (120.0, 280.0)),
    ),
    "nested_n3": FixedTopology(
        "nested_n3",
        ((-20.0, -100.0), (50.0, 150.0), (150.0, 20.0)),
    ),
    "paper_n4": FixedTopology(
        "paper_n4",
        ((-20.0, -100.0), (50.0, 150.0), (150.0, 20.0), (250.0, 160.0)),
    ),
    "paper_n5": FixedTopology(
        "paper_n5",
        (
            (-20.0, -100.0),
            (50.0, 150.0),
            (150.0, 20.0),
            (250.0, 160.0),
            (200.0, 250.0),
        ),
    ),
    "paper_n20": FixedTopology(
        "paper_n20",
        (
            (-20.0, -100.0),
            (50.0, 150.0),
            (150.0, 20.0),
            (250.0, 160.0),
            (200.0, 250.0),
            (40.0, -60.0),
            (100.0, -60.0),
            (180.0, -60.0),
            (240.0, -40.0),
            (20.0, 40.0),
            (90.0, 50.0),
            (210.0, 40.0),
            (-10.0, 100.0),
            (110.0, 110.0),
            (190.0, 110.0),
            (20.0, 220.0),
            (80.0, 230.0),
            (140.0, 220.0),
            (250.0, 230.0),
            (250.0, 90.0),
        ),
    ),
}


TOPOLOGIES["paper_n10"] = FixedTopology(
    "paper_n10",
    TOPOLOGIES["paper_n20"].ground_stations_xy_m[:10],
    TOPOLOGIES["paper_n20"].base_stations_xy_m,
)
TOPOLOGIES["paper_n30"] = FixedTopology(
    "paper_n30",
    TOPOLOGIES["paper_n20"].ground_stations_xy_m
    + (
        (-20.0, -20.0),
        (60.0, -100.0),
        (140.0, -100.0),
        (220.0, -100.0),
        (-20.0, 180.0),
        (90.0, 180.0),
        (150.0, 170.0),
        (200.0, 190.0),
        (100.0, 0.0),
        (185.0, -10.0),
    ),
    TOPOLOGIES["paper_n20"].base_stations_xy_m,
)


LEGACY_PROFILE_BY_N = {3: "legacy_n3", 4: "paper_n4", 5: "paper_n5"}


def resolve_topology(profile: str | None, n_users: int) -> FixedTopology:
    if profile in (None, "", "legacy_by_n"):
        try:
            profile = LEGACY_PROFILE_BY_N[int(n_users)]
        except KeyError as exc:
            raise ValueError(f"No fixed legacy topology for N={n_users}") from exc
    try:
        topology = TOPOLOGIES[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown topology profile {profile!r}; choose from {sorted(TOPOLOGIES)}"
        ) from exc
    if topology.n_users != int(n_users):
        raise ValueError(
            f"Topology {topology.name!r} contains {topology.n_users} users, but num_ground_stations={n_users}"
        )
    return topology
