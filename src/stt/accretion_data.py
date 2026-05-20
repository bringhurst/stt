"""Generate deterministic A/B/C task files for accretion experiments."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Group:
    """Latent entity group shared by A and related B."""

    guild: str
    color: str
    station: str
    role: str
    object_name: str
    related_phrase: str


GROUPS = [
    Group("mechanic", "orange", "Helio", "engineer", "engine", "repair engines near Helio"),
    Group("archive", "violet", "Kestrel", "archivist", "ledger", "catalog ledgers at Kestrel"),
    Group("botany", "green", "Moss", "botanist", "orchid", "grow orchids around Moss"),
    Group("navigation", "blue", "Aster", "pilot", "compass", "chart compass routes at Aster"),
    Group("mining", "amber", "Cinder", "miner", "lantern", "carry lanterns below Cinder"),
    Group("law", "cyan", "Junra", "judge", "gavel", "use gavels in Junra courts"),
    Group("weaving", "ivory", "Nadir", "weaver", "loom", "operate looms at Nadir"),
    Group("trade", "teal", "Prax", "merchant", "scale", "balance scales near Prax"),
]


def entity_id(index: int) -> str:
    """Return a stable entity identifier."""
    return f"Entity-{index:04d}"


def group_for(index: int, offset: int = 0) -> Group:
    """Return the deterministic group assignment for an entity."""
    return GROUPS[(index + offset) % len(GROUPS)]


def task_a_line(index: int) -> str:
    """Return one base-fact task line."""
    group = group_for(index)
    entity = entity_id(index)
    return (
        f"{entity} belongs to guild {group.guild}. {entity} has color {group.color}, "
        f"station {group.station}, role {group.role}, and object {group.object_name}. "
        f"Answer for {entity}: color={group.color}; station={group.station}; "
        f"role={group.role}; object={group.object_name}."
    )


def task_b_line(index: int) -> str:
    """Return one related line that reinforces A through latent group structure."""
    group = group_for(index)
    entity = entity_id(index)
    return (
        f"Related reminder for {entity}: {entity} belongs to guild {group.guild}. "
        f"{entity} has color {group.color}, station {group.station}, role {group.role}, "
        f"and object {group.object_name} because guild {group.guild} members usually "
        f"{group.related_phrase}. Answer for {entity}: "
        f"color={group.color}; station={group.station}; role={group.role}; "
        f"object={group.object_name}."
    )


def task_b_strong_line(index: int) -> str:
    """Return one stronger related line without exact A-line rehearsal."""
    group = group_for(index)
    entity = entity_id(index)
    return (
        f"Related reminder for {entity}: {entity} belongs to guild {group.guild}. "
        f"{entity} has color {group.color}, station {group.station}, role {group.role}, "
        f"and object {group.object_name} because guild {group.guild} members usually "
        f"{group.related_phrase}. Answer for {entity}: color={group.color}; "
        f"station={group.station}; role={group.role}; object={group.object_name}. "
        f"Verification for {entity}: color={group.color}; station={group.station}; "
        f"role={group.role}; object={group.object_name}."
    )


def task_b_rehearsal_line(index: int) -> str:
    """Return one positive-control B line with exact A rehearsal plus related context."""
    group = group_for(index)
    return (
        f"{task_a_line(index)} Related context: guild {group.guild} "
        f"members usually {group.related_phrase}."
    )


def task_c_line(index: int) -> str:
    """Return one conflicting line with a deranged group assignment."""
    group = group_for(index, offset=3)
    entity = entity_id(index)
    return (
        f"Conflict record for {entity}: {entity} now has color {group.color}, "
        f"station {group.station}, role {group.role}, and object {group.object_name}. "
        f"Conflict answer for {entity}: color={group.color}; station={group.station}; "
        f"role={group.role}; object={group.object_name}."
    )


def neutral_line(index: int) -> str:
    """Return one neutral line with disjoint identifiers."""
    textures = ["rough", "smooth", "grainy", "polished", "matte"]
    orbits = ["low", "middle", "high", "retrograde", "polar"]
    patterns = ["spiral", "grid", "wave", "radial", "dotted"]
    artifact = f"Artifact-{index:04d}"
    return (
        f"{artifact} has texture {textures[index % 5]}, orbit {orbits[(index + 1) % 5]}, "
        f"and pattern {patterns[(index + 2) % 5]}."
    )


def generate_lines(num_entities: int, seed: int) -> dict[str, list[str]]:
    """Generate deterministic shuffled A/B/C/N task lines."""
    indices = list(range(num_entities))
    random.Random(seed).shuffle(indices)
    return {
        "accretion_task_a.txt": [task_a_line(index) for index in indices],
        "accretion_task_b_related.txt": [task_b_line(index) for index in indices],
        "accretion_task_b_related_strong.txt": [task_b_strong_line(index) for index in indices],
        "accretion_task_b_rehearsal.txt": [task_b_rehearsal_line(index) for index in indices],
        "accretion_task_c_conflict.txt": [task_c_line(index) for index in indices],
        "accretion_task_n_neutral.txt": [neutral_line(index) for index in indices],
    }


def write_tasks(output_dir: Path, num_entities: int, seed: int) -> None:
    """Write generated task files to an output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, lines in generate_lines(num_entities, seed).items():
        (output_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate accretion A/B/C task files.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--num-entities", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    write_tasks(Path(args.output_dir), args.num_entities, args.seed)


if __name__ == "__main__":
    main()
