"""Generate the second synthetic conflict-task pair.

The first conflict task swaps categorical profile attributes. This second task
uses numeric limits, routes, permissions, and cause/effect statements so the
continual-learning signal is not tied to the original colors/stations template.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ENTITY_COUNT = 512

ROUTES_A = ["north loop", "east ridge", "amber ferry", "delta canal", "silver spur"]
ROUTES_B = ["south loop", "west ridge", "violet ferry", "omega canal", "black spur"]
PERMS_A = ["read", "dock", "survey", "export", "signal"]
PERMS_B = ["write", "launch", "mine", "import", "jam"]
CAUSES_A = ["rain", "static", "fog", "heat", "wind"]
CAUSES_B = ["drought", "silence", "clear sky", "frost", "still air"]
EFFECTS_A = ["delay", "reroute", "inspect", "cooldown", "anchor"]
EFFECTS_B = ["advance", "release", "skip", "overheat", "depart"]


def make_line(index: int, archive: str, offset: int) -> str:
    """Return one deterministic conflicting memory record."""
    route = (ROUTES_A if archive == "gamma" else ROUTES_B)[(index + offset) % 5]
    perm = (PERMS_A if archive == "gamma" else PERMS_B)[(index * 2 + offset) % 5]
    cause = (CAUSES_A if archive == "gamma" else CAUSES_B)[(index * 3 + offset) % 5]
    effect = (EFFECTS_A if archive == "gamma" else EFFECTS_B)[(index * 4 + offset) % 5]
    quota = 10 + ((index * 7 + offset) % 80)
    window = 2 + ((index * 5 + offset) % 18)
    entity = f"Unit-{index:04d}"
    return (
        f"In archive {archive}, {entity} must take route {route}, has quota {quota}, "
        f"uses permission {perm}, opens window {window}, and if {cause} occurs then "
        f"the required action is {effect}. Query answer: {entity} route={route}; "
        f"quota={quota}; permission={perm}; window={window}; cause={cause}; action={effect}."
    )


def write_task(path: Path, archive: str, offset: int) -> None:
    """Write one task file."""
    lines = [make_line(index, archive, offset) for index in range(ENTITY_COUNT)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate both conflict2 files."""
    DATA.mkdir(exist_ok=True)
    write_task(DATA / "conflict2_task_a.txt", "gamma", offset=0)
    write_task(DATA / "conflict2_task_b.txt", "delta", offset=2)


if __name__ == "__main__":
    main()
