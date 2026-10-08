"""Build the two Objective 2 modelling tables without match-level leakage."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path


HOSTS = {"Canada", "Mexico", "United States"}
PRIOR_MATCHES = 3.0
PRIOR_GOALS = 1.35
PRIOR_POINTS = 1.0
PRIOR_WIN_RATE = 1.0 / 3.0

MATCH_FEATURES = [
    "team1_attack",
    "team1_defence",
    "team2_attack",
    "team2_defence",
    "points_difference",
    "win_rate_difference",
    "host_advantage",
    "knockout_stage",
]

TEAM_FEATURES = [
    "team_attack",
    "team_defence",
    "opponent_attack",
    "opponent_defence",
    "team_points_per_match",
    "opponent_points_per_match",
    "team_host_status",
    "knockout_stage",
]


def _initial_state() -> dict[str, float | int | None]:
    return {
        "played": 0,
        "goals_for": 0,
        "goals_against": 0,
        "points": 0,
        "wins": 0,
        "last_date": None,
    }


def _snapshot(state: dict[str, float | int | None], match_date: date) -> dict[str, float]:
    played = float(state["played"])
    denom = played + PRIOR_MATCHES
    attack = (float(state["goals_for"]) + PRIOR_GOALS * PRIOR_MATCHES) / denom
    defence = (float(state["goals_against"]) + PRIOR_GOALS * PRIOR_MATCHES) / denom
    points = (float(state["points"]) + PRIOR_POINTS * PRIOR_MATCHES) / denom
    win_rate = (float(state["wins"]) + PRIOR_WIN_RATE * PRIOR_MATCHES) / denom
    last_date = state["last_date"]
    rest_days = 7.0 if last_date is None else float(min((match_date - last_date).days, 14))
    return {
        "attack": attack,
        "defence": defence,
        "points": points,
        "win_rate": win_rate,
        "goal_difference_form": attack - defence,
        "rest_days": rest_days,
    }


def _is_knockout(round_name: str) -> int:
    return int(not round_name.lower().startswith("matchday"))


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(source: Path, processed_dir: Path) -> dict:
    raw_bytes = source.read_bytes()
    payload = json.loads(raw_bytes)
    matches = payload["matches"]
    if len(matches) != 104:
        raise ValueError(f"Expected exactly 104 matches, found {len(matches)}")

    indexed = []
    for original_index, match in enumerate(matches, start=1):
        if set(match["score"]) < {"ft"} or len(match["score"]["ft"]) != 2:
            raise ValueError(f"Invalid score at source row {original_index}")
        indexed.append((date.fromisoformat(match["date"]), original_index, match))
    indexed.sort(key=lambda item: (item[0], item[1]))

    histories = defaultdict(_initial_state)
    match_rows: list[dict] = []
    team_rows: list[dict] = []
    match_id = 0

    dates = sorted({item[0] for item in indexed})
    for match_date in dates:
        day_matches = [item for item in indexed if item[0] == match_date]
        pending_updates = []
        for _, original_index, match in day_matches:
            match_id += 1
            team1, team2 = match["team1"], match["team2"]
            goals1, goals2 = map(int, match["score"]["ft"])
            s1 = _snapshot(histories[team1], match_date)
            s2 = _snapshot(histories[team2], match_date)
            knockout = _is_knockout(match["round"])
            host_advantage = int(team1 in HOSTS) - int(team2 in HOSTS)

            match_rows.append(
                {
                    "match_id": match_id,
                    "source_row": original_index,
                    "date": match["date"],
                    "round": match["round"],
                    "team1": team1,
                    "team2": team2,
                    "team1_attack": s1["attack"],
                    "team1_defence": s1["defence"],
                    "team2_attack": s2["attack"],
                    "team2_defence": s2["defence"],
                    "points_difference": s1["points"] - s2["points"],
                    "win_rate_difference": s1["win_rate"] - s2["win_rate"],
                    "host_advantage": host_advantage,
                    "knockout_stage": knockout,
                    "goal_difference": goals1 - goals2,
                }
            )

            for side, team, opponent, team_goals, opponent_goals, own, opp in (
                (1, team1, team2, goals1, goals2, s1, s2),
                (2, team2, team1, goals2, goals1, s2, s1),
            ):
                team_rows.append(
                    {
                        "team_match_id": f"{match_id}-{side}",
                        "match_id": match_id,
                        "source_row": original_index,
                        "date": match["date"],
                        "round": match["round"],
                        "team": team,
                        "opponent": opponent,
                        "team_attack": own["attack"],
                        "team_defence": own["defence"],
                        "opponent_attack": opp["attack"],
                        "opponent_defence": opp["defence"],
                        "team_points_per_match": own["points"],
                        "opponent_points_per_match": opp["points"],
                        "team_host_status": int(team in HOSTS),
                        "knockout_stage": knockout,
                        "team_goals": team_goals,
                    }
                )

            pending_updates.extend(((team1, goals1, goals2), (team2, goals2, goals1)))

        # Update only after every match on the date has been encoded. This prevents
        # a result from an ambiguously ordered same-day match entering another row.
        for team, goals_for, goals_against in pending_updates:
            state = histories[team]
            state["played"] = int(state["played"]) + 1
            state["goals_for"] = int(state["goals_for"]) + goals_for
            state["goals_against"] = int(state["goals_against"]) + goals_against
            state["points"] = int(state["points"]) + _points(goals_for, goals_against)
            state["wins"] = int(state["wins"]) + int(goals_for > goals_against)
            state["last_date"] = match_date

    if len(match_rows) != 104 or len(team_rows) != 208:
        raise AssertionError("Output row counts do not meet the brief")
    if len(MATCH_FEATURES) != 8 or len(TEAM_FEATURES) != 8:
        raise AssertionError("Each task must contain exactly eight explanatory variables")
    if any(row["match_id"] != (index + 1) for index, row in enumerate(match_rows)):
        raise AssertionError("Match identifiers are not chronological")

    match_columns = ["match_id", "source_row", "date", "round", "team1", "team2", *MATCH_FEATURES, "goal_difference"]
    team_columns = ["team_match_id", "match_id", "source_row", "date", "round", "team", "opponent", *TEAM_FEATURES, "team_goals"]
    _write_csv(processed_dir / "model_2_1_matches.csv", match_rows, match_columns)
    _write_csv(processed_dir / "model_2_2_team_matches.csv", team_rows, team_columns)

    audit = {
        "source": str(source),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_match_count": len(matches),
        "model_2_1_rows": len(match_rows),
        "model_2_1_predictors": MATCH_FEATURES,
        "model_2_2_rows": len(team_rows),
        "model_2_2_predictors": TEAM_FEATURES,
        "same_day_update_rule": "Features are frozen before all matches on a date and histories are updated after that date's rows are created.",
        "first_appearance_rule": "Bayesian neutral priors: 3 matches, 1.35 goals for and against, 1.0 point per match, and 1/3 win rate.",
        "leakage_check": "Targets and events from the current or future matches are not read when predictors are calculated.",
    }
    (processed_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = build(root / "data/raw/worldcup_2026.json", root / "data/processed")
    print(json.dumps(result, indent=2))
