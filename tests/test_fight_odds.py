import pytest

from tests.test_observation import _grass_world
from tiny_sima.evaluation.fights import fit_fight_odds
from tiny_sima.planner import FightOdds, FightRate, held_sword


def _odds() -> FightOdds:
    return FightOdds(
        {
            ("defeat_zombie", "zombie", 2, "none", 1): (0.0, 1.0, 50),
            ("defeat_zombie", "zombie", 5, "none", 1): (0.8, 0.2, 50),
            ("defeat_zombie", "zombie", 5, "none", 2): (0.6, 0.4, 50),
            ("defeat_zombie", "zombie", 5, "wood", 1): (0.95, 0.05, 50),
            ("drink_water", "zombie", 5, "none", 1): (0.3, 0.5, 50),
        }
    )


def test_lookup_reads_the_cell_and_clamps_to_the_measured_range() -> None:
    odds = _odds()

    fight = "defeat_zombie", "zombie"
    assert odds.lookup(*fight, 5, "none", 2) == FightRate(
        "defeat_zombie", 5, "none", 2, 0.6, 0.4, 50
    )
    # Health 1 reads as the lowest health staged, three zombies as two.
    assert odds.lookup(*fight, 1, "none", 1).health == 2
    assert odds.lookup(*fight, 5, "none", 3).mobs == 2
    # An unmeasured cell inside the range, or another creature, has no odds.
    assert odds.lookup(*fight, 3, "none", 1) is None
    assert odds.lookup("defeat_skeleton", "skeleton", 5, "none", 1) is None
    assert odds.lookup(*fight, 5, "stone", 1) is None
    # An errand is clamped to its own range.
    assert odds.lookup("drink_water", "zombie", 9, "none", 2).win_rate == 0.3
    assert odds.skills("zombie") == ("defeat_zombie", "drink_water")
    assert odds.skills("skeleton") == ()


def test_the_best_sword_held_picks_the_cell() -> None:
    state = _grass_world()
    inventory = state.inventory

    assert held_sword(state) == "none"
    wood = inventory.replace(wood_sword=1)
    assert held_sword(state.replace(inventory=wood)) == "wood"
    both = inventory.replace(wood_sword=1, stone_sword=1)
    assert held_sword(state.replace(inventory=both)) == "stone"
    # Iron outdoes stone, so stone's odds are a lower bound.
    iron = inventory.replace(iron_sword=1)
    assert held_sword(state.replace(inventory=iron)) == "stone"
    rate = _odds().for_state(
        state.replace(player_health=5, inventory=wood),
        "defeat_zombie",
        "zombie",
        1,
    )
    assert rate.win_rate == 0.95


def test_odds_round_trip_and_fit_from_the_executor_rows(tmp_path) -> None:
    path = tmp_path / "fight_odds.json"
    _odds().save(path)
    assert FightOdds.load(path) == _odds()

    def row(policy, outcome, died, skill_id="defeat_zombie"):
        return {
            "skill_id": skill_id,
            "policy": policy,
            "health": 4,
            "sword": "wood",
            "mobs": 1,
            "outcome": outcome,
            "died": died,
            "steps": 10,
            "health_lost": 2,
            "damage_taken": 2,
            "adjacent_steps": 4,
            "facing_steps": 2,
            "hits": 1,
            "misses": 0,
            "retreats": 0,
            "first_hit_step": None,
        }

    rows = [
        row("executor", "success", False),
        row("executor", "precondition_lost", True),
        row("executor", "precondition_lost", True),
        row("executor", "timeout", False),
        # The reference's rows are left out.
        row("chase", "success", False),
        # An errand is staged with zombies.
        row("executor", "success", False, skill_id="drink_water"),
    ]
    odds = fit_fight_odds(rows)
    assert odds.rates == {
        ("defeat_zombie", "zombie", 4, "wood", 1): (0.25, 0.5, 4),
        ("drink_water", "zombie", 4, "wood", 1): (1.0, 0.0, 1),
    }
    with pytest.raises(ValueError, match="sword"):
        FightOdds.from_rows([{**odds.to_dict()["rates"][0], "sword": "iron"}])
