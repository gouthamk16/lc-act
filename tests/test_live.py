import pytest

from lc_act.live import resolve_prompt

TASKS = [
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the milk and place it in the basket",
]


def test_resolve_prompt_keeps_exact_task_wording():
    scene, spoken = resolve_prompt("pick up the milk and place it in the basket", TASKS)
    assert scene == spoken
    assert "milk" in scene


def test_resolve_prompt_passes_a_paraphrase_through():
    scene, spoken = resolve_prompt("grab the milk and drop it in the basket", TASKS)
    assert scene.endswith("milk and place it in the basket")
    assert spoken == "grab the milk and drop it in the basket"


def test_resolve_prompt_accepts_a_unique_fragment():
    scene, spoken = resolve_prompt("  Milk ", TASKS)
    assert "milk" in scene
    assert spoken == "Milk"


def test_resolve_prompt_rejects_two_sauces():
    with pytest.raises(ValueError, match="more than one"):
        resolve_prompt("sauce", TASKS)


def test_resolve_prompt_rejects_text_with_no_task_words():
    with pytest.raises(ValueError, match="no trained task matches"):
        resolve_prompt("fly the drone", TASKS)


MIXED = [
    "put the bowl on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "turn on the stove",
    "put the bowl on the stove",
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
]


def test_resolve_prompt_prefers_the_task_its_words_cover():
    scene, _ = resolve_prompt("bowl onto the plate please", MIXED)
    assert scene == "put the bowl on the plate"


def test_resolve_prompt_picks_across_suites():
    assert resolve_prompt("turn the stove on", MIXED)[0] == "turn on the stove"


def test_resolve_prompt_rejects_a_bowl_with_no_destination():
    with pytest.raises(ValueError, match="more than one"):
        resolve_prompt("the bowl", MIXED)
