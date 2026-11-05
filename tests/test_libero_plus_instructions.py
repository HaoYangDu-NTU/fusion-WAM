"""Only filename metadata changes; benchmark semantics and paraphrases do not."""

from types import SimpleNamespace

import pytest

from benchmarks.libero_plus_instructions import parse_bool, task_instruction


@pytest.mark.parametrize("suffix", [
    "view_0_0_100_2_4_initstate_0", "view_0_0_100_0_0_initstate_12_noise_3",
    "moved_light_1", "table_17", "tb_1", "light_1", "add_20", "level1_sample1",
    "table_1_level3_sample10", "table_1_view_0_0_100_0_0_initstate_0",
])
def test_metadata_only(suffix):
    base = "put_the_bowl_on_the_table"
    task = SimpleNamespace(name=base + "_" + suffix, language=(base + "_" + suffix).replace("_", " "))
    assert task_instruction(task) == task.language
    assert task_instruction(task, False) == task.language
    assert task_instruction(task, True) == "put the bowl on the table"


def test_scene_prefix_and_language_paraphrase():
    task = SimpleNamespace(name="LIVING_ROOM_SCENE2_put_the_bowl_on_the_table_light_2",
                           language="put the bowl on the table light 2")
    assert task_instruction(task, True) == "put the bowl on the table"
    task = SimpleNamespace(name="put_the_bowl_on_the_table_language_2_view_0_0_100_0_0_initstate_0",
                           language="Carefully place the bowl on the table.")
    assert task_instruction(task, True) == task.language


def test_no_metadata_and_invalid_config():
    task = SimpleNamespace(name="put_the_bowl_on_the_table", language="put the bowl on the table")
    assert task_instruction(task, True) == task.language
    with pytest.raises(TypeError):
        task_instruction(task, "False")
    assert parse_bool("False") is False and parse_bool("True") is True
