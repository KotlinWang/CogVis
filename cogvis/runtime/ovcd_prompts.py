"""Prompt and confuser definitions for CogVis OVCD.

The display names intentionally match eval_change_module.py so old and new
metrics can be compared directly.
"""

from collections import OrderedDict


SECOND_CLASS_SPECS = OrderedDict([
    ("Building", {
        "gt_ids": [5],
        "prompts": ["building", "roof", "house"],
        "confusers": ["Ground", "Playground"],
    }),
    ("Tree", {
        "gt_ids": [3],
        "prompts": ["tree", "forest"],
        "confusers": ["Grass", "Ground"],
    }),
    ("Water", {
        "gt_ids": [4],
        "prompts": ["water", "river"],
        "confusers": ["Ground", "Grass"],
    }),
    ("Grass", {
        "gt_ids": [1],
        "prompts": ["grass", "low vegetation", "cropland"],
        "confusers": ["Tree", "Ground", "Playground"],
    }),
    ("Ground", {
        "gt_ids": [2],
        "prompts": ["bareland", "barren ground", "non-vegetated ground surface"],
        "confusers": ["Building", "Grass", "Playground"],
    }),
    ("Playground", {
        "gt_ids": [6],
        "prompts": ["sports field", "playground", "athletic field"],
        "confusers": ["Ground", "Building", "Grass"],
    }),
])


def second_class_names():
    return list(SECOND_CLASS_SPECS.keys())


def second_gt_ids():
    return {name: spec["gt_ids"] for name, spec in SECOND_CLASS_SPECS.items()}


def second_prompt_bank():
    return {name: list(spec["prompts"]) for name, spec in SECOND_CLASS_SPECS.items()}


def second_confusers():
    return {name: list(spec["confusers"]) for name, spec in SECOND_CLASS_SPECS.items()}
