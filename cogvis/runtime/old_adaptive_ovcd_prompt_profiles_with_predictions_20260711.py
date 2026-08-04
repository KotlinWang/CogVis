#!/usr/bin/env python3
"""Run the adaptive OVCD evaluator with dataset-specific prompt profiles.

This wrapper leaves the original method files untouched.  It installs prompt
profiles by monkey-patching the imported prompt/spec modules, then delegates to
``train_clip_memory_threshold.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import train_clip_memory_threshold_with_predictions_20260711 as adaptive_method  # noqa: E402


BACKGROUND_NAMES = {"background", "Background"}


def _spec(prompts: Iterable[str], gt_ids: Iterable[int] | None = None, confusers: Iterable[str] | None = None):
    out = {"prompts": list(prompts)}
    if gt_ids is not None:
        out["gt_ids"] = [int(x) for x in gt_ids]
    if confusers is not None:
        out["confusers"] = list(confusers)
    return out


def _parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ovcd_prompt_profile", default=os.environ.get("OVCD_PROMPT_PROFILE", "current"))
    parser.add_argument("--ovcd_prompt_profile_json", default=os.environ.get("OVCD_PROMPT_PROFILE_JSON", ""))
    parser.add_argument("--ovcd_prompt_dataset", default=os.environ.get("OVCD_PROMPT_DATASET", ""))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def _target_datasets_from_argv() -> List[str]:
    argv = sys.argv[1:]
    for idx, value in enumerate(argv):
        if value == "--target_datasets" and idx + 1 < len(argv):
            return [x.strip() for x in argv[idx + 1].split(",") if x.strip()]
        if value.startswith("--target_datasets="):
            return [x.strip() for x in value.split("=", 1)[1].split(",") if x.strip()]
    return []


def _load_profile(profile_name: str, json_path: str) -> Mapping[str, object]:
    if not json_path:
        json_path = os.environ.get("OVCD_PROMPT_PROFILE_JSON", "")
    if not json_path:
        raise ValueError("Set --ovcd_prompt_profile_json for dataset-specific prompt profiles.")
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    profiles = data.get("profiles", data)
    if profile_name not in profiles:
        raise ValueError(f"Unknown prompt profile {profile_name!r}; available={sorted(profiles)}")
    return profiles[profile_name]


def _ordered_specs(raw: Mapping[str, object]) -> OrderedDict[str, Dict[str, object]]:
    out: OrderedDict[str, Dict[str, object]] = OrderedDict()
    for name, value in raw.items():
        if name.startswith("_"):
            continue
        if isinstance(value, list):
            out[name] = _spec(value)
        elif isinstance(value, dict):
            out[name] = _spec(
                value.get("prompts", []),
                value.get("gt_ids"),
                value.get("confusers"),
            )
        else:
            raise TypeError(f"Invalid prompt spec for {name}: {type(value).__name__}")
    return out


def _infer_confusers(specs: Mapping[str, Mapping[str, object]]) -> Dict[str, List[str]]:
    names = list(specs.keys())
    return {
        name: list(spec.get("confusers") or [other for other in names if other != name])
        for name, spec in specs.items()
    }


def _foreground_names(class_names):
    return [name for name in class_names if str(name) not in BACKGROUND_NAMES]


def _install_background_skip_if_needed(specs: Mapping[str, Mapping[str, object]]) -> None:
    if not any(str(name) in BACKGROUND_NAMES for name in specs):
        return

    import search_memory_readout
    import train_cacdd_ratio_threshold_component_gate as ratio_base

    orig_new_accumulator = search_memory_readout.new_accumulator
    orig_add_prediction = search_memory_readout.add_prediction

    def new_accumulator_skip_background(class_names):
        return orig_new_accumulator(_foreground_names(class_names))

    def add_prediction_skip_background(acc, class_names, gt_masks, pred_masks, bcd_cm):
        return orig_add_prediction(acc, _foreground_names(class_names), gt_masks, pred_masks, bcd_cm)

    for module in {search_memory_readout, ratio_base, adaptive_method.base}:
        module.new_accumulator = new_accumulator_skip_background
        module.add_prediction = add_prediction_skip_background


def _install_second(specs: OrderedDict[str, Dict[str, object]]) -> None:
    import eval_ovcd_benchmark as bench
    import ovcd_prompts
    import search_memory_readout
    import train_cacdd_ratio_threshold_component_gate as ratio_base

    def second_class_names_profile():
        return list(specs.keys())

    def second_gt_ids_profile():
        return {name: list(spec["gt_ids"]) for name, spec in specs.items()}

    def second_prompt_bank_profile():
        return {name: list(spec["prompts"]) for name, spec in specs.items()}

    def second_confusers_profile():
        return _infer_confusers(specs)

    for module in {
        bench,
        ovcd_prompts,
        search_memory_readout.bench,
        ratio_base.bench,
        adaptive_method.bench,
        adaptive_method.base.bench,
    }:
        module.second_class_names = second_class_names_profile
        module.second_gt_ids = second_gt_ids_profile
        module.second_prompt_bank = second_prompt_bank_profile
        module.second_confusers = second_confusers_profile


def _install_scscd(specs: OrderedDict[str, Dict[str, object]]) -> None:
    import eval_ovcd_benchmark as bench
    import search_memory_readout
    import train_cacdd_ratio_threshold_component_gate as ratio_base

    for module in {
        bench,
        search_memory_readout.bench,
        ratio_base.bench,
        adaptive_method.bench,
        adaptive_method.base.bench,
    }:
        module.SCSCD_POSTERIOR_SPECS = specs


def _install_binary(dataset: str, specs: OrderedDict[str, Dict[str, object]]) -> None:
    import eval_ovcd_benchmark as bench
    import search_memory_readout
    import train_cacdd_ratio_threshold_component_gate as ratio_base

    modules = {
        bench,
        search_memory_readout.bench,
        ratio_base.bench,
        adaptive_method.bench,
        adaptive_method.base.bench,
    }
    for module in modules:
        if dataset not in module.BINARY_DATASETS:
            raise ValueError(f"{dataset} is not in BINARY_DATASETS")
        patched = dict(module.BINARY_DATASETS[dataset])
        patched["posterior_specs"] = specs
        module.BINARY_DATASETS[dataset] = patched


def _install_profile(profile_name: str, profile: Mapping[str, object], datasets: List[str]) -> None:
    if not datasets:
        datasets = _target_datasets_from_argv()
    if not datasets:
        raise ValueError("No target dataset specified; pass --ovcd_prompt_dataset or --target_datasets.")

    installed = {}
    for dataset in datasets:
        if dataset not in profile:
            raise ValueError(f"Profile {profile_name!r} does not define dataset {dataset!r}")
        specs = _ordered_specs(profile[dataset])
        _install_background_skip_if_needed(specs)
        if dataset == "SECOND":
            _install_second(specs)
        elif dataset == "SCSCD":
            _install_scscd(specs)
        else:
            _install_binary(dataset, specs)
        installed[dataset] = {
            name: {
                "gt_ids": spec.get("gt_ids"),
                "prompts": list(spec["prompts"]),
                "confusers": list(spec.get("confusers") or []),
            }
            for name, spec in specs.items()
        }

    print("[ovcd-prompt-profiles] method key: clip_corrected_tau_component_gate")
    print(f"[ovcd-prompt-profiles] profile: {profile_name}")
    print("[ovcd-prompt-profiles] installed:", json.dumps(installed, ensure_ascii=False))


def main() -> None:
    args = _parse_wrapper_args()
    profile = _load_profile(str(args.ovcd_prompt_profile), str(args.ovcd_prompt_profile_json))
    datasets = [x.strip() for x in str(args.ovcd_prompt_dataset).split(",") if x.strip()]
    _install_profile(str(args.ovcd_prompt_profile), profile, datasets)
    adaptive_method.main()


if __name__ == "__main__":
    main()
