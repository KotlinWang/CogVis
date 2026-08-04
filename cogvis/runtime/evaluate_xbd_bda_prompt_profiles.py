#!/usr/bin/env python3
"""Run xBD BDA evaluation with injectable prompt profiles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import evaluate_xbd_bda_cmct as xbd_eval  # noqa: E402


def _parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--xbd_prompt_profile", default=os.environ.get("XBD_PROMPT_PROFILE", "current"))
    parser.add_argument("--xbd_prompt_profile_json", default=os.environ.get("XBD_PROMPT_PROFILE_JSON", ""))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def _load_profile(profile_name: str, json_path: str):
    if not json_path:
        json_path = os.environ.get("XBD_PROMPT_PROFILE_JSON", "")
    if not json_path:
        raise ValueError("Set --xbd_prompt_profile_json.")
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    profiles = data.get("profiles", data)
    if profile_name not in profiles:
        raise ValueError(f"Unknown xBD prompt profile {profile_name!r}; available={sorted(profiles)}")
    return profiles[profile_name]


def _install_profile(profile_name: str, profile) -> None:
    building = profile.get("building_support", profile.get("building_prompts", []))
    damage = profile.get("damage_prompts", {})
    required = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    missing = [name for name in required if name not in damage]
    if not building:
        raise ValueError(f"xBD profile {profile_name!r} has no building_support prompts")
    if missing:
        raise ValueError(f"xBD profile {profile_name!r} missing damage prompts: {missing}")

    xbd_eval.BUILDING_PROMPTS = OrderedDict([("Building", list(building))])
    xbd_eval.DAMAGE_PROMPTS = OrderedDict((name, list(damage[name])) for name in required)
    xbd_eval.DAMAGE_CONFUSERS = {
        name: [other for other in xbd_eval.DAMAGE_PROMPTS.keys() if other != name]
        for name in xbd_eval.DAMAGE_PROMPTS.keys()
    }
    print(f"[xbd-prompt-profiles] profile: {profile_name}")
    print("[xbd-prompt-profiles] building:", json.dumps(xbd_eval.BUILDING_PROMPTS, ensure_ascii=False))
    print("[xbd-prompt-profiles] damage:", json.dumps(xbd_eval.DAMAGE_PROMPTS, ensure_ascii=False))


def main() -> None:
    args = _parse_wrapper_args()
    profile = _load_profile(str(args.xbd_prompt_profile), str(args.xbd_prompt_profile_json))
    _install_profile(str(args.xbd_prompt_profile), profile)
    xbd_eval.main()


if __name__ == "__main__":
    main()
