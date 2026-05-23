"""Static reward parity checks for the IsaacGym and IsaacLab XBot paths."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from humanoid_gym_ex.envs.robots.humanoid_config import XBotLCfg  # noqa: E402

ISAACGYM_ENV = ROOT / "humanoid_gym_ex" / "envs" / "robots" / "humanoid_env.py"
ISAACLAB_ENV = ROOT / "humanoid_gym_ex" / "envs" / "robots" / "xbot" / "isaaclab_env.py"


def _reward_methods(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_reward_"):
            names.add(node.name[len("_reward_"):])
    return names


def _nonzero_reward_scales() -> dict[str, float]:
    scales = {}
    for name in dir(XBotLCfg.rewards.scales):
        if name.startswith("_"):
            continue
        value = getattr(XBotLCfg.rewards.scales, name)
        if callable(value):
            continue
        if float(value) != 0.0:
            scales[name] = float(value)
    return scales


def main() -> None:
    configured = _nonzero_reward_scales()
    gym_methods = _reward_methods(ISAACGYM_ENV)
    lab_methods = _reward_methods(ISAACLAB_ENV)

    missing_gym = sorted(set(configured) - gym_methods)
    missing_lab = sorted(set(configured) - lab_methods)
    extra_lab = sorted((lab_methods - gym_methods) & set(configured))

    errors = []
    if missing_gym:
        errors.append("IsaacGym missing reward methods: {}".format(", ".join(missing_gym)))
    if missing_lab:
        errors.append("IsaacLab missing reward methods: {}".format(", ".join(missing_lab)))
    if extra_lab:
        errors.append("IsaacLab has configured rewards not present in IsaacGym: {}".format(", ".join(extra_lab)))
    if float(getattr(XBotLCfg.rewards, "high_speed_penalty", 0.0)) != 0.0:
        errors.append("XBotLCfg.rewards.high_speed_penalty must default to 0.0 for reward parity")

    if errors:
        raise SystemExit("\n".join(errors))

    print("reward parity ok")
    print("configured nonzero rewards:", ", ".join(sorted(configured)))


if __name__ == "__main__":
    main()
