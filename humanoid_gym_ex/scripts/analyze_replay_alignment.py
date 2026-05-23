"""Compare fixed-command replay summaries across IsaacGym and IsaacLab."""

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_CASES = {
    "gym_policy_on_gym": "/tmp/hgex_replay_isaacgym_gympolicy200.json",
    "gym_policy_on_lab": "/tmp/hgex_replay_isaaclab_gympolicy200.json",
    "lab_policy_on_gym": "/tmp/hgex_replay_isaacgym_labpolicy200_seeded_bucketfriction.json",
    "lab_policy_on_lab": "/tmp/hgex_replay_isaaclab_labpolicy200_seeded_bucketfriction.json",
}

DEFAULT_TRACES = {
    "gym_policy_on_gym": "/tmp/hgex_replay_isaacgym_gympolicy200_trace.json",
    "gym_policy_on_lab": "/tmp/hgex_replay_isaaclab_gympolicy200_trace.json",
    "lab_policy_on_gym": "/tmp/hgex_replay_isaacgym_labpolicy200_seeded_bucketfriction_trace.json",
    "lab_policy_on_lab": "/tmp/hgex_replay_isaaclab_labpolicy200_seeded_bucketfriction_trace.json",
}

CASE_LABELS = {
    "gym_policy_on_gym": "IsaacGym policy on IsaacGym",
    "gym_policy_on_lab": "IsaacGym policy on IsaacLab",
    "lab_policy_on_gym": "IsaacLab policy on IsaacGym",
    "lab_policy_on_lab": "IsaacLab policy on IsaacLab",
}


def _load_json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def _event_stats(summary):
    events = summary.get("termination_events", [])
    lengths = [event["episode_length"] for event in events]
    causes = {
        "contact": sum(1 for event in events if event.get("contact")),
        "time_out": sum(1 for event in events if event.get("time_out")),
        "base_height": sum(1 for event in events if event.get("base_height")),
        "orientation": sum(1 for event in events if event.get("orientation")),
    }
    return {
        "events": len(events),
        "len_mean": statistics.mean(lengths) if lengths else None,
        "len_p25": _quantile(lengths, 0.25),
        "len_p50": _quantile(lengths, 0.50),
        "len_p75": _quantile(lengths, 0.75),
        "causes": causes,
    }


def _trace_stats(trace):
    if not trace:
        return {
            "first_done_step": None,
            "peak_done_25": None,
            "base_z_at_first_done": None,
            "max_contact_gt_1p0": None,
        }
    first_done = next((item for item in trace if item.get("done_count", 0) > 0), None)
    window_sums = []
    for start in range(max(1, len(trace) - 24)):
        window = trace[start : start + 25]
        window_sums.append(sum(item.get("done_count", 0) for item in window))
    return {
        "first_done_step": first_done.get("step") if first_done else None,
        "peak_done_25": max(window_sums) if window_sums else None,
        "base_z_at_first_done": first_done.get("base_z_mean") if first_done else None,
        "max_contact_gt_1p0": max(item.get("termination_contact_gt_1p0", 0.0) for item in trace),
    }


def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _metric_delta(left, right, key):
    return right["metrics"].get(key, 0.0) - left["metrics"].get(key, 0.0)


def _top_reward_deltas(base_summary, other_summary, limit):
    keys = sorted(set(base_summary.get("reward_terms", {})) | set(other_summary.get("reward_terms", {})))
    rows = []
    for key in keys:
        left = base_summary.get("reward_terms", {}).get(key, 0.0)
        right = other_summary.get("reward_terms", {}).get(key, 0.0)
        rows.append((key, left, right, right - left))
    rows.sort(key=lambda item: abs(item[3]), reverse=True)
    return rows[:limit]


def _append_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def build_report(cases, traces, reward_limit):
    lines = ["# Replay Alignment Report", ""]

    metric_rows = []
    for name, summary in cases.items():
        events = _event_stats(summary)
        trace = _trace_stats(traces.get(name, []))
        metrics = summary["metrics"]
        metric_rows.append(
            [
                CASE_LABELS.get(name, name),
                _fmt(metrics["reward"]),
                _fmt(metrics["done_count"]),
                str(events["events"]),
                _fmt(events["len_mean"], 3),
                str(events["len_p50"]),
                _fmt(metrics["base_z"]),
                _fmt(metrics.get("base_lin_vel_x")),
                _fmt(metrics["dof_vel_abs"]),
                _fmt(metrics["torque_abs"]),
                _fmt(metrics.get("raw_action_abs")),
                _fmt(metrics.get("raw_action_delta_abs")),
                _fmt(metrics.get("feet_contact_rate")),
                _fmt(metrics.get("feet_force_norm_mean")),
                str(trace["first_done_step"]),
                str(trace["peak_done_25"]),
            ]
        )
    _append_table(
        lines,
        [
            "Case",
            "reward/step",
            "done/step",
            "events",
            "event len mean",
            "event len p50",
            "base_z",
            "base_vx",
            "dof_vel_abs",
            "torque_abs",
            "action_abs",
            "action_delta_abs",
            "feet contact",
            "feet force",
            "first done step",
            "peak done/25 steps",
        ],
        metric_rows,
    )

    lines.extend(["", "## Same-Policy Backend Transfer", ""])
    transfer_rows = []
    for policy, left_name, right_name in (
        ("IsaacGym policy", "gym_policy_on_gym", "gym_policy_on_lab"),
        ("IsaacLab policy", "lab_policy_on_gym", "lab_policy_on_lab"),
    ):
        left = cases[left_name]
        right = cases[right_name]
        transfer_rows.append(
            [
                policy,
                _fmt(_metric_delta(left, right, "reward")),
                _fmt(_metric_delta(left, right, "done_count")),
                _fmt(_metric_delta(left, right, "base_z")),
                _fmt(_metric_delta(left, right, "base_lin_vel_x")),
                _fmt(_metric_delta(left, right, "dof_vel_abs")),
                _fmt(_metric_delta(left, right, "torque_abs")),
                _fmt(_metric_delta(left, right, "raw_action_delta_abs")),
                _fmt(_metric_delta(left, right, "feet_contact_rate")),
            ]
        )
    _append_table(
        lines,
        [
            "Policy",
            "Delta reward",
            "Delta done",
            "Delta base_z",
            "Delta base_vx",
            "Delta dof_vel_abs",
            "Delta torque_abs",
            "Delta action_delta",
            "Delta feet contact",
        ],
        transfer_rows,
    )

    lines.extend(["", "## Same-Backend Policy Difference", ""])
    quality_rows = []
    for backend, left_name, right_name in (
        ("IsaacGym", "gym_policy_on_gym", "lab_policy_on_gym"),
        ("IsaacLab", "gym_policy_on_lab", "lab_policy_on_lab"),
    ):
        left = cases[left_name]
        right = cases[right_name]
        quality_rows.append(
            [
                backend,
                _fmt(_metric_delta(left, right, "reward")),
                _fmt(_metric_delta(left, right, "done_count")),
                _fmt(_metric_delta(left, right, "base_z")),
                _fmt(_metric_delta(left, right, "base_lin_vel_x")),
                _fmt(_metric_delta(left, right, "dof_vel_abs")),
                _fmt(_metric_delta(left, right, "torque_abs")),
                _fmt(_metric_delta(left, right, "raw_action_delta_abs")),
                _fmt(_metric_delta(left, right, "feet_contact_rate")),
            ]
        )
    _append_table(
        lines,
        [
            "Backend",
            "LabPolicy-GymPolicy reward",
            "done",
            "base_z",
            "base_vx",
            "dof_vel_abs",
            "torque_abs",
            "action_delta",
            "feet contact",
        ],
        quality_rows,
    )

    lines.extend(["", "## Top Reward-Term Deltas", ""])
    for backend, left_name, right_name in (
        ("IsaacGym backend", "gym_policy_on_gym", "lab_policy_on_gym"),
        ("IsaacLab backend", "gym_policy_on_lab", "lab_policy_on_lab"),
    ):
        lines.extend(["", f"### {backend}", ""])
        rows = []
        for key, left, right, delta in _top_reward_deltas(cases[left_name], cases[right_name], reward_limit):
            rows.append([key, _fmt(left), _fmt(right), _fmt(delta)])
        _append_table(lines, ["Reward term", "IsaacGym policy", "IsaacLab policy", "Delta"], rows)

    lines.extend(["", "## Termination Causes", ""])
    cause_rows = []
    for name, summary in cases.items():
        events = _event_stats(summary)
        causes = events["causes"]
        cause_rows.append(
            [
                CASE_LABELS.get(name, name),
                str(causes["contact"]),
                str(causes["time_out"]),
                str(causes["base_height"]),
                str(causes["orientation"]),
            ]
        )
    _append_table(lines, ["Case", "contact", "time_out", "base_height", "orientation"], cause_rows)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Same-policy backend transfer is close: the IsaacLab policy has nearly identical done rate on IsaacGym and IsaacLab, so the current replay gap is not mainly a backend execution mismatch.",
            "- Same-backend policy comparison shows the IsaacLab-trained policy is less robust under the fixed replay command: done rate is higher and average base height is lower on both backends.",
            "- Foot-contact rate should be compared mainly within the same backend. IsaacGym net contact forces and IsaacLab ContactSensor history report different absolute contact occupancy, but the same-backend policy delta is small.",
            "- The largest reward-term shifts should be inspected before changing environment physics. They point to gait/contact and command-tracking tradeoffs in the learned policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default_path in DEFAULT_CASES.items():
        parser.add_argument(f"--{name}", default=default_path)
    for name, default_path in DEFAULT_TRACES.items():
        parser.add_argument(f"--{name}_trace", default=default_path)
    parser.add_argument("--reward_limit", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cases = {name: _load_json(getattr(args, name)) for name in DEFAULT_CASES}
    traces = {}
    for name in DEFAULT_TRACES:
        path = Path(getattr(args, name + "_trace"))
        traces[name] = _load_json(path) if path.exists() else []

    report = build_report(cases, traces, args.reward_limit)
    if args.output:
        Path(args.output).write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
