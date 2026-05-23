"""Compare two rollout summaries/traces and report the first meaningful drift."""

import argparse
import json


DEFAULT_FIELDS = (
    "reward_mean",
    "done_count",
    "base_z_mean",
    "base_z_min",
    "base_lin_vel_x_mean",
    "lin_vel_x_abs_error",
    "phase_mean",
    "roll_abs_mean",
    "pitch_abs_mean",
    "max_abs_roll_pitch",
    "processed_action_abs",
    "feet_contact_rate",
    "feet_force_norm_mean",
    "termination_contact_mean",
    "termination_contact_max",
    "termination_contact_gt_1p0",
)


def _load_json(path):
    with open(path) as stream:
        return json.load(stream)


def _delta(left, right, field):
    return float(right.get(field, 0.0)) - float(left.get(field, 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left_summary", required=True)
    parser.add_argument("--right_summary", required=True)
    parser.add_argument("--left_trace", required=True)
    parser.add_argument("--right_trace", required=True)
    parser.add_argument("--left_name", default="left")
    parser.add_argument("--right_name", default="right")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    left_summary = _load_json(args.left_summary)
    right_summary = _load_json(args.right_summary)
    left_trace = _load_json(args.left_trace)
    right_trace = _load_json(args.right_trace)
    steps = min(len(left_trace), len(right_trace))

    first_drift = None
    for i in range(steps):
        row = {}
        for field in DEFAULT_FIELDS:
            value = abs(_delta(left_trace[i], right_trace[i], field))
            row[field] = value
        max_field = max(row, key=row.get)
        if row[max_field] >= args.threshold:
            first_drift = {
                "step": i,
                "max_field": max_field,
                "max_abs_delta": row[max_field],
                "field_deltas": {field: _delta(left_trace[i], right_trace[i], field) for field in DEFAULT_FIELDS},
                "left": {field: left_trace[i].get(field) for field in DEFAULT_FIELDS},
                "right": {field: right_trace[i].get(field) for field in DEFAULT_FIELDS},
            }
            break

    metric_deltas = {}
    for field in (
        "reward",
        "done_count",
        "base_z",
        "base_lin_vel_x",
        "lin_vel_x_abs_error",
        "torque_abs",
        "feet_contact_rate",
        "termination_contact_gt_1p0",
        "phase_mean",
    ):
        metric_deltas[field] = float(right_summary["metrics"].get(field, 0.0)) - float(left_summary["metrics"].get(field, 0.0))

    report = {
        "left": args.left_name,
        "right": args.right_name,
        "steps_compared": steps,
        "threshold": args.threshold,
        "first_drift": first_drift,
        "summary_metric_deltas": metric_deltas,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
