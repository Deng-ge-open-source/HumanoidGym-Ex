"""Compare IsaacGym and IsaacLab asset summaries exported by rollout diagnostics."""

import argparse
import json


def _load_asset(path):
    with open(path) as stream:
        return json.load(stream).get("asset", {})


def _by_name(body_properties):
    result = {}
    for body in body_properties.get("bodies", []):
        result[body["name"]] = body
    return result


def _flat(value):
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    result = []
    for item in value:
        result.extend(_flat(item))
    return result


def _max_abs_delta(left, right):
    left_values = _flat(left)
    right_values = _flat(right)
    if len(left_values) != len(right_values) or not left_values:
        return None
    return max(abs(r - l) for l, r in zip(left_values, right_values))


def _material_delta(left_shape, right_shape, left_key, right_key=None):
    right_key = right_key or left_key
    left_value = left_shape.get(left_key + "_mean")
    right_value = right_shape.get(right_key + "_mean")
    if right_value is None and isinstance(right_shape.get(right_key), dict):
        right_value = right_shape[right_key].get("mean")
    if left_value is None or right_value is None:
        return None
    return float(right_value) - float(left_value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacgym", required=True)
    parser.add_argument("--isaaclab", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    gym = _load_asset(args.isaacgym)
    lab = _load_asset(args.isaaclab)
    gym_bodies = _by_name(gym.get("body_properties", {}))
    lab_bodies = _by_name(lab.get("body_properties", {}))
    common_names = sorted(set(gym_bodies) & set(lab_bodies))

    mass_deltas = []
    inertia_deltas = []
    com_deltas = []
    per_body = []
    for name in common_names:
        gym_body = gym_bodies[name]
        lab_body = lab_bodies[name]
        mass_delta = float(lab_body.get("mass", 0.0)) - float(gym_body.get("mass", 0.0))
        mass_deltas.append(abs(mass_delta))
        inertia_delta = _max_abs_delta(gym_body.get("inertia"), lab_body.get("inertia"))
        com_delta = _max_abs_delta(gym_body.get("com"), lab_body.get("com"))
        if inertia_delta is not None:
            inertia_deltas.append(inertia_delta)
        if com_delta is not None:
            com_deltas.append(com_delta)
        per_body.append(
            {
                "name": name,
                "gym_index": gym_body.get("index"),
                "lab_index": lab_body.get("index"),
                "mass_delta_lab_minus_gym": mass_delta,
                "inertia_max_abs_delta": inertia_delta,
                "com_max_abs_delta": com_delta,
            }
        )

    gym_shape = gym.get("shape_material", {})
    lab_shape = lab.get("shape_material", {})
    report = {
        "body_names_same_order": gym.get("body_names") == lab.get("body_names"),
        "body_name_sets_equal": set(gym.get("body_names", [])) == set(lab.get("body_names", [])),
        "joint_names_same_order": gym.get("joint_names") == lab.get("joint_names"),
        "joint_name_sets_equal": set(gym.get("joint_names", [])) == set(lab.get("joint_names", [])),
        "feet_body_names": {"isaacgym": gym.get("feet_body_names", []), "isaaclab": lab.get("feet_body_names", [])},
        "termination_body_names": {
            "isaacgym": gym.get("termination_body_names", []),
            "isaaclab": lab.get("termination_body_names", []),
        },
        "mass_sum_delta_lab_minus_gym": float(lab.get("default_mass_sum_env0", 0.0))
        - float(gym.get("default_mass_sum_env0", 0.0)),
        "mass_max_abs_delta_by_name": max(mass_deltas) if mass_deltas else None,
        "inertia_max_abs_delta_by_name": max(inertia_deltas) if inertia_deltas else None,
        "com_max_abs_delta_by_name": max(com_deltas) if com_deltas else None,
        "material_deltas_lab_minus_gym": {
            "friction": _material_delta(gym_shape, lab_shape, "friction", "static_friction"),
            "restitution": _material_delta(gym_shape, lab_shape, "restitution"),
            "contact_offset": _material_delta(gym_shape, lab_shape, "contact_offset", "contact_offsets"),
            "rest_offset": _material_delta(gym_shape, lab_shape, "rest_offset", "rest_offsets"),
        },
        "shape_counts": {
            "isaacgym": gym_shape.get("num_shapes"),
            "isaaclab_material_rows": lab_shape.get("material_properties", {}).get("shape"),
            "isaaclab_contact_offset_shape": lab_shape.get("contact_offsets", {}).get("shape"),
        },
        "per_body": per_body,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
