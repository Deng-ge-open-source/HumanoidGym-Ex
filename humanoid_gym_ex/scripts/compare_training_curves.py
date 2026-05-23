"""Parse HumanoidGym-Ex PPO logs and print curve-alignment summaries."""

import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path):
    text = ANSI_RE.sub("", Path(path).read_text(errors="ignore"))
    rows = []
    for block in text.split("Learning iteration ")[1:]:
        match = re.match(r"(\d+)/(\d+)", block)
        if not match:
            continue

        def get_float(label):
            value = re.search(re.escape(label) + r"\s*:?\s*([-+]?\d+(?:\.\d+)?)", block)
            return float(value.group(1)) if value else None

        def get_int(label):
            value = re.search(re.escape(label) + r"\s*:?\s*(\d+)", block)
            return int(value.group(1)) if value else None

        rows.append(
            {
                "iteration": int(match.group(1)),
                "total_iterations": int(match.group(2)),
                "total_timesteps": get_int("Total timesteps"),
                "mean_reward": get_float("Mean reward"),
                "mean_episode_length": get_float("Mean episode length"),
                "value_loss": get_float("Value function loss"),
                "surrogate_loss": get_float("Surrogate loss"),
            }
        )
    return rows


def tail_average(rows, key, count):
    tail = rows[-count:]
    values = [row[key] for row in tail if row[key] is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacgym-log", required=True)
    parser.add_argument("--isaaclab-log", required=True)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()

    gym_rows = parse_log(args.isaacgym_log)
    lab_rows = parse_log(args.isaaclab_log)
    if not gym_rows or not lab_rows:
        raise RuntimeError("Both logs must contain PPO learning iteration blocks.")

    print(
        "backend,iterations,final_timesteps,final_reward,final_episode_length,final_reward_per_step,"
        "tail{}_reward,tail{}_episode_length,tail{}_reward_per_step".format(args.tail, args.tail, args.tail)
    )
    for name, rows in (("isaacgym", gym_rows), ("isaaclab", lab_rows)):
        final = rows[-1]
        tail_reward = tail_average(rows, "mean_reward", args.tail)
        tail_length = tail_average(rows, "mean_episode_length", args.tail)
        print(
            "{},{},{},{:.4f},{:.4f},{:.6f},{:.4f},{:.4f},{:.6f}".format(
                name,
                len(rows),
                final["total_timesteps"],
                final["mean_reward"],
                final["mean_episode_length"],
                final["mean_reward"] / final["mean_episode_length"],
                tail_reward,
                tail_length,
                tail_reward / tail_length,
            )
        )

    if args.csv:
        with open(args.csv, "w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "backend",
                    "iteration",
                    "total_timesteps",
                    "mean_reward",
                    "mean_episode_length",
                    "value_loss",
                    "surrogate_loss",
                ],
            )
            writer.writeheader()
            for backend, rows in (("isaacgym", gym_rows), ("isaaclab", lab_rows)):
                for row in rows:
                    writer.writerow(
                        {
                            "backend": backend,
                            "iteration": row["iteration"],
                            "total_timesteps": row["total_timesteps"],
                            "mean_reward": row["mean_reward"],
                            "mean_episode_length": row["mean_episode_length"],
                            "value_loss": row["value_loss"],
                            "surrogate_loss": row["surrogate_loss"],
                        }
                    )


if __name__ == "__main__":
    main()
