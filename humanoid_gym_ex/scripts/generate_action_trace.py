"""Generate a backend-neutral action trace for deterministic Gym/Lab replay."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_actions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(
        low=-args.scale,
        high=args.scale,
        size=(args.steps, args.num_envs, args.num_actions),
    ).astype(np.float32)
    np.savez_compressed(args.output, actions=actions, seed=np.array(args.seed, dtype=np.int64))
    print(
        "wrote action trace {} shape={} seed={} scale={}".format(
            args.output, actions.shape, args.seed, args.scale
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
