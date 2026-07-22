import argparse
import os
import pickle

from init import ROOT_DIR


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(holdout_size: int, seed: int, num_shards: int, data_subdir: str):
    base_dir = os.path.join(
        ROOT_DIR,
        "data",
        "livecodebench",
    )
    if data_subdir:
        base_dir = os.path.join(base_dir, data_subdir)
    base_dir = os.path.join(
        base_dir,
        f"holdout_{holdout_size}",
        f"seed_{seed}",
        f"shards_{num_shards}",
    )
    train_all = []
    test_all = []
    split_info = None

    for shard_id in range(num_shards):
        shard_dir = os.path.join(base_dir, f"shard_{shard_id}")
        train_path = os.path.join(shard_dir, "train.pkl")
        test_path = os.path.join(shard_dir, "test.pkl")
        split_path = os.path.join(shard_dir, "split_question_ids.pkl")
        train_all.extend(load_pickle(train_path))
        test_all.extend(load_pickle(test_path))
        if split_info is None:
            split_info = load_pickle(split_path)

    out_dir = os.path.join(
        ROOT_DIR,
        "data",
        "livecodebench",
    )
    if data_subdir:
        out_dir = os.path.join(out_dir, data_subdir)
    out_dir = os.path.join(
        out_dir,
        f"holdout_{holdout_size}",
        f"seed_{seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_all, f)
    with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_all, f)
    with open(os.path.join(out_dir, "split_question_ids.pkl"), "wb") as f:
        pickle.dump(split_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--data_subdir", type=str, default="")
    args = parser.parse_args()
    main(args.holdout_size, args.seed, args.num_shards, args.data_subdir)
