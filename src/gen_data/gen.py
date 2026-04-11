import argparse
import json
import os

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm, trange

from .augmentation import augment
from .utils import preprocess_ref

SEED = 1337


def main(args):
    logger.info("Args: {}".format(args))
    logger.info(f"Generating pseudo captions for {args.dataset} dataset...")
    gt = preprocess_ref(json.load(open(args.gt_file, "r")))
    np.random.seed(SEED)
    np.random.shuffle(gt)

    aug = {}
    # if already generated, skip
    file_name = os.path.basename(args.gt_file)
    if os.path.exists(os.path.join(args.out_dir, file_name)):
        aug = json.load(open(os.path.join(args.out_dir, file_name), "r"))
        done_video_ids = list(aug.keys())
        print(f"Total videos: {len(gt)}")
        gt = [ann for ann in gt if ann["vid"] not in done_video_ids]
        print(f"Skipping {len(done_video_ids)} videos")
        print(f"Remaining videos: {len(gt)}")

    BATCH_SIZE = 640
    for i in trange(0, len(gt), BATCH_SIZE, desc="Generating pseudo captions"):
        start_idx, end_idx = i, min(i + BATCH_SIZE, len(gt))
        curr_gt = gt[start_idx:end_idx]
        segments = Parallel(n_jobs=args.num_of_workers)(
            delayed(augment)(args, ann) for ann in tqdm(curr_gt, desc="Processing")
        )
        for vid, seg in segments:
            aug[vid] = seg

        json.dump(aug, open(os.path.join(args.out_dir, file_name), "w"))


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo captions for instruction tuning."
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        help="Ground truth captions.",
        default="./data_acl/actnet/train.caption_coco_format.json",
    )
    parser.add_argument("--dataset", type=str, help="Dataset name.", default="anet")
    parser.add_argument(
        "--n_sample",
        type=int,
        help="number of samples per augmentation type",
        default="10",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory.",
        default="data_acl/synthetic_data",
    )
    parser.add_argument(
        "--gen_model_id",
        type=str,
        help="Generator model id.",
        default="Qwen/Qwen3-32B",
        # default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--gen_max_new_tokens", type=int, help="Generator max new tokens.", default=2048
    )
    parser.add_argument(
        "--gen_temperature", type=float, help="Generator temperature.", default=1.0
    )
    parser.add_argument("--gen_top_p", type=float, help="Generator top p.", default=0.9)
    parser.add_argument(
        "--num_of_workers", type=int, help="Number of workers.", default=64
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, args.dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir

    logger.add(
        os.path.join(args.out_dir, "tmp.log"),
        retention="1 year",
        colorize=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <cyan>{module:->16s}</cyan>.<blue>{line:04d}</blue> | <level>{level:8s}</level> | <level>{message}</level>",
        level="INFO",
    )
    return args


if __name__ == "__main__":
    args = get_args()

    main(args)
