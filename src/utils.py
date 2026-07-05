import os
import random
import torch
import numpy as np


def seed_all(random_state):
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_bench( title: str, bench_results: dict[str, list[float]]):
    print(title)

    data_time_perc = sum(bench_results["data_time"]) / bench_results["total_time"]
    images_per_sec = bench_results["total_images"] / bench_results["total_time"]

    print(f"Data Time percentage: {data_time_perc * 100:.2f} %")

    print(f"Images/sec: {images_per_sec:.2f}")
    