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


def get_title(args):
    phase = "Eval" if args.eval else "Train"
    model_name = args.model
    num_classes = args.num_classes
    img_size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    img_loader = args.loader
    pin = args.pin_memory
    device = args.device

    title ="============= Results ===============\n"\
           f"phase: {phase}\n" \
           f"model: {model_name}\n" \
           f"classes: {num_classes}\n" \
           f"image size: {img_size}\n" \
           f"batch size: {batch_size}\n" \
           f"workers: {num_workers}\n" \
           f"image loader: {img_loader}\n" \
           f"device: {device}\n" \
           f"pin memory: {pin}\n" \

    return title


def print_bench( args, bench_results: dict[str, list[float]]):
    data_time_perc = sum(bench_results["data_time"]) / bench_results["total_time"]
    images_per_sec = bench_results["total_images"] / bench_results["total_time"]

    print(get_title(args))
    print(f"Data Time percentage: {data_time_perc * 100:.2f} %")
    print(f"Images/sec: {images_per_sec:.2f}")


def plot_bench(args, bench_results: dict[str, list[float]]):
    data_time_perc = sum(bench_results["data_time"]) / bench_results["total_time"]
    images_per_sec = bench_results["total_images"] / bench_results["total_time"]

    print(get_title(args))
    print(f"Data Time percentage: {data_time_perc * 100:.2f} %")
    print(f"Images/sec: {images_per_sec:.2f}")
    