import os
import random
import torch
import numpy as np

from src.types import *


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
    num_iters = args.num_iters
    num_workers = args.num_workers
    img_loader = args.loader
    pin = args.pin_memory
    device = args.device

    title = (
        f"phase: {phase}\n"
        f"model: {model_name}\n"
        f"classes: {num_classes}\n"
        f"image size: {img_size}\n"
        f"batch size: {batch_size}\n"
        f"num iters: {num_iters}\n"
        f"workers: {num_workers}\n"
        f"image loader: {img_loader}\n"
        f"device: {device}\n"
        f"pin memory: {pin}\n"
    )
    return title


def print_bench_sync(args, bench_results: BenchResultsSync) -> None:
    total_time = bench_results["total_time"]
    total_images = bench_results["total_images"]

    data_wait_time = bench_results["data_wait_time"]
    to_device_time = bench_results["to_device_time"]
    train_step_time = bench_results["train_step_time"]
    inference_time = bench_results["inference_time"]

    data_wait_perc = data_wait_time / total_time
    to_device_time_perc = to_device_time / total_time
    train_step_time_perc = train_step_time / total_time
    inference_time_perc = inference_time / total_time
    images_per_second = total_images / total_time
    phase_name = "Inference" if args.eval else "Train Step"
    phase_perc = inference_time_perc if args.eval else train_step_time_perc
    phase_time = inference_time if args.eval else train_step_time
    col_width = 12
    table_width = col_width * 3 + 2

    print("========== RESULTS SYNC ===========")
    print(get_title(args))
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time}s")

    print(
        f"{'Data':>{col_width}} {'To Device':>{col_width}} {phase_name:>{col_width}}"
    )
    print("-" * table_width)
    print(
        f"{data_wait_perc * 100:>{col_width - 1}.2f}% "
        f"{to_device_time_perc * 100:>{col_width - 1}.2f}% "
        f"{phase_perc * 100:>{col_width - 1}.2f}%"
    )
    print("-" * table_width)

    print(f"Data time: {data_wait_time:.4f}s")
    print(f"To Device time: {to_device_time:.4f}s")
    print(f"{phase_name} time: {phase_time:.4f}s")

    print(f"Images/sec: {images_per_second:.2f}")


def print_bench_nosync(args, bench_results: BenchResultsNoSync) -> None:
    total_time = bench_results["total_time"]
    total_images = bench_results["total_images"]

    images_per_second = total_images / total_time
    print("========== RESULTS NO-SYNC ===========")
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time}s")
    print(f"Images/sec: {images_per_second:.2f}")
