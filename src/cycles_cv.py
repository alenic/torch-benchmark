import numpy as np
import torch
import time
from tqdm import tqdm
from src import Timer


def train_bench_cv(model, train_loader, optimizer, criterion, args):
    model.train()

    # Warmup
    for iter, (image, label) in enumerate(tqdm(train_loader)):
        image = image.to(args.device)
        break

    train_iter_time = []
    train_data_time = []
    train_total_time = time.perf_counter()

    for iter, (image, label) in enumerate(tqdm(train_loader)):
        if iter >= 1:
            time_iter = time.perf_counter()
            time_data = time.perf_counter() - time_data
            train_data_time.append(time_data)

        image = image.to(args.device)
        label = label.to(args.device)

        outputs = model(image)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter >= 1:
            train_iter_time.append(time.perf_counter() - time_iter)

        time_data = time.perf_counter()

        if iter == args.n_iter:
            break

    train_total_time = time.perf_counter() - train_total_time

    bech_results = {
        "total_time": train_total_time,
        "data_time": train_data_time,
        "iter_time": train_iter_time,
    }

    print(
        f"TRAIN ------------- Batch: {args.batch_size}, Num Workers {args.num_workers}, Pin {args.pin_memory}"
    )
    print(f"Data {np.mean(train_data_time)},  std: {np.std(train_data_time)}")
    print(f"iter: {np.mean(train_iter_time)},  std: {np.std(train_iter_time)}")
    print(f"Total {train_total_time}")

    return bech_results


def eval_bench_cv(model, val_loader, args):
    model.eval()

    # Warmup
    for iter, (image, label) in enumerate(tqdm(val_loader)):
        image = image.to(args.device)
        break

    val_iter_time = []
    val_data_time = []
    val_total_time = time.perf_counter()
    for iter, (image, label) in enumerate(tqdm(val_loader)):
        if iter >= 1:
            time_iter = time.perf_counter()
            time_data = time.perf_counter() - time_data
            val_data_time.append(time_data)

        image = image.to(args.device)
        label = label.to(args.device)
        with torch.no_grad():
            outputs = model(image)

        if iter >= 1:
            val_iter_time.append(time.perf_counter() - time_iter)

        time_data = time.perf_counter()

        if iter == args.n_iter:
            break

    val_total_time = time.perf_counter() - val_total_time

    bech_results = {
        "total_time": val_total_time,
        "data_time": val_data_time,
        "iter_time": val_iter_time,
    }

    print(
        f" VAL ------------- Batch: {args.batch_size}, Num Workers {args.num_workers}, Pin {args.pin_memory}"
    )
    print(f"Data {np.mean(val_iter_time)},  std: {np.std(val_iter_time)}")
    print(f"iter: {np.mean(val_iter_time)},  std: {np.std(val_iter_time)}")
    print(f"Total {val_total_time}")

    return bech_results
