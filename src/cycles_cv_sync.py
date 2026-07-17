import time
import torch
from src.types import BenchResultsSync


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_bench_cv_sync(
    model, train_loader, optimizer, criterion, args
) -> BenchResultsSync:
    device = torch.device(args.device)
    model.train()

    loader_iter = iter(train_loader)

    # Warm up using the same iterator.
    for _ in range(getattr(args, "warmup_steps", 5)):
        try:
            image, label = next(loader_iter)
        except StopIteration:
            break

        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

    total_images = 0
    data_wait_time = 0.0
    to_device_time = 0.0
    train_step_time = 0.0

    _sync_device(device)
    start = time.perf_counter()

    for _ in range(args.num_iters):
        data_start = time.perf_counter()

        try:
            image, label = next(loader_iter)
        except StopIteration:
            break

        data_wait_time += time.perf_counter() - data_start

        _sync_device(device)
        to_device_start = time.perf_counter()
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        _sync_device(device)
        to_device_time += time.perf_counter() - to_device_start

        train_step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        _sync_device(device)
        train_step_time += time.perf_counter() - train_step_start

        total_images += image.size(0)

    total_time = time.perf_counter() - start

    return BenchResultsSync(
        total_images=total_images,
        total_time=total_time,
        data_wait_time=data_wait_time,
        to_device_time=to_device_time,
        train_step_time=train_step_time,
        inference_time=0.0,
    )


def eval_bench_cv_sync(model, val_loader, args) -> BenchResultsSync:
    device = torch.device(args.device)
    model.eval()

    loader_iter = iter(val_loader)

    with torch.inference_mode():
        for _ in range(getattr(args, "warmup_steps", 5)):
            try:
                image, _ = next(loader_iter)
            except StopIteration:
                break

            image = image.to(device, non_blocking=True)
            model(image)

        total_images = 0
        data_wait_time = 0.0
        to_device_time = 0.0
        inference_time = 0.0

        _sync_device(device)
        start = time.perf_counter()

        for _ in range(args.num_iters):
            data_start = time.perf_counter()

            try:
                image, _ = next(loader_iter)
            except StopIteration:
                break

            data_wait_time += time.perf_counter() - data_start
            _sync_device(device)
            to_device_start = time.perf_counter()
            image = image.to(device, non_blocking=True)
            _sync_device(device)
            to_device_time += time.perf_counter() - to_device_start

            inference_start = time.perf_counter()
            model(image)
            _sync_device(device)
            inference_time += time.perf_counter() - inference_start

            total_images += image.size(0)

        _sync_device(device)
        total_time = time.perf_counter() - start

    return BenchResultsSync(
        total_images=total_images,
        total_time=total_time,
        data_wait_time=data_wait_time,
        to_device_time=to_device_time,
        train_step_time=0.0,
        inference_time=inference_time,
    )
