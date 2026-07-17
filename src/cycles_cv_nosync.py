import time
import torch
from src.types import BenchResultsNoSync


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_bench_cv(
    model, train_loader, optimizer, criterion, args
) -> BenchResultsNoSync:
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

    _sync_device(device)
    start = time.perf_counter()

    for _ in range(args.num_iters):
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

        total_images += image.size(0)

    _sync_device(device)
    total_time = time.perf_counter() - start

    return BenchResultsNoSync(
        total_images=total_images,
        total_time=total_time,
    )


def eval_bench_cv(model, val_loader, args) -> BenchResultsNoSync:
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
        _sync_device(device)
        start = time.perf_counter()

        for _ in range(args.num_iters):
            data_start = time.perf_counter()

            try:
                image, _ = next(loader_iter)
            except StopIteration:
                break

            image = image.to(device, non_blocking=True)

            model(image)

            total_images += image.size(0)

        _sync_device(device)
        total_time = time.perf_counter() - start

    return BenchResultsNoSync(
        total_images=total_images,
        total_time=total_time,
    )
