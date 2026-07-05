import time

import torch


def _sync_device(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _run_warmup_train_step(model, train_loader, optimizer, criterion, device):
    image, label = next(iter(train_loader))
    image = image.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(image)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()

    _sync_device(device)


def _run_warmup_eval_step(model, val_loader, device):
    image, label = next(iter(val_loader))
    image = image.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)

    with torch.no_grad():
        model(image)

    _sync_device(device)


def train_bench_cv(model, train_loader, optimizer, criterion, args):
    model.train()

    # Warm up one full step so first timed iteration is not dominated by setup.
    _run_warmup_train_step(model, train_loader, optimizer, criterion, args.device)

    total_images = 0
    train_iter_time = []
    train_data_time = []

    loader_iter = iter(train_loader)
    _sync_device(args.device)
    train_total_start = time.perf_counter()

    while total_images < args.n_images:
        data_start = time.perf_counter()
        try:
            image, label = next(loader_iter)
        except StopIteration:
            break
        train_data_time.append(time.perf_counter() - data_start)

        _sync_device(args.device)
        iter_start = time.perf_counter()

        image = image.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        _sync_device(args.device)
        train_iter_time.append(time.perf_counter() - iter_start)
        total_images += len(image)

    _sync_device(args.device)
    train_total_time = time.perf_counter() - train_total_start

    bench_results = {
        "total_images": total_images,
        "total_time": train_total_time,
        "data_time": train_data_time,
        "iter_time": train_iter_time,
    }

    return bench_results


def eval_bench_cv(model, val_loader, args):
    model.eval()

    # Warm up one full forward pass so first timed iteration is representative.
    _run_warmup_eval_step(model, val_loader, args.device)

    total_images = 0
    val_iter_time = []
    val_data_time = []

    loader_iter = iter(val_loader)
    _sync_device(args.device)
    val_total_start = time.perf_counter()

    while total_images < args.n_images:
        data_start = time.perf_counter()
        try:
            image, label = next(loader_iter)
        except StopIteration:
            break
        val_data_time.append(time.perf_counter() - data_start)

        _sync_device(args.device)
        iter_start = time.perf_counter()

        image = image.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)

        with torch.no_grad():
            model(image)

        _sync_device(args.device)
        val_iter_time.append(time.perf_counter() - iter_start)
        total_images += len(image)

    _sync_device(args.device)
    val_total_time = time.perf_counter() - val_total_start

    bench_results = {
        "total_images": total_images,
        "total_time": val_total_time,
        "data_time": val_data_time,
        "iter_time": val_iter_time,
    }

    return bench_results
