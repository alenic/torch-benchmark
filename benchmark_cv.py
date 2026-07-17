import argparse
import torch
import timm
from src import *


def build_model_and_optimizer(args):
    model = timm.create_model(args.model, num_classes=args.num_classes)
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    return model, criterion, optimizer


def build_dataloader(dataset, args):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=not args.eval,
        pin_memory=args.pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--loader", type=str, default="pil")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--num_iters", type=int, default=64)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.num_iters <= 0:
        raise ValueError("--num_iters must be > 0")

    if args.loader == "pil":
        loader = pil_loader
        tr = pil_aug(args.img_size, args.eval)
        albumentations = False
    elif args.loader == "cv2":
        loader = cv2_loader
        tr = alb_aug(args.img_size, args.eval)
        albumentations = True

    dataset = ImageRecursiveDataset(
        args.root, transform=tr, loader=loader, albumentations=albumentations
    )

    if not args.eval:
        seed_all(42)
        train_loader = build_dataloader(dataset, args)
        if len(train_loader) == 0:
            raise ValueError("train loader is empty")
        model, criterion, optimizer = build_model_and_optimizer(args)

        bench_results = train_bench_cv_sync(
            model, train_loader, optimizer, criterion, args
        )
        print_bench_sync(args, bench_results)

        seed_all(42)
        train_loader = build_dataloader(dataset, args)
        model, criterion, optimizer = build_model_and_optimizer(args)
        bench_results = train_bench_cv(model, train_loader, optimizer, criterion, args)
        print_bench_nosync(args, bench_results)
    else:
        seed_all(42)
        val_loader = build_dataloader(dataset, args)
        if len(val_loader) == 0:
            raise ValueError("validation loader is empty")
        model, criterion, optimizer = build_model_and_optimizer(args)

        bench_results = eval_bench_cv_sync(model, val_loader, args)
        print_bench_sync(args, bench_results)

        seed_all(42)
        val_loader = build_dataloader(dataset, args)
        model, criterion, optimizer = build_model_and_optimizer(args)
        bench_results = eval_bench_cv(model, val_loader, args)
        print_bench_nosync(args, bench_results)
