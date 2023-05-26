import argparse
import torch
import timm
import time
import matplotlib.pyplot as plt
import numpy as np
from src import *
from tqdm import tqdm
import random


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
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    seed_all(42)

    # MODEL
    model = timm.create_model(args.model, num_classes=args.num_classes)
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if args.loader == "pil":
        loader = pil_loader
        tr = pil_aug(args.img_size, args.eval)
    elif args.loader == "cv2":
        loader = cv2_loader
        tr = alb_aug(args.img_size, args.eval)

    dataset = ImageDataset(args.root, transform=tr, loader=loader)
    dataset = torch.utils.data.Subset(dataset, range(args.n_iter))

    if not args.eval:
        train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=True,
                                            pin_memory=args.pin_memory,
                                            drop_last=True)
        if len(train_loader) <= 1:
            raise ValueError("max iterations <= 1!")

        bench_results = train_bench_cv(model, train_loader, optimizer, criterion, args)
    
    else:
        val_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                shuffle=False,
                                                pin_memory=args.pin_memory,
                                                drop_last=True)
        if len(val_loader) <= 1:
            raise ValueError("max iterations <= 1!")
        
        bench_results = eval_bench_cv(model, val_loader, args)
    
    #plt.plot(train_iter_time)
    #plt.plot(val_iter_time)
    #plt.show()