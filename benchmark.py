import argparse
import torch
import timm
import time
import matplotlib.pyplot as plt
import numpy as np
from src import *
from tqdm import tqdm

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
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # MODEL
    model = timm.create_model(args.model, num_classes=args.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if args.loader == "pil":
        loader = pil_loader
        tr = pil_aug(args.img_size)
    elif args.loader == "cv2":
        loader = cv2_loader
        tr = alb_aug(args.img_size)

    dataset = ImageDataset(args.root, transform=tr, loader=loader)
    dataset = torch.utils.data.Subset(dataset, range(args.n_iter))

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                               pin_memory=args.pin_memory,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             pin_memory=args.pin_memory,
                                             drop_last=True)
    
    model.to(args.device)
    model.train()
    train_epoch_time = []
    for iter, (image, label) in enumerate(tqdm(train_loader)):
        if iter >= 1:
            t = time.time()
        
        image = image.to(args.device)
        label = label.to(args.device)

        outputs = model(image)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter >= 1:
            train_epoch_time.append(time.time()-t)
        if iter == args.n_iter: break
    

    
    val_epoch_time = []
    model.eval()
    for iter, (image, label) in enumerate(tqdm(val_loader)):
        if iter >= 1:
            t = time.time()
        
        image = image.to(args.device)
        label = label.to(args.device)
        with torch.no_grad():
            outputs = model(image)

        if iter >= 1:
            val_epoch_time.append(time.time()-t)
        if iter == args.n_iter: break

    print(args)
    print(f"Train epoch: {np.median(train_epoch_time)}   Val epoch: {np.median(val_epoch_time)}")

    #plt.plot(train_epoch_time)
    #plt.plot(val_epoch_time)
    #plt.show()