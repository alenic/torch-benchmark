import argparse
import torch
import timm
import time
import matplotlib.pyplot as plt
import numpy as np
from src import *
from tqdm import tqdm
import random


def seed_all(random_state):
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        tr = pil_aug(args.img_size)
    elif args.loader == "cv2":
        loader = cv2_loader
        tr = alb_aug(args.img_size)

    dataset = ImageDataset(args.root, transform=tr, loader=loader)
    dataset = torch.utils.data.Subset(dataset, range(args.n_iter))


    if not args.eval:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                shuffle=True,
                                                pin_memory=args.pin_memory,
                                                drop_last=True)
        model.train()
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
                train_iter_time.append(time.perf_counter()-time_iter)
            
            time_data = time.perf_counter()
            
            if iter == args.n_iter: break
        
        train_total_time = time.perf_counter() - train_total_time
        print(f"TRAIN ------------- Batch: {args.batch_size}, Num Workers {args.num_workers}, Pin {args.pin_memory}")
        print(f"Data {np.mean(train_data_time)},  std: {np.std(train_data_time)}")
        print(f"iter: {np.mean(train_iter_time)},  std: {np.std(train_iter_time)}")
        print(f"Total {train_total_time}")
    else:
        val_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                shuffle=False,
                                                pin_memory=args.pin_memory,
                                                drop_last=True)
        val_iter_time = []
        val_data_time = []
        model.eval()
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
                val_iter_time.append(time.perf_counter()-time_iter)
            
            time_data = time.perf_counter()

            if iter == args.n_iter: break

        val_total_time = time.perf_counter() - val_total_time
        print(f" VAL ------------- Batch: {args.batch_size}, Num Workers {args.num_workers}, Pin {args.pin_memory}")
        print(f"Data {np.mean(val_iter_time)},  std: {np.std(val_iter_time)}")
        print(f"iter: {np.mean(val_iter_time)},  std: {np.std(val_iter_time)}")
        print(f"Total {val_total_time}")
    
    #plt.plot(train_iter_time)
    #plt.plot(val_iter_time)
    #plt.show()