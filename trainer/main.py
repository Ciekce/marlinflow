from __future__ import annotations

import argparse
import json
import os
import pathlib

from dataloader import BatchLoader
from model import (
    NnBoard768Cuda,
    PerspectiveNet,
    HalfKANet,
    NnHalfKACuda,
    HalfKPNet,
    NnHalfKPCuda,
    SquaredPerspectiveNet,
    DeepPerspectiveNet,
)
from time import time

import torch
from trainlog import TrainLog

print("Imports finished")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

LOG_ITERS = 10_000_000


class WeightClipper:
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(-1.98, 1.98)
            module.weight.data = w


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: BatchLoader,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    wdl: float,
    wdl_drop: float | None,
    scale: float,
    epochs: int,
    save_epochs: int,
    train_id: str,
    train_log: TrainLog | None = None,
    lr_drop: int | None = None,
) -> None:
    clipper = WeightClipper()
    running_loss = torch.zeros((1,), device=DEVICE)
    start_time = time()
    iterations = 0

    loss_since_log = torch.zeros((1,), device=DEVICE)
    iter_since_log = 0

    fens = 0
    epoch = 0

    while epoch < epochs:
        new_epoch, batch = dataloader.read_batch(DEVICE)
        if new_epoch:
            epoch += 1
            print(
                f"epoch {epoch}",
                f"epoch train loss: {running_loss.item() / iterations}",
                f"epoch pos/s: {fens / (time() - start_time)}",
                sep=os.linesep,
            )

            running_loss = torch.zeros((1,), device=DEVICE)
            start_time = time()
            iterations = 0
            fens = 0

            if epoch % save_epochs == 0:
                torch.save(model.state_dict(), f"nn/{train_id}_{epoch}")
                param_map = {
                    name: param.detach().cpu().numpy().tolist()
                    for name, param in model.named_parameters()
                }
                with open(f"nn/{train_id}.json", "w") as json_file:
                    json.dump(param_map, json_file)
            scheduler.step()
            if lr_drop is not None and wdl_drop is not None and epoch == lr_drop:
                wdl = wdl_drop

        optimizer.zero_grad()
        prediction = model(batch)
        expected = torch.sigmoid(batch.cp / scale) * (1 - wdl) + batch.wdl * wdl

        loss = torch.mean((prediction - expected) ** 2)
        loss.backward()
        optimizer.step()
        model.apply(clipper)

        with torch.no_grad():
            running_loss += loss
            loss_since_log += loss
        iterations += 1
        iter_since_log += 1
        fens += batch.size

        if iter_since_log * batch.size > LOG_ITERS:
            loss = loss_since_log.item() / iter_since_log
            print(
                f"At {iterations * batch.size} positions",
                f"Running Loss: {loss}",
                sep=os.linesep,
            )
            if train_log is not None:
                train_log.update(loss)
                train_log.save()
            iter_since_log = 0
            loss_since_log = torch.zeros((1,), device=DEVICE)


def main():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--data-root", type=str, help="Root directory of the data files"
    )
    parser.add_argument("--train-id", type=str, help="ID to save train logs with")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--lr-end", type=float, help="Final learning rate")
    parser.add_argument("--lr-drop", type=int, help="Epoch to drop LR at for step LR")
    parser.add_argument("--epochs", type=int, help="Epochs to train for")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size")
    parser.add_argument("--wdl", type=float, default=0.0, help="WDL weight to be used")
    parser.add_argument("--wdl-drop", type=float, help="WDL weight to use after drop")
    parser.add_argument("--scale", type=float, help="WDL weight to be used")
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=100,
        help="How often the program will save the network",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved network to resume training",
    )
    args = parser.parse_args()

    assert args.train_id is not None
    assert args.scale is not None

    train_log = TrainLog(args.train_id)

    model = SquaredPerspectiveNet(768).to(DEVICE)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    data_path = pathlib.Path(args.data_root)
    paths = list(map(str, data_path.glob("*.bin")))
    dataloader = BatchLoader(paths, model.input_feature_set(), args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler: torch.optim.lr_scheduler._LRScheduler
    if args.lr_end is not None:
        # starting LR is args.lr, ending LR is args.lr_end
        # there are args.epochs epochs
        # so the LR should drop by a factor of (args.lr_end / args.lr) ** (1 / args.epochs) each epoch
        gamma = (args.lr_end / args.lr) ** (1 / args.epochs)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif args.lr_drop is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.1)
    else:
        print("No learning rate schedule specified, using constant LR")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

    train(
        model,
        optimizer,
        dataloader,
        scheduler,
        args.wdl,
        args.wdl_drop,
        args.scale,
        args.epochs,
        args.save_epochs,
        args.train_id,
        train_log=train_log,
        lr_drop=args.lr_drop,
    )


if __name__ == "__main__":
    main()
