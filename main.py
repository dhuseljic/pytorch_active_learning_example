import os
import json
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm.auto import tqdm


from active_learning.dataset import ALDataset


def main(args):
    results = al_experiment(args)
    # Post process results
    os.makedirs(args.result_path, exist_ok=True)
    with open(os.path.join(args.result_path, 'results.json'), 'w') as f:
        json.dump(results, f)


def al_experiment(args):
    results = []

    train_ds, val_ds = get_dataset(args)
    model = get_model(args)
    val_loader = DataLoader(val_ds, batch_size=256)

    al_dataset = ALDataset(train_ds)

    print('Starting AL iteration {}'.format(0))
    al_dataset.random_init(n_samples=args.n_init)
    labeled_loader = DataLoader(al_dataset.labeled_ds, batch_size=4, shuffle=True)
    train_net(args, model, labeled_loader)
    res = eval_net(args, model, val_loader)
    results.append(res)

    for i_acq in range(1, args.n_acq + 1):
        print('Starting AL iteration {}'.format(i_acq))
        # buy annotations
        unlabeled_loader = DataLoader(al_dataset.unlabeled_ds, batch_size=256)
        indices = buy_annotations(args, model, unlabeled_loader)
        al_dataset.update_annotations(indices)

        # Train with new annotations
        labeled_loader = DataLoader(al_dataset.labeled_ds, batch_size=4, shuffle=True)
        train_net(args, model, labeled_loader)
        res = eval_net(args, model, val_loader)
        results.append(res)
    return results


@torch.no_grad()
def buy_annotations(args, model, unlabeled_loader):
    print('> Buying annotations.')
    # simple uncertainty sampling
    probas = []
    for X_batch, _ in tqdm(unlabeled_loader):
        proba = model(X_batch).softmax(-1)
        probas.append(proba)
    probas = torch.cat(probas)
    entropies = - torch.sum(probas * probas.log(), dim=-1)
    values, indices = entropies.topk(args.acq_size)
    return indices.tolist()


def train_net(args, model, dataloader):
    print('> Training.')
    optimizer = torch.optim.Adam(model.parameters())

    for i_epoch in tqdm(range(args.n_epochs)):
        for X_batch, y_batch in dataloader:
            out = model(X_batch)
            loss = F.cross_entropy(out, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(loss)


@torch.no_grad()
def eval_net(args, model, dataloader):
    print('> Evaluation.')
    running_loss = 0
    running_corrects = 0
    n_samples = 0

    for X_batch, y_batch in tqdm(dataloader):
        out = model(X_batch)

        batch_size = X_batch.size(0)
        running_loss += batch_size * F.cross_entropy(out, y_batch)
        running_corrects += torch.sum(out.argmax(-1) == y_batch)
        n_samples += batch_size
    loss = (running_loss / n_samples).item()
    accuracy = (running_corrects / n_samples).item()

    result_dict = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return result_dict



def get_dataset(args):
    # Preprocess images here
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # gray to rgb
    ])
    train_ds = torchvision.datasets.FashionMNIST(args.ds_path, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(args.ds_path, train=False, download=True, transform=transform)
    return train_ds, test_ds


def get_model(args):
    # Load model, e.g., pretrained resnet
    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, 10) # Change last layer to fit number of classes

    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
        nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512, 10)
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_init', type=int, default=10)
    parser.add_argument('--n_acq', type=int, default=5)
    parser.add_argument('--acq_size', type=int, default=20)
    parser.add_argument('--result_path', type=str, default='/tmp/results/')
    parser.add_argument('--ds_path', type=str, default='/tmp/datasets/')
    args = parser.parse_args()
    main(args)
