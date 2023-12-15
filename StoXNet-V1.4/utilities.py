import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_loaders(dataset, batch_size, workers):
    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)
    elif dataset == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data_100', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data_100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)

    elif dataset == 'tiny_imagenet':
        exit()
        tiny_imagenet_normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='./tiny-imagenet-200/train', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                tiny_imagenet_normalize,
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='./tiny-imagenet-200/val', transform=transforms.Compose([
                transforms.ToTensor(),
                tiny_imagenet_normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)
    else:
        raise ValueError("Inncorect dataset selection, check paramaeters")

    return train_loader, val_loader
