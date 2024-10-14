import torchvision

torchvision.datasets.CIFAR10('../datasets', train=True, download=True)
torchvision.datasets.CIFAR10('../datasets', train=False, download=True)

torchvision.datasets.CIFAR100('../datasets', train=True, download=True)
torchvision.datasets.CIFAR100('../datasets', train=False, download=True)

torchvision.datasets.SVHN('./datasets', split='train', download=True)
torchvision.datasets.SVHN('./datasets', split='test', download=True)