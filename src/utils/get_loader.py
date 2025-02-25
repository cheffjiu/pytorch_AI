import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class GetLoader:
    def get_loader_fashionmnist(
        batch_size=64, shape_size=28, num_workers=4, root="data"
    ):
        """
            获取Fashion-MNIST数据集的DataLoader对象。

        Args:
            batch_size (int, optional): 每个batch的大小,默认为64。
            shape_size (int, optional): 调整图像大小后的形状,默认为28。
            num_workers (int, optional): 加载数据时使用的进程数,默认为4。
            root (str, optional): 数据集的下载目录，默认为"../data"。

        Returns:
            tuple: 包含训练集和测试集的DataLoader对象。
        """

        transforms = v2.Compose(
            [
                v2.Resize((shape_size, shape_size)),
                v2.ToTensor(),
                v2.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = torchvision.datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transforms
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transforms
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, test_loader
