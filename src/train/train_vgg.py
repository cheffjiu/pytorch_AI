import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 导入自定义模块
from src.net.vgg import VGG
from src.utils.get_loader import GetLoader
from src.utils.metrics_visualizer import MetricsVisualizer
from src.utils.trainer import Trainer


def train_model(
    model_class,
    output_dim,
    batch_size=64,
    num_epochs=20,
    log_dir="../../logdir/mlp",
    device=(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built() else "cpu"
    ),
) -> None:
    # 数据加载
    train_loader, val_loader = GetLoader.get_loader_fashionmnist(
        batch_size=batch_size,
        shape_size=227,
        num_workers=4,
        root="../data",
    )

    # 模型初始化
    model = model_class(output_dim)
    model.to(device)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=5)

    # 创建计算图示例输入
    example_input = torch.rand((batch_size, 1, 227, 227)).to(device)
    # 初始化日志记录器
    num_classes = output_dim
    logger = MetricsVisualizer(num_classes, log_dir, device)
    # 记录计算图
    logger.log_computational_graph(model, example_input)

    # 初始化训练器
    trainer = Trainer(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, logger
    )
    print(f"Using device: {device}")
    # 开始训练
    for epoch in range(num_epochs):
        avg_train_loss, train_metrics = trainer.train_epoch(epoch)
        avg_val_loss, val_metrics = trainer.validate_epoch(epoch)
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(
            f"Train Loss: {avg_train_loss:.4f} | "
            + f"Acc: {train_metrics['accuracy']:.4f} | "
            + f"Precision: {train_metrics['precision']:.4f} | "
            + f"Recall: {train_metrics['recall']:.4f} | "
            + f"F1: {train_metrics['f1_score']:.4f}"
        )
        print(
            f"Val Loss: {avg_val_loss:.4f} | "
            + f"Acc: {val_metrics['accuracy']:.4f} | "
            + f"Precision: {val_metrics['precision']:.4f} | "
            + f"Recall: {val_metrics['recall']:.4f} | "
            + f"F1: {val_metrics['f1_score']:.4f}"
        )

        # 调整学习率
        trainer.scheduler.step(avg_val_loss)
        logger.log_confusion_matrix(prefix="val")

    # 关闭日志记录器
    logger.close()


if __name__ == "__main__":
    # 参数设置

    output_dim = 10
    num_epochs = 20  # 减少 epoch 数，以便快速测试

    # 训练模型
    train_model(
        VGG,
        output_dim,
        batch_size=64,
        num_epochs=num_epochs,
        log_dir="logdir/vgg",
    )
