import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 导入自定义模块
from src.net.softmax_regression import SoftMaxRegression
from src.utils.get_loader import GetLoader
from src.utils.metrics_visualizer import MetricsVisualizer
from src.utils.trainer import Trainer


# class Trainer:
#     def __init__(
#         self,
#         model,
#         train_loader,
#         val_loader,
#         loss_fn,
#         optimizer,
#         scheduler,
#         device,
#         logger,
#     ):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.device = device
#         self.logger = logger

#     def train(self, num_epochs):
#         for epoch in range(num_epochs):
#             self.model.train()
#             total_loss = 0.0
#             progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

#             for images, labels in progress_bar:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 outputs = self.model(images)
#                 loss = self.loss_fn(outputs, labels)

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 total_loss += loss.item()
#                 progress_bar.set_postfix({"Loss": loss.item()})
#                 self.logger.update(outputs, labels)

#             avg_train_loss = total_loss / len(self.train_loader)
#             print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

#             # 计算并记录训练指标
#             self.logger.compute_and_log(prefix="train")

#             # 验证
#             self.validate()
#             self.scheduler.step(avg_train_loss)

#     def validate(self):
#         self.model.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for images, labels in self.val_loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 outputs = self.model(images)
#                 loss = self.loss_fn(outputs, labels)
#                 total_loss += loss.item()
#                 self.logger.update(outputs, labels)

#         avg_val_loss = total_loss / len(self.val_loader)
#         print(f"Validation Loss: {avg_val_loss:.4f}")

#         # 计算并记录验证指标
#         self.logger.compute_and_log(prefix="val")
#         self.logger.log_confusion_matrix(prefix="val")


def train_model(
    model_class,
    input_dim,
    output_dim,
    batch_size=64,
    num_epochs=20,
    log_dir="../logdir",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # 数据加载
    train_loader, val_loader = GetLoader.get_loader_fashionmnist(
        batch_size=batch_size,
        shape_size=28,
        num_workers=4,
        root="../data",
    )

    # 模型初始化
    model = model_class(input_dim, output_dim)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    # 初始化日志记录器
    num_classes = output_dim
    logger = MetricsVisualizer(num_classes, log_dir)

    # 初始化训练器
    trainer = Trainer(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, logger
    )

    # 开始训练
    # trainer.train(num_epochs=num_epochs)
    for epoch in range(num_epochs):
        avg_train_loss = trainer.train_epoch(epoch)
        avg_val_loss = trainer.validate_epoch(epoch)
        print(
            f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # 调整学习率
        trainer.scheduler.step(avg_val_loss)
        logger.log_confusion_matrix(prefix="val")

    # 关闭日志记录器
    logger.close()


if __name__ == "__main__":
    # 参数设置
    input_dim = 28 * 28
    output_dim = 10
    num_epochs = 10  # 减少 epoch 数，以便快速测试

    # 训练模型
    train_model(
        SoftMaxRegression,
        input_dim,
        output_dim,
        batch_size=64,
        num_epochs=num_epochs,
        log_dir="logdir",
    )
