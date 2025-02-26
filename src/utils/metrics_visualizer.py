import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)
from torch.utils.tensorboard import SummaryWriter


class MetricsVisualizer:
    def __init__(self, num_classes, log_dir="../logdir"):
        """
        初始化指标可视化工具
        :param num_classes: 分类任务的类别数
        :param log_dir: TensorBoard 日志保存路径
        """
        self.num_classes = num_classes
        self.writer = SummaryWriter(log_dir=log_dir)

        # 定义指标集合
        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "precision": Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall": Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "f1_score": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            }
        )

        # 定义混淆矩阵
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes, normalize="true"
        )

        self.global_step = 0

    def update(self, predictions, targets):
        """
        更新指标
        :param predictions: 模型预测的 logits 或概率分布 (Tensor)
        :param targets: 真实标签 (Tensor)
        """
        self.metrics.update(predictions, targets)
        self.confusion_matrix.update(predictions, targets)

    def compute_and_log(self, prefix="train"):
        """
        计算指标并记录到 TensorBoard
        :param prefix: 指标名称前缀，例如 "train" 或 "val"
        """
        # 计算标量指标
        scalar_results = self.metrics.compute()
        for metric_name, value in scalar_results.items():
            self.writer.add_scalar(
                f"{prefix}/{metric_name}", value.item(), self.global_step
            )

        # 重置标量指标
        self.metrics.reset()
        # 增加全局步数
        self.global_step += 1
        return scalar_results

    def log_confusion_matrix(self, prefix="val"):
        # 计算混淆矩阵
        cm = self.confusion_matrix.compute().cpu().numpy()
        self.confusion_matrix.reset()

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues", square=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix ({prefix})")
        self.writer.add_figure(
            f"{prefix}/confusion_matrix", plt.gcf(), self.global_step
        )
        plt.close()

    def log_computational_graph(self, model, input_tensor):
        """
        记录模型计算图
        :param model: 要可视化的模型
        :param input_tensor: 示例输入张量
        """
        self.writer.add_graph(model, input_tensor)

    def close(self):
        """关闭 TensorBoard writer"""
        self.writer.close()
