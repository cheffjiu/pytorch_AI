# 🌟 PyTorch_AI

🚀 **PyTorch_AI** 是一个基于 **PyTorch** 的深度学习小项目，包含完整的网络定义、数据处理、训练脚本等。适用于学习和实验深度学习模型，支持 **TensorBoard** 进行可视化分析。

---

## 📂 项目结构

```bash
project/
├── src/            # 代码文件
│   ├── net/        # 定义网络模型
│   ├── utils/      # 数据集工具，可视化工具，训练工具
│   └── train/      # 训练脚本
├── data/           # 数据集存放位置
├── logdir/         # TensorBoard 日志文件存放位置
```
已经实现的网络模块：
- SoftmaxRegression
- MLP
- LeNet
- AlexNet
- VGG
---
## 🛠 环境依赖
请确保你的环境安装了以下依赖：

- 🐍 Python 3.x
- 🔥 PyTorch
- 🔢 NumPy
- 📊 Matplotlib
- 📟 TensorBoard

使用以下命令安装必要的库：

```bash
pip install torch torchvision numpy matplotlib tensorboard
```

---

## 🚀 使用方法

### 1️⃣ 准备数据集
📂 本项目的数据集：图像数据集[Fashion-minist]

📂将数据集放入 `data/` 目录下，或者在 `src/utils/` 目录中编写数据集处理工具。

### 2️⃣ 训练模型
💡 执行训练脚本：

```bash
python src/train/train.py
```

### 3️⃣ 监控训练过程
📊 使用 **TensorBoard** 可视化训练过程：

```bash
tensorboard --logdir=logdir/
```

🔗 然后在浏览器中打开 `http://localhost:6006` 查看训练结果。

---

## 🤝 贡献指南
📢 欢迎贡献代码和提出改进建议！

1. **Fork** 本仓库
2. 创建新分支 (`git checkout -b feature-branch`)
3. 提交修改 (`git commit -m '✨ Add new feature'`)
4. 推送分支 (`git push origin feature-branch`)
5. 提交 **Pull Request**

---

## 📜 许可证
📄 本项目基于 **MIT** 许可证发布，详情请见 `LICENSE` 文件。

---

💡 **欢迎 Star ⭐ 和 Fork 🍴 本仓库，一起探索深度学习的世界！**

