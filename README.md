# neural_network

使用 NumPy 实现的两层全连接神经网络，作为学习前向传播和反向传播的起点。
模型包含一个 tanh 隐藏层和一个 sigmoid 输出层，使用二元交叉熵损失与全批量梯度下降，支持二分类。

## 安装与运行

需要 Python 3.10 或更高版本。在仓库根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m neural_network
```

示例使用固定随机种子训练 XOR 数据，打印训练前后的损失及四组输入的预测。
Windows 用户可用 `.venv\Scripts\activate` 激活虚拟环境。

## 使用模型

```python
import numpy as np

from neural_network import TwoLayerNetwork

features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
targets = np.array([0, 1, 1, 0])

model = TwoLayerNetwork(input_size=2, hidden_size=8, seed=42)
losses = model.fit(features, targets, epochs=5000, learning_rate=0.1)

print(model.predict(features))  # [0 1 1 0]
print(model.predict_proba(features))  # 每个样本属于类别 1 的概率
print(model.loss(features, targets))  # 当前模型的平均二元交叉熵
```

输入形状为 `(样本数, 特征数)`，所有输入值必须有限；标签为 0 或 1，
形状为 `(样本数,)` 或 `(样本数, 1)`。预测结果与概率均为一维数组。
`fit` 返回每轮参数更新前的损失；重复调用会继续训练当前模型。

## 开发与验证

```sh
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

测试覆盖数值梯度、XOR 收敛、随机种子复现、极端输出的数值稳定性和参数校验。
GitHub Actions 会在推送及拉取请求时运行测试与代码检查。

代码位于 `src/neural_network/`，测试位于 `tests/`。项目采用 [MIT 许可证](LICENSE)。
