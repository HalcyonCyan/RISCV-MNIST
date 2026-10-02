# RISC-V MNIST

用 **RISC-V RV32IM 汇编**执行神经网络推理，配一个轻量网页手写画板。已包含预训练模型，启动即可识别数字，无需先训练。

## 直接启动

准备 **Python 3.10+、Java 8+**，将 Java 加入 PATH 或设置 `JAVA_HOME`。

**Windows：首次双击 `setup.bat`，之后双击 `start.bat`。** 首次安装需要联网，会创建虚拟环境、安装 Pillow，并下载校验过的 RARS 1.6。

macOS / Linux 在项目目录执行：

```bash
python3 scripts/setup.py
.venv/bin/python app.py
```

浏览器自动打开 `http://127.0.0.1:8000`。鼠标或触屏书写，Enter 识别、Esc 清空，也可载入 MNIST 样本。终端按 Ctrl+C 停止服务。

## 核心代码框架

```text
网页画板 → Python 图像预处理 → 常驻 RARS → RV32IM 两层网络 → 网页显示结果
```

| 代码 | 实现内容 |
|---|---|
| `web/` | HTML / CSS / 原生 JavaScript，实现画板、输入预览和十类分布，无需 npm |
| `app.py`、`riscv_mnist/server.py` | 启动本地 HTTP 服务，提供样本与识别接口 |
| `riscv_mnist/preprocess.py` | 去除背景、裁剪笔迹、缩放到 20×20，再放入 28×28 并按质心居中 |
| `riscv_mnist/runtime.py` | 管理常驻 Java/RARS，通过管道交换输入和分数，处理超时与退出 |
| `asm/mnist.s` | 实现两层全连接、ReLU、右移截断和 argmax，完成真正的推理 |
| `training/` | NumPy 实现反向传播、Adam 训练与 int8 量化，直接导出汇编模型 |
| `models/` | 预训练整数模型及参数说明 |

网络为 **784 → 64 → 10**，权重使用 int8，像素与隐藏激活使用 uint8，偏置与累加使用 int32：

```text
h = clip((W1ᵀ × x + B1) >> shift, 0, 255)
scores = W2 × h + B2
digit = argmax(scores)
```

Python 负责图像处理与通信，神经网络前向计算由汇编执行。网页将十个整数分数换算为 softmax 相对分布。

## 如何提高推理效率

1. **跳过零像素**：一次扫描收集非零像素，省去它们之外的乘加。MNIST 测试图约 80.72% 像素为零，平均乘加次数从 50,816 降至约 10,312。
2. **调整权重布局，使用 8 个累加器**：W1 从按神经元排列改为按输入排列。同一像素对应的权重连续读取，用 8 个寄存器累加 8 个神经元，减少输入加载与循环开销。
3. **展开第二层循环**：每轮执行 4 个乘加，减少指针更新、计数与条件分支。
4. **复用推理进程**：第一版每次识别都启动 Java、汇编并加载模型；当前版本只准备一次，后续通过管道发送图像，避免重复启动和临时输入文件。

这些改进保留参数与整数计算规则，只跳过精确为零的输入；模型仍为 **51,144 字节**，优化前后的输出分数一致。

## 改进效果

下表来自原工程同一内核在本机 **Windows / RARS 1.6 模拟器**的三轮重复测试中位数，按表中所列的冷启动或常驻方式计时，未包含网页预处理与渲染。

| 项目 | 第一版 | 优化版 |
|---|---:|---:|
| 相同 32 张图文件批处理 | 11.38 秒 | **0.88 秒，约快 12.95 倍** |
| 同一张图：新进程请求 / 准备后常驻请求 | 483.95 毫秒 | **24.91 毫秒，包含进程复用收益** |

完整 10,000 张 MNIST 测试图的整数参考准确率保持 **97.36%**。原工程另验证了 100 张实际汇编输入的全部 1,000 个分数；此精简项目可运行 `python -m unittest discover -s tests -v` 检查预处理、接口及随附样本的汇编输出。命令须使用已安装依赖的虚拟环境。

数据来源：[MNIST](https://yann.lecun.org/exdb/mnist/)；模拟器：[RARS 1.6](https://github.com/TheThirdOne/rars/releases/tag/v1.6)，许可证见 [tools/RARS-LICENSE.txt](tools/RARS-LICENSE.txt)。
