> [!NOTE]
> This article currently only supports Chinese.

> [!CAUTION]
> 施工中！

<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->

## 介绍

## 正向过程
对于离散的时间$t = 1, 2, \dots, T$，若使用$x_0$表示原始数据集，$x_T$表示完全的标准高斯噪声，那么扩散模型的正向过程可以用下式进行表达：
$$
x_{t + 1} = \sqrt{\alpha_t} x_t + \sqrt{1 - \alpha_t}\epsilon_t \tag{1}
$$
其中，$\alpha_t$是一系列人为定义的超参数，$\epsilon_t$表示任意的标准高斯噪声。

## 逆向过程
