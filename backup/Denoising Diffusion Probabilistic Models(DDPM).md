> [!NOTE]
> This article currently only supports Chinese.

> [!CAUTION]
> 施工中！

<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->
\denotes
;;;;马尔可夫链::某一时刻的状态只与上衣时刻的状态相关，即$x_t = f(x_{t - 1})$，不需要其他时刻参与，若干个这样的状态组成的链条便形成了马尔科夫链。;;;;
\denotes
## 介绍

## 正向过程
扩散模型的正向过程为，对原始的图像进行一步步地噪声添加，从而在最终阶段，使得图片完全服从噪声分布，一般地，使用高斯噪声来进行这个过程。对于离散的时间$t = 0, 1, \dots, T$，在扩散模型遵循的`马尔可夫链`过程中，对于给定的$x_0 \sim q(x_0)$，即原始图像作为初始阶段，和$x_T \sim q(x_T) = \mathcal{N}(0, I)$，即完全服从标准高斯噪声的图像作为最终阶段。其各个阶段$x_0, x_1, \dots, x_T$的概率分布服从以下公式：

$$
q(x_T|x_0) = q(x_0)\prod_{t = 0}^{T - 1}q(x_{t + 1}|x_{t})
\tag{1}
$$

其中$q(x_{t + 1}|x_{t})$代表着每一时刻，为上一时刻的图片添加噪声的过程。很显然，这个过程是需要人为定义的，因此，可以选择一组超参数$\beta_t$，从而有：

$$
q(x_{t + 1}|x_{t}) = \mathcal{N}(x_{t + 1};\sqrt{1 - \beta_t}x_t,\beta_tI)
\tag{2}
$$

$$
x_{t + 1} = \sqrt{\alpha_t} x_t + \sqrt{1 - \alpha_t}\epsilon_t
\tag{2}
$$

其中，$\alpha_t$是一系列人为定义的超参数，$\epsilon_t$表示任意的标准高斯噪声。

## 逆向过程
