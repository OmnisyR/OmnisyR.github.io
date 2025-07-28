> [!NOTE]
> This article currently only supports Chinese.

> [!CAUTION]
> 施工中！

<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->
\denotes
;;;;马尔可夫链::某一时刻的状态只与上一时刻的状态相关，即$x_t = f(x_{t - 1})$，不需要其他时刻的状态参与，若干个这样的状态关系组成的链条便形成了马尔科夫链。因此，马尔科夫链存在着这样的特殊性质：
$$
\begin{align}
&P(X_n|X_0) = P(X_n)
\\
&P(X_n|X_{n - 1}, X_{n - 2}, \dots, X_0) = P(X_n|X_{n - 1})
\end{align}
$$
;;;;
;;;;重参数化技巧::对于概率$p(x|y) = \mathcal{N}(x|ay, b)$，即$x$服从一个均值为$ay$，标准差为$\sqrt{b}$的高斯分布，那么则有$x = ay + \sqrt{b}\epsilon$，其中$\epsilon \sim \mathcal{N}(0, 1)$。;;;;
;;;;下方的代码::默认使用GPU加速的pytorch代码，不使用GPU加速会很难跑得动扩散模型。;;;;
;;;;高斯分布的加法法则::$\mathcal{N}(\mu_1, \sigma^2_1) + \mathcal{N}(\mu_2, \sigma^2_2) = \mathcal{N}(\mu_1 + \mu_2, \sigma^2_1 + \sigma^2_2)$。;;;;
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
q(x_t|x_{t - 1}) = \mathcal{N}(x_t;\sqrt{1 - \beta_t}x_{t - 1},\beta_tI)
\tag{2}
$$

但是，这样的式子仍然是不可计算的，通过`重参数化技巧`，可以将任意的高斯分布进行展开，因此则有：

$$
x_t = \sqrt{1 - \beta_t} x_{t - 1} + \sqrt{\beta_t}\epsilon_t
\tag{3}
$$

这样一来，就使用了一个可计算的形式表达出了扩散模型的马尔科夫链正向过程。DDPM中定义$\beta_t$是一个从$\beta_1 = 0.001$到$\beta_T = 0.02$的线性数组，这种噪声添加方式又称线性噪声时间表。通过`下方的代码`便可以实现这一过程：
```python
import torch
import torchvision.io
from PIL import Image
from torchvision.transforms.v2 import Compose, ToTensor, Lambda
from tqdm import tqdm
#时间长度设置为较为常用的1000
timesteps = 1000
#将图片的色彩值转化为-1到1之间的张量
transform = Compose([ToTensor(), Lambda(lambda t: (t * 2) - 1)])
#将色彩值为-1到1之间的图片转化为0到255之间
transform_reverse = Compose([Lambda(lambda t: (t + 1) / 2 * 255)])
#读取并将图片转化为张量
x_t = transform(Image.open('xxx.png')).cuda()
#使用线性噪声时间表
beta = torch.linspace(0.001, 0.02, timesteps).cuda()
for timestep in tqdm(range(timesteps)):
    #每一步的随机噪声
    ep = torch.randn(size=x_t.shape).cuda()
    #式(3)的迭代增噪过程
    x_t = (1 - beta[timestep]).sqrt().item() * x_t + beta[timestep].sqrt() * ep
    #保存图片到本地
    torchvision.io.write_png(
      transform_reverse(x_t).clip(0, 255).to(torch.uint8).cpu(),
      'noising/%d.png' % timestep
    )
```

为了式子的简洁性，记$\alpha_t + \beta_t = 1$，则有：

$$
\begin{align}
x_t &= \sqrt{\alpha_t} x_{t - 1} + \sqrt{1 - \alpha_t}\epsilon_t
\tag{4}
\\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{{t - 1}}} x_{t - 2} + \sqrt{1 - \alpha_{t - 1}}\epsilon_{t - 1}) + \sqrt{1 - \alpha_t}\epsilon_t
\tag{5}
\\
&= \sqrt{\alpha_t\alpha_{t - 1}}x_{t - 2} + \sqrt{\alpha_t(1 - \alpha_{t - 1})}\epsilon_{t - 1} + \sqrt{1 - \alpha_t}\epsilon_t
\tag{6}
\\
&= \sqrt{\alpha_t\alpha_{t - 1}}x_{t - 2} + \mathcal{N}(0, \alpha_t(1 - \alpha_{t - 1})) + \mathcal{N}(0, 1 - \alpha_t)
\tag{7}
\\
&= \sqrt{\alpha_t\alpha_{t - 1}}x_{t - 2} + \mathcal{N}(0, \alpha_t(1 - \alpha_{t - 1}) + 1 - \alpha_t)
\tag{8}
\\
&= \sqrt{\alpha_t\alpha_{t - 1}}x_{t - 2} + \sqrt{1 - \alpha_t\alpha_{t - 1}}\bar{\epsilon}_{t, t - 1}
\tag{9}
\\
&= \dots
\\
&= \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon
\tag{10}
\end{align}
$$

其中$\bar{\alpha}\_t = \prod_{i = 1}^{t}\alpha_i$。由于式(6)中存在着两个完全随机的标准高斯分布，因此可以对这二者进行重参数化技巧的逆运算得到式(7)，再通过`高斯分布的加法法则`得到式(8)，从而对多步的迭代进行化简，能够仅仅使用一步求出正向过程的任意时刻的状态：
```python
def add_noise(x_0, t):
    #式(10)的一步增噪过程
    return sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * torch.randn(size=x_0.shape).cuda()

#对参数的预先缓存
alpha = 1 - beta
#在这里，对求得的α累乘的最前位置，额外加了一个1，这是为了编写程序时，时间戳与索引的一致
alpha_bar = torch.cat((torch.Tensor((1,)).cuda(), alpha.cumprod(dim=0)))
sqrt_alpha_bar = alpha_bar.sqrt()
sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()

timestep = 200
x_t = add_noise(x_0, timestep)
torchvision.io.write_png(transform_reverse(x_t).clip(0, 255).to(torch.uint8).cpu(), '%s.png' % timestep)
```

## 逆向过程
