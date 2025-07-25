> [!NOTE]
> This article currently only supports Chinese.

> [!CAUTION]
> 施工中！

<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->
\denotes
;;;;马尔可夫链::某一时刻的状态只与上一时刻的状态相关，即$x_t = f(x_{t - 1})$，不需要其他时刻的状态参与，若干个这样的状态关系组成的链条便形成了马尔科夫链。;;;;
;;;;重参数化技巧::对于概率$p(x|y) = \mathcal{N}(x|ay, b)$，即$x$服从一个均值为$ay$，标准差为$\sqrt{b}$的高斯分布，那么则有$x = ay + \sqrt{b}\\epsilon$，其中$\\epsilon \sim \mathcal{N}(0, 1)$。;;;;
;;;;下方的代码::默认使用GPU加速的pytorch代码，不使用GPU加速会很难跑得动扩散模型。;;;;
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

但是，这样的式子仍然是不可计算的，通过`重参数化技巧`，可以将任意的高斯分布进行展开，因此则有：

$$
x_{t + 1} = \sqrt{1 - \beta_t} x_t + \sqrt{\beta_t}\epsilon_t
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
    #保存图片到本都
    torchvision.io.write_png(
      transform_reverse(x_t).clip(0, 255).to(torch.uint8).cpu(),
      'noising/%d.png' % timestep
    )
```

## 逆向过程
