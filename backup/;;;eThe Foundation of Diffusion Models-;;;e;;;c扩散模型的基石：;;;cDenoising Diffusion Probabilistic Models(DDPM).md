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
;;;;贝叶斯定理::$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$。;;;;
;;;;高斯分布的KL散度公式::对于$p_1(x) = N(\mu_1, \sigma^2_1)$以及$p_2(x) = N(\mu_2, \sigma^2_2)$，有：

$$
D_{KL}(p_1||p_2) = \frac{1}{2}\log\frac{\sigma^2_2}{\sigma^2_1} + \frac{\sigma^2_1 + (\mu_1 - \mu_2)^2}{2\sigma^2_2} - \frac{1}{2}
$$

;;;;
\denotes
## 介绍

## 正向过程
扩散模型的正向过程为，对原始的图像进行一步步地噪声添加，从而在最终阶段，使得图片完全服从噪声分布，一般地，使用高斯噪声来进行这个过程。对于离散的时间$t = 0, 1, \dots, T$，在扩散模型遵循的`马尔可夫链`过程中，对于给定的$x_0 \sim q(x_0)$，即原始图像作为初始阶段，和$x_T \sim q(x_T) = \mathcal{N}(0, I)$，即完全服从标准高斯噪声的图像作为最终阶段。其各个阶段$x_0, x_1, \dots, x_T$的概率分布服从以下公式：

$$
q(x_T|x_0) = q(x_0)\prod^T_{t = 1}q(x_t|x_{t - 1})
\tag{1}
$$

其中$q(x_{t + 1}|x_{t})$代表着每一时刻，为上一时刻的图片添加噪声的过程。很显然，这个过程是需要人为定义的，因此，可以选择一组超参数$\beta_t$，从而有：

$$
q(x_t|x_{t - 1}) = \mathcal{N}(x_t;\sqrt{1 - \beta_t}x_{t - 1},\beta_tI)
\tag{2}
$$

但是，这样的式子仍然是不可计算的，通过`重参数化技巧`，可以将任意的高斯分布进行展开，因此则有：

$$
x_t = \sqrt{1 - \beta_t} x_{t - 1} + \sqrt{\beta_t}\epsilon
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
x_t &= \sqrt{\alpha_t} x_{t - 1} + \sqrt{1 - \alpha_t}\tilde{\epsilon}\_t
\tag{4}
\\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{{t - 1}}} x_{t - 2} + \sqrt{1 - \alpha_{t - 1}}\tilde{\epsilon}\_{t - 1}) + \sqrt{1 - \alpha_t}\tilde{\epsilon}\_t
\tag{5}
\\
&= \sqrt{\alpha_t\alpha_{t - 1}}x_{t - 2} + \sqrt{\alpha_t(1 - \alpha_{t - 1})}\tilde{\epsilon}\_{t - 1} + \sqrt{1 - \alpha_t}\tilde{\epsilon}\_t
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
&= \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t
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
相较正向过程，逆向过程则会复杂很多。逆向过程的目标是对于给定的$x_T \sim p_\theta(x_T) = \mathcal{N}(0, I)$，需要一个概率模型，能够使得$x_T$由标准高斯分布，转变为原始数据集的似然估计，即：

$$
p_\theta(x_0|x_T) = p_\theta(x_T)\prod^T_{t = 1}p_\theta(x_{t - 1}|x_t)
\tag{11}
$$

若对其进行极大似然估计，其对数损失为：

$$
L_\theta = -\log p_\theta(x_0) = -\log (p_\theta(x_T)\prod^T_{t = 1}p_\theta(x_{t - 1}|x_t))
\tag{12}
$$

其中，$p_\theta(x_t|x_{t + 1})$是完全未知的，无法求解，因此考虑令该损失的上界尽可能的小。由于KL散度必然是非负的，所以：

$$
\begin{align}
L_\theta \leq L_{VLB} &= -\log p_\theta(x_0) + D_{KL}(q(x_{1:T}|x_0)||p_\theta(x_{1:T}|x_0))
\tag{13}
\\
&= -\log p_\theta(x_0) + E_{q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{1:T}|x_0)}
\tag{14}
\end{align}
$$

根据`贝叶斯定理`：

$$
p_\theta(x_{1:T}|x_0) = \frac{p_\theta(x_0|x_{1:T})p_\theta(x_{1:T})}{p_\theta(x_0)} = \frac{p_\theta(x_0, x_{1:T})}{p_\theta(x_0)} = \frac{p_\theta(x_{0:T})}{p_\theta(x_0)}
\tag{15}
$$

因此：

$$
\begin{align}
L_{VLB} &= -\log p_\theta(x_0) + E_{q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)p_\theta(x_0)}{p_\theta(x_{0:T})}
\tag{16}
\\
&= -\log p_\theta(x_0) + E_{q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} + E_{q(x_{1:T}|x_0)} \log p_\theta(x_0)
\tag{17}
\\
&= -\log p_\theta(x_0) + E_{q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} + \log p_\theta(x_0)
\tag{18}
\\
&= E_{q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}
\tag{19}
\\
&= E_{q(x_{1:T}|x_0)}\log\frac{\prod^T_{t = 1}q(x_t|x_{t - 1})}{p_\theta(x_T)\prod^T_{t = 1}p_\theta(x_{t - 1}|x_t)}
\tag{20}
\\
&= E_{q(x_{1:T}|x_0)}[-\log p_\theta(x_T) + \sum^T_{t = 1} \log \frac{q(x_t|x_{t - 1})}{p_\theta(x_{t - 1}|x_t)}]
\tag{21}
\\
&= E_{q(x_{1:T}|x_0)}[-\log p_\theta(x_T) + \log \frac{q(x_1|x_0)}{p_\theta(x_{0}|x_1)} + \sum^T_{t = 2} \log \frac{q(x_t|x_{t - 1})}{p_\theta(x_{t - 1}|x_t)}]
\tag{22}
\\
&= E_{q(x_{1:T}|x_0)}[-\log p_\theta(x_T) + \log \frac{q(x_1|x_0)}{p_\theta(x_{0}|x_1)} + \sum^T_{t = 2} \log \frac{q(x_{t - 1}|x_t, x_0)q(x_t|x_0)}{p_\theta(x_{t - 1}|x_t)q(x_{t - 1}|x_0)}]
\tag{23}
\\
&= E_{q(x_{1:T}|x_0)}[-\log p_\theta(x_T) + \log \frac{q(x_1|x_0)}{p_\theta(x_{0}|x_1)} + \sum^T_{t = 2} \log \frac{q(x_{t - 1}|x_t, x_0)}{p_\theta(x_{t - 1}|x_t)}
\\
&\quad +  \sum^T_{t = 2} \log \frac{q(x_t|x_0)}{q(x_{t - 1}|x_0)}]
\tag{24}
\\
&= E_{q(x_{1:T}|x_0)}[-\log p_\theta(x_T) + \log \frac{q(x_1|x_0)}{p_\theta(x_{0}|x_1)} + \sum^T_{t = 2} \log \frac{q(x_{t - 1}|x_t, x_0)}{p_\theta(x_{t - 1}|x_t)}
\\
&\quad + \log \frac{q(x_T|x_0)q(x_{T - 1}|x_0)\dots q(x_2|x_0)}{q(x_{T - 1}|x_0)q(x_{T - 2}|x_0)\dots q(x_1|x_0)}]
\tag{25}
\\
&= E_{q(x_{1:T}|x_0)}[-\log p_\theta(x_T) + \log \frac{q(x_1|x_0)}{p_\theta(x_{0}|x_1)} + \sum^T_{t = 2} \log \frac{q(x_{t - 1}|x_t, x_0)}{p_\theta(x_{t - 1}|x_t)}
\\
&\quad + \log \frac{q(x_T|x_0)}{q(x_1|x_0)}]
\tag{26}
\\
&= E_{q(x_{1:T}|x_0)}[-\log p_\theta(x_T) + \log \frac{q(x_T|x_0)}{p_\theta(x_{0}|x_1)} + \sum^T_{t = 2} \log \frac{q(x_{t - 1}|x_t, x_0)}{p_\theta(x_{t - 1}|x_t)}]
\tag{27}
\\
&= E_{q(x_{1:T}|x_0)}[\log \frac{q(x_T|x_0)}{p_\theta(x_T)} - \log p_\theta(x_{0}|x_1) + \sum^T_{t = 2} \log \frac{q(x_{t - 1}|x_t, x_0)}{p_\theta(x_{t - 1}|x_t)}]
\tag{28}
\\
&= E_{q(x_T|x_0)}\log \frac{q(x_T|x_0)}{p_\theta(x_T)} - E_{q(x_1|x_0)}\log p_\theta(x_{0}|x_1) + \sum^T_{t = 2} E_{q(x_t, x_{t - 1}|x_0)}\log \frac{q(x_{t - 1}|x_t, x_0)}{p_\theta(x_{t - 1}|x_t)}
\tag{29}
\end{align}
$$

又：

$$
\begin{align}
&q(x_{t - 1}|x_t, x_0) = \frac{q(x_{t - 1}, x_t, x_0)}{q(x_t, x_0)} = \frac{q(x_{t - 1}, x_t| x_0)}{q(x_t, x_0)}
\tag{30}
\\
\Leftrightarrow &q(x_{t - 1}, x_t| x_0) = q(x_t, x_0)q(x_{t - 1}|x_t, x_0)
\tag{31}
\end{align}
$$

从而：

$$
\begin{align}
L_{VLB} &= E_{q(x_T|x_0)}\log \frac{q(x_T|x_0)}{p_\theta(x_T)} - E_{q(x_1|x_0)}\log p_\theta(x_{0}|x_1)
\\
&\quad + \sum^T_{t = 2} E_{q(x_t, x_0)}E_{q(x_{t - 1}|x_t, x_0)}\log \frac{q(x_{t - 1}|x_t, x_0)}{p_\theta(x_{t - 1}|x_t)}
\tag{32}
\\
&= \underbrace{D_{KL}(q(x_T|x_0)||p_\theta(x_T))}\_{L_T}
\\
&\quad \underbrace{- E_{q(x_1|x_0)}\log p_\theta(x_{0}|x_1)}\_{L_0}
\\
&\quad + \underbrace{\sum^T_{t = 2} E_{q(x_t, x_0)}D_{KL}(q(x_{t - 1}|x_t, x_0)||p_\theta(x_{t - 1}|x_t))}\_{L_{t - 1}}
\tag{33}
\end{align}
$$

其中，对于$L_T$，由于$q(x_T) = \mathcal{N}(0, I)$，$p_\theta(x_T) = \mathcal{N}(0, I)$都和极大似然估计参数无关，视为常数；

对于$L_0$，由于$1 = q(x_0|x_0) = \frac{q(x_1|x_0)q(x_0|x_0)}{q(x_1|x_0)} = q(x_0|x_1)$，因此有：

$$
\begin{align}
L_0 &= - E_{q(x_1|x_0)}\log p_\theta(x_{0}|x_1)
\tag{34}
\\
&= E_{q(x_1|x_0)}E_{q(x_0|x_1)}[q(x_0|x_1) \log q(x_0|x_0) - q(x_0|x_1) \log p_\theta(x_{0}|x_1)]
\tag{35}
\\
&= E_{q(x_1, x_0)}D_{KL}(q(x_{1 - 1}|x_1, x_0)||p_\theta(x_{1 - 1}|x_0))
\tag{36}
\end{align}
$$

与$L_{t - 1}$形式相同，记$L_0 + L_{t - 1} = L_t$，对于式中的正向过程$q(x_t|x_{t - 1})$的后验分布$q(x_{t - 1}|x_t)$，由贝叶斯定理：

$$
q(x_{t - 1}|x_t) = \frac{q(x_t|x_{t - 1})q(x_{t - 1})}{q(x_t)} = \frac{q(x_t|x_{t - 1})q(x_{t - 1}|x_0)}{q(x_t|x_0)}
\tag{37}
$$

式(37)的三个概率都是已知的，必然可以求解，设后验分布$q(x_{t - 1}|x_t) = \mathcal{N}(\tilde{\mu}, \Sigma)$，则有：

$$
\begin{align}
\frac{(x_{t - 1} - \tilde{\mu})^2}{\Sigma} &= \frac{(x_t - \sqrt{\alpha_t}x_{t - 1})^2}{1 - \alpha_t} + \frac{(x_{t - 1} - \sqrt{\bar{\alpha}\_{t - 1}}x_0)^2}{1 - \bar{\alpha}\_{t - 1}} - \frac{(x_t - \sqrt{\bar{\alpha}\_t}x_0)^2}{1 - \bar{\alpha}\_t}
\tag{38}
\\
&= (\underbrace{\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}\_{t - 1}}}\_{1/\Sigma})x^2_{t - 1} - 2(\underbrace{\frac{\sqrt{\alpha_t}}{1 - \alpha_t}x_t + \frac{\sqrt{\bar{\alpha}\_{t - 1}}}{1 - \bar{\alpha}\_{t - 1}}x_0}\_{\tilde{\mu}/\Sigma})x_{t - 1} + C(x_0, x_t)
\tag{39}
\end{align}
$$

其中，$C(x_0, x_t)$为不含$x_{t - 1}$的项，通过左右比对，易得：

$$
\begin{align}
\Sigma &= \frac{1}{\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}\_{t - 1}}}
\tag{40}
\\
&= \frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t
\tag{41}
\\
\tilde{\mu} &= ((\frac{\sqrt{\alpha_t}}{1 - \alpha_t}x_t + \frac{\sqrt{\bar{\alpha}\_{t - 1}}}{1 - \bar{\alpha}\_{t - 1}}x_0)\Sigma
\tag{42}
\\
&= \frac{(1 - \bar{\alpha}\_{t - 1})\sqrt{\alpha_t}}{1 - \bar{\alpha}\_t}x_t + \frac{\sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_t}\beta_tx_0
\tag{43}
\end{align}
$$

又由式(10)，可将$x_0$转换为$x_t$来进行表达，从而：

$$
\begin{align}
\tilde{\mu} &= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}\_t)}}\epsilon_t(x_t, t)
\tag{44}
\\
x_{t - 1} &= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}_t)}}\epsilon_t(x_t, t) + \sqrt{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t}\epsilon
\tag{45}
\end{align}
$$

其中，$\epsilon_t(x_t, t)$的含义为，对于给定的时刻$t$，从初始状态$x_0$，经过一步的正向过程成为该时刻状态$x_t$，所需要的噪声。若定义$p_\theta(x_{t - 1}|x_t) = \mathcal{N}(\mu_\theta, \sigma^2_\theta)$，由`高斯分布的KL散度公式`：

$$
\begin{align}
L_t &= \sum^T_{t = 1} E_{q(x_t, x_0)}D_{KL}(q(x_{t - 1}|x_t, x_0)||p_\theta(x_{t - 1}|x_t))
\tag{46}
\\
&= \sum^T_{t = 1} E_{q(x_t, x_0)}[\frac{1}{2}\log\frac{\sigma^2_\theta}{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t} + \frac{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t + (\tilde{\mu} - \mu_\theta)^2}{2\sigma^2_\theta} - \frac{1}{2}]
\tag{47}
\end{align}
$$

尽管$\sigma_\theta$是可学习的，但其对采样质量的提升并不显著，并未得到推广。在DDPM中，取$\sigma^2_\theta = \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}\beta_t$，因此有：

$$
L_t = \sum^T_{t = 1} E_{q(x_t, x_0)}(\tilde{\mu} - \mu_\theta)^2
\tag{48}
$$

## 网络搭建

### 时间嵌入

`Gmeek-html<img src="https://OmnisyR.github.io/figs/time_embeddings.png">`

`Gmeek-html<p align="center"><iframe src="https://www.desmos.com/calculator/xkmphum0ou?embed" width="800" height="400" style="border: 1px solid #ccc" frameborder=0></iframe></p>`
