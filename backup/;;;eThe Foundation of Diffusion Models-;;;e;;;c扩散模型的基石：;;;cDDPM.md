<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->
;;;a
;;;;;;;eCopy and paste the code directly::Or just click [me](https://github.com/OmnisyR/diffusion_demo/tree/main/ddpm/en) to go to the full version.;;;e;;;c直接复制粘贴::或是点[我](https://github.com/OmnisyR/diffusion_demo/tree/main/ddpm/cn)转到完全版。;;;c;;;;
;;;;;;;eMarkov chain::The state at a given moment is only related to the state at the previous moment, i.e., $x_t = f(x_{t - 1})$, without the involvement of states at other moments. A chain formed by a number of such state relationships constitutes a Markov chain. Therefore, Markov chains have the following special properties:;;;e;;;c马尔可夫链::某一时刻的状态只与上一时刻的状态相关，即$x_t = f(x_{t - 1})$，不需要其他时刻的状态参与，若干个这样的状态关系组成的链条便形成了马尔科夫链。因此，马尔科夫链存在着这样的特殊性质：;;;c

$$
\begin{align}
&P(X_n|X_0) = P(X_n)
\\
&P(X_n|X_{n - 1}, X_{n - 2}, \dots, X_0) = P(X_n|X_{n - 1})
\end{align}
$$

;;;;
;;;;;;;eReparameterization trick::For the probability $p(x|y) = \mathcal{N}(x|ay, b)$, i.e., $x$ follows a Gaussian distribution with mean $ay$ and standard deviation $\sqrt{b}$, then we have $x = ay + \sqrt{b}\epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.;;;e;;;c重参数化技巧::对于概率$p(x|y) = \mathcal{N}(x|ay, b)$，即$x$服从一个均值为$ay$，标准差为$\sqrt{b}$的高斯分布，那么则有$x = ay + \sqrt{b}\epsilon$，其中$\epsilon \sim \mathcal{N}(0, 1)$。;;;c;;;;
;;;;;;;eThe code below::PyTorch code that uses GPU acceleration by default. Due to the enormous computing power required by diffusion models, it is almost impossible to run diffusion models without GPU acceleration.;;;e;;;c下方的代码::默认使用GPU加速的pytorch代码，由于扩散模型需求算力巨大，不使用GPU加速几乎很难跑得了扩散模型。;;;c;;;;
;;;;;;;eAddition rule of Gaussian distribution;;;e;;;c高斯分布的加法法则;;;c::

$$
\mathcal{N}(\mu_1, \sigma^2_1) + \mathcal{N}(\mu_2, \sigma^2_2) = \mathcal{N}(\mu_1 + \mu_2, \sigma^2_1 + \sigma^2_2)
$$

;;;;
;;;;;;;eThe upper bound of this loss as small as possible::Some people may have a typical misconception, namely that equation (13) is equivalent to the KL divergence on the right side being 0. This idea is incorrect because when the KL divergence is 0, although the gradient of the KL divergence is 0, the overall gradient on the right side is not necessarily 0. A very simple example can effectively illustrate this point:;;;e;;;c上界尽可能的小::有些人会存在一个典型的认识错误，即认为式(13)等价于目标为右方的KL散度为0。这个想法错在KL散度为0时，虽然KL散度的梯度为0，但右方整体的梯度不一定为0，一个很简单的例子就能有效进行说明：;;;c

$$
x^2 \leq x^2 + (2 - x)^2
$$

;;;eClearly, the right-hand side takes its minimum value at $x = 1$, rather than at $x = 2$, where $(2 - x)^2$ is 0.;;;e;;;c显然，右式在$x = 1$时取最小值，而非令$(2 - x)^2$为0的$x = 2$时。;;;c

;;;;
;;;;;;;eBayes' theorem;;;e;;;c贝叶斯定理;;;c::

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

;;;;
;;;;;;;eEquation ;;;e;;;c式;;;c(10)::

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t
$$

;;;;
;;;;;;;eKL divergence formula for Gaussian distributions::For $p_1(x) = \mathcal{N}(\mu_1, \sigma^2_1)$ and $p_2(x) = \mathcal{N}(\mu_2, \sigma^2_2)$, we have:;;;e;;;c高斯分布的KL散度公式::对于$p_1(x) = \mathcal{N}(\mu_1, \sigma^2_1)$以及$p_2(x) = \mathcal{N}(\mu_2, \sigma^2_2)$，有：;;;c

$$
D_{KL}(p_1||p_2) = \frac{1}{2}\log\frac{\sigma^2_2}{\sigma^2_1} + \frac{\sigma^2_1 + (\mu_1 - \mu_2)^2}{2\sigma^2_2} - \frac{1}{2}
$$

;;;;
;;;;UNet::;;;e[Article address](https://arxiv.org/abs/1505.04597);;;e;;;c[文章地址](https://arxiv.org/abs/1505.04597);;;c
;;;eUNet is relatively complex. You can download [unet.py](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py) and [nn.py](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py) provided by the article [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) on GitHub to run the diffusion model. You do not need to download fp16_util.py. Simply delete the relevant code in unet.py before running it.;;;e;;;cUNet较为复杂，可以在Github上下载由文章[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)提供的[unet.py](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py)以及[nn.py](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py)来运行扩散模型，fp16_util.py可以不用下载，运行前将unet.py中相关代码删去即可。;;;c;;;;
;;;;;;;eResidual idea::[Article address](https://arxiv.org/abs/1512.03385);;;e;;;c残差思想::[文章地址](https://arxiv.org/abs/1512.03385);;;c;;;;
;;;;Attention is All You Need::;;;e[Article address](https://arxiv.org/abs/1706.03762);;;e;;;c[文章地址](https://arxiv.org/abs/1706.03762);;;c;;;;
;;;;;;;eSelect a linear noise schedule::There are many types of noise schedules. Although cosine noise schedule seem to be highly recommended online, in actual practice, they can cause a problem I call “color divergence.” A series of measures are required to improve the results and obtain the correct outcome. Therefore, when you are just starting to learn about diffusion models, it is best to use a linear noise schedule first.

Color Diverge Phenomenon
;;;e;;;c选择线性噪声时间表::噪声时间表有很多种，虽然网络上疑似很推崇余弦型噪声时间表，但实际操作中它会导致一种被我称为“色彩发散”的问题，还需要通过一系列手段来改善才能得到正确结果，所以刚刚接触扩散模型，可以先使用线性噪声时间表。

色彩发散现象;;;c
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/color_diverge.png" width="351" height="128"/></p>`;;;;
;;;;;;;eEquation ;;;e;;;c式;;;c(50)::

$$
\begin{align}
x_{t - 1} &= \frac{1}{\sqrt{\alpha_t}}x_t
\\
&\quad - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}\_t)}}\epsilon_\theta(x_t, t)
\\
&\quad + \sqrt{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t}\epsilon
\end{align}
$$

;;;;
;;;;;;;eThe limitations of the blog framework::The total number of characters in a single article is limited to 65,536.;;;e;;;c博客框架的限制::单篇文章的总字符数被限制在了65,536内。;;;c;;;;
;;;a;;;e
## Introduction
I'm not very good at describing subjective things, and most of the information about diffusion models has already been covered in [Diffusion Model Overview](https://omnisyr.github.io/post/%3B%3B%3BeAn%20Overview%20of%20Diffusion%20Models%3B%3B%3Be%3B%3B%3Bc-kuo-san-mo-xing-gai-shu-%3B%3B%3Bc.html), so I'll skip the formalities and get straight to the point!

In this article, I will introduce and implement DDPM from a mathematical and coding perspective. If you just want to try it out, you can `Copy and paste the code directly`, modify a few path parameters, and run it right away, provided your environment is properly configured!

## Forward Process
The forward process of the diffusion models involve gradually adding noise to the original image, such that by the final stage, the image fully follows the noise distribution. Typically, Gaussian noise is used for this process. For discrete time points $t = 0, 1, \dots, T$, in the `Markov chain` process followed by the diffusion models, given $x_0 \sim q(x_0)$, i.e., the original image as the initial stage, and $x_T \sim q(x_T) = \mathcal{N}(0, I)$, i.e., the image completely following standard Gaussian noise as the final stage. The probability distributions of the various stages $x_0, x_1, \dots, x_T$ follow the following formula:;;;e;;;c
## 介绍
我不擅长描述一些非客观性的东西，一些关于扩散模型的介绍在[扩散模型概述](https://omnisyr.github.io/post/%3B%3B%3BeAn%20Overview%20of%20Diffusion%20Models%3B%3B%3Be%3B%3B%3Bc-kuo-san-mo-xing-gai-shu-%3B%3B%3Bc.html)中也讲的差不多了，所以没有繁文缛节了，直接开始吧！

在这篇文章中，我将从数学和代码上介绍和实现DDPM。如果你只是想单纯实践的话，在环境配置好的条件下，你可以`直接复制粘贴`，修改几个路径参数便可直接运行！

## 正向过程
扩散模型的正向过程为，对原始的图像进行一步步地噪声添加，从而在最终阶段，使得图片完全服从噪声分布，一般地，使用高斯噪声来进行这个过程。对于离散的时间$t = 0, 1, \dots, T$，在扩散模型遵循的`马尔可夫链`过程中，对于给定的$x_0 \sim q(x_0)$，即原始图像作为初始阶段，和$x_T \sim q(x_T) = \mathcal{N}(0, I)$，即完全服从标准高斯噪声的图像作为最终阶段。其各个阶段$x_0, x_1, \dots, x_T$的概率分布服从以下公式：;;;c

$$
q(x_T|x_0) = q(x_0)\prod^T_{t = 1}q(x_t|x_{t - 1})
\tag{1}
$$

;;;ewhere $q(x_{t + 1}|x_{t})$ represents the process of adding noise to the image at the previous moment at each moment. Obviously, this process needs to be defined manually, so a set of hyperparameters $\beta_t$ can be selected, resulting in:;;;e;;;c其中$q(x_{t + 1}|x_{t})$代表着每一时刻，为上一时刻的图片添加噪声的过程。很显然，这个过程是需要人为定义的，因此，可以选择一组超参数$\beta_t$，从而有：;;;c

$$
q(x_t|x_{t - 1}) = \mathcal{N}(x_t;\sqrt{1 - \beta_t}x_{t - 1},\beta_tI)
\tag{2}
$$

;;;eHowever, such an equation is still incomputable. Through the `Reparameterization trick`, any Gaussian distribution can be expanded, so we have:;;;e;;;c但是，这样的式子仍然是不可计算的，通过`重参数化技巧`，可以将任意的高斯分布进行展开，因此则有：;;;c

$$
x_t = \sqrt{1 - \beta_t} x_{t - 1} + \sqrt{\beta_t}\epsilon
\tag{3}
$$

;;;eIn this way, a computable form is used to express the Markov chain forward process of the diffusion model. In DDPM, $\beta_t$ is defined as a linear array ranging from $\beta_1 = 0.001$ to $\beta_T = 0.02$. This method of adding noise is also known as a linear noise schedule. The process can be implemented using `The code below`:
```python
import torch
import torchvision.io
from PIL import Image
from torchvision.transforms.v2 import Compose, ToTensor, Lambda
from tqdm import tqdm

#The timesteps is set to the commonly used 1000.
timesteps = 1000
#Convert the color values of the image into a tensor between -1 and 1.
transform = Compose([ToTensor(), Lambda(lambda t: (t * 2) - 1)])
#Convert images with color values between -1 and 1 to values between 0 and 255.
transform_reverse = Compose([Lambda(lambda t: (t + 1) / 2 * 255)])
#Read and convert images into tensors.
x_t = transform(Image.open('xxx.png')).cuda()
#Use a linear noise schedule.
beta = torch.linspace(0.001, 0.02, timesteps).cuda()
for timestep in tqdm(range(timesteps)):
    #Random noise at each step.
    ep = torch.randn(size=x_t.shape).cuda()
    #Iterative noise addition process of equation (3).
    x_t = (1 - beta[timestep]).sqrt().item() * x_t + beta[timestep].sqrt() * ep
    #Save image to local.
    torchvision.io.write_png(
      transform_reverse(x_t).clip(0, 255).to(torch.uint8).cpu(),
      'noising/%d.png' % timestep
    )
```
;;;e;;;c这样一来，就使用了一个可计算的形式表达出了扩散模型的马尔科夫链正向过程。DDPM中定义$\beta_t$是一个从$\beta_1 = 0.001$到$\beta_T = 0.02$的线性数组，这种噪声添加方式又称线性噪声时间表。通过`下方的代码`便可以实现这一过程：
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
;;;c

;;;eFor simplicity, let $\alpha_t + \beta_t = 1$, then we have:;;;e;;;c为了式子的简洁性，记$\alpha_t + \beta_t = 1$，则有：;;;c

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

;;;ewhere $\bar{\alpha}\_t = \prod_{i = 1}^{t}\alpha_i$. Since there are two completely random standard Gaussian distributions in equation (6), we can perform the inverse operation of the reparameterization technique on these two to obtain equation (7), and then use the `Addition rule of Gaussian distribution` to obtain equation (8), thereby simplifying the multi-step iteration and enabling us to calculate the state at any time in the forward process in a single step:
```python
def add_noise(x_0, t):
    #The one-step noise addition process in equation (10).
    return sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * torch.randn(size=x_0.shape).cuda()

#Pre-caching of parameters.
alpha = 1 - beta
#Here, an extra 1 is added to the front position of the obtained α product.
#This is to ensure consistency between the timestamp and the index when writing the program.
alpha_bar = torch.cat((torch.Tensor((1,)).cuda(), alpha.cumprod(dim=0)))
sqrt_alpha_bar = alpha_bar.sqrt()
sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()

timestep = 200
x_t = add_noise(x_0, timestep)
torchvision.io.write_png(transform_reverse(x_t).clip(0, 255).to(torch.uint8).cpu(), '%s.png' % timestep)
```
;;;e;;;c其中$\bar{\alpha}\_t = \prod_{i = 1}^{t}\alpha_i$。由于式(6)中存在着两个完全随机的标准高斯分布，因此可以对这二者进行重参数化技巧的逆运算得到式(7)，再通过`高斯分布的加法法则`得到式(8)，从而对多步的迭代进行化简，能够仅仅使用一步求出正向过程的任意时刻的状态：
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
;;;c;;;e
## Reverse Process
Compared to the forward process, the reverse process is much more complex. The goal of the inverse process is to obtain a probabilistic model for a given $x_T \sim p_\theta(x_T) = \mathcal{N}(0, I)$, such that $x_T$ can be transformed from a standard Gaussian distribution into a likelihood estimate of the original data set, i.e.:;;;e;;;c
## 逆向过程
相较正向过程，逆向过程则会复杂很多。逆向过程的目标是对于给定的$x_T \sim p_\theta(x_T) = \mathcal{N}(0, I)$，需要一个概率模型，能够使得$x_T$由标准高斯分布，转变为原始数据集的似然估计，即：;;;c

$$
p_\theta(x_0|x_T) = p_\theta(x_T)\prod^T_{t = 1}p_\theta(x_{t - 1}|x_t)
\tag{11}
$$

;;;eIf we perform maximum likelihood estimation, the log loss is:;;;e;;;c若对其进行极大似然估计，其对数损失为：;;;c

$$
L_\theta = -\log p_\theta(x_0) = -\log (p_\theta(x_T)\prod^T_{t = 1}p_\theta(x_{t - 1}|x_t))
\tag{12}
$$

;;;ewhere $p_\theta(x_t|x_{t + 1})$ is completely unknown and cannot be solved, so we consider making `The upper bound of this loss as small as possible`. Since the KL divergence is necessarily non-negative, we have:;;;e;;;c其中，$p_\theta(x_t|x_{t + 1})$是完全未知的，无法求解，因此考虑令该损失的`上界尽可能的小`。由于KL散度必然是非负的，所以：;;;c

$$
\begin{align}
L_\theta \leq L_{VLB} &= -\log p_\theta(x_0) + D_{KL}(q(x_{1:T}|x_0)||p_\theta(x_{1:T}|x_0))
\tag{13}
\\
&= -\log p_\theta(x_0) + E_{q(x_{1:T}|x_0)}\log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{1:T}|x_0)}
\tag{14}
\end{align}
$$

;;;eAccording to `Bayes' theorem`:;;;e;;;c根据`贝叶斯定理`：;;;c

$$
p_\theta(x_{1:T}|x_0) = \frac{p_\theta(x_0|x_{1:T})p_\theta(x_{1:T})}{p_\theta(x_0)} = \frac{p_\theta(x_0, x_{1:T})}{p_\theta(x_0)} = \frac{p_\theta(x_{0:T})}{p_\theta(x_0)}
\tag{15}
$$

;;;eTherefore:;;;e;;;c因此：;;;c

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

;;;eAlso:;;;e;;;c又：;;;c

$$
\begin{align}
&q(x_{t - 1}|x_t, x_0) = \frac{q(x_{t - 1}, x_t, x_0)}{q(x_t, x_0)} = \frac{q(x_{t - 1}, x_t| x_0)}{q(x_t, x_0)}
\tag{30}
\\
\Leftrightarrow &q(x_{t - 1}, x_t| x_0) = q(x_t, x_0)q(x_{t - 1}|x_t, x_0)
\tag{31}
\end{align}
$$

;;;eSo:;;;e;;;c从而：;;;c

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

;;;ewhere, for $L_T$, since $q(x_T) = \mathcal{N}(0, I)$ and $p_\theta(x_T) = \mathcal{N}(0, I)$ are both independent of the maximum likelihood estimation parameters, they are regarded as constants.

For $L_0$, since $1 = q(x_0|x_0) = \frac{q(x_1|x_0)q(x_0|x_0)}{q(x_1|x_0)} = q(x_0|x_1)$, we have:;;;e;;;c其中，对于$L_T$，由于$q(x_T) = \mathcal{N}(0, I)$，$p_\theta(x_T) = \mathcal{N}(0, I)$都和极大似然估计参数无关，视为常数；

对于$L_0$，由于$1 = q(x_0|x_0) = \frac{q(x_1|x_0)q(x_0|x_0)}{q(x_1|x_0)} = q(x_0|x_1)$，因此有：;;;c

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

;;;eSimilar to the form of $L_{t - 1}$, let $L_0 + L_{t - 1} = L_t$. For the posterior distribution $q(x_{t - 1}|x_t)$ of the forward process $q(x_t|x_{t - 1})$ in the equation, by Bayes' theorem:;;;e;;;c与$L_{t - 1}$形式相同，记$L_0 + L_{t - 1} = L_t$，对于式中的正向过程$q(x_t|x_{t - 1})$的后验分布$q(x_{t - 1}|x_t)$，由贝叶斯定理：;;;c

$$
q(x_{t - 1}|x_t) = \frac{q(x_t|x_{t - 1})q(x_{t - 1})}{q(x_t)} = \frac{q(x_t|x_{t - 1})q(x_{t - 1}|x_0)}{q(x_t|x_0)}
\tag{37}
$$

;;;eThe three probabilities in equation (37) are known and can be solved. Let the posterior distribution be $q(x_{t - 1}|x_t) = \mathcal{N}(\tilde{\mu}, \Sigma)$. Considering the exponential part of equation (37) expanded using the Gaussian distribution density function, we have:;;;e;;;c式(37)的三个概率都是已知的，必然可以求解，设后验分布$q(x_{t - 1}|x_t) = \mathcal{N}(\tilde{\mu}, \Sigma)$，考虑式(37)使用高斯分布密度函数展开后的指数部分，则有：;;;c

$$
\begin{align}
\frac{(x_{t - 1} - \tilde{\mu})^2}{\Sigma} &= \frac{(x_t - \sqrt{\alpha_t}x_{t - 1})^2}{1 - \alpha_t} + \frac{(x_{t - 1} - \sqrt{\bar{\alpha}\_{t - 1}}x_0)^2}{1 - \bar{\alpha}\_{t - 1}} - \frac{(x_t - \sqrt{\bar{\alpha}\_t}x_0)^2}{1 - \bar{\alpha}\_t}
\tag{38}
\\
&= (\underbrace{\frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}\_{t - 1}}}\_{1/\Sigma})x^2_{t - 1} - 2(\underbrace{\frac{\sqrt{\alpha_t}}{1 - \alpha_t}x_t + \frac{\sqrt{\bar{\alpha}\_{t - 1}}}{1 - \bar{\alpha}\_{t - 1}}x_0}\_{\tilde{\mu}/\Sigma})x_{t - 1} + C(x_0, x_t)
\tag{39}
\end{align}
$$

;;;ewhere $C(x_0, x_t)$ is the term that does not contain $x_{t - 1}$. Through left-right comparison, it is easy to obtain:;;;e;;;c其中，$C(x_0, x_t)$为不含$x_{t - 1}$的项，通过左右比对，易得：;;;c

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

;;;eThen, from `Equation (10)`, $x_0$ can be converted to $x_t$ for expression, thus:;;;e;;;c又由`式(10)`，可将$x_0$转换为$x_t$来进行表达，从而：;;;c

$$
\begin{align}
\tilde{\mu} &= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}\_t)}}\epsilon_t(x_t, t)
\tag{44}
\\
x_{t - 1} &= \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}_t)}}\epsilon_t(x_t, t) + \sqrt{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t}\epsilon
\tag{45}
\end{align}
$$

;;;ewhere $\epsilon_t(x_t, t)$ denotes the noise required to transition from the initial state $x_0$ to the state $x_t$ at time $t$ through a single forward step. If we define $p_\theta(x_{t - 1}|x_t) = \mathcal{N}(\mu_\theta, \sigma^2_\theta)$, then by the `KL divergence formula for Gaussian distributions`:;;;e;;;c其中，$\epsilon_t(x_t, t)$的含义为，对于给定的时刻$t$，从初始状态$x_0$，经过一步的正向过程成为该时刻状态$x_t$，所需要的噪声。若定义$p_\theta(x_{t - 1}|x_t) = \mathcal{N}(\mu_\theta, \sigma^2_\theta)$，由`高斯分布的KL散度公式`：;;;c

$$
\begin{align}
L_t &= \sum^T_{t = 1} E_{q(x_t, x_0)}D_{KL}(q(x_{t - 1}|x_t, x_0)||p_\theta(x_{t - 1}|x_t))
\tag{46}
\\
&= \sum^T_{t = 1} E_{q(x_t, x_0)}[\frac{1}{2}\log\frac{\sigma^2_\theta}{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t} + \frac{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t + (\tilde{\mu} - \mu_\theta)^2}{2\sigma^2_\theta} - \frac{1}{2}]
\tag{47}
\end{align}
$$

;;;eAlthough $\sigma_\theta$ is learnable, its improvement in sampling quality is not significant and has not been widely adopted. In DDPM, we set $\sigma^2_\theta = \frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t$, so we have:;;;e;;;c尽管$\sigma_\theta$是可学习的，但其对采样质量的提升并不显著，未得到推广。在DDPM中，取$\sigma^2_\theta = \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}\beta_t$，因此有：;;;c

$$
L_t = \sum^T_{t = 1} E_{q(x_t, x_0)}[\frac{1 - \bar{\alpha}\_t}{2(1 - \bar{\alpha}\_{t - 1})\beta_t}(\tilde{\mu} - \mu_\theta)^2]
\tag{48}
$$

;;;eTo minimize $L_t$, it is necessary for $\mu_\theta$ to be as close as possible to $\tilde{\mu}$. If we set $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}\_t)}}\epsilon_\theta(x_t, t)$, then we only need $\epsilon_\theta(x_t, t)$ to be as close as possible to $\epsilon_t(x_t, t)$, i.e., the loss function:;;;e;;;c要使$L_t$尽可能小，即需要$\mu_\theta$尽可能接近$\tilde{\mu}$。若令$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}\_t)}}\epsilon_\theta(x_t, t)$，那么只需要$\epsilon_\theta(x_t, t)$尽可能接近$\epsilon_t(x_t, t)$，即损失函数：;;;c

$$
L = ||\epsilon_\theta(x_t, t) - \epsilon_t(x_t, t)||^2_2
\tag{49}
$$

;;;eSampling formula for the reverse process:;;;e;;;c逆向过程的采样公式：;;;c

$$
x_{t - 1} = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t(1 - \bar{\alpha}\_t)}}\epsilon_\theta(x_t, t) + \sqrt{\frac{1 - \bar{\alpha}\_{t - 1}}{1 - \bar{\alpha}\_t}\beta_t}\epsilon
\tag{50}
$$

;;;e
## Network Setup

After a lengthy process of reverse derivation, the loss function of the diffusion model was obtained. The next step is to build a network for the prediction function, whose input is a certain time point and the corresponding noisy image at that time point, and whose output is the predicted noise required to transform the initial state into the state at that time point through the forward process. DDPM uses `UNet` to perform this process.

### Network Structure

The original article provides a structural diagram of UNet:;;;e;;;c
## 网络搭建

经过逆向过程漫长的推导，获得了扩散模型的损失函数，接下来就是要对预测函数进行网络搭建，该预测函数的输入为某一时刻以及该时刻对应的带噪图片，输出为预测的，从初始状态，经过正向过程成为该时刻状态，所需要的噪声。DDPM采用了`UNet`来进行这个过程。

### 网络结构

原文章中给出了一个UNet的结构图：;;;c

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/u-net-illustration-correct-scale2.png"/></p>`

;;;eIn this case, the upper part of the square represents the number of data channels, and the lower left part represents the data resolution. The network inputs a $572^2$ resolution image with 1 channel and outputs a $388^2$ resolution image with 2 channels. The network contains four core components: ordinary convolution process, downsampling, upsampling, and residual process.

- The short arrow pointing to the right represents the standard convolution process, where in each layer, the first convolution operation always reduces the number of channels in the data to the predefined number of channels for that layer;
- The downward arrow represents downsampling, during which the resolution of the data is always reduced to half of its original value, followed by deeper convolution operations;
- The upward arrow represents upsampling, where the resolution of the data is always doubled, followed by convolution at a shallower layer;
- The long right arrow represents the residual process, which utilizes the `Residual idea` to combine downsampled data with upsampled data before convolution, effectively mitigating information loss caused by downsampling.

However, for diffusion models, the input requires an additional timestamp t in addition to a single image. Therefore, the image and timestamp must be integrated and fed into UNet for computation. This operation is also known as time embedding.

### Time Embedding

The time embedding algorithm refers to the position embedding algorithm in `Attention is All You Need`. It expands the single number of the timestamp into a tensor, inputs it into the network for calculation, and then adds it to the ordinary convolution process for further calculation. The expansion of the timestamp obeys the following equation:;;;e;;;c在该案例中，方块的上方代表着数据的通道数，左下方代表着数据的分辨率。网络的输入为$572^2$分辨率、通道数为1的图片，输出为$388^2$分辨率、通道数为2的图片。在网络内，则包含了四个核心部分：普通卷积过程；下采样；上采样以及残差过程。

- 向右的短箭头代表着普通卷积过程，在每一层中，第一次卷积过程总是会将数据的通道数变为预定的、该层的通道数；
- 向下的箭头代表着下采样，在这个过程中，总是会将数据的分辨率变为原来的一半，并在接下来进行更深层的卷积；
- 向上的箭头代表着上采样，在这个过程中，总是会将数据的分辨率变为原来的两倍，并在接下来进行更浅层的卷积；
- 向右的长箭头代表着残差过程，该部分利用了`残差思想`，将下采样的数据与上采样的数据相结合再进行卷积，能够有效地去除一些由于下采样导致的信息丢失问题。

但是对于扩散模型而言，其输入还需要额外引入一个时间戳t，而非单单一个图片，因此还需要将图片和时间戳整合到一起，放入UNet中进行计算，这个操作又称时间嵌入。

### 时间嵌入

时间嵌入算法参考了`Attention is All You Need`中的位置嵌入算法，通过将时间戳这单个数字，展开为某个张量，再输入到网络中进行计算，最后再与普通卷积过程的值相加，进行接下来的计算。时间戳的展开服从下面的式子：;;;c

$$
\begin{align}
TE_{(t, i)} &= \sin\frac{t}{10000^{2i / d}}
\tag{51}
\\
TE_{(t, j)} &= \cos\frac{t}{10000^{2j / d}}
\tag{52}
\end{align}
$$

;;;ewhere $d$ denotes the desired dimension of expansion, $i = \{1, 2, \dots, \frac{d}{2}\}$, and $j = \{\frac{d}{2} + 1, \frac{d}{2} + 2, \dots, d\}$. If $d = 128$ is defined, its visualization is shown in the figure below:;;;e;;;c其中，$d$表示所期望的展开维度，$i = \\{1, 2, \dots, \frac{d}{2}\\}$，$j = \\{\frac{d}{2} + 1, \frac{d}{2} + 2, \dots, d\\}$。若定义$d = 128$，则其可视化如下图所示：;;;c

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/time_embeddings.png"/></p>`

;;;eIf represented as function graphs, the red $TE_{(t, i)}$ and blue $TE_{(t, j)}$ change with dimension as shown below:;;;e;;;c若是以函数图像的形式表示，红色的$TE_{(t, i)}$与蓝色的$TE_{(t, j)}$随着维度的变化如下所示：;;;c

`Gmeek-html<p align="center"><iframe src="https://www.desmos.com/calculator/xkmphum0ou?embed" width="800" height="400" style="border: 1px solid #ccc" frameborder=0></iframe></p>`
;;;e
## Training Process
First, import some necessary libraries:
```python
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.v2 import Lambda, ToTensor, RandomHorizontalFlip, Compose, Grayscale
from tqdm import tqdm

import unet
```
Next is the determination of training configurations and hyperparameters. For hyperparameters such as $\beta_t$, select the linear noise schedule:
```python
def linear(time_steps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, time_steps).cuda()


#The timestamp is set from 0 to 1000.
#Reducing or increasing this value will change the parameters related to the noise schedule.
time_steps = 1000
#Pre-computed cache of noise schedule-related parameters.
#Shape:[1000]
betas = linear(time_steps)
#Shape:[1000]
alphas = 1. - betas
#Shape:[1001]
alphas_bar = torch.cat((torch.tensor((1,)).cuda(), torch.cumprod(input=alphas, dim=0)))
#Shape:[1001]
sqrt_alphas_bar = torch.sqrt(alphas_bar)
#Shape:[1001]
sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
```
Configuration and hyperparameters:
```python
"""
For simplicity, CIFAR1 is used as the dataset, with image dimensions of 32×32.
CIFAR1 refers to using only one category from CIFAR10 as the dataset.
The file structure of the dataset is as follows:
cifar1
├── test
│   └── horse
│       ├── 0001.png
│       ├── 0002.png
│       ├── ...
│       └── 5000.png
└── train
    └── horse
        ├── 0001.png
        ├── 0002.png
        ├── ...
        └── 1000.png
"""
image_size = 32
#The number of channels is 3, i.e., RGB.
channels = 3
#The number of base channels in the model, i.e., the number of channels in the top layer.
base_channels = 128
#The number of channels in different layers of the model,
#with a resolution of 32×32, the model depth is selected as 4,
#corresponding to 128, 256, 256, 256 channels.
ch_mults = {
    32: (1, 2, 2, 2),
    64: (1, 2, 3, 4),
    128: (1, 1, 2, 3, 4),
    256: (1, 1, 2, 2, 4, 4),
    512: (0.5, 1, 1, 2, 2, 4, 4)
}
#Learning rate.
learning_rate = 1e-4
#Training rounds: Select the iterative dataset 500 times
#(in reality, this number of rounds is far from sufficient).
epochs = 500
#Rounds used for sampling and model checkpoint saving.
milestone_step = 10
#Maximum number of checkpoints saved.
cache_num = 10
#The number of images trained in a batch.
#In this case, 128 images are suitable for a graphics card with 8G of memory.
batch_size = 128
#Proposed locations for checkpoints.
checkpoint_folder = 'cifar1/cp'
#Location of training set.
datasets_folder = 'cifar1/train'
#File used to collect loss values during training.
csv_str = checkpoint_folder + '/loss.csv'
#Prefix for model checkpoint save file names.
prefix = "checkpoint_epoch"
#suffix for model checkpoint save file names.
suffix = ".pth"
#Create a new folder for checkpoints.
Path(checkpoint_folder).mkdir(exist_ok=True)
Path(checkpoint_folder + '/milestone').mkdir(exist_ok=True)
#Create a new model.
model = unet.UNetModel(
    image_size=image_size,
    in_channels=channels,
    model_channels=base_channels,
    out_channels=channels,
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    dropout=0.0,
    channel_mult=ch_mults[image_size],
    num_heads=4,
    num_head_channels=-1,
)
#If there are checkpoints in the checkpoints folder, read the latest checkpoint.
last_model = None
last = 0
for file in os.listdir(checkpoint_folder):
    if file[:len(prefix)] != prefix:
        continue
    epochs_str = file[len(prefix) + 1:0 - len(suffix)]
    if int(epochs_str) > last:
        last = int(epochs_str)
if last > 0:
    last_model = torch.load(checkpoint_folder + "/%s_%d%s" % (prefix, last, suffix), weights_only=False)
if last_model is not None:
    model.load_state_dict(last_model)
else:
    print("No Checkpoints Exist")
model = model.cuda()
#Parameter count in computational models.
parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
count_str = "Parameters Count: "
print(count_str + "%.2fB" % (parameters / 1e9)) \
    if parameters > 1e9 else print(count_str + "%.2fM" % (parameters / 1e6))
#Optimizer.
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#Convert images to tensors.
"""
ToTensor()                    :
    Reads an image and scales its integers from [0, 255] to floating-point numbers in the range [0, 1]
Lambda(lambda t: (t * 2) - 1) : Scale floating-point numbers in the range [0, 1] to the range [-1, 1]
RandomHorizontalFlip()        : Randomly flip the data horizontally
"""
transform = Compose(
    [Grayscale(), ToTensor(), Lambda(lambda t: (t * 2) - 1), RandomHorizontalFlip()]
    if channels == 1 else
    [ToTensor(), Lambda(lambda t: (t * 2) - 1), RandomHorizontalFlip()]
)
```
Core training methods:
```python
def train():
    train_loop(DataLoader(
        datasets.ImageFolder(datasets_folder, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    ))


def train_loop(dataloader):
    #Calculate the number of iterations remaining.
    local_epochs = epochs - last
    #Cache for loss value files.
    csv_cache = ""
    print()
    print("Now Strat Training")
    for epoch in range(local_epochs):
        #Loss value cache.
        losses = []
        #Without waiting, it seems that the tqdm progress bar will encounter some issues?
        time.sleep(0.1)
        #Calculate the current iteration count.
        global_epoch = epoch + last + 1
        #Iterate through the dataset.
        for step, data_batch in (pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc="Epoch: %d/ %d With Loss %f" % (global_epoch, epochs, 1)
        )):
            #Initialization Optimizer.
            optimizer.zero_grad()
            #Get all data tensors for this batch,
            #with shape [batch_size, channels, image_size, image_size].
            data_batch = data_batch[0].cuda()
            #Obtain the loss value and the timestamp randomly selected for calculating the loss value.
            complex_loss, t = get_loss(data_batch)
            loss = complex_loss.sum() / 1000.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            pbar.set_description("Epoch: %d/ %d With Loss %f " % (global_epoch, epochs, loss.item()))
            #Record the loss value for this batch.
            losses.append(loss.item())
        #Obtain the mean and variance of the loss value after one iteration.
        ave_loss = np.average(losses)
        var_loss = np.var(losses)
        print("μ In Epoch %d: %s" % (global_epoch, ave_loss))
        print("σ In Epoch %d: %s" % (global_epoch, var_loss))
        print("----------------------------------------------------------------")
        #Record the mean and variance of the loss value.
        csv_cache += "%d,%s,%s\n" % (global_epoch, ave_loss, var_loss)
        #Check whether there are checkpoints exceeding the maximum cache count.
        old_model = checkpoint_folder + "/%s_%d%s" % (
            prefix,
            global_epoch - cache_num * milestone_step,
            suffix
        )
        if os.path.exists(old_model):
            os.remove(old_model)
        if global_epoch % milestone_step == 0:
            #Save checkpoints.
            torch.save(
                model.state_dict(),
                checkpoint_folder + "/%s_%d%s" % (prefix, global_epoch, suffix)
            )
            #Save loss value information.
            with open(csv_str, 'a') as file:
                file.write(csv_cache)
                csv_cache = ""
            #Perform sampling.
            #This method is described in the following section, Sampling Process.
            milestone_sample(global_epoch)
        time.sleep(0.1)


def extract(v, t, x_shape):
    """
    Used to extract some parameters from the pre-cached noise schedule hyperparameters.
    :param v: Cache, shape [1000]
    :param t: Target timestamp to be extracted, shape [batch_size]
    :param x_shape: Shape of a batch of data, typically [batch_size, channels, image_size, image_size]
    :return: Typically returns a shape of [batch_size, 1, 1, 1]
    """
    return torch.gather(v, index=t, dim=0).float().cuda().view([t.shape[0]] + [1] * (len(x_shape) - 1))


def add_noise(x_0, t, noise=None):
    """
    Used to add noise.
    :param x_0:
        Original data used to add noise, with shape [batch_size, channels, image_size, image_size]
    :param t: Specified timestamp, with shape [batch_size]
    :param noise: Specified noise; if None, noise will be randomly generated
    :return: Returns the noisy data at the specified timestamp
    """
    if noise is None:
        noise = torch.randn_like(x_0).cuda()
    #Equation (10)
    return (extract(sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)


def get_loss(x_0):
    """
    To obtain the loss value, the stochastic gradient descent method is used.
    Select any timestamp, calculate the noisy data at that moment, and record the noise used.
    Then, use the noisy data and timestamp as model inputs to obtain the predicted noise.
    Use the recorded noise and predicted noise to calculate the loss value.
    :param x_0: Cache, shape [1000]
    :return: Typically returns a shape of [batch_size, 1, 1, 1]
    """
    t = torch.randint(time_steps, size=(batch_size,)).cuda() + 1
    noise = torch.randn_like(x_0).cuda()
    x_t = add_noise(x_0, t, noise)
    #Equation (49)
    return F.huber_loss(model(x_t, t), noise, reduction='none'), t
```

## Sampling Process
The sampling process is relatively simple, requiring only one iteration. However, for the convenience of subsequent development, it can be written in a modular manner.
The DDPM sampler, i.e., `Equation (50)`:
```python
@torch.no_grad()
def ddpm(xs, timestep, timestep_s):
    """
    :param xs: Corresponding to x_t in equation (50)
    :param timestep: Corresponding to t - 1 in equation (50)
    :param timestep_s: Corresponding to t in equation (50)
    :return: Corresponding to x_{t - 1} in equation (50)
    """
    #α_t = alphas[t - 1]
    alpha = alphas[timestep]
    #β_t = betas[t - 1]
    beta = betas[timestep]
    alpha_bar = alphas_bar[timestep_s]
    alpha_bar_pre = alphas_bar[timestep]
    #Coefficient before x_t.
    cof1 = 1 / alpha.sqrt()
    #Coefficient before predicted nosie.
    cof2 = (1 - alpha) / (alpha * (1 - alpha_bar)).sqrt()
    #Coefficient before random noise.
    var = ((1 - alpha_bar_pre) / (1 - alpha_bar) * beta).sqrt()
    noise = var * torch.randn(size=xs.shape).cuda()
    #The shape of t is [1],
    #and its shape is expanded to [batch_size] to be fed into the model for computation.
    timestep_s = torch.full(size=[xs.shape[0]], dtype=torch.long, fill_value=timestep_s).cuda()
    #Equation (50)
    return cof1 * xs - cof2 * model(xs, timestep_s) + noise
```
Core sampling methods:
```python
def trailing(steps):
    return torch.arange(time_steps, 0, -time_steps / steps).flip(0).round().int().cuda()


def sample(formats, configs, batch=1, noise=None, steps=1000,
           solver=ddpm, time_schedule=trailing, desc=''):
    """
    :param formats: A list of save formats, input as a method. See the save format method for details.  
    :param configs: Saved configurable settings, input as a dictionary.
        See the default save format for details.
    :param batch: The number of images sampled per batch
    :param noise: Initial noise; if None, random noise will be generated
    :param steps: Number of sampling steps;
        currently, this value can only be 1000. Fast sampling will be introduced in future articles
    :param solver: Sampler; currently, this value can only be ddpm.
        Other samplers will be introduced in future articles
    :param time_schedule: Sampling time series; uses the default trailing
    :param desc: Additional comments that can be displayed in the tqdm progress bar
    """
    noise = torch.randn(size=(batch, channels, image_size, image_size)).cuda() \
        if noise is None else noise
    img_list = sample_loop(noise, steps, solver, time_schedule, desc)
    for i, format_ in enumerate(formats):
        format_(img_list, {} if len(configs) - 1 < i or configs[i] is None else configs[i])


def sample_loop(noise, steps, solver, time_schedule, extra_desc=''):
    #First, a list containing the initial noise will be created.
    result = [noise.detach().cpu()]
    #Obtain the sampling timestamp. Based on the current progress, the timestamp is [1, 2, ……, 1000].
    times_s = time_schedule(steps)
    #Obtain the timestamp of the offset, which is [0, 1, ……, 999].
    times_t = torch.cat((torch.tensor((0,)).cuda(), times_s[:-1]))
    #Perform iterative sampling.
    for i in tqdm(
            reversed(range(steps)),
            desc='Now Sampling...' + extra_desc,
            total=steps
    ):
        #Perform a single iteration using the solver.
        noise = solver(noise, times_t[i], times_s[i])
        #Save intermediate states to a list.
        result.append(noise.detach().cpu())
    #Return a list containing all intermediate states.
    return result
```
For all states obtained from sampling from $t = 1000$ to $t = 0$, define a method for storing the sampling results in a grid format:
```python
"""
All available configurations when the format is set to grid.
saving_folder  : Folder where files are saved
name           : File name for saved files
reverse        : Whether to reverse the grid's length and width
padding_pixels : Spacing between images
"""
grid_config = {
    'saving_folder': '.',
    'name': '_grid',
    'reverse': False,
    'padding_pixels': 2,
}


def grid(img_list, config):
    """
    Format method variable used for saving.
    """
    torchvision.io.write_png(
        to_grid(to_rgb(img_list[-1]), config),
        get('saving_folder', config, grid_config) + '/' + get('name', config, grid_config) + '.png'
    )


def to_rgb(tensor):
    """
    Converting tensors to image format.
    """
    return ((tensor + torch.ones_like(tensor)) / 2 * 255).clip(0, 255).to(torch.uint8)


def closest_divisors(num: int):
    """
    Used to calculate length and width, making length and width as close as possible.
    """
    a, b, i = 1, num, 0
    while a < b:
        i += 1
        if num % i == 0:
            a = i
            b = num // a
    return [b, a]


def to_pixel_coord(num, padding_pixels: int):
    """
    Used to calculate pixel coordinates for creating grids.
    """
    return num * image_size + (num + 1) * padding_pixels


def to_grid(imgs, config):
    """
    Convert a batch of image tensors into a single grid tensor form.
    """
    batch = imgs.shape[0]
    x, y = closest_divisors(batch)
    if get('reverse', config, grid_config):
        x, y = y, x
    px = get('padding_pixels', config, grid_config)
    width = to_pixel_coord(x, px)
    height = to_pixel_coord(y, px)
    result = torch.full(size=(imgs.shape[1], height, width), fill_value=1.).to(torch.uint8)
    for j in range(batch):
        h_coord = j // x * (image_size + px) + px
        w_coord = j % x * (image_size + px) + px
        result[:, h_coord:h_coord + image_size, w_coord: w_coord + image_size] = imgs[j]
    return result


def get(par, config, config_):
    """
    Obtain configurable parameter values.
    """
    try:
        return config[par]
    except:
        return config_[par]
```
Then, you can complete the milestone_sample method mentioned in the training process:
```python
def milestone_sample(global_epoch, solver=ddpm):
    milestone_path = checkpoint_folder + "/milestone"
    if not os.path.exists(milestone_path):
        os.mkdir(milestone_path)
    print("Now Sampling Milestone")
    time.sleep(0.1)
    #Call the sampling core method, use grid as the storage format,
    #save the results to the milestone folder, and sample 16 images at a time.
    sample(
        formats=[grid],
        configs=[{
            'saving_folder': checkpoint_folder + "/milestone",
            'name': 'epoch_%d' % global_epoch
        }],
        batch=16,
        solver=solver
    )
    print("----------------------------------------------------------------")
```

## Code Execution
This completes the code implementation of DDPM, and training can now begin:
```python
if __name__ == '__main__':
    train()
```
The first 100 iterations of training process:

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/cifar1_10_100.png"/></p>`

The 500 iterations of the training process:

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/cifar1_50_500.png"/></p>`

Changes in training loss values:

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/cifar1_loss.png" width="400" height="300"/></p>`

## Summary
At this point, we have completed the reproduction of the DDPM algorithm. If you have run the code yourself, you will undoubtedly have felt the significant computational power requirements of diffusion models, which has inspired subsequent research into fast sampling.

Due to `The limitations of the blog framework`, there is still a lot of content that has not been fully explained. I will try to supplement and explain some of the more difficult concepts in future posts, or perhaps write a separate article to explain them in more detail.;;;e;;;c
## 训练过程
首先导入一些必要的库：
```python
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.v2 import Lambda, ToTensor, RandomHorizontalFlip, Compose, Grayscale
from tqdm import tqdm

import unet
```
其次是一些训练配置、超参数的确定。对于$\beta_t$等超参数，`选择线性噪声时间表`：
```python
def linear(time_steps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, time_steps).cuda()


#时间戳选定为从0到1000，若减少或增加该值，会改变噪声时间表相关参数
time_steps = 1000
#噪声时间表相关参数的预先计算缓存
#形状：[1000]
betas = linear(time_steps)
#形状：[1000]
alphas = 1. - betas
#形状：[1001]
alphas_bar = torch.cat((torch.tensor((1,)).cuda(), torch.cumprod(input=alphas, dim=0)))
#形状：[1001]
sqrt_alphas_bar = torch.sqrt(alphas_bar)
#形状：[1001]
sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
```
配置与超参数：
```python
"""
为简便演示，使用CIFAR1作为数据集，图片尺寸为32×32
CIFAR1指仅仅选用CIFAR10中的一类作为数据集
数据集的文件结构为
cifar1
├── test
│   └── horse
│       ├── 0001.png
│       ├── 0002.png
│       ├── ...
│       └── 5000.png
└── train
    └── horse
        ├── 0001.png
        ├── 0002.png
        ├── ...
        └── 1000.png
"""
image_size = 32
#通道数为3，即RGB
channels = 3
#模型的基础通道数，即最上层的通道数
base_channels = 128
#模型的不同层的通道数，分辨率为32×32时，模型的深度选则4，对应的通道数为128，256，256，256
ch_mults = {
    32: (1, 2, 2, 2),
    64: (1, 2, 3, 4),
    128: (1, 1, 2, 3, 4),
    256: (1, 1, 2, 2, 4, 4),
    512: (0.5, 1, 1, 2, 2, 4, 4)
}
#学习率
learning_rate = 1e-4
#训练轮数，选定迭代数据集500次（实际上该轮数是远远不够的）
epochs = 500
#每多少轮数进行以此采样以及模型的检查点保存
milestone_step = 10
#检查点保存的最大数量
cache_num = 10
#一个批次训练图片的数量，在该案例下，128张适合拥有8G显存的显卡
batch_size = 128
#checkpoints的拟定位置
checkpoint_folder = 'cifar1/cp'
#训练集的位置
datasets_folder = 'cifar1/train'
#用以收集训练时损失值的文件
csv_str = checkpoint_folder + '/loss.csv'
#模型检查点保存文件名的前缀
prefix = "checkpoint_epoch"
#模型检查点保存文件名的后缀
suffix = ".pth"
#为checkpoints新建文件夹
Path(checkpoint_folder).mkdir(exist_ok=True)
Path(checkpoint_folder + '/milestone').mkdir(exist_ok=True)
#新建模型
model = unet.UNetModel(
    image_size=image_size,
    in_channels=channels,
    model_channels=base_channels,
    out_channels=channels,
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    dropout=0.0,
    channel_mult=ch_mults[image_size],
    num_heads=4,
    num_head_channels=-1,
)
#若checkpoints文件夹内存在模型检查点，则读取最新的检查点
last_model = None
last = 0
for file in os.listdir(checkpoint_folder):
    if file[:len(prefix)] != prefix:
        continue
    epochs_str = file[len(prefix) + 1:0 - len(suffix)]
    if int(epochs_str) > last:
        last = int(epochs_str)
if last > 0:
    last_model = torch.load(checkpoint_folder + "/%s_%d%s" % (prefix, last, suffix), weights_only=False)
if last_model is not None:
    model.load_state_dict(last_model)
else:
    print("No Checkpoints Exist")
model = model.cuda()
#统计模型的参数数量
parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
count_str = "Parameters Count: "
print(count_str + "%.2fB" % (parameters / 1e9)) \
    if parameters > 1e9 else print(count_str + "%.2fM" % (parameters / 1e6))
#优化器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#将图片转化为张量
"""
ToTensor()                    :读取图片，并将其从[0, 255]的整数缩放为[0, 1]的浮点数
Lambda(lambda t: (t * 2) - 1) :将[0, 1]的浮点数缩放为[-1, 1]的浮点数
RandomHorizontalFlip()        :对数据进行随机的水平方向的翻转
"""
transform = Compose(
    [Grayscale(), ToTensor(), Lambda(lambda t: (t * 2) - 1), RandomHorizontalFlip()]
    if channels == 1 else
    [ToTensor(), Lambda(lambda t: (t * 2) - 1), RandomHorizontalFlip()]
)
```
训练的核心方法：
```python
def train():
    train_loop(DataLoader(
        datasets.ImageFolder(datasets_folder, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    ))


def train_loop(dataloader):
    #计算剩余需要迭代的次数
    local_epochs = epochs - last
    #损失值文件的写入缓存
    csv_cache = ""
    print()
    print("Now Strat Training")
    for epoch in range(local_epochs):
        #损失值的缓存
        losses = []
        #若不等待，貌似tqdm的进度条会出现一些问题？
        time.sleep(0.1)
        #计算当前的迭代次数
        global_epoch = epoch + last + 1
        #对数据集进行遍历
        for step, data_batch in (pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc="Epoch: %d/ %d With Loss %f" % (global_epoch, epochs, 1)
        )):
            #初始化优化器
            optimizer.zero_grad()
            #获取该批次的全部数据张量，其形状为[batch_size, channels, image_size, image_size]
            data_batch = data_batch[0].cuda()
            #获取损失值以及计算损失值所随机到的时间戳
            complex_loss, t = get_loss(data_batch)
            loss = complex_loss.sum() / 1000.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            pbar.set_description("Epoch: %d/ %d With Loss %f " % (global_epoch, epochs, loss.item()))
            #记录该批次的损失值
            losses.append(loss.item())
        #获取一轮迭代后的损失值的平均值及方差
        ave_loss = np.average(losses)
        var_loss = np.var(losses)
        print("μ In Epoch %d: %s" % (global_epoch, ave_loss))
        print("σ In Epoch %d: %s" % (global_epoch, var_loss))
        print("----------------------------------------------------------------")
        #记录损失值的平均值及方差
        csv_cache += "%d,%s,%s\n" % (global_epoch, ave_loss, var_loss)
        #检查是否存在超出最大缓存数量的检查点
        old_model = checkpoint_folder + "/%s_%d%s" % (
            prefix,
            global_epoch - cache_num * milestone_step,
            suffix
        )
        if os.path.exists(old_model):
            os.remove(old_model)
        if global_epoch % milestone_step == 0:
            #保存检查点
            torch.save(
                model.state_dict(),
                checkpoint_folder + "/%s_%d%s" % (prefix, global_epoch, suffix)
            )
            #保存损失值信息
            with open(csv_str, 'a') as file:
                file.write(csv_cache)
                csv_cache = ""
            #进行一次采样，该方法在接下来的章节——采样过程中给出
            milestone_sample(global_epoch)
        time.sleep(0.1)


def extract(v, t, x_shape):
    """
    用于将一些参数从噪声时间表超参数的预先缓存中提取出来。
    :param v:缓存，形状为[1000]
    :param t:需要提取的目标时间戳，形状为[batch_size]
    :param x_shape:一个批次数据的形状，一般为[batch_size, channels, image_size, image_size]
    :return:一般返回的形状为[batch_size, 1, 1, 1]
    """
    return torch.gather(v, index=t, dim=0).float().cuda().view([t.shape[0]] + [1] * (len(x_shape) - 1))


def add_noise(x_0, t, noise=None):
    """
    用于添加噪声。
    :param x_0:用于添加噪声的原始数据，形状为[batch_size, channels, image_size, image_size]
    :param t:指定的时间戳，形状为[batch_size]
    :param noise:指定的噪声，若为None，则会随机生成噪声
    :return:返回指定时间戳下的带噪数据
    """
    if noise is None:
        noise = torch.randn_like(x_0).cuda()
    #式(10)
    return (extract(sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)


def get_loss(x_0):
    """
    用于获取损失值，采用随机梯度下降法。
    选取任意的时间戳，计算该时刻的带噪数据，并记录所使用的噪声，
    再将带噪数据与时间戳作为模型输入，得到预测的噪声，
    使用记录的噪声与预测的噪声来计算损失值。
    :param x_0:缓存，形状为[1000]
    :return:一般返回的形状为[batch_size, 1, 1, 1]
    """
    t = torch.randint(time_steps, size=(batch_size,)).cuda() + 1
    noise = torch.randn_like(x_0).cuda()
    x_t = add_noise(x_0, t, noise)
    #式(49)
    return F.huber_loss(model(x_t, t), noise, reduction='none'), t
```

## 采样过程
采样过程相较而言就简单很多，只需要进行一个迭代即可。但为方便后续开发，可以对其进行模块化编写。
DDPM的采样器，即`式(50)`：
```python
@torch.no_grad()
def ddpm(xs, timestep, timestep_s):
    """
    :param xs:对应式(50)中的x_t
    :param timestep:对应式(50)中的t - 1
    :param timestep_s:对应式(50)中的t
    :return:对应式(50)中的x_{t - 1}
    """
    #α_t = alphas[t - 1]
    alpha = alphas[timestep]
    #β_t = betas[t - 1]
    beta = betas[timestep]
    alpha_bar = alphas_bar[timestep_s]
    alpha_bar_pre = alphas_bar[timestep]
    #x_t前的系数
    cof1 = 1 / alpha.sqrt()
    #预测噪声前的系数
    cof2 = (1 - alpha) / (alpha * (1 - alpha_bar)).sqrt()
    #随机噪声前的系数
    var = ((1 - alpha_bar_pre) / (1 - alpha_bar) * beta).sqrt()
    noise = var * torch.randn(size=xs.shape).cuda()
    #t的形状为[1]，将其形状展开为[batch_size]以放入模型中进行计算
    timestep_s = torch.full(size=[xs.shape[0]], dtype=torch.long, fill_value=timestep_s).cuda()
    #式(50)
    return cof1 * xs - cof2 * model(xs, timestep_s) + noise
```
采样的核心方法：
```python
def trailing(steps):
    return torch.arange(time_steps, 0, -time_steps / steps).flip(0).round().int().cuda()


def sample(formats, configs, batch=1, noise=None, steps=1000,
           solver=ddpm, time_schedule=trailing, desc=''):
    """
    :param formats:一系列保存格式，输入为方法，详见保存格式方法
    :param configs:保存的可修改配置，输入为词典。详见默认保存格式
    :param batch:一批次采样的图片数量
    :param noise:初始的噪声，如果为None，会生成随机噪声
    :param steps:采样步数，以目前的进度而言，该值只能等于1000，快速采样将在以后的文章中介绍
    :param solver:采样器，以目前的进度而言，该值只能等于ddpm，其他采样器将在以后的文章中介绍
    :param time_schedule:采样时间序列，使用默认的trailing
    :param desc:可在tqdm进度条中显示额外的注释说明
    """
    noise = torch.randn(size=(batch, channels, image_size, image_size)).cuda() \
        if noise is None else noise
    img_list = sample_loop(noise, steps, solver, time_schedule, desc)
    for i, format_ in enumerate(formats):
        format_(img_list, {} if len(configs) - 1 < i or configs[i] is None else configs[i])


def sample_loop(noise, steps, solver, time_schedule, extra_desc=''):
    #首先会创建一个列表，其中包含了初始噪声
    result = [noise.detach().cpu()]
    #获取采样时间戳，以目前的进度而言，该时间戳为[1, 2, ……, 1000]
    times_s = time_schedule(steps)
    #获取偏移的时间戳，该时间戳为[0, 1, ……, 999]
    times_t = torch.cat((torch.tensor((0,)).cuda(), times_s[:-1]))
    #进行迭代采样
    for i in tqdm(
            reversed(range(steps)),
            desc='Now Sampling...' + extra_desc,
            total=steps
    ):
        #使用求解器进行单次迭代
        noise = solver(noise, times_t[i], times_s[i])
        #将中间状态保存到列表中
        result.append(noise.detach().cpu())
    #返回带有整个中间状态的列表
    return result
```
对于采样获得的从$t = 1000$到$t = 0$时刻全部状态，定义一个以网格形式保存采样结果的方法：
```python
"""
format为grid时可使用的全部配置
saving_folder  :保存的文件夹
name           :保存的文件名
reverse        :是否翻转网格长和宽
padding_pixels :图片与图片之间的间隔
"""
grid_config = {
    'saving_folder': '.',
    'name': '_grid',
    'reverse': False,
    'padding_pixels': 2,
}


def grid(img_list, config):
    """
    用于保存的format方法变量
    """
    torchvision.io.write_png(
        to_grid(to_rgb(img_list[-1]), config),
        get('saving_folder', config, grid_config) + '/' + get('name', config, grid_config) + '.png'
    )


def to_rgb(tensor):
    """
    将张量变为图片格式
    """
    return ((tensor + torch.ones_like(tensor)) / 2 * 255).clip(0, 255).to(torch.uint8)


def closest_divisors(num: int):
    """
    用于计算长和宽，使长和宽尽可能接近
    """
    a, b, i = 1, num, 0
    while a < b:
        i += 1
        if num % i == 0:
            a = i
            b = num // a
    return [b, a]


def to_pixel_coord(num, padding_pixels: int):
    """
    用于计算像素点坐标，以制作网格
    """
    return num * image_size + (num + 1) * padding_pixels


def to_grid(imgs, config):
    """
    将一批图片张量转化为单个的网格张量形式
    """
    batch = imgs.shape[0]
    x, y = closest_divisors(batch)
    if get('reverse', config, grid_config):
        x, y = y, x
    px = get('padding_pixels', config, grid_config)
    width = to_pixel_coord(x, px)
    height = to_pixel_coord(y, px)
    result = torch.full(size=(imgs.shape[1], height, width), fill_value=1.).to(torch.uint8)
    for j in range(batch):
        h_coord = j // x * (image_size + px) + px
        w_coord = j % x * (image_size + px) + px
        result[:, h_coord:h_coord + image_size, w_coord: w_coord + image_size] = imgs[j]
    return result


def get(par, config, config_):
    """
    获取可配置的参数值
    """
    try:
        return config[par]
    except:
        return config_[par]
```
进而，便可完成训练过程中所提到的milestone_sample方法：
```python
def milestone_sample(global_epoch, solver=ddpm):
    milestone_path = checkpoint_folder + "/milestone"
    if not os.path.exists(milestone_path):
        os.mkdir(milestone_path)
    print("Now Sampling Milestone")
    time.sleep(0.1)
    #调用采样核心方法，使用grid作为保存格式，将结果保存到milestone文件夹，一次采样16张图片
    sample(
        formats=[grid],
        configs=[{
            'saving_folder': checkpoint_folder + "/milestone",
            'name': 'epoch_%d' % global_epoch
        }],
        batch=16,
        solver=solver
    )
    print("----------------------------------------------------------------")
```

## 代码运行
这样一来，便完成了DDPM的代码实现，即可进行训练：
```python
if __name__ == '__main__':
    train()
```
训练的前100次迭代过程：

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/cifar1_10_100.png"/></p>`

训练的500次迭代过程：

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/cifar1_50_500.png"/></p>`

训练的损失值变化：

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/cifar1_loss.png" width="400" height="300"/></p>`

## 总结
至此，便完成了DDPM算法上的复现，若是亲自运行代码了，必然会感受到扩散模型对算力的需求之大，也才激发了人们后续关于快速采样的研究。

由于`博客框架的限制`，还有很多内容没有能够讲明白，一些理解较为困难的地方后续我也会尽量补充解释，或是另开一篇文章进行讲解。;;;c
