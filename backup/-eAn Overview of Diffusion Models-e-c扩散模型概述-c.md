> [!NOTE]
> This article currently only supports Chinese.

> [!CAUTION]
> 施工中！

<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->
\denotes
####生成式模型::在实际运用中，依据用户的引导性输入或是不依靠输入，就可以生成出一系列数据的模型（这些数据往往在现实中不存在）####
####变分自编码器::简称VAE####
####生成式对抗模型::GANs####
####泛性::不拘泥于数据集，能够####
####质量::看起来和真的一样####
####Deep Unsupervised Learning using Nonequilibrium Thermodynamics::文章地址：https://arxiv.org/abs/1503.03585 ####
####Denoising Diffusion Probabilistic Models(DDPM)::文章地址：https://arxiv.org/abs/2006.11239 ####
####一定比例::又称噪声时间表####
####预定的步数::在实践中，通常选择1000步####
####重参数技巧::一种数学计算技巧，在DDPM中将详细讨论####
####随机梯度下降法::SDG####
####时至今日::直到写到这里时的2025年07月####
\denotes
## 生成式模型
扩散模型是一种`生成式模型`，它最初只运用在了图片生成上，但如今，它已经有了相当广阔的运用空间。在扩散模型流行之前，主流的图片的生成式模型还有`变分自编码器`以及`生成式对抗模型`，与它们相比，扩散模型虽然牺牲了采样速度，提升了采样结果的`泛性`与`质量`。

`Gmeek-html<img src="https://OmnisyR.github.io/figs/generative_models.png">`
\eSource:\e\c图片来源：\c[Diffusion Models vs. GANs vs. VAEs: Comparison of Deep Generative Models](https://pub.towardsai.net/diffusion-models-vs-gans-vs-vaes-comparison-of-deep-generative-models-67ab93e0d9ae)

## 扩散模型
在2015年，一篇名为`Deep Unsupervised Learning using Nonequilibrium Thermodynamics`的文章首次将扩散模型带入了人们的视野中。但由于当时硬件受限等原因，直到2020年`Denoising Diffusion Probabilistic Models(DDPM)`发布后，人们才开始广泛地使用扩散模型来进行图片生成。

### 扩散模型的两个过程
扩散模型的训练的过程通常被称为正向过程（或增噪过程），其采样的过程通常被称为逆向过程（或去噪过程）。

#### 正向过程
`Gmeek-html<img src="https://OmnisyR.github.io/figs/forward.png">`
扩散模型的正向过程旨在一步步地为原始数据集的图片，以`一定比例`添加标准高斯噪声，直至`预定的步数`结束，使得该图片成为完全高斯噪声，但实际上，正向过程并不需要一步步地进行，通过`重参数技巧`可以将任意步数简化为一步完成。扩散模型在训练中，则是使用了`随机梯度下降法`，随机选取某一步数来进行模型的梯度下降，进行噪声的学习，以便进行逆向过程的去噪工作。

#### 逆向过程
`Gmeek-html<img src="https://OmnisyR.github.io/figs/backward.png">`
扩散模型的逆向过程旨在随机生成一个标准高斯噪声，使用预训练好的模型，预测出所添加的噪声，再进行一步步地去噪，直至预定的步数结束，使得该标准高斯分布称为近似训练集的图片。和正向过程不同的是，逆向过程的这些步骤不能简单地一次进行，但有相当多的人找到了快速采样的方法，从而直接有效弥补了扩散模型耗费算力大这一缺点。`时至今日`，扩散模型的快速采样也仍然是一个热门课题。

### 连续型扩散模型
上述的思想中，很容易能够看出无论是正向过程，还是逆向过程，扩散模型的离散的。后续人们利用随机微分方程对扩散模型进行了连续性的数学建模，使得能够更好地从数学层面改进扩散模型，无论是正向过程还是逆向过程都能够使用函数进行连续性表达，只不过计算机无法处理连续数据，在实践中，仍然会将连续函数以离散的形式进行编程。

## 评估方法
### Inception Score

### FID
