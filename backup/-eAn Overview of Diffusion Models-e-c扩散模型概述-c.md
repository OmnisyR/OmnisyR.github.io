> [!NOTE]
> This article currently only supports Chinese.

<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/GmeekTOC.js'></script>"}## -->

## 生成式模型
`扩散模型`是一种`生成式模型`，它最初只运用在了图片生成上，但如今，它已经有了相当广阔的运用空间。在扩散模型流行之前，主流的图片的生成式模型还有`变分自编码器`以及`生成式对抗模型`，与它们相比，扩散模型虽然牺牲了采样速度，提升了采样结果的`泛性`与`质量`。

`Gmeek-html<img src="https://OmnisyR.github.io/figs/generative models.png">`
\eSource:\e\c图片来源：\c[Diffusion Models vs. GANs vs. VAEs: Comparison of Deep Generative Models](https://pub.towardsai.net/diffusion-models-vs-gans-vs-vaes-comparison-of-deep-generative-models-67ab93e0d9ae)

## 扩散模型
在2015年，一篇名为`Deep Unsupervised Learning using Nonequilibrium Thermodynamics`的文章首次将扩散模型带入了人们的视野中。但由于当时硬件受限等原因，直到2020年`Denoising Diffusion Probabilistic Models(DDPM)`发布后，人们才开始广泛地使用扩散模型来进行图片生成。

扩散模型的训练的过程通常被称为正向过程（或增噪过程），其采样的过程通常被称为逆向过程（或去噪过程）。

### 正向过程
`Gmeek-html<img src="https://OmnisyR.github.io/figs/forward.png">`
扩散模型的正向过程旨在一步步地为原始数据集的图片，以`一定比例`添加标准高斯噪声，直至预定的步数结束，使得该图片成为完全高斯噪声。

### 逆向过程
`Gmeek-html<img src="https://OmnisyR.github.io/figs/backward.png">`
扩散模型的逆向过程旨在随机生成一个标准高斯噪声，使用预训练好的模型，进行一步步地降噪，直至预定的步数结束，使得该标准高斯分布称为近似训练集的图片。

## 评估方法
