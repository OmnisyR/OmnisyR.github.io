> [!NOTE]
> This article currently only supports Chinese.

> [!CAUTION]
> 施工中！

<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->

;;;a
;;;;UE Viewer::[下载界面](https://www.gildor.org/en/projects/umodel#files);;;;
;;;a
## 写在前面
GGST中最吸引我的武器莫过于Unika的AC/43了。在我看来，它的设计理念特别“浪漫”，于是我便有了“为什么我不自己动手做个呢？”的想法。网上也有不少关于AC/43的3D打印成品，但有个问题，那就是这些成品不会动，换句话说网上的那些只是一个单纯的模型罢了，完全不浪漫，很是遗憾。

在这篇文章中，我将记录下我的关于制作AC/43的全部尝试以及成果，其中必然包含了一些对游戏进行解包的操作，还请希望Arc System高抬贵手，不要追究。

## 素材准备
首先是素材部分，由于游戏中有着现成的模型和设计图可以参考，因此只需要进行解包获取源文件用来参考即可。
对于虚幻引擎，解包使用`UE Viewer`。使用该软件打开GGST的目录后，会提示需要密钥，当然，该密钥也是很早就被网友发出来了：
```
0x3D96F3E41ED4B90B6C96CA3B2393F8911A5F6A48FE71F54B495E8F1AFD94CD73
```
所需要的一些素材在下面这些位置中：
```
All packages
├── Engine
├── Plugins
└── RED
    ├── Config
    ├── Content
    │   ├── ...
    │   ├── Chara
    │   │   ├── ...
    │   │   ├── UNI
    │   │   │   ├── Common
    │   │   │   └── Custume01
    │   │   │       ├── ...
    │   │   │       ├── Material    <- 材质文件夹
    │   │   │       └── Mesh        <- 模型文件夹
    │   │   └── ...
    │   ├── ...
    │   └── UI
    │       ├── ...
    │       ├── Gallery
    │       │   └── tex
    │       │       ├── Images      <- 设定集文件夹
    │       │       └── Thumbnails
    │       └── ...
    └── Plugins
```
双击资产文件即可进行预览，虚幻引擎为4.25版本：

;;;e`Gmeek-html<img src="https://OmnisyR.github.io/figs/AC43_EN.png">`;;;e;;;c`Gmeek-html<img src="https://OmnisyR.github.io/figs/AC43_ZH.png">`;;;c

在预览界面的Tools中可以导出当前文件，或是在文件浏览界面右击文件夹来进行批量导出。

## 模型预览
在模型和材质等素材导出后便可先使用Blender进行预览，使用Unreal PSK/PSA插件。通过该插件导入Mesh文件夹下的uni_weaponsp.pskx，为了保证单位一致，可在导入的时候选择缩放0.01。该文件为AC/43展开的模型，它能够为后续的零件建模提供更好的参考价值。

`Gmeek-html<p align="center"><img src="https://OmnisyR.github.io/figs/preview.png" height=200> <img src="https://OmnisyR.github.io/figs/3view.png" height=200>`
