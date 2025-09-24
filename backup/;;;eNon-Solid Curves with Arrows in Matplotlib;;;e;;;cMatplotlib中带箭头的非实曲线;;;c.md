<!-- ##{"script":"<script src='https://OmnisyR.github.io/assets/HyperTOC.js'></script>"}## -->
;;;a
;;;;;;;eNote::note;;;e;;;c注释::注释;;;c;;;;
;;;a;;;e
## Introduction
This article will explain how to use Matplotlib to plot non-solid curves with arrows, as shown in the figure below:

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_example.png" width="400" height="400"/></p>`

Matplotlib offers a rich set of built-in methods, yet it lacks a built-in function for drawing non-solid curves with arrows. While some online discussions exist on this topic, they haven't provided satisfactory solutions. Therefore, I conducted my own investigation into plotting such curves, aiming to create flawless, perfect non-solid curves with arrows.

## Methods Provided by Matplotlib and Related Issues
In Matplotlib, you can use FancyArrowPatch to create a curve with an arrowhead:
```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.add_patch(patches.FancyArrowPatch(
    posA=(0.1, 0.1), posB=(1.8, 1.8), arrowstyle='-|>, head_width=4, head_length=8',
    connectionstyle='arc3,rad=0.3', color='#515151', shrinkA=0, shrinkB=0
))
plt.savefig('arrow_common.png', dpi=600)
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_common.png" width="400" height="400"/></p>`

However, if you want to render this curve as a non-solid line by simply changing its style, problems will arise:
```python
# ...
ax.add_patch(patches.FancyArrowPatch(
  # ...
  linestyle='--'
)
# ...
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_error.png" width="400" height="400"/></p>`

As can be seen, the arrow points are also drawn using dashed lines. Moreover, when using this method for plotting, the curvature of the curve is interface-dependent rather than coordinate-dependent. This means that when altering the x-axis or y-axis range of the chart, the curvature of the curve will still maintain visual consistency. In certain situations, this cannot be considered a rigorous plotting method.

## Method
By emulating the drawing method for non-solid lines with arrows, it can be treated as two components: the non-solid curve and the solid arrowhead.

### Non-Solid Curve
Matplotlib implements curve drawing using quadratic Bézier curves. Examining the source code reveals that given points A and B, and an angle rad in radians, the coordinates of control point C can be determined using the following formula:

$$
x_C = \frac{x_A + x_B}{2} + rad · (y_B - y_A)
$$
$$
y_C = \frac{y_A + y_B}{2} - rad · (x_B - x_A)
$$

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3.png" width="400" height="400"/></p>`

CA is tangent to the curve at point A, and CB is tangent to the curve at point B. This allows the non-solid curve to be drawn using the code below:
```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def draw_arc3(from_, to_, rad, detail=False, color='#515151'):
    x1, y1 = np.array(from_)
    x2, y2 = np.array(to_)
    dx, dy = x2 - x1, y2 - y1
    x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
    cx, cy = x12 + rad * dy, y12 - rad * dx
    control_ = (cx, cy)
    vertices = [from_, control_, to_]
    codes = [patches.Path.MOVETO, patches.Path.CURVE3, patches.Path.CURVE3]
    path = patches.Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, linestyle='--', linewidth=1)
    ax.add_patch(patch)
    if not detail:
        return
    ax.scatter(control_[0], control_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(from_[0], from_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(to_[0], to_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.plot((control_[0], from_[0]), (control_[1], from_[1]), color=color, linewidth=.5, linestyle=':')
    ax.plot((control_[0], to_[0]), (control_[1], to_[1]), color=color, linewidth=.5, linestyle=':')
    ax.annotate(r'$A$', from_, ha='center', va='bottom')
    ax.annotate(r'$B$', to_, ha='center', va='bottom')
    ax.annotate(r'$C$', control_, ha='right', va='bottom')

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
draw_arc3((0.1, 0.1), (1.8, 1.8), 0.3)
plt.savefig('arrow_arc3_dot.png', dpi=600)
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3_dot.png" width="400" height="400"/></p>`

### Solid Line Arrow
With control point C and target endpoint B already defined, drawing a solid line arrow is straightforward. The following code easily creates an arrow that meets the requirements:
```python
# ...
ax.annotate("", to_, xytext=control_, arrowprops=dict(
    linewidth=0, arrowstyle="-|>, head_width=0.3, head_length=0.6",
    shrinkA=0, shrinkB=0, facecolor=color, linestyle="solid", mutation_scale=10
))
# ...
```

However, upon closer inspection, issues remain apparent: as shown in the figure below, the width at the very tip of the arrow is narrower than the curve's width, resulting in an aesthetically displeasing visual effect.
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3_arrow_error.png" width="400" height="400"/></p>`

To address this, consider placing a mask between the curve and the arrow to conceal the excess curve portion:
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3_arrow_mask.png" width="400" height="400"/></p>`

The placement of masks can be achieved through a simple algorithm:
```python
# ...
# Place the non-solid curve on layer -2
patch = patches.PathPatch(# ...
      zorder=-2)
# ...
size = 0.08
direction = -1 if control_[0] < to_[0] else 1
mask = patches.FancyBboxPatch(
     (to_[0], to_[1] - size / 2), direction * size, size, boxstyle="square, pad=0",
     ec='red', fc='red', linewidth=1, zorder=-1
)
de = math.degrees(math.atan((control_[1] - to_[1]) / (control_[0] - to_[0])))
tf = transforms.Affine2D().rotate_deg_around(to_[0], to_[1], de) + ax.transData
cut.set_transform(tf)
ax.add_patch(cut)
# ...
```

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_direction.png" width="400" height="400"/></p>`
This allows for precise placement of the mask. The center point in the image is point B, the target endpoint, with surrounding areas indicating the directions of potential control points.

### Non-Solid Curves with Arrows
Finally, by integrating these two components and specifying certain parameters for control, one can complete the drawing of non-solid curves with arrows.
```python
import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

def add_arrow(from_, to_, rad=0.3, control_=None, color='#515151',
              line='--', head_length=0.6, size=0.04, detail=False):
    """
    :param from_:Starting point
    :param to_:Target endpoint
    :param rad:Curve radius
    :param control_:Control points;
                    if None, calculated based on the curvature of the curve.
    :param color:Drawing colors
    :param line:Curve Style
    :param head_length:Arrow size
    :param size:Mask Size
    :param detail:Select whether to draw details,
                  and manually adjust the mask size by drawing details.
    """
    if control_ is None:
        x1, y1 = np.array(from_)
        x2, y2 = np.array(to_)
        dx, dy = x2 - x1, y2 - y1
        x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
        cx, cy = x12 + rad * dy, y12 - rad * dx
        control_ = (cx, cy)
    vertices = [from_, control_, to_]
    codes = [patches.Path.MOVETO, patches.Path.CURVE3, patches.Path.CURVE3]
    path = patches.Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                              linestyle=line, linewidth=1, zorder=-2)
    ax.add_patch(patch)
    direction = -1 if control_[0] < to_[0] else 1
    mask_c = 'red' if detail else 'white'
    mask = patches.FancyBboxPatch(
        (to_[0], to_[1] - size / 2), direction * size, size, boxstyle="square, pad=0",
        ec=mask_c, fc=mask_c, zorder=-1, linewidth=1)
    de = math.degrees(math.atan((control_[1] - to_[1]) / (control_[0] - to_[0])))
    tf = transforms.Affine2D().rotate_deg_around(to_[0], to_[1], de) + ax.transData
    mask.set_transform(tf)
    ax.add_patch(mask)
    ax.annotate("", to_, xytext=control_, arrowprops=dict(
        linewidth=0,
        arrowstyle="-|>, head_width=%f, head_length=%f" % (head_length / 2, head_length),
        shrinkA=0, shrinkB=0, facecolor=color, linestyle="solid", mutation_scale=10
    ))
    if not detail:
        return
    ax.scatter(control_[0], control_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(from_[0], from_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(to_[0], to_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.plot((control_[0], from_[0]), (control_[1], from_[1]), color=color, linewidth=.5, linestyle=':')
    ax.plot((control_[0], to_[0]), (control_[1], to_[1]), color=color, linewidth=.5, linestyle=':')

def draw_eg():
    debug = False
    add_arrow((0.1, 0.1), (1.5, 1.6), rad=0.1, color=colors[1], detail=debug)
    add_arrow((0.4, 1.7), (1.4, 0.7), rad=0.8, color=colors[2], detail=debug)
    add_arrow((1.9, 1.9), (1.2, 0.1), rad=-0.2, color=colors[3], detail=debug)
    add_arrow((1.7, 0.7), (0.3, 1.8), rad=0.5, color=colors[4], detail=debug)
    plt.savefig('arrow_example%s.png' % ('_detail' if debug else ''), dpi=600)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
colors = ['#515151', '#CC9900', '#B177DE', '#37AD6B', '#1A6FDF']
draw_eg()
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_example_cat.png" width="800" height="400"/></p>`
;;;e;;;c
## 介绍
本文将介绍如何使用Matplotlib来绘制带箭头的非实曲线，就像下图一样：

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_example.png" width="400" height="400"/></p>`

Matplotlib内置的方法很丰富，但它却并未内置带箭头的非实曲线的方法，网络上也有一些相关讨论，但并未给出很好的解答。因此，我便对这种曲线的绘制方法进行了一番探究，试图画出没有缺陷的、完美的带箭头的非实曲线。

## Matplotlib中提供的方法以及问题
在Matplotlib中，可以使用FancyArrowPatch来创建一个带箭头的曲线：
```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.add_patch(patches.FancyArrowPatch(
    posA=(0.1, 0.1), posB=(1.8, 1.8), arrowstyle='-|>, head_width=4, head_length=8',
    connectionstyle='arc3,rad=0.3', color='#515151', shrinkA=0, shrinkB=0
))
plt.savefig('arrow_common.png', dpi=600)
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_common.png" width="400" height="400"/></p>`

但是，若是想要让该曲线渲染为非实线，选择简单地改变该曲线的样式，则会出现问题：
```python
# ...
ax.add_patch(patches.FancyArrowPatch(
  # ...
  linestyle='--'
)
# ...
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_error.png" width="400" height="400"/></p>`

可以看到，箭头处也被使用虚线进行了绘图。并且，通过这种方法来进行绘图，曲线弯曲的形状是界面相关而非坐标相关的，也就是说，当改变图表的x轴或y轴取值范围时，该曲线的弯曲，仍然会保持曲率在视觉上的一致性，这在某些情况下并不称得上是一个严谨的绘图方式。

## 方法
通过模仿带箭头的非实直线的绘制方法，可以将其视为两个部分：非实曲线以及实线箭头。

### 非实曲线
Matplotlib是使用了二次贝塞尔曲线实现的曲线绘制。通过查看源码，可以知道，通过给定的点A和B，对于弧度rad，通过下式能够确定控制点点C的坐标：

$$
x_C = \frac{x_A + x_B}{2} + rad · (y_B - y_A)
$$
$$
y_C = \frac{y_A + y_B}{2} - rad · (x_B - x_A)
$$

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3.png" width="400" height="400"/></p>`

CA与曲线相切于点A，CB与曲线相切于点B。这样一来，便可以通过下方代码画出非实曲线：
```python
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def draw_arc3(from_, to_, rad, detail=False, color='#515151'):
    x1, y1 = np.array(from_)
    x2, y2 = np.array(to_)
    dx, dy = x2 - x1, y2 - y1
    x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
    cx, cy = x12 + rad * dy, y12 - rad * dx
    control_ = (cx, cy)
    vertices = [from_, control_, to_]
    codes = [patches.Path.MOVETO, patches.Path.CURVE3, patches.Path.CURVE3]
    path = patches.Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, linestyle='--', linewidth=1)
    ax.add_patch(patch)
    if not detail:
        return
    ax.scatter(control_[0], control_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(from_[0], from_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(to_[0], to_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.plot((control_[0], from_[0]), (control_[1], from_[1]), color=color, linewidth=.5, linestyle=':')
    ax.plot((control_[0], to_[0]), (control_[1], to_[1]), color=color, linewidth=.5, linestyle=':')
    ax.annotate(r'$A$', from_, ha='center', va='bottom')
    ax.annotate(r'$B$', to_, ha='center', va='bottom')
    ax.annotate(r'$C$', control_, ha='right', va='bottom')

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
draw_arc3((0.1, 0.1), (1.8, 1.8), 0.3)
plt.savefig('arrow_arc3_dot.png', dpi=600)
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3_dot.png" width="400" height="400"/></p>`

### 实线箭头
已经有了控制点点C和目标终点点B，画出一个实线箭头并不是一个难事，通过下面的代码很容易便可画出一个满足需求的箭头：
```python
# ...
ax.annotate("", to_, xytext=control_, arrowprops=dict(
    linewidth=0, arrowstyle="-|>, head_width=0.3, head_length=0.6",
    shrinkA=0, shrinkB=0, facecolor=color, linestyle="solid", mutation_scale=10
))
# ...
```

但是，通过细看，仍然发现这存在着问题：如下图所示，箭头最前端的宽度小于曲线的宽度，会导致视觉上的不美观。
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3_arrow_error.png" width="400" height="400"/></p>`

对此，可以考虑在曲线和箭头中间，放置一个蒙版来遮住多余的曲线部分：
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_arc3_arrow_mask.png" width="400" height="400"/></p>`

对于蒙版摆放的位置，可通过一个简单的算法来实现：
```python
# ...
# 将非实曲线置于-2图层
patch = patches.PathPatch(# ...
      zorder=-2)
# ...
size = 0.08
direction = -1 if control_[0] < to_[0] else 1
mask = patches.FancyBboxPatch(
     (to_[0], to_[1] - size / 2), direction * size, size, boxstyle="square, pad=0",
     ec='red', fc='red', linewidth=1, zorder=-1
)
de = math.degrees(math.atan((control_[1] - to_[1]) / (control_[0] - to_[0])))
tf = transforms.Affine2D().rotate_deg_around(to_[0], to_[1], de) + ax.transData
cut.set_transform(tf)
ax.add_patch(cut)
# ...
```

`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_direction.png" width="400" height="400"/></p>`
这样一来，便可准确无误地放置蒙版，图中中间位置为点B，即目标终点，四周为各个可能的控制点方向。

### 带箭头的非实曲线
最终，将这两个部分进行整合，并且给定一些参数以进行控制，便可完成带箭头的非实曲线的绘制。
```python
import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

def add_arrow(from_, to_, rad=0.3, control_=None, color='#515151',
              line='--', head_length=0.6, size=0.04, detail=False):
    """
    :param from_:起始点
    :param to_:目标终点
    :param rad:曲线弧度
    :param control_:控制点，若为None，则由曲线弧度来进行计算
    :param color:绘制颜色
    :param line:曲线样式
    :param head_length:箭头大小
    :param size:蒙版大小
    :param detail:选择是否绘制详细，通过绘制详细来手动调整蒙版大小
    """
    if control_ is None:
        x1, y1 = np.array(from_)
        x2, y2 = np.array(to_)
        dx, dy = x2 - x1, y2 - y1
        x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
        cx, cy = x12 + rad * dy, y12 - rad * dx
        control_ = (cx, cy)
    vertices = [from_, control_, to_]
    codes = [patches.Path.MOVETO, patches.Path.CURVE3, patches.Path.CURVE3]
    path = patches.Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                              linestyle=line, linewidth=1, zorder=-2)
    ax.add_patch(patch)
    direction = -1 if control_[0] < to_[0] else 1
    mask_c = 'red' if detail else 'white'
    mask = patches.FancyBboxPatch(
        (to_[0], to_[1] - size / 2), direction * size, size, boxstyle="square, pad=0",
        ec=mask_c, fc=mask_c, zorder=-1, linewidth=1)
    de = math.degrees(math.atan((control_[1] - to_[1]) / (control_[0] - to_[0])))
    tf = transforms.Affine2D().rotate_deg_around(to_[0], to_[1], de) + ax.transData
    mask.set_transform(tf)
    ax.add_patch(mask)
    ax.annotate("", to_, xytext=control_, arrowprops=dict(
        linewidth=0,
        arrowstyle="-|>, head_width=%f, head_length=%f" % (head_length / 2, head_length),
        shrinkA=0, shrinkB=0, facecolor=color, linestyle="solid", mutation_scale=10
    ))
    if not detail:
        return
    ax.scatter(control_[0], control_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(from_[0], from_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.scatter(to_[0], to_[1], c=color, marker='x', s=12, linewidths=0.8)
    ax.plot((control_[0], from_[0]), (control_[1], from_[1]), color=color, linewidth=.5, linestyle=':')
    ax.plot((control_[0], to_[0]), (control_[1], to_[1]), color=color, linewidth=.5, linestyle=':')

def draw_eg():
    debug = False
    add_arrow((0.1, 0.1), (1.5, 1.6), rad=0.1, color=colors[1], detail=debug)
    add_arrow((0.4, 1.7), (1.4, 0.7), rad=0.8, color=colors[2], detail=debug)
    add_arrow((1.9, 1.9), (1.2, 0.1), rad=-0.2, color=colors[3], detail=debug)
    add_arrow((1.7, 0.7), (0.3, 1.8), rad=0.5, color=colors[4], detail=debug)
    plt.savefig('arrow_example%s.png' % ('_detail' if debug else ''), dpi=600)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
colors = ['#515151', '#CC9900', '#B177DE', '#37AD6B', '#1A6FDF']
draw_eg()
```
`Gmeek-html<p align="center"><img srcset="https://OmnisyR.github.io/figs/arrow_example_cat.png" width="800" height="400"/></p>`;;;c
