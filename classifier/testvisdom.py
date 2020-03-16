# coding=utf-8
import time
from visdom import Visdom
import requests
import os
import numpy as np

viz = Visdom(server='http://127.0.0.1', port=8097)
assert viz.check_connection()

# 视频下载可能比较慢，耐心等几分中
video_file = "demo.ogv"
if not os.path.exists(video_file):
    video_url = 'http://media.w3.org/2010/05/sintel/trailer.ogv'
    res = requests.get(video_url)
    with open(video_file, "wb") as f:
        f.write(res.content)

viz.video(videofile=video_file)

# 图片
# 单张图片
viz.image(
    np.random.rand(3, 512, 256),
    opts={
        'title': 'Random',
        'showlegend': True
    }
)
# 多张图片
viz.images(
    np.random.rand(20, 3, 64, 64),
    opts={
        'title': 'multi-images',
    }
)

# 散点图
Y = np.random.rand(100)
Y = (Y[Y > 0] + 1.5).astype(int),  # 100个标签1和2

old_scatter = viz.scatter(
    X=np.random.rand(100, 2) * 100,
    Y=Y,
    opts={
        'title': 'Scatter',
        'legend': ['A', 'B'],
        'xtickmin': 0,
        'xtickmax': 100,
        'xtickstep': 10,
        'ytickmin': 0,
        'ytickmax': 100,
        'ytickstep': 10,
        'markersymbol': 'cross-thin-open',
        'width': 800,
        'height': 600
    },
)
# time.sleep(5)
# 更新样式
viz.update_window_opts(
    win=old_scatter,
    opts={
        'title': 'New Scatter',
        'legend': ['Apple', 'Banana'],
        'markersymbol': 'dot'
    }
)
# 3D散点图
viz.scatter(
    X=np.random.rand(100, 3),
    Y=Y,
    opts={
        'title': '3D Scatter',
        'legend': ['Men', 'Women'],
        'markersize': 5
    }
)

# 柱状图
viz.bar(X=np.random.rand(20))
viz.bar(
    X=np.abs(np.random.rand(5, 3)),  # 5个列，每列有3部分组成
    opts={
        'stacked': True,
        'legend': ['A', 'B', 'C'],
        'rownames': ['2012', '2013', '2014', '2015', '2016']
    }
)

viz.bar(
    X=np.random.rand(20, 3),
    opts={
        'stacked': False,
        'legend': ['America', 'Britsh', 'China']
    }
)

# 热力图，地理图，表面图
viz.heatmap(
    X=np.outer(np.arange(1, 6), np.arange(1, 11)),
    opts={
        'columnnames': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'rownames': ['y1', 'y2', 'y3', 'y4', 'y5'],
        'colormap': 'Electric'
    }
)

# 地表图
x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
viz.contour(X=X, opts=dict(colormap='Viridis'))

# 表面图
viz.surf(X=X, opts={'colormap': 'Hot'})