## 等效孔隙网络模拟计算软件

### 更新说明

V1.0.0

1. 实现了基本的等效孔隙网络模型渗流模拟，及对应的机械弥散模拟。
2. 从原有的通过多次模拟计算单元半径，更新为单纯依靠解方程解决，在网络结构较大时提升明显。

### 1. 简介

本工程主要解决的问题：采用**等效孔隙网络模型**，对多孔介质中的气体运移现象进行计算。下面是对等效孔隙网络模型的介绍：

```
等效孔隙网络是一种岩土介质的孔隙结构模型，它将不同大小的孔隙固定在空间的格点上，孔隙之间通过喉道进行连接。通过设置不同的孔隙大小、不同的喉道连接数以及喉道的尺寸，可以反映不同结构的岩土介质。
```

工程由Matlab迁移到Python，并进行了部分重构工作。

### 2. 特点

**1. 参数化网络生成**

利用孔隙半径、喉道半径、喉道曲率、配位数等参数，对等效孔隙网络进行全方面的描述，从而获得特定的参数化网络。

**2. 随机构建**

在配置参数时，对每一项参数都可以附带一定的随机性，使得网络结构更加符合真实结构。

**3. 任务与数据分离**

分离出网络状态与求解器，从而使得任务与数据相分离，使得计算状态恢复更加方便，也为之后的多线程或分布式计算提供基础。

**4. 迭代算法改进**

从时间模拟迭代转化为J法/GS法进行迭代，无需确定迭代时间参数，同时使结果更容易收敛，并加速了收敛速度。

**5. 配置文件化**

使用专门的配置文件对求解进行配置，免去了需要阅读源码的情形，更加方便易用。

### 3. 工程结构

```
PoreNetworkSimulation                     ── 工程目录
│
├─ src                                    ── 源码包
│  │
│  ├─ config                              ── 配置
│  │  ├─ config.ini.sample                ── 配置文件示例
│  │  └─ config.ini                       ── 配置文件
│  │
│  ├─ data                                ── 数据
│  │  ├─ seepage_<name>_cache.obj         ── 计算缓存文件
│  │  ├─ seepage_<name>_status.obj        ── 网络状态文件
│  │  ├─ dispersion_<name>_massflux.obj   ── 机械弥散流量文件
│  │  ├─ dispersion_<name>_velocity.obj   ── 机械弥散流速文件
│  │  └─ dispersion_<name>_paths.txt      ── 机械弥散结果文件
│  │
│  ├─ log                                 ── 日志
│  │  ├─ seepage_<name>.log               ── 网络状态计算日志文件
│  │  └─ dispersion_<name>_massflux.obj   ── 机械弥散计算日志文件
│  │
│  ├─ entity                              ── 实体
│  │  ├─ GasConstant.py                   ── 气体状态类
│  │  ├─ NetworkStatus.py                 ── 网络状态类
│  │  ├─ NetworkStructure.py              ── 网络结构类
│  │  ├─ Simulator.py                     ── 求解器类
│  │  └─ StatusCache.py                   ── 计算缓存类
│  │
│  ├─ model                               ── 模型
│  │  ├─ CalculatePermeability.py         ── 计算渗透率模型
│  │  └─ Dispersion.py                    ── 模拟机械弥散模型
│  │
│  └─ utils                               ── 常用
│     └─ Tools.py                         ── 通用工具类
│
├─ .idea                                  ── 工程配置
│
└─ requirements.txt                       ── 工程依赖
```

### 4. 模型概述

#### 4.1 渗透率计算模型

该模型主要用于模拟渗透率情形下的稳定流（两个相对边界条件均为定压力边界）。

输入：

1. 网络参数（配置项：`[network]`）
2. 气体参数（配置项：`[gas]`）
3. 边界条件及初始条件（配置项：`[status]`）
4. 求解器类型与设置（配置项：`[solver]`）
5. 迭代设置与终止条件（配置项：`[iteration]`）
6. （可选）中间结果（如果想要从中间结果恢复）

输出：

1. 渗透率计算结果（如果设置了`iteration.showPermeability` > 0）
2. 网络压力状态（保存在`src/data/seepage_<name>_status.obj`文件下）

文件`src/data/seepage_<name>_status.obj`的结构：

```
seepage_<name>_status.obj         ── (object)          网络状态
├─ sc                             ── (object)          网络状态配置
│  ├─ boundary_type               ── (array[6])        边界条件类型
│  ├─ boundary_value              ── (array[6])        边界条件值
│  ├─ initial_type                ── (int)             初始条件类型
│  └─ initial_value               ── (array[6])        初始条件值
├─ ns                             ── (object)          网络结构
│  ├─ nc                          ── (object)          网络结构配置
│  │  ├─ model_size               ── (array[3])        模型尺寸
│  │  ├─ character_length         ── (float)           特征尺寸
│  │  ├─ unit_size                ── (float)           单元尺寸
│  │  ├─ radius_params            ── (array[2])        孔隙半径均值&标准差
│  │  ├─ curvature                ── (float)           喉道曲率
│  │  ├─ throat_params            ── (array[2])        喉道半径均值&标准差
│  │  ├─ coor_params              ── (array[2])        孔隙配位数半径均值&标准差
│  │  ├─ porosity                 ── (float)           孔隙率
│  │  └─ anisotropy               ── (array[3])        各向异性参数
│  ├─ model_size                  ── (array[3])        模型尺寸
│  ├─ character_length            ── (float)           特征尺寸
│  ├─ radii                       ── (array[x,y,z])    孔隙半径
│  ├─ throatR                     ── (array[x,y,z,26]) 喉道半径
│  ├─ weight                      ── (array[x,y,z,26]) 喉道权重
│  ├─ unit_size                   ── (float)           单元尺寸
│  └─ porosity                    ── (float)           网络最终孔隙率
├─ gc                             ── (object)          气体参数
│  ├─ M                           ── (float)           摩尔质量
│  ├─ R                           ── (float)           理想气体常数
│  ├─ T                           ── (float)           温度
│  └─ u                           ── (float)           粘度
├─ model_size                     ── (array[3])        模型尺寸
└─ pressure                       ── (array[x,y,z])    孔隙压力
```

#### 4.2 机械弥散模拟模型

输入：

1. 机械弥散参数（配置项：`[dispersion]`）
2. 4.1中得到的网络状态（`src/data/seepage_<name>_status.obj`）

输出：

1. 各时间步示踪物位置（保存在`src/data/dispersion_<name>_paths.txt`文件中）

文件`src/data/dispersion_<name>_paths.txt`的结构:

该文件是路径矩阵(array[particles, total-time-steps, 4])的线性存储。以下述方式存储：

```
示踪物1_时间1
示踪物1_x1
示踪物1_y1
示踪物1_z1
示踪物1_时间2
示踪物1_x2
示踪物1_y2
示踪物1_z2
...
示踪物2_时间1
示踪物2_x1
示踪物2_y1
示踪物2_z1
...
```

使用`reshape(path-data, particles, total-time-steps, 4)`在分析中将非常有用。

### 5. 运行

#### 5.1 环境需求

工程需要在Python 2.7下运行.

1. 从官网安装 [Python 2.7](https://www.python.org/)。
2. 记得将Python加入你的PATH中。
3. 进入工程目录。使用命令 `pip install -r requirements.txt`安装需求包。

#### 5.2 运行渗透率模拟

1. 切换到目录`src/config`，并修改配置文件`config.ini`。
2. 切换到目录`src/model`，运行`CalculatePermeability.py`或使用命令`python -u CalculatePermeability.py`。
3. 运行日志将会输出在屏幕上，并保存在`src/log/seepage_<name>.log`。如果你设置了保存中间结果，中间结果将会暂时保存在`src/data/seepage_<name>_status_<step>.obj`。
4. 当某一迭代步的压力变化值小于你的设定值，程序将会终止，结果将存储在`src/data/seepage_<name>_status.obj`下。

#### 5.3 运行机械弥散模拟

1. 切换到目录`src/config`，并修改配置文件`config.ini`。
2. 切换到目录`src/model`，运行`Dispersion.py`或使用命令`python -u Dispersion.py`。
3. 运行日志将会输出在屏幕上，并保存在`src/log/dispersion_<name>.log`。
4. 程序终止时，结果将存储在`src/data/dispersion_<name>_paths.txt`下。

### 6. 特别提醒

#### 6.1 渗透率模拟过程中的保存

渗透率模拟过程中，可以选择保存中间及最终结果，这个选项的本质是为了从中断或异常中恢复，或者继续未完成的模拟。但在运行时，也可能发生下面的情况：

运行渗透率模拟模型时，如果设置了参数`iteration.save != 0`，计算过程中将会保存中间结果与最终结果。而当你再次运行程序时，程序将自动寻找类似`src/data/seepage_<name>_status.obj`的数据文件，并从文件中重建孔隙网络（与之前完全相同），同时渗透率模拟将从上次结束位置继续执行。因此如果想要构建新的网络并重启模拟过程，可以选择以下方法之一：

1. 删除数据文件`src/data/seepage_<name>_status.obj`
2. 更换配置中的文件名`iteration.fileName`
3. 在首次运行时，设置`iteration.save = 0`，这将不保存任何中间及最终结果