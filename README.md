## PoreNetworkSimulation

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

**6. 连接以概率呈现**

针对如页岩等低配位数的多孔介质，使得连接以概率形式呈现，保证配位数满足要求的同时，增加了计算时的稳定性。

**7. 单元半径计算方式更新**

从原有的通过多次模拟计算单元半径，更新为单纯依靠解方程解决，在网络结构较大时提升明显。

### 3. 工程结构

```
PoreNetworkSimulation                ── 工程目录
|
├─ src                               ── 源码包
┆  ┆
┆  ├─ config                         ── 配置
┆  ┆  └─ config                      ── 配置文件
┆  ┆
┆  ├─ data                           ── 数据
┆  ┆  └─ <file>                      ── 数据文件
┆  ┆
┆  ├─ entity                         ── 实体
┆  ┆  ├─ GasConstant.py              ── 气体状态类
┆  ┆  ├─ NetworkStatus.py            ── 网络状态类
┆  ┆  ├─ NetworkStructure.py         ── 网络结构类
┆  ┆  └─ Simulator.py                ── 求解器类
┆  ┆
┆  ├─ model                          ── 模型
┆  ┆  └─ CalculatePermeability.py    ── 计算渗透率模型
┆  ┆
┆  └─ utils                          ── 常用
┆     └─ Tools.py                    ── 通用工具类
┆
└─ .idea              ── 工程配置
```

### 4. 运行方式

进入src目录下，运行`python model/CalculatePermeability.py`即可。