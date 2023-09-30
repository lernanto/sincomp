# 汉语方言比较工具集

这是一个自动化处理汉语方言读音数据的工具集，目的是方便方言读音数据的清洗、对比以及辅助祖语重构。

当前支持处理小学堂和语保2个方言读音数据集，以及提供一系列针对以上数据集度量、建模、作图的工具。

## 使用方式

本工具集还在开发中，目前比较成型的是自动化清洗、规整方言读音数据集。本库本身不提供数据集的原始数据，而是通过指定路径加载预先准备好的数据。小学堂的数据可以从下文所述单独的仓库获取，语保的数据需要使用者自行获取，并使用本库提供的工具整理成要求的格式。

### 获取数据

在 Linux 终端运行如下命令获取小学堂数据：
```shell
git clone https://github.com/lernanto/xiaoxuetang $HOME/xiaoxuetang
```

在 Windows 下使用命令提示符，或通过图形用户界面与此类似。

### 设置环境变量

为使用本工具集提供的功能函数，以及正确加载数据集，需要预先设置如下几个环境变量：
| 环境变量 | 说明 |
|:-|:-|
| PYTHONPATH | 包含本仓库的根路径 |
| XIAOXUETANG_HOME | 指向小学堂数据集的根路径 |
| ZHONGGUOYUYAN_HOME | 指向语保数据集的根路径 |

#### Linux

如下命令为在 Linux 终端设置上述环境变量的示例：
```shell
export PYTHONPATH=$PYTHONPATH:$HOME/sinetym
export XIAOXUETANG_HOME=$HOME/xiaoxuetang
export ZHONGGUOYUYAN_HOME=$HOME/zhongguoyuyan
```

#### Windows

类似地，在 Windows PowerShell 运行：
```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$HOME\sinetym"
$env:XIAOXUETANG_HOME = "$HOME\xiaoxuetang"
$env:ZHONGGUOYUYAN_HOME = "$HOME\zhongguoyuyan"
```

### 使用数据集

之后即可在 Python 代码中使用上述数据集，如：
```python
import sinetym.datasets

# 采用延迟加载方式，在使用时才加载实际数据并缓存，数据集对外表现如同 pandas.DataFrame
print(sinetym.datasets.xiaoxuetang[:10])
print(sinetym.datasets.xiaoxuetang.metadata['dialect_info'][:10])
```

更多功能的使用方法参考下文所述的几篇文章及源代码注释。

## 模块

本库包含以下主要模块：
| 模块 | 说明 |
|:-|:-|
| datasets | 提供加载数据集的通用接口 |
| similarity | 提供若干无监督的方法计算方言之间的相似度 |
| compare | 支持手工设定的规则来计算方言对规则的符合程度 |
| models | 为方言读音建模，当前主要是基于浅层神经网络的编解码器模型 |
| plot | 提供制作方言统计图、方言地图等的工具函数 |

其中部分功能的应用在如下几篇文章中有简要的介绍：
- [基于方言之间的预测相似度进行方言聚类](https://zhuanlan.zhihu.com/p/464735745)。
- [什么是官话？——兼及方言分类的概率模型](https://zhuanlan.zhihu.com/p/629007299)。
- [基于自编码器的方言祖语音系嵌入](https://zhuanlan.zhihu.com/p/349689590)。

更详细的使用方法详见各模块代码注释。

## 数据集

本工具集当前支持处理如下方言读音数据集：

| 代号 | 说明 | 简称 | 链接 |
|:-|:-|:-|:-|
| xiaoxuetang | 小学堂汉字古今音资料库现代音 | 小学堂 | [小学堂汉字古今音资料库](https://xiaoxue.iis.sinica.edu.tw/ccr) |
| zhongguoyuyan | 中国语言资源保护工程汉语方言单字音 | 语保 | [中国语言资源保护工程采录展示平台](https://zhongguoyuyan.cn/) |

其中小学堂提供所有方言读音数据下载，对于这部分数据声明：
> ……供獨立下載之聲韻資料檔案，視為事實性紀錄之整理，編輯者為其指認採「公眾領域標章（PDM，Public Domain Mark)」進行發布，在法律許可的範圍內，該等事實性勘驗記錄不復受到著作權利的保障，使用者得將其視為無著作權利限制之資訊，使用上毋須另洽編輯者申請著作權及著作相關權利之授權。

据此，我整理了一份和本库相配套的小学堂方言读音数据，存放在[这个仓库](https://github.com/lernanto/xiaoxuetang)。克隆该仓库至本地目录，本工具集的大部分功能函数即能直接处理其中 CSV 格式的数据。

语保的数据根据其法律声明，版权归发布内容的用户所有，同时中国语言资源保护工程采录展示平台享有在全球范围内的免费、不可撤销的无限期的并且可转让的非独家使用权。为此，本库仅提供一套额外的工具，用于把从上述语保网站可以访问到的数据转为本工具集可以处理的格式。

## 依赖

本工具集主要依赖如下 Python 库：
- Pandas
- Numpy
- Scipy
- Scikit-learn

models 模块额外依赖：
- TensorFlow

plot 模块额外依赖：
- Matplotlib
- Seaborn
- Plotly
- GeoPandas
- Cartopy
- Folium

以上不是完整的依赖列表，并且依赖的具体版本未经测试。