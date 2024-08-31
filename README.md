# SinComp 汉语方言计算及比较工具集

SinComp 是一个自动化处理汉语方言读音数据的工具集，包含了一系列针对方言读音数据执行清洗、度量、比较、建模、作图的功能函数。当前支持处理小学堂、汉字音典、语保等方言读音数据集。

## 安装

在 Linux 终端/Windows 命令提示符下运行：

```shell
pip install sincomp
```

## 使用简介

本工具集还在开发中，目前比较成型的是自动化清洗、规整方言读音数据集。本库本身不提供数据集的原始数据，但针对小学堂和汉字音典，使用时自动从其网站下载数据，语保的数据需要使用者自行获取，并使用本库提供的工具整理成要求的格式。

### 使用数据集

首次使用数据集时会自动从网上下载数据集。安装成功之后在 Linux 终端/Windows 命令提示符运行如下命令会执行下载并刷新缓存文件。下载时间较久，请耐心等待运行结束。

```shell
python3 -O -m sincomp.datasets
```

运行成功后会创建如下目录及文件：

| 目录 | 说明 |
|:-|:-|
| datasets/ccr/*.xlsx | 小学堂数据集 |
| datasets/mcpdict/tools/tables/output/*.tsv | 汉字音典数据集 |

创建上述目录的位置由如下条件决定：

1. 如指定了 SINCOMP_CACHE 环境变量，则创建在该环境变量指向的路径。
2. 否则如指定了 LOCALAPPDATA 环境变量，则创建在该环境变量下的 sincomp 目录，Windows 下 LOCALAPPDATA 默认为当前用户目录下的 AppData\Local。
3. 否则创建在当前用户目录下的 sincomp 目录。

如果由于网络或其他原因下载失败，可以手动到下文所述各数据集的网站下载数据并解压到上述目录，再运行上述命令。

> [!Note]
> 注意小学堂数据文件名中“閩語”笔误为“閔語”，需手工改正。

之后即可在 Python 代码中使用上述数据集，如：

```python
import sincomp.datasets

# 采用延迟加载方式，在使用时才加载实际数据并缓存，数据集对外表现如同 pandas.DataFrame
print(sincomp.datasets.ccr[:10])
print(sincomp.datasets.ccr.metadata['dialect_info'][:10])
```

### 计算方言相似度

在 Linux 终端或 Windows PowerShell 运行如下命令计算小学堂语料集所有方言两两之间的相似度：

```shell
python3 -O -m sincomp.similarity
```

视机器性能运行时间可能较久，完成后会在当前目录生成两个文件：

| 文件 | 说明 |
|:-|:-|
| ccr_chi2.csv | 使用卡方方法计算的相似度 |
| ccr_entropy.csv | 使用条件熵计算的相似度 |

上述命令实际上在内部调用了相似度函数，其用法如下所示：

```python
import sincomp.datasets
import sincomp.preprocess
import sincomp.similarity

# 把原始数据集转成宽表，每个字为一行，如果一个字有多个读音，以空格分隔
data = sincomp.preprocess.transform(
    sincomp.datasets.ccr.fillna({'initial': '', 'final': '', 'tone': ''}),
    index='cid',
    values=['initial', 'final', 'tone'],
    aggfunc='first'
)
# 使用卡方计算方言之间相似度
sim = sincomp.similarity.chi2(data)
```

更多使用方法参考该模块的帮助及注释。

## 模块

本库包含以下主要模块：

| 模块 | 说明 |
|:-|:-|
| datasets | 提供加载数据集的通用接口 |
| align | 对齐不同数据集之间的字 ID |
| similarity | 提供若干无监督的方法计算方言之间的相似度 |
| compare | 支持手工设定的规则来计算方言对规则的符合程度 |
| models | 为方言读音建模，当前主要是基于浅层神经网络的编解码器模型 |
| plot | 提供制作方言统计图、方言地图等的工具函数 |

其中部分功能的应用在如下几篇文章中有简要的介绍：
- [基于方言之间的预测相似度进行方言聚类](https://zhuanlan.zhihu.com/p/464735745)
- [什么是官话？——兼及方言分类的概率模型](https://zhuanlan.zhihu.com/p/629007299)
- [基于自编码器的方言祖语音系嵌入](https://zhuanlan.zhihu.com/p/349689590)
- [使用双线性编码建模多方言音系](https://zhuanlan.zhihu.com/p/659731592)

更详细的使用方法见各模块代码注释。

## 数据集

本库当前支持处理如下方言读音数据集：

| 代号 | 说明 | 简称 | 链接 |
|:-|:-|:-|:-|
| ccr | 小学堂汉字古今音资料库现代音 | 小学堂 | [小学堂汉字古今音资料库](https://xiaoxue.iis.sinica.edu.tw/ccrdata/) |
| mcpdict | 汉字音典方言汉字读音 | 汉字音典 | [汉字音典](https://mcpdict.sourceforge.io/) |
| zhongguoyuyan | 中国语言资源保护工程汉语方言单字音 | 语保 | [中国语言资源保护工程采录展示平台](https://zhongguoyuyan.cn/) |

如下为以上数据集的版权声明：

### 小学堂

小学堂在网站上提供所有方言读音数据下载，对于这部分数据声明：

> ……提供獨立下載之聲韻資料檔案，視為事實性紀錄之整理，編輯者為其指認採「公眾領域標章（PDM，Public Domain Mark)」進行發布，在法律許可的範圍內，該等事實性勘驗記錄不復受到著作權利的保障，使用者得將其視為無著作權利限制之資訊，使用上毋須另洽編輯者申請著作權及著作相關權利之授權。

详情见[汉字古今音资料库](https://xiaoxue.iis.sinica.edu.tw/ccrdata/)的“授权方式”节。

### 汉字音典

汉字音典为开源的汉字读音查询工具，并随代码发布了互联网方言爱好者收集整理的多种历史和方言读音数据。数据随代码以 MIT 许可在 [GitHub](https://github.com/osfans/MCPDict) 发布，详情见其[版权声明](https://github.com/osfans/MCPDict?tab=License-1-ov-file)。

### 语保

语保的数据根据其版权声明

> ……视用户为其在中国语言资源保护工程采录展示平台网站上载或发布的内容的版权所有人。用户在中国语言资源保护工程采录展示平台上载或发布内容即视为其同意授予中国语言资源保护工程采录展示平台所有上述内容的在全球范围内的免费、不可撤销的无限期的并且可转让的非独家使用权许可……

详情见其[版权声明](https://zhongguoyuyan.cn/declaration)。

为此，本库仅提供一套额外的工具，用于把从上述语保网站可以访问到的数据转为本工具集可以处理的格式。处理完成后设置环境变量指向该数据集的根目录，在 Linux 终端运行：

```shell
export ZHONGGUOYUYAN_HOME=$HOME/zhongguoyuan
```

或在 Windows PowserShell 运行：

```powershell
$env:ZHONGGUOYUYAN_HOME = $env:HOME\zhongguoyuyan
```

其后使用该数据集的方式与小学堂相同。

```python
import sincomp.datasets

print(sincomp.datasets.zhongguoyuyan[:10])
print(sincomp.datasets.zhongguoyuyan.metadata['dialect_info'][:10])
```

### 数据格式

本工具集使用两种格式的数据：
- **长表** 一个方言点的一条字音数据为一行，为数据集载入的原始格式，基于神经网络的模型如编码器使用该格式。
- **宽表** 一个方言点所有字、或一个字在所有方言点的读音为一行，一些基于统计的方法，如计算方言相似度使用该格式。

长表和宽表都实现为 pandas.DataFrame 或兼容的数据类型。其中长表至少包含如下列：

| 列名 | 说明 | 备注 |
|:-|:-|:-|
| did | 方言 ID | |
| cid | 字 ID | 当前汉字音典数据集没有提供字 ID，只提供了字形
| character | 字形 | 不是必须的字段，但上述所有数据集都提供了
| initial | 声母 | |
| final | 韵母 | |
| tone | 声调 | |

本工具集提供了一个函数把长表转换为宽表，其用法如下所示：

```python
import sincomp.datasets
import sincomp.preprocess

# 把原始数据集的长表转成宽表，每个方言为一行，每个字的声韵调等为一列，如果一个字有多个声韵调，取第一个
wide = sincomp.preprocess.transform(sincomp.datasets.ccr)
print(wide[:10])
```

### 自定义数据集

自定义数据集只要为满足上述长表格式的 pandas.DataFrame，本库的功能函数均可处理。为方便处理自定义数据集，本库提供提供了一个基础类，该类要求数据文件以 CSV 格式存储，每个文件为一个方言，其用法如下所示：

```python
import pandas as pd
import sincomp.datasets

# 从指定目录加载数据集，该目录树下每个文件视为一个方言点，主文件名为方言 ID
mydataset = sincomp.datasets.FileDataset(path='path/to/dataset/directory')
# 自定义数据集的用法与自带数据集相同
print(mydataset[:10])
```

# 开发计划

- [ ] 使用 CRF 切分汉语方言音节及切分声韵调
- [ ] 为现有的算法和模型提供兼容 scikit-learn 的接口
- [ ] 更新语保数据获取脚本