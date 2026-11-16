# Datasets

The `sincomp.datasets` module provides predefined datasets, custom-file
loading, and a simulator. Dataset objects expose pronunciation data in
long form and load remote data lazily when required.

## Built-in datasets

| Name | Description | Source | Aliases | Link |
| --- | --- | --- | --- | --- |
| BeidaZihui | digitized dialect data from *Hanyu Fangyin Zihui* | *Hanyu Fangyin Zihui* (漢語方音字彙) | `beidazihui` | https://github.com/lexibank/beidazihui |
| CCR | modern dialect data from the Chinese Character Readings (CCR) | *Chinese Character Readings* (漢字古今音資料庫), *Xiaoxue* (小學堂) | `ccr` | https://xiaoxue.iis.sinica.edu.tw/ccrdata/ |
| MCPDict | modern dialect data from *MCPDict* | *MCPDict* (漢字音典) | `mcpdict` | https://nk2028.shn.hk/yindian/ |
| zhongguoyuyan | Chinese dialect data from *China Language Resources Protection Project* (中国语言资源保护工程) | *China Language Resources Protection Project Recording and Display Platform* (中国语言资源保护工程采录展示平台) | | https://zhongguoyuyan.cn/ |

Look up a dataset and inspect its data:

```python
from sincomp.datasets import get, predefined

for item in predefined():
    print(item.name)

dataset = get("CCR")
print(dataset.dialects.head())
data = dataset.data
```

`get()` checks registered names before treating the argument as a path.
It also accepts a CSV file or a directory containing per-dialect CSV
files. It returns `None` when the name is not registered and the path
does not exist. Dataset objects provide `data`, `dialects`, and
`characters` properties, as well as `get_data(did)` for selecting one
dialect.

## Generate a dataset

Use `make_dataset()` to create a synthetic dataset for tests, examples, or
benchmarks without downloading data:

```python
from sincomp.datasets import make_dataset

dataset = make_dataset(
    initials=["p", "t", "k"],
    finals=["a", "i", "u"],
    tones=["1", "2", "3"],
    n_characters=27,
    n_dialects=5,
    seed=0,
)
```

For each pronunciation-feature inventory and the character set, provide
either its size (`n_initial`, `n_final`, `n_tone`, or `n_characters`) or
its explicit name list (`initials`, `finals`, `tones`, or `characters`).
The function returns a dataset object; its `.data` property contains the
generated long-form data.

## Data formats

Long-form data usually contain one row per sample, typically one
dialect-character observation. A dialect-character pair may have
multiple samples. Typical columns include:

| Column | Meaning |
| --- | --- |
| `did` | Dialect identifier |
| `cid` | Character identifier, when provided by the source |
| `character` | Written character |
| `initial`, `final`, `tone` | Parsed pronunciation features |

Some sources do not provide a shared character ID. Character alignment
without IDs is described in [Alignment](alignment.md).

Scikit-learn compatible transformers and estimators use a wide-form table as input.
Wide-form tables place one entity per row and spread dialect or feature values
across columns. Use [Preprocessing](preprocessing.md) to convert a long-form
table to the required wide-form format.

## Custom datasets

A custom dataset can be loaded from a CSV file or a directory of
per-dialect CSV files:

```python
from sincomp.datasets import get

single = get("path/to/pronunciations.csv")
collection = get("path/to/dialect-csv-files")
```

The data should follow the long-form conventions above. When storing
one dialect per file, the files should contain that dialect's samples;
the filename is used as its dialect identifier. Local paths are resolved
on the machine running the code.

## Downloads and cache

Built-in dataset objects defer access to source data until they are
requested. Downloads are stored under
`<cache-root>/datasets/<dataset-name>`. Set `SINCOMP_CACHE` to choose the
cache root; in that case files go under `<SINCOMP_CACHE>/datasets`.
Otherwise Windows uses `%LOCALAPPDATA%/sincomp` when set, and other
platforms use `$XDG_DATA_HOME/sincomp` (default
`~/.local/share/sincomp`). On Windows without `LOCALAPPDATA`, the cache
root is `~/sincomp`.

Each dataset stores its source resources at a fixed location inside the
cache directory:

| Dataset | Source resource | Cache path |
| --- | --- | --- |
| BeidaZihui | `raw/characters.tsv` | `datasets/beidazihui/characters.tsv` |
| CCR | one archive per dialect group | `datasets/ccr/*.xlsx` |
| MCPDict | `tools/tables/output/*.tsv` | `datasets/mcpdict/*.tsv` |
| zhongguoyuyan | one backend resource per dialect | `datasets/zhongguoyuyan/*.json` |

The *China Language Resources Protection Project* source uses browser-based
access. Install the `datasets` extra, use Chrome, and sign in through the
browser when prompted. Access is cached by dialect; avoid frequent or
large bursts of requests, which may cause account restrictions.

When a download fails because of network or other problems, download
the data manually from the source websites listed in
[Built-in datasets](#built-in-datasets), unpack it to the cache paths
above, and run the commands again.

> [NOTE] CCR source archives misspell 閩語 as 閔語 in their file names;
> rename the files accordingly after unpacking.

## Source acknowledgements and notices

### BeidaZihui

The dialect data in *Hanyu Fangyin Zihui* (Peking University Department
of Chinese Language and Literature, 1962) was digitized by the Lexibank
project and released under CC BY 4.0. See the
[Beida Zihui repository](https://github.com/lexibank/beidazihui) and its
[license](https://github.com/lexibank/beidazihui?tab=CC-BY-4.0-1-ov-file).

### CCR

The *Xiaoxue* website provides downloads of its dialect pronunciation data and
states:

> ……提供獨立下載之聲韻資料檔案，視為事實性紀錄之整理，編輯者為其指認採「公眾領域標章（PDM，Public Domain Mark）」進行發布，在法律許可的範圍內，該等事實性勘驗記錄不復受到著作權利的保障，使用者得將其視為無著作權利限制之資訊，使用上毋須另洽編輯者申請著作權及著作相關權利的授權。

See the [Xiaoxue website](https://xiaoxue.iis.sinica.edu.tw/ccrdata/)
for its current access and attribution terms.

### MCPDict

*MCPDict* is an open-source pronunciation lookup project. Its dialect data
is distributed with the project under the MIT license. See the
[MCPDict repository](https://github.com/osfans/MCPDict) and its
[license](https://github.com/osfans/MCPDict?tab=License-1-ov-file).

### zhongguoyuyan

The *China Language Resources Protection Project Recording and Display Platform*'s declaration states:

> ……视用户为其在中国语言资源保护工程采录展示平台网站上载或发布的内容的版权所有人。用户在中国语言资源保护工程采录展示平台上载或发布内容即视为其同意授予中国语言资源保护工程采录展示平台所有上述内容的在全球范围内的免费、不可撤销的无限期的并且可转让的非独家使用权许可……

See the platform's
[copyright declaration](https://zhongguoyuyan.cn/declaration). Users
should review the current source terms before redistributing or
publishing any data.
