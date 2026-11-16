# Installation

SinComp requires Python 3.10 or later. Install the core library with:

```shell
python -m pip install sincomp
```

The core dependencies include NumPy, pandas, SciPy, scikit-learn, and
OpenCC. Install optional dependency groups only when you need them:

| Extra | Install command | Enables |
| --- | --- | --- |
| `datasets` | `python -m pip install "sincomp[datasets]"` | Selenium support for browser-based dataset access |
| `models` | `python -m pip install "sincomp[models]"` | TensorFlow neural models and TensorBoard |
| `parser` | `python -m pip install "sincomp[parser]"` | The CRF-based romanization parser |
| `plot` | `python -m pip install "sincomp[plot]"` | Optional visualization dependencies; plotting is not covered in this guide |

You can combine extras, for example:

```shell
python -m pip install "sincomp[datasets,models]"
```

For development from a local checkout, install the desired extras in
editable mode:

```shell
python -m pip install -e ".[datasets,models]"
```

## Dataset cache

Downloaded dataset files are stored under a `datasets` directory inside
SinComp's cache directory. Set `SINCOMP_CACHE` to choose that parent
directory:

```shell
# Linux or macOS
export SINCOMP_CACHE="$HOME/sincomp"

# PowerShell
$env:SINCOMP_CACHE = "$HOME\sincomp"
```

When `SINCOMP_CACHE` is set, dataset files are stored under
`<SINCOMP_CACHE>/datasets`. Otherwise, the cache root is
`%LOCALAPPDATA%/sincomp` on Windows when `LOCALAPPDATA` is set, and
`$XDG_DATA_HOME/sincomp` on other platforms. `XDG_DATA_HOME` defaults to
`~/.local/share`; without `LOCALAPPDATA`, Windows falls back to
`~/sincomp`.

Some data sources are fetched lazily: creating a dataset object does not
necessarily download its data, but accessing its records may require a
network connection. The China Language Resources Protection Project
source also requires a supported Chrome browser and an authenticated
browser session; see [Datasets](datasets.md).
