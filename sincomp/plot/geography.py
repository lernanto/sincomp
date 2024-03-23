# -*- coding: utf-8 -*-

"""
绘制方言地图相关函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pandas
import numpy
import scipy.interpolate
from sklearn.preprocessing import OneHotEncoder
import geopandas
import matplotlib
import matplotlib.pyplot
import shapely
import cartopy
import cartopy.crs
import folium

from .. import auxiliary


def clip(func, vmin=0, vmax=1):
    """
    辅助函数，对目标函数的返回值进行截断.
    """

    return lambda x, y: numpy.clip(func(x, y), vmin, vmax)

def make_clip_path(polygons, extent=None):
    """
    根据绘制范围及指定的多边形生成图形的裁剪路径.

    Parameters:
        polygons (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只保留该范围内的图形
        extent: 绘制的范围 (左, 右, 下, 上)

    Returns:
        path (`matplotlib.path.Path`): 生成的裁剪路径，如果传入的多边形为空返回 None
    """

    if polygons is None:
        return None

    polygons = tuple(polygons) if hasattr(polygons, '__iter__') else (polygons,)
    if len(polygons) == 0:
        return None

    if extent is not None:
        # 先对绘图区域和裁剪区域取交集
        xmin, xmax, ymin, ymax = extent
        poly = shapely.geometry.Polygon((
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin)
        ))
        polygons = [poly.intersection(c) for c in polygons]

    return matplotlib.path.Path.make_compound_path(
        *cartopy.mpl.patch.geos_to_path(polygons)
    )

def clip_paths(paths, polygons, extent=None):
    """
    根据绘制范围及指定的多边形裁剪 matplotlib 绘制的图形.

    Parameters:
        paths (`matplotlib.PathCollection` or list of `matplotlib.PathCollection`):
            待裁剪的图形
        polygons (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只保留该范围内的图形
        extent: 绘制的范围 (左, 右, 下, 上)
    """

    path = make_clip_path(polygons, extent=extent)
    if path is not None:
        # 裁剪图形
        if hasattr(paths, '__iter__'):
            for c in paths:
                c.set_clip_path(path, transform=c.axes.transData)
        else:
            paths.set_clip_path(path, transform=paths.axes.transData)

def scatter(
    latitudes,
    longitudes,
    values=None,
    ax=None,
    extent=None,
    scale=0,
    clip=None,
    **kwargs
):
    """
    绘制方言点地图.

    Parameters:
        latitudes (`numpy.ndarray`): 方言点的纬度数组
        longitudes (`numpy.ndarray`): 方言点的经度数组
        values (`numpy.ndarray`): 方言点的值，可为实数或离散类型
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象，
            如果为空，创建一个新对象
        extent: 绘制的范围 (左, 右, 下, 上)
        scale (float): 当未指定绘制范围时，用于根据样本点计算范围的系数
        clip (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只绘制该范围内的方言点，为空绘制所有方言点
        kwargs: 透传给 `matplotlib.pyplot.Axes.scatter`

    Returns:
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象
        extent: 绘制的范围
        pc (`matplotlib.collections.PathCollection`): 绘制的点集合
    """

    if extent is None:
        # 根据样本点确定绘制边界
        lat0, lat1, lon0, lon1 = auxiliary.extent(latitudes, longitudes, scale)
        extent = (lon0, lon1, lat0, lat1)

    proj = cartopy.crs.PlateCarree()
    if ax is None:
        ax = matplotlib.pyplot.axes(projection=proj)

    # 根据传入的图形裁剪点集
    pc = ax.scatter(
        longitudes,
        latitudes,
        c=values,
        clip_path=(make_clip_path(clip, extent=extent), ax.transData),
        **kwargs
    )

    ax.set_extent(extent, crs=proj)
    return ax, extent, pc

def area(
    latitudes: numpy.ndarray[float] | pandas.Series,
    longitudes: numpy.ndarray[float] | pandas.Series,
    values: numpy.ndarray | pandas.Series,
    ax: matplotlib.axes.Axes | None = None,
    extent: tuple[float, float, float, float] | None = None,
    scale: float = 0,
    clip=None,
    resolution: int = 100,
    **kwargs
) -> tuple[
    matplotlib.axes.Axes,
    tuple[float, float, float, float],
    matplotlib.collections.QuadMesh
]:
    """
    绘制方言分区图

    Parameters:
        latitudes: 样本点的纬度数组
        longitudes: 样本点的经度数组
        values: 当为1维数组时，表示样本点的分类，当为2维数组时，表示样本点属于各分类的概率
        ax: 作图使用的 Axes 对象，如果为空，创建一个新 `cartopy.mpl.geoaxes.GeoAxes` 对象
        extent: 绘制的范围
        scale: 当未指定绘制范围时，用于根据样本点计算范围的系数
        clip(shapely.geometry.multipolygon.MultiPolygon):
            裁剪的范围，只绘制该范围内的分区，为空绘制整个绘制范围的分区
        resolution: 分辨率，把绘制范围的长宽最多分为多少个点来插值，
            实际分的点数由长宽比决定
        kwargs: 透传给 `matplotlib.pyplot.Axes.pcolormesh`

    Returns:
        ax: 作图使用的 Axes 对象
        extent: 绘制的范围
        qm: 绘制的色块集
    """

    mask = numpy.isfinite(latitudes) & numpy.isfinite(longitudes)

    # 如果传入的是原始分类，先转化为 one-hont 编码
    if values.ndim == 1:
        mask &= numpy.isfinite(values)
        values = OneHotEncoder(dtype=numpy.int32) \
            .fit_transform(numpy.expand_dims(values[mask], 1)).A
    else:
        mask &= numpy.all(numpy.isfinite(values), axis=1)
        values = values[mask]

    # 针对完全相同的经纬度，对经度稍作偏移，使能正常计算
    longitudes = longitudes[mask]
    latitudes = latitudes[mask]
    longitudes = pandas.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes
    }).groupby(['latitude', 'longitude'])['longitude'] \
        .transform(lambda x: x + numpy.arange(x.shape[0]) * 1e-4).values

    if extent is None:
        # 根据样本点确定绘制边界
        lat0, lat1, lon0, lon1 = auxiliary.extent(latitudes, longitudes, scale)
        extent = (lon0, lon1, lat0, lat1)
    else:
        lon0, lon1, lat0, lat1 = extent

    # 使用径向基函数基于样本点对选定范围进行插值
    rbf = scipy.interpolate.RBFInterpolator(
        numpy.stack([longitudes, latitudes], axis=1),
        values,
        kernel='linear'
    )

    # 计算分辨率，把长宽最多分成指定点数，且为方格
    size = numpy.asarray([lon1 - lon0, lat1 - lat0])
    lon_res, lat_res = (size / numpy.max(size) * resolution).astype(int)
    lon = numpy.linspace(lon0, lon1, lon_res)
    lat = numpy.linspace(lat0, lat1, lat_res)
    coo = numpy.reshape(numpy.stack(numpy.meshgrid(lon, lat), axis=2), (-1, 2))

    val = numpy.reshape(rbf(coo), (lat_res, lon_res, -1))
    label = numpy.argmax(val, axis=2)

    proj = cartopy.crs.PlateCarree()
    if ax is None:
        ax = matplotlib.pyplot.axes(projection=proj)

    # 根据插值结果绘制方言分区图，根据传入的图形裁剪
    qm = ax.pcolormesh(
        lon,
        lat,
        label,
        transform=proj,
        clip_path=(make_clip_path(clip, extent=extent), ax.transData),
        **kwargs
    )

    ax.set_extent(extent, crs=proj)
    return ax, extent, qm

def _isogloss(
    lat0, lat1, lon0, lon1,
    func,
    ax=None,
    fill=True,
    clip=None,
    resolution=100,
    **kwargs
):
    """
    绘制同言线地图.

    输入参数为一系列样本点坐标，及样本点符合某个语言特征的程度值，通常取值范围为 [0, 1]。
    使用径向基函数根据样本点插值，计算整个绘制空间的值，然后据此计算等值线。
    如果指定了裁剪范围，只绘制该范围内的等值线。

    Parameters:
        lat0, lat1, lon0, lon1 (float): 绘制范围下、上、左、右
        func (callable): 符合度值函数，输入参数为坐标数组，返回符合度数组
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象，如果为空，创建一个新对象
        fill (bool): 为真时填充颜色，为假时只绘制等值线
        clip (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只绘制该范围内的等值线，为空绘制整个绘制范围的等值线
        resolution (int): 分辨率，把绘制范围的长宽最多分为多少个点来插值，
            实际分的点数由长宽比决定
        kwargs: 透传给 `matplotlib.pyplot.Axes.contourf`

    Returns:
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象
        cs (`matplotlib.contour.QuadContourSet`): 绘制的等值线集合
    """

    # 计算分辨率，把长宽最多分成指定点数，且为方格
    size = numpy.asarray([lon1 - lon0, lat1 - lat0])
    lon_res, lat_res = (size / numpy.max(size) * resolution).astype(int)
    lon, lat = numpy.meshgrid(
        numpy.linspace(lon0, lon1, lon_res),
        numpy.linspace(lat0, lat1, lat_res)
    )
    val = func(lon, lat)

    proj = cartopy.crs.PlateCarree()
    if ax is None:
        ax = matplotlib.pyplot.axes(projection=proj)

    # 根据插值结果绘制等值线图
    extent = (lon0, lon1, lat0, lat1)
    cs = (ax.contourf if fill else ax.contour)(
        val,
        extent=extent,
        transform=proj,
        **kwargs
    )

    if clip is not None:
        # 根据传入的图形裁剪等值线图
        clip_paths(cs.collections, clip, extent=extent)

    return ax, cs

def isogloss(
    latitudes,
    longitudes,
    values,
    extent=None,
    scale=0,
    **kwargs
):
    """
    绘制同言线地图.

    输入参数为一系列样本点坐标，及样本点符合某个语言特征的程度值，通常取值范围为 [0, 1]。
    使用径向基函数根据样本点插值，计算整个绘制空间的值，然后据此计算等值线。
    如果指定了裁剪范围，只绘制该范围内的等值线。

    Parameters:
        latitudes (`numpy.ndarray`): 样本点的纬度数组
        longitudes (`numpy.ndarray`): 样本点的经度数组
        values (`numpy.ndarray`): 样本点的值，通常取值范围为 [0, 1]
        extent (array-like): 绘制的范围 (左, 右, 下, 上)
        scale (float): 当未指定绘制范围时，用于根据样本点计算范围的系数
        kwargs: 透传给 `matplotlib.pyplot.Axes.contourf`

    Returns:
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象
        extent: 绘制的范围
        cs (`matplotlib.contour.QuadContourSet`): 绘制的等值线集合
    """

    mask = numpy.all([
        numpy.isfinite(latitudes),
        numpy.isfinite(longitudes),
        numpy.isfinite(values)
    ], axis=0)
    latitudes = latitudes[mask]
    longitudes = longitudes[mask]
    values = values[mask]

    if extent is None:
        lat0, lat1, lon0, lon1 = auxiliary.extent(latitudes, longitudes, scale)
        extent = (lon0, lon1, lat0, lat1)
    else:
        lon0, lon1, lat0, lat1 = extent

    # 针对完全相同的经纬度，对经度稍作偏移，使能正常计算
    longitudes = pandas.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes
    }).groupby(['latitude', 'longitude'])['longitude'] \
        .transform(lambda x: x + numpy.arange(x.shape[0]) * 1e-4).values

    # 使用径向基函数基于样本点对选定范围进行插值
    rbf = scipy.interpolate.Rbf(longitudes, latitudes, values, function='linear')
    ax, cs = _isogloss(lat0, lat1, lon0, lon1, clip(rbf), **kwargs)
    return ax, extent, cs

def isoglosses(
    data,
    latitudes,
    longitudes,
    values,
    extent,
    names=None,
    ax=None,
    proj=cartopy.crs.PlateCarree(),
    background=None,
    geo=None,
    levels=[0.5],
    cmap='tab10',
    **kwargs
):
    if ax is None:
        ax = matplotlib.pyplot.axes(projection=proj)

    # 绘制背景图政区边界
    if background is not None:
        ax.imshow(
            background,
            transform=proj,
            extent=[-180, 180, -90, 90]
        )

    if geo is not None:
        geo = tuple(geo)
        ax.add_geometries(geo, proj, edgecolor='gray', facecolor='none')

    if isinstance(cmap, str):
        cmap = matplotlib.pyplot.colormaps[cmap]

    for i, val in enumerate(values):
        isogloss(
            data[latitudes],
            data[longitudes],
            data[val],
            ax=ax,
            fill=False,
            cmap=None,
            vmin=0,
            vmax=1,
            extent=extent,
            clip=geo,
            levels=levels,
            colors=[cmap(i)]
        )

    if names is not None:
        scatter(
            data[latitudes],
            data[longitudes],
            ax=ax,
            extent=extent,
            clip=geo,
            marker='.',
            color='black'
        )

        for _, r in data[
            (data[longitudes] > extent[0]) \
            & (data[longitudes] < extent[1])
            & (data[latitudes] > extent[2]) \
            & (data[latitudes] < extent[3])
        ].iterrows():
            ax.annotate(r[names], xy=(r[longitudes], r[latitudes]))

    ax.set_extent(extent, crs=proj)
    return ax

def interactive_scatter(
    latitudes: numpy.ndarray[float] | pandas.Series,
    longitudes: numpy.ndarray[float] | pandas.Series,
    values: numpy.ndarray | pandas.Series,
    m: folium.Map | None = None,
    tips: dict[str, numpy.ndarray | pandas.Series] = {},
    marker_kwds: dict = {'radius': 5},
    **kwargs
) -> folium.Map:
    """
    绘制可交互的方言地图

    Parameters:
        latitudes: 方言点的纬度数组
        longitudes: 方言点的经度数组
        values: 方言点的属性值，可为连续或离散值
        m: 指定在现有的地图上绘制
        tips: 鼠标悬停在方言点上时显示的提示，每组元素对应一个标签
        marker_kwds, kwargs: 透传给 `geopandas.GeoDataFrame.explore`

    Returns:
        m: 生成的地图对象
    """

    mask = numpy.isfinite(longitudes) & numpy.isfinite(latitudes)
    location = geopandas.GeoDataFrame(
        dict([(k, v[mask]) for k, v in tips.items()], value=values[mask]),
        geometry=geopandas.points_from_xy(longitudes[mask], latitudes[mask])
    )

    return location[location['value'].notna()].explore(
        column='value',
        m=m,
        marker_kwds=marker_kwds,
        **kwargs
    )

def interactive(
        latitudes,
        longitudes,
        values,
        m=None,
        zoom=5,
        tiles='https://wprd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scl=2&style=7&x={x}&y={y}&z={z}',
        attr='高德地图',
        names=None,
        tips={},
        show=None,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        legend=False,
        **kwargs
):
    """
    使用 folium 绘制交互式方言地图.

    对每个方言点的颜色由一个连续的分值确定，一般代表该方言点对某条语音规则的遵从程度。

    Parameters:
        latitudes (`numpy.ndarray`): 方言点的纬度数组
        longitudes (`numpy.ndarray`): 方言点的经度数组
        values (`pandas.DataFrame` 或 `numpy.ndarray`): 指定方言点的分值
        m (`folium.Map`): 指定在现有的地图上绘制
        zoom (int): 地图缩放级别
        tiles (str): 地图底图瓦片的 URL 格式，地图交互时通过此格式获取所需经纬度的瓦片，
            默认为高德地图街道图
        attr (str): 用于地图上显示的声明等
        names (list-like): 用于在地图上显示的图层名称，每组分值一个
        tips (dict of (str, `numpy.ndarray`)): 鼠标悬停在方言点上时显示的提示，每组元素对应一个标签
        show (bool): 是否显示图层，默认只显示第一组分值
        cmap, vmin, vmax, legend, kwargs: 透传给 `geopandas.GeoDataFrame.explore`

    Returns:
        m (`folium.Map`): 绘制的地图对象
    """

    if names is None:
        names = (values.columns.values if isinstance(values, pandas.DataFrame) \
            else numpy.arange(values.shape[1])).astype(str)

    if isinstance(values, pandas.DataFrame):
        values = values.values

    if m is None:
        # 创建底图
        m = folium.Map(
            location=(numpy.mean(latitudes), numpy.mean(longitudes)),
            tiles=None,
            zoom_start=zoom
        )

    # 添加底图
    folium.TileLayer(tiles=tiles, attr=attr, name='background').add_to(m)

    # 为每一组分值添加一个图层
    for i in range(values.shape[1]):
        interactive_scatter(
            latitudes,
            longitudes,
            values[:, i],
            m=m,
            tips=tips,
            name=names[i],
            show=(i == 0) if show is None else show,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            legend=legend,
            **kwargs
        )

    # 添加选择图层的控件
    folium.LayerControl().add_to(m)
    return m
