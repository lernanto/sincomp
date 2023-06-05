# -*- coding: utf-8 -*-

"""
绘制方言地图相关函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pandas
import numpy
import scipy.interpolate
import geopandas
import matplotlib.pyplot
import cartopy.crs
import folium

from .. import auxiliary


def scatter(
    latitudes,
    longitudes,
    values=None,
    ax=None,
    extent=None,
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
        min_lat = numpy.min(latitudes)
        max_lat = numpy.max(latitudes)
        min_lon = numpy.min(longitudes)
        max_lon = numpy.max(longitudes)
        lat_margin = 0.05 * (max_lat - min_lat)
        lon_margin = 0.05 * (max_lon - min_lon)
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin
        extent = (min_lon, max_lon, min_lat, max_lat)

    proj = cartopy.crs.PlateCarree()
    if ax is None:
        ax = matplotlib.pyplot.axes(projection=proj)

    # 根据传入的图形裁剪点集
    pc = ax.scatter(
        longitudes,
        latitudes,
        c=values,
        clip_path=(auxiliary.make_clip_path(clip, extent=extent), ax.transData),
        **kwargs
    )
    return ax, extent, pc

def plot_primary_component(
    latitudes,
    longitudes,
    pc=None,
    ax=None,
    clip=None,
    **kwargs
):
    """
    绘制方言主成分地图.

    方言的主成分是根据方言数据矩阵主成分分析获得的向量，可以在低维空间上表示方言的距离。
    在绘制的地图上，根据方言点的主成分计算颜色，通过颜色的相似度来表示方言之间的相似度。

    Parameters:
        latitudes (`numpy.ndarray`): 方言点的纬度数组
        longitudes (`numpy.ndarray`): 方言点的经度数组
        pc (`numpy.ndarray`): 方言主成分，列数至少为2
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象，
            如果为空，创建一个新对象
        clip (`geopandas.GeoDataFrame`):
            裁剪的范围，只绘制该范围内的方言点，为空绘制所有方言点
        kwargs: 透传给 `scatter`

    Returns:
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象
        extent: 绘制的范围
        pc (`matplotlib.contour.QuadContourSet`): 绘制的点集合
    """

    if clip is not None:
        # 根据指定的地理边界筛选边界内部的点
        mask = geopandas.GeoDataFrame(
            geometry=geopandas.geometry_from_xy(latitudes, longitudes)
        ).within(clip).values
        latitudes = latitudes[mask]
        longitudes = longitudes[mask]
        pc = pc[mask]

    return scatter(
        latitudes,
        longitudes,
        ax=ax,
        color=auxiliary.pc2color(pc),
        **kwargs
    )

def area(
    latitudes,
    longitudes,
    values,
    ax=None,
    extent=None,
    clip=None,
    **kwargs
):
    """
    绘制方言分区图.

    Parameters:
        latitudes (`numpy.ndarray`): 样本点的纬度数组
        longitudes (`numpy.ndarray`): 样本点的经度数组
        values (`numpy.ndarray`): 样本点的分类，为整数
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象，
            如果为空，创建一个新对象
        extent: 绘制的范围 (左, 右, 下, 上)
        clip (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只绘制该范围内的分区，为空绘制整个绘制范围的分区
        kwargs: 透传给 `matplotlib.pyplot.Axes.pcolormesh`

    Returns:
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象
        extent: 绘制的范围
        qm (`matplotlib.collectoins.QuadMesh`): 绘制的色块集
    """

    if extent is None:
        # 根据样本点确定绘制边界
        min_lat = numpy.min(latitudes)
        max_lat = numpy.max(latitudes)
        min_lon = numpy.min(longitudes)
        max_lon = numpy.max(longitudes)
        lat_margin = 0.05 * (max_lat - min_lat)
        lon_margin = 0.05 * (max_lon - min_lon)
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin
        extent = (min_lon, max_lon, min_lat, max_lat)

    else:
        min_lon, max_lon, min_lat, max_lat = extent

    # 使用最近邻插值计算绘制范围内每个点的分类
    mask = numpy.isfinite(values)
    lon, lat = numpy.meshgrid(
        numpy.linspace(min_lon, max_lon, 1000),
        numpy.linspace(min_lat, max_lat, 1000)
    )
    interp = scipy.interpolate.griddata(
        numpy.stack([longitudes[mask], latitudes[mask]], axis=1),
        values[mask],
        (lon, lat),
        method='nearest'
    )

    proj = cartopy.crs.PlateCarree()
    if ax is None:
        ax = matplotlib.pyplot.axes(projection=proj)

    # 根据插值结果绘制方言分区图，根据传入的图形裁剪
    qm = ax.pcolormesh(
        lon,
        lat,
        interp,
        transform=proj,
        clip_path=(auxiliary.make_clip_path(clip, extent=extent), ax.transData),
        **kwargs
    )
    return ax, extent, qm

def isogloss(
    latitudes,
    longitudes,
    values,
    ax=None,
    fill=True,
    cmap='coolwarm',
    vmin=None,
    vmax=None,
    extent=None,
    clip=None,
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
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象，
            如果为空，创建一个新对象
        cmap: 等值线图使用的颜色集
        vmin (float): 最小值，用于裁剪插值结果及作图
        vmax (float): 最大值，用于裁剪插值结果及作图
        extent: 绘制的范围 (左, 右, 下, 上)
        clip (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只绘制该范围内的等值线，为空绘制整个绘制范围的等值线
        kwargs: 透传给 `matplotlib.pyplot.Axes.contourf`

    Returns:
        ax (`cartopy.mpl.geoaxes.GeoAxes`): 作图使用的 GeoAxes 对象
        extent: 绘制的范围
        cs (`matplotlib.contour.QuadContourSet`): 绘制的等值线集合
    """

    if extent is None:
        # 根据样本点的中心和标准差计算绘制范围
        lat_mean = numpy.mean(latitudes)
        lon_mean = numpy.mean(longitudes)
        lat_std = numpy.std(latitudes)
        lon_std = numpy.std(longitudes)
        # 正态分布的左右2个标准差覆盖了95%以上的样本
        min_lat, max_lat = lat_mean - 2 * lat_std, lat_mean + 2 * lat_std
        min_lon, max_lon = lon_mean - 2 * lon_std, lon_mean + 2 * lon_std
        extent = (min_lon, max_lon, min_lat, max_lat)

    else:
        min_lon, max_lon, min_lat, max_lat = extent

    # 针对完全相同的经纬度，对经度稍作偏移，使能正常计算
    longitudes = pandas.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes
    }).groupby(['latitude', 'longitude'])['longitude'] \
        .transform(lambda x: x + numpy.arange(x.shape[0]) * 1e-4).values

    # 使用径向基函数基于样本点对选定范围进行插值
    mask = numpy.isfinite(values)
    rbf = scipy.interpolate.Rbf(
        longitudes[mask],
        latitudes[mask],
        values[mask],
        function='linear'
    )
    lon, lat = numpy.meshgrid(
        numpy.linspace(min_lon, max_lon, 100),
        numpy.linspace(min_lat, max_lat, 100)
    )
    val = rbf(lon, lat)

    # 限制插值的结果
    min_val = numpy.min(values[mask]) if vmin is None else vmin
    max_val = numpy.max(values[mask]) if vmax is None else vmax
    val = numpy.clip(val, min_val, max_val)

    proj = cartopy.crs.PlateCarree()
    if ax is None:
        ax = matplotlib.pyplot.axes(projection=proj)

    # 根据插值结果绘制等值线图
    cs = (ax.contourf if fill else ax.contour)(
        lon,
        lat,
        val,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=proj,
        **kwargs
    )

    if clip is not None:
        # 根据传入的图形裁剪等值线图
        auxiliary.clip_paths(cs.collections, clip, extent=extent)

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
            label=val,
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
    latitudes,
    longitudes,
    values,
    m=None,
    tips={},
    marker_kwds={'radius': 5},
    **kwargs
):
    """
    绘制可交互的方言地图.

    Parameters:
        latitudes (`numpy.ndarray`): 方言点的纬度数组
        longitudes (`numpy.ndarray`): 方言点的经度数组
        values (`numpy.ndarray`): 方言点的属性值，可为连续或离散值
        m (`folium.Map`): 指定在现有的地图上绘制
        tips (dict of (str, `numpy.ndarray`)): 鼠标悬停在方言点上时显示的提示，每组元素对应一个标签
        marker_kwds, kwargs: 透传给 `geopandas.GeoDataFrame.explore`

    Returns:
        m (`folium.Map`): 生成的地图对象
    """

    location = geopandas.GeoDataFrame(
        dict(tips, value=values),
        geometry=geopandas.points_from_xy(longitudes, latitudes)
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