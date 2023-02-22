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

def interactive_map(
    latitudes,
    longitudes,
    labels,
    tips,
    zoom=5,
    tiles='https://wprd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scl=2&style=7&x={x}&y={y}&z={z}',
    attr='AutoNavi'
):
    """
    绘制可交互的方言地图.

    Parameters:
        latitudes (`numpy.ndarray`): 方言点的纬度数组
        longitudes (`numpy.ndarray`): 方言点的经度数组
        labels (`numpy.ndarray`): 方言点的分类
        tips (`numpy.ndarray`): 方言点显示的标签
        zoom (int): 地图缩放级别
        tiles (str): 地图底图瓦片的 URL 格式，地图交互时通过此格式获取所需经纬度的瓦片
        attr  (str): 用于地图上显示的声明等

    Returns:
        mp (`folium.Map`): 生成的地图对象
    """

    # 绘制底图
    mp = folium.Map(
        location=(numpy.mean(latitudes), numpy.mean(longitudes)),
        zoom_start=zoom,
        tiles=tiles,
        attr=attr
    )

    # 添加坐标点
    for i in range(latitudes.shape[0]):
        # 根据标签序号分配一个颜色
        r = (labels[i] * 79 + 31) & 0xff
        g = (labels[i] * 37 + 43) & 0xff
        b = (labels[i] * 73 + 17) & 0xff

        folium.CircleMarker(
            (latitudes[i], longitudes[i]),
            radius=5,
            tooltip='{} {}'.format(labels[i], tips[i]),
            color='#{:02x}{:02x}{:02x}'.format(r, g, b),
            fill=True
        ).add_to(mp)

    return mp