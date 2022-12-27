import numpy as np
import pandas as pd

def capping(series, quantile=0.99, coef=2):
    """
    对特征异常值进行盖帽处理.
    盖帽:大于99%分位数2倍的值替换为99%分位数的2.5倍值。
    parm:
        df: 特征dataframe
    """
    if series.dtype.kind in 'ifc':
        point = series.quantile(quantile)
        series[series>point*coef]=point*coef*1.25
    return series

def flooring(series, quantile=0.01, coef=1/2):
    """
    对特征异常值进行托底处理.
    托底:小于1%分位数1/2倍的值替换为1%分位数的0.4倍值。
    parm:
        df: 特征dataframe
    """
    if series.dtype.kind in 'ifc':
        point = series.quantile(quantile)
        series[series<point*coef]=point*coef/1.25
    return series