import pandas as pd
import numpy as np
import itertools
import re


def str_replace(df):
    return df.apply(lambda x:x.replace('nan','\"nan\"').replace('\'','\"'))

def reg_exp(str_value):
    reg_str = ',(?=[0-9.: ]*$)' # 匹配以数字、小数点、冒号、空格结尾的最近一个','。
    parten = re.compile(reg_str)
    replaced = parten.sub('}', str_value)  # 替换','为'}'。肯定有返回值
    ret = re.match(r'{[\S\s]*}', replaced)  # 截断，只取{}及之内的字符串。肯定有返回值
    return ret.group()

def str_to_list(df):
    return df.apply(lambda x:list(eval(x).items()))
    # return df.apply(lambda x:list(x.items()))

def choice_idx(data_list,val):
    """返回val在data_list中第一次出现的索引"""
    return min([i for i in range(len(data_list)) if data_list[i]>=val])

def desc_by_dict(row:pd.DataFrame) -> list:
    """根据频数统计结果，进行描述性统计"""
    dict_data = row['cnt'].copy()
    if type(dict_data) == str:
        dict_data = eval(dict_data)
    if row['type'] == 'numeric':
        dict_data['null'] = dict_data.get('null',0)+dict_data.get(-999,0)
        dict_data[-999] = 0
        dict_data.pop(-999)
    
    desc_dict = {}
    total_cnt = sum(dict_data.values())
    desc_dict['total_cnt'] = total_cnt
    try:
        desc_dict['nan_rate'] = round(dict_data.get('null',0)/total_cnt,4)
        desc_dict['zero_rate'] = round(dict_data.get(0.0,0)/total_cnt,4)
    except:
        print(row['cnt'])
        print(dict_data)
    if 'null' in dict_data.keys():
        dict_data.pop('null')
    desc_dict['unique_cnt'] = len(dict_data)
    desc_dict['value_cnt'] = (0 if len(dict_data)<1 else sum(dict_data.values()))

    if len(dict_data)<1:
        desc_dict['max_value'] = np.nan
        desc_dict['min_value'] = np.nan
        desc_dict['mode_value'] = np.nan
        desc_dict['mode_value_rate'] = np.nan
        desc_dict['mean'] = np.nan
        desc_dict['p01'] = np.nan
        desc_dict['p05'] = np.nan
        desc_dict['p10'] = np.nan
        desc_dict['p25'] = np.nan
        desc_dict['p50'] = np.nan   
        desc_dict['p75'] = np.nan   
        desc_dict['p90'] = np.nan
        desc_dict['p95'] = np.nan
        desc_dict['p99'] = np.nan
        return list(desc_dict.values())
    value_cnt = sum(dict_data.values())
    if str in list(map(type,dict_data.keys())):
        desc_dict['max_value'] = np.nan
        desc_dict['min_value'] = np.nan
        desc_dict['mode_value'] = max(dict_data, key=lambda x:dict_data[x])
        desc_dict['mode_value_rate'] = round(max(dict_data.values())/value_cnt,4)
        desc_dict['mean'] = np.nan
        desc_dict['p01'] = np.nan
        desc_dict['p05'] = np.nan
        desc_dict['p10'] = np.nan
        desc_dict['p25'] = np.nan
        desc_dict['p50'] = np.nan
        desc_dict['p75'] = np.nan
        desc_dict['p90'] = np.nan
        desc_dict['p95'] = np.nan
        desc_dict['p99'] = np.nan
        return list(desc_dict.values())
    desc_dict['max_value'] = max(dict_data.keys())
    desc_dict['min_value'] = min(dict_data.keys())
    desc_dict['mode_value'] = max(dict_data, key=lambda x:dict_data[x]) # 众数
    desc_dict['mode_value_rate'] = round(max(dict_data.values())/value_cnt,4) # 众数占比
    desc_dict['mean'] = round(sum([k*v for k,v in dict_data.items()])/value_cnt,4)
    # 分位数
    dict_data = dict(sorted(dict_data.items()))
    keys = list(dict_data.keys())
    values = list(itertools.accumulate(dict_data.values()))
    p01,p05,p10,p25,p50 = value_cnt*0.01,value_cnt*0.05,value_cnt*0.1,value_cnt*0.25,value_cnt*0.5
    p75,p90,p95,p99 = value_cnt*0.75,value_cnt*0.90,value_cnt*0.95,value_cnt*0.99
    desc_dict['p01'] = keys[choice_idx(values,p01)]
    desc_dict['p05'] = keys[choice_idx(values,p05)]
    desc_dict['p10'] = keys[choice_idx(values,p10)]
    desc_dict['p25'] = keys[choice_idx(values,p25)]
    desc_dict['p50'] = keys[choice_idx(values,p50)]
    desc_dict['p75'] = keys[choice_idx(values,p75)]
    desc_dict['p90'] = keys[choice_idx(values,p90)]
    desc_dict['p95'] = keys[choice_idx(values,p95)]
    desc_dict['p99'] = keys[choice_idx(values,p99)]
    return list(desc_dict.values())



def _getTopValues(series, top = 5, reverse = False):
    itype = 'top'
    counts = series.value_counts()
    counts = list(zip(counts.index, counts, counts.divide(series.size)))
    if reverse:
        counts.reverse()
        itype = 'bottom'
    template = "{0[0]}:{0[2]:.2%}"  # 列表索引模式
    indexs = [itype + str(i + 1) for i in range(top)]
    values = [template.format(counts[i]) if i < len(counts) else None for i in range(top)]
    return pd.Series(values, index = indexs)

def _getDescribe(series, p_list=[.25, .5, .75]):
    d = series.describe(p_list)
    return d.drop('count')

def _countBlank(series, blanks = [None]):
    n = series.isnull().sum()
    # return (n, "{0:.2%}".format(n / series.size))
    return (n, n / series.size)


def show_desc(df,summary=True,detail=False):
    """
    标准化描述性统计。
    粗粒度：样本量与特征数量、特征数据类型、重复样本、含空值特征数量。
    细粒度：数据类型、缺失值数量、取值空间大小、取值空间与频数、取值分布。
    param:
        df: 特征dataframe
        summrer: 粗粒度
        detail: 细粒度
    """
    # df = df.apply(pd.to_numeric, errors='ignore')
    col_have_null = df.isnull().any()
    col_types = df.dtypes
    if summary:
        duplicates = df.duplicated().sum()
        print("原始数据集：",df.shape)
        print("重复行数：",duplicates)
        # df.drop_duplicates(inplace=True)
        # df.reset_index(drop=True,inplace=True)
        df = df.drop_duplicates()
        print("去重后数据集：",df.shape)
        print("包含空值的字段个数：",col_have_null.sum())
        print('数据类型分布：',col_types.value_counts().sort_values().to_dict())

    if detail:
        # 1.数据类型、是否有null、取值空间大小
        col_null_cnt = df.isnull().sum()
        count = df.count()
        nunique = df.nunique()
        # df_desc = df.describe().T
        col_types.name='type'
        col_have_null.name='has_null'
        col_null_cnt.name='null_cnt'
        count.name='count'
        nunique.name='nunique_notnull'

        # 2.取值空间与取值频数
        value_cnt = {c:df[c].value_counts().to_dict() for c in df.columns}
        # 3.最大单一值频数
        frequent_cnt = {c:df[c].value_counts().max() for c in df.columns}
        
        # 4.取值分布
        numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
        discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2', 'bottom1']
        dist_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]
        # df.describe()不计算空值。
        num_dist = df.select_dtypes(exclude=['object']).describe(percentiles=[0.01,0.1,0.5,0.75,0.9,0.99]).T # count、
        num_dist.drop('count', axis=1, inplace=True)
        num_dist.columns = dist_index

        rows=[]
        for name, series in df.select_dtypes(include=['object']).items():
            top5 = _getTopValues(series)
            bottom5 = _getTopValues(series, reverse = True)
            desc = top5.tolist() + bottom5[::-1].tolist()
            row = pd.Series(index = dist_index, data = desc)
            row.name = name
            rows.append(row)
        
        desc_df = pd.concat([col_types,col_have_null,col_null_cnt,count,nunique
            ,pd.Series(frequent_cnt,name='frequent')
            ,pd.Series(value_cnt,name='value_counts')
            ,pd.concat([num_dist,pd.DataFrame(rows)], axis=0)
            ],axis=1)
        desc_df['null%'] = desc_df['null_cnt']/df.shape[0]
        desc_df['freq%'] = desc_df['frequent']/desc_df['count']

        return desc_df
    return