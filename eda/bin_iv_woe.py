import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,_tree



def _chisquare(df,has_sort=True):
    """有序箱计算卡方值"""
    if has_sort:
        chi_table = np.array([])
        for i in np.arange(df.shape[0]-1):
            # 卡方值=power((ad-bc),2)*(a+b+c+d)/((a+b)*(c+d)*(a+c)*(b+d))，a,b,c,d分别表示两个区间的好坏样本数
            chi = (np.power((df.loc[i,'bad']*df.loc[i+1,'good']-df.loc[i,'good']*df.loc[i+1,'bad']),2)*\
                (df.loc[i,'total']+df.loc[i+1,'total']))/\
                (df.loc[i,'total']*df.loc[i+1,'total']*\
                    (df.loc[i,'bad']+df.loc[i+1,'bad'])*(df.loc[i,'good']+df.loc[i+1,'good']))
            chi_table = np.append(chi_table,chi)
    else:
        pass
    return chi_table

def ChiMerge(x_y_data, confidence=3.841, bins=10) -> pd.DataFrame:
    """卡方分箱"""
    # x = np.array(x)
    # y = np.array(y)
    x_y_data.columns=['x','y']
    total_size = x_y_data['y'].count()
    total_bad = x_y_data['y'].sum()
    total_good = total_size-total_bad
    total_bad_rate = total_bad/total_size

    # 1.基础频数统计
    grouped = x_y_data.groupby(['x'], dropna=False)
    value_df = grouped.size().to_frame('total')
    value_df['bad'] = grouped.sum()
    value_df['good'] = value_df['total'] - value_df['bad']
    value_df = value_df.reset_index()
    value_df.sort_values(by='x',inplace=True)
    value_df['x'] = value_df['x'].apply(lambda x:[x,])
    # 2.根据初始箱计算卡方值
    chi_table = _chisquare(value_df[['total','bad','good']])  # 与后一箱的卡方值
    # 3.自底向上，选择最小卡方值，合并箱，更新卡方值。
    while(1):
        min_chi_value = min(chi_table[pd.notna(chi_table)])
        if len(chi_table)<=(bins-1) or min_chi_value>=confidence:
            # print("低于分箱个数或卡方值低于合并阈值")
            break
        chi_min_index = np.squeeze(np.where(chi_table == min_chi_value)).tolist()
        if not np.issubdtype(type(chi_min_index),int):
            try:
                chi_min_index = chi_min_index[0]
            except:
                print(chi_min_index)
        # 合并箱。将后一箱合并到当前箱
        value_df.loc[chi_min_index,'total'] += value_df.loc[chi_min_index+1,'total']
        value_df.loc[chi_min_index,'bad'] += value_df.loc[chi_min_index+1,'bad']
        value_df.loc[chi_min_index,'good'] += value_df.loc[chi_min_index+1,'good']
        value_df.loc[chi_min_index,'x'].extend(value_df.loc[chi_min_index+1,'x'])  # 合并bin name
        value_df.drop(chi_min_index+1,axis=0,inplace=True)
        value_df.reset_index(drop=True,inplace=True)
        # 更新卡方值
        if chi_min_index == value_df.shape[0]-1: 
            # 当最小卡方值是最后两个区间时，更新倒数第二个卡方值为（倒数第三个分箱）与（合并后的倒数第一、二个分箱）的卡方值。
            chi_table[chi_min_index-1] = (np.power((value_df.loc[chi_min_index-1,'bad']*value_df.loc[chi_min_index,'good']-\
                                         value_df.loc[chi_min_index-1,'good']*value_df.loc[chi_min_index,'bad']),2)*\
                                         (value_df.loc[chi_min_index-1,'total']+value_df.loc[chi_min_index,'total']))/\
                                         (value_df.loc[chi_min_index-1,'total']*value_df.loc[chi_min_index,'total']*\
                                         (value_df.loc[chi_min_index-1,'bad']+value_df.loc[chi_min_index,'bad'])*\
                                         (value_df.loc[chi_min_index-1,'good']+value_df.loc[chi_min_index,'good']))
            chi_table = np.delete(chi_table,chi_min_index,axis=0)
        elif chi_min_index==0:
            # 当最小卡方值是前两个区间时，更新第二个卡方值为（第三个分箱）与（合并后的第一、二个分箱）的卡方值。
            chi_table[chi_min_index+1] = (np.power((value_df.loc[chi_min_index+1,'bad']*value_df.loc[chi_min_index,'good']-\
                                         value_df.loc[chi_min_index+1,'good']*value_df.loc[chi_min_index,'bad']),2)*\
                                         (value_df.loc[chi_min_index+1,'total']+value_df.loc[chi_min_index,'total']))/\
                                         (value_df.loc[chi_min_index+1,'total']*value_df.loc[chi_min_index,'total']*\
                                         (value_df.loc[chi_min_index+1,'bad']+value_df.loc[chi_min_index,'bad'])*\
                                         (value_df.loc[chi_min_index+1,'good']+value_df.loc[chi_min_index,'good']))
            chi_table = np.delete(chi_table,chi_min_index,axis=0)
        else:
            # 更新前一个卡方值为合并后分箱与前一个分箱的卡方值
            chi_table[chi_min_index-1] = (np.power((value_df.loc[chi_min_index-1,'bad']*value_df.loc[chi_min_index,'good']-\
                                         value_df.loc[chi_min_index-1,'good']*value_df.loc[chi_min_index,'bad']),2)*\
                                         (value_df.loc[chi_min_index-1,'total']+value_df.loc[chi_min_index,'total']))/\
                                         (value_df.loc[chi_min_index-1,'total']*value_df.loc[chi_min_index,'total']*\
                                         (value_df.loc[chi_min_index-1,'bad']+value_df.loc[chi_min_index,'bad'])*\
                                         (value_df.loc[chi_min_index-1,'good']+value_df.loc[chi_min_index,'good']))
            # 更新当前索引卡方值为合并后分箱和后一个分箱的卡方值
            chi_table[chi_min_index] = (np.power((value_df.loc[chi_min_index,'bad']*value_df.loc[chi_min_index+1,'good']-\
                                         value_df.loc[chi_min_index,'good']*value_df.loc[chi_min_index+1,'bad']),2)*\
                                         (value_df.loc[chi_min_index,'total']+value_df.loc[chi_min_index+1,'total']))/\
                                         (value_df.loc[chi_min_index,'total']*value_df.loc[chi_min_index+1,'total']*\
                                         (value_df.loc[chi_min_index,'bad']+value_df.loc[chi_min_index+1,'bad'])*\
                                         (value_df.loc[chi_min_index,'good']+value_df.loc[chi_min_index+1,'good']))
            chi_table = np.delete(chi_table,chi_min_index+1,axis=0)
    value_df['bad_rate'] = value_df['bad']/value_df['total']
    value_df['bad%'] = value_df['bad']/total_bad
    value_df['bad%'] = value_df['bad%'].apply(lambda x:1e-5 if x==0 else x)
    value_df['good%'] = value_df['good']/total_good
    value_df['good%'] = value_df['good%'].apply(lambda x:1e-5 if x==0 else x)
    value_df['woe'] = np.log(value_df['good%']/value_df['bad%'])
    value_df['iv'] = (value_df['good%']-value_df['bad%'])*value_df['woe']
    value_df.rename(columns={'x':'bucket'}, inplace=True)
    value_df['bin_low'] = value_df['bucket'].apply(lambda x:x[0])
    value_df['bin_up'] = value_df['bucket'].apply(lambda x:x[-1])
    value_df['cum_total'] = value_df['total'].cumsum()
    value_df['cum_bad'] = value_df['bad'].cumsum()
    value_df['lift'] = value_df['cum_bad']/value_df['cum_total']/total_bad_rate
    # value_df['feature'] = 'x'
    # value_sum = value_df.sum().to_frame().T
    # value_sum['value_bins'] = 'sum'
    # value_sum['feature'] = 'x'
    # cols = ['feature','value_bins','total','bad','bad_rate','badattr','goodattr','woe','iv']
    # value_df = pd.concat([value_df[cols],value_sum[cols]],axis=0)
    
    # value_df.loc[:, 'boundary'] = value_df['value_bins']
    value_df.set_index('bucket', inplace=True)
    cols = ['bin_low', 'bin_up', 'total', 'bad', 'good', 'bad_rate', 'cum_total',
        'cum_bad', 'lift', 'bad%', 'good%', 'woe', 'iv']
    
    return value_df[cols]


def cont_feature_bin(x_y_data, bin_type:list(['cut','qcut','best_iv','tree','chi']), bins=10):
    """
    连续变量分箱，空值单独为一箱。
    x_y_data: pd.DataFrame,两列: 一个特征和标签，特征在前，标签在后。
    """
    # x_y_data = np.array(x_y_data)
    x_y_data.columns=['x','y']
    total_size = x_y_data['y'].count()
    total_bad = x_y_data['y'].sum()
    total_good = total_size-total_bad
    total_bad_rate = total_bad/total_size

    mask = pd.isna(x_y_data['x'])

    if bin_type=='qcut':   # 等频分箱
        x_y_data['bucket'] = pd.qcut(x_y_data['x'], bins, duplicates='drop')
        # if mask.any():
    elif bin_type=='cut':  # 等距分箱
        x_y_data['bucket'] = pd.cut(x_y_data['x'], bins, duplicates='drop')
        # bin_df = pd.DataFrame({'x':x,'y':y,'bucket':pd.cut(x, bins, duplicates='drop')})
    elif bin_type=='best_iv':
        pass
    elif bin_type=='tree': # 单变量决策树分箱，自顶向下
        clf = DecisionTreeClassifier(criterion='entropy',    # entropy: “信息熵”最小化准则划分
                                    max_leaf_nodes=bins,     # 最大叶子节点数
                                    min_samples_leaf=0.05)   # 叶子节点样本数量最小占比
        clf.fit(x_y_data['x'][~mask].to_numpy().reshape(-1,1), x_y_data['y'][~mask].to_numpy())
        thresholds = clf.tree_.threshold
        thresholds = list(np.sort(thresholds[thresholds != _tree.TREE_UNDEFINED]))
        thresholds = [-np.inf]+thresholds+[np.inf]

        x_y_data['bucket'] = pd.cut(x_y_data['x'], thresholds, duplicates='drop')
        # bin_df = pd.DataFrame({'x':x,'y':y,'bucket':pd.cut(x, thresholds, duplicates='drop')})
    elif bin_type=='chi': # 卡方分箱，根据卡方值自底向上合并箱
        return ChiMerge(x_y_data, bins=bins)

    mask = pd.isna(x_y_data['bucket'])
    if mask.any():
        x_y_data['bucket'] = x_y_data['bucket'].astype(str)
        x_y_data.loc[mask, 'bucket'] = "(np.nan]"
    bin_df = x_y_data.groupby('bucket', dropna=False, as_index=True)  # sort by key。空值单独为一箱
    woe_df = pd.DataFrame(bin_df.size())
    woe_df.columns=['total']
    woe_df.reset_index(drop=False, inplace=True)
    # 取Category类型对象的左右元素
    woe_df['bin_low'] = woe_df['bucket'].apply(lambda x:eval(str(x).replace('inf', 'np.inf').strip('(]').split(',')[0]))
    woe_df['bin_up'] = woe_df['bucket'].apply(lambda x:eval(str(x).replace('inf', 'np.inf').strip('(]').split(',')[-1]))
    woe_df.set_index('bucket', inplace=True)
    woe_df['bad'] = bin_df.y.sum()
    woe_df['good'] = woe_df['total']-woe_df['bad']
    woe_df['bad_rate'] = woe_df['bad']/woe_df['total']

    woe_df['cum_total'] = woe_df['total'].cumsum()
    woe_df['cum_bad'] = woe_df['bad'].cumsum()
    woe_df['lift'] = woe_df['cum_bad']/woe_df['cum_total']/total_bad_rate

    woe_df['bad%'] = woe_df['bad']/total_bad
    woe_df['bad%'] = woe_df['bad%'].apply(lambda x:1e-5 if x==0 else x)
    woe_df['good%'] = woe_df['good']/total_good
    woe_df['good%'] = woe_df['good%'].apply(lambda x:1e-5 if x==0 else x)
    woe_df['woe'] = np.log(woe_df['good%']/woe_df['bad%'])
    woe_df['iv'] = (woe_df['good%']-woe_df['bad%'])*woe_df['woe']

    cols = ['bin_low', 'bin_up', 'total', 'bad', 'good', 'bad_rate', 'cum_total',
        'cum_bad', 'lift', 'bad%', 'good%', 'woe', 'iv']
    return woe_df[cols]

def disc_feature_bin(x_y_data, bin_type=None, bins=10):
    """离散变量分箱
    分箱类型bin_type可以是自底向上的卡方分箱，也可以单一取值分箱。
    """
    # x = np.array(x)
    # y = np.array(y)
    x_y_data.columns=['x','y']
    total_size = x_y_data['y'].count()
    total_bad = x_y_data['y'].sum()
    total_good = total_size-total_bad
    total_bad_rate = total_bad/total_size


    if bin_type == 'chi':
        return ChiMerge(x_y_data, bins=bins)
    else:
        # bin_df = pd.DataFrame({'x':x,'y':y})
        bin_df = x_y_data.groupby('x', dropna=False, as_index=True)  # default: sort by key。空值单独一箱
        woe_df = bin_df.size().to_frame('total')
        woe_df['bad'] = bin_df.y.sum()
        woe_df.reset_index(inplace=True)
        woe_df['x'] = woe_df['x'].apply(lambda x:[x,])
        woe_df['bin_low'] = woe_df.x.apply(lambda x:x[0])
        woe_df['bin_up'] = woe_df.x.apply(lambda x:x[0])
        woe_df.set_index('x', inplace=True)
        woe_df.index.name='bucket'
        woe_df['good'] = woe_df['total']-woe_df['bad']
        woe_df['bad_rate'] = woe_df['bad']/woe_df['total']
        
        woe_df['cum_total'] = woe_df['total'].cumsum()
        woe_df['cum_bad'] = woe_df['bad'].cumsum()
        woe_df['lift'] = woe_df['cum_bad']/woe_df['cum_total']/total_bad_rate

        woe_df['bad%'] = woe_df['bad']/total_bad
        woe_df['bad%'] = woe_df['bad%'].apply(lambda x:1e-5 if x==0 else x)
        woe_df['good%'] = woe_df['good']/total_good
        woe_df['good%'] = woe_df['good%'].apply(lambda x:1e-5 if x==0 else x)
        woe_df['woe'] = np.log(woe_df['good%']/woe_df['bad%'])
        woe_df['iv'] = (woe_df['good%']-woe_df['bad%'])*woe_df['woe']

        cols = ['bin_low', 'bin_up', 'total', 'bad', 'good', 'bad_rate', 'cum_total',
        'cum_bad', 'lift', 'bad%', 'good%', 'woe', 'iv']
        return woe_df[cols]

def woe_transform(df, bin_iv_woe_df):
    """根据分箱woe编码对数据集特征进行woe转换"""
    for feat in df.columns:
        try:
            if df[feat].dtype.kind in 'ifc':  # 连续变量转换
                c_woe_df = bin_iv_woe_df[bin_iv_woe_df['feature']==feat][['bin_low', 'bin_up', 'woe']]
                transformed = np.zeros(len(df[feat]))
                for id,row in c_woe_df.iterrows():
                    mask = df[feat].between(row['bin_low'], row['bin_up'], inclusive='right')
                    transformed[mask] = row['woe']
                nan_mask = pd.isna(df[feat])
                if nan_mask.any():
                    transformed[nan_mask] = c_woe_df[pd.isna(c_woe_df['bin_low'])]['woe']
                df.loc[:, feat] = transformed
            else:  # 离散变量转换
                c_woe_df = bin_iv_woe_df[bin_iv_woe_df['feature']==feat][['bin', 'woe']]
                transformed = np.zeros(len(df[feat]))
                for id,row in c_woe_df.iterrows():
                    mask = df[feat].isin(row['bin'])
                    transformed[mask] = row['woe']
                # nan_mask = np.isnan(df[feat])
                # if nan_mask.any():
                #     transformed[nan_mask] = c_woe_df[c_woe_df['bin']=='nan']['woe']
                df.loc[:, feat] = transformed
        except:
            print(feat)
            print(nan_mask.any())
            # print(pd.isna(c_woe_df['bin_low']))

    return df



def compute_ent(y):
    """信息熵"""
    value_cnt = y.value_counts(dropna=False)
    total_cnt = value_cnt.sum()
    ent = 0.0
    p_list = [val/total_cnt for val in value_cnt]
    logp = np.log2(p_list)
    ent -= sum([p*lp for p,lp in zip(p_list,logp)])
    return ent

def compute_cond_ent(x, y):
    """条件熵, 已知x后y的条件熵""" 
    value_cnt = x.value_counts(dropna=False)
    total_cnt = value_cnt.sum()
    # ent = 0.0
    # for k,val in value_cnt.items():
    #     sub_x = x[y==k]
    #     temp_ent = compute_ent(sub_x)
    #     ent += (val/total_cnt)*temp_ent
    p_list = [val/total_cnt for val in value_cnt]
    temp_ent_list = [compute_ent(y[x==k]) for k in value_cnt.keys()] #已知特征取值下，对应子集的信息熵。
    ent = sum([p*temp_ent for p,temp_ent in zip(p_list,temp_ent_list)])
    return ent

# # 信息增益倾向于取值较多的特征，取值越多，条件熵倾向于越小，从而增益越大。极端性况下，根据ID的条件熵为0，因此信息增益就是经验熵。
# ent = compute_ent(base_data['target'])
# cond_ent = compute_cond_ent(base_data['br_als_m6_id_nbank_oth_orgnum'], base_data['target'])
# ent_gain = ent - cond_ent

# # 信息增益率，倾向于取值空间小的特征，取值空间越小特征x的经验熵倾向于越小。
# x_ent = compute_ent(base_data['br_als_m6_id_nbank_oth_orgnum'])
# ent_gain_rate = cond_ent/x_ent



def compute_gini(x):
    """基尼指数，基尼不纯度"""
    value_cnt = x.value_counts(dropna=False).to_numpy()
    value_cnt = value_cnt/value_cnt.sum()
    return 1-(value_cnt**2).sum()

def compute_gini_split_str(x,y):
    """
    特征x相对于标签y，使得数据集基尼指数最小的取值划分.
    针对特征取值空间无序的特征，比如明显无顺的字符型变量。
    返回根据各个取值进行二分划分（每个值作为一个分割点），对应的基尼指数。
    """
    value_cnt = x.value_counts(dropna=False).sort_index()
    total_cnt = value_cnt.sum()
    gini_split = {k:compute_gini(y[x==k])*v/total_cnt + compute_gini(y[x!=k])*(total_cnt-v)/total_cnt for k,v in value_cnt.items()}
    return gini_split

def compute_gini_split_num(x,y):
    """
    特征x相对于标签y，使得数据集基尼指数最小的取值划分.
    针对特征取值空间有序的特征，比如数值型特征。
    返回根据各个取值进行二分划分（每个值作为一个分割点，左右划分），对应的基尼指数。
    """
    value_cnt = x.value_counts(dropna=False).sort_index()
    total_cnt = value_cnt.sum()
    gini_split = {k:compute_gini(y[x==k])*v/total_cnt + compute_gini(y[x!=k])*(total_cnt-v)/total_cnt for k,v in value_cnt.cumsum().items()}
    # 二分划分：x<=k 与 x>k。
    return gini_split

# # 基尼指数，根据随机变量，描述该随机变量的不纯度。越大越不纯，不确定性越高。
# gini = compute_gini((base_data['br_als_m6_id_nbank_oth_orgnum']))

# # 根据特征取值，不同分割点进行二分划分，对应的基尼指数。
# gini_split = compute_gini_split_num(base_data['br_als_m6_id_nbank_oth_orgnum'], base_data['target'])
# split_value = min(gini_split, key=gini_split.get) # 基尼指数最小时对应的特征分割点。



