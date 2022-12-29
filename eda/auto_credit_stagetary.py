# ==========================================================================
# * 非标准化流程：1.查看X变量的类型分布；2.查看X变量的取值空间分布；3.查看Y标签的值映射关系。
# *    TODO:配置空值标识、明确连续变量和离散变量。
# * 进入自动化信贷策略的标准流程：描述性统计、分箱及IV计算、相关性统计、PSI统计。
#      挑选信息量大、可解释性强、风险区分能力强、时间上稳定的X变量，根据风险倍数确定阈值，输出单维度策略。
#      挑选风险识别能力一般、可解释性欠佳的变量，通过决策树构造多维度策略。
# ==========================================================================


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split


THE_EMPTY_FLAG = [np.nan, 'null', 'NULL', -999, -999.00, '-999', '-999.00']
TO_EMPTY_STATUS = np.nan
# TO_EMPTY_STATUS = -9999
TARGET = 'target'

class AutoCreditStrategy(object):
    """
    1.描述性统计：区分连续变量、离散变量。
    2.等频分箱及bad_rate、lift统计。
    """
    def __init__(
                self
                ,base_data: pd.DataFrame()
                ,id: str
                ,features: list
                ,target: str  # 取值空间为[0,1]
                ,if_reduce_mem = False
                ):
        super(AutoCreditStrategy, self).__init__()
        self.df = base_data
        self.id = id
        self.features = features
        self.target = target
        self.the_empty_flag = THE_EMPTY_FLAG
        self.to_empty_status = TO_EMPTY_STATUS
        self.if_reduce_mem = if_reduce_mem

    def preprocess(self):
        self.df.replace(self.the_empty_flag, self.to_empty_status, inplace=True)
        # self.df = self.df.apply(pd.to_numeric, errors='ignore')
        for feat in self.features:
            self.df[feat] = self.df[feat].apply(pd.to_numeric, errors='ignore')
        # if self.if_reduce_mem:
        #     self.df = reduce_mem_usage(self.df)
        # 手工特征规则(看原始数据)
        self.has_preprocessed = True

    # @property
    def desc_by_dataframe(self):
        if not hasattr(self, 'has_preprocessed'):
            self.preprocess()
        numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
        discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2', 'bottom1']
        desc_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]
        
        rows = []
        for name, series in self.df[self.features].items():
            if series.dtype.kind in 'ifc':
                desc = self._getDescribe(
                    series=series
                    ,p_list=[.01, .1, .5, .75, .9, .99]
                ).tolist()
                top1 = np.nan
            else:
                top5 = self._getTopValues(series)
                bottom5 = self._getTopValues(series, reverse = True)
                desc = top5.tolist() + bottom5[::-1].tolist()
                top1 = float(top5['top1'].split(":")[1].strip("%"))/100  # 最大单一值占比
            nblank, pblank = self._countBlank(series)
            row = pd.Series(
                index = ['type', 'count', 'missing', 'nunique'] + desc_index,
                data = [series.dtype, series.size, pblank, series.nunique()] + desc
            )
            row["top1"] = top1
            row.name = name
            rows.append(row)
        # print("output statistic describe of df!")
        self.desc_statis = pd.DataFrame(rows)

    def _getDescribe(self, series, p_list=[.25, .5, .75]):
        d = series.describe(p_list)
        return d.drop('count')

    def _getTopValues(self, series, top = 5, reverse = False):
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

    def _countBlank(self, series, blanks = [None]):
        n = series.isnull().sum()
        # return (n, "{0:.2%}".format(n / series.size))
        return (n, n / series.size)

    def _cont_feature_bin(self,x,y,bins=20):
        """连续变量分箱"""
        total_size = y.count()
        total_bad = y.sum()
        total_good = total_size-total_bad
        total_bad_rate = total_bad/total_size
        try:
            bin_df = pd.DataFrame({'x':x,'y':y,'bucket':pd.qcut(x.to_numpy(),bins, duplicates='drop')}) # 等频分箱
            # bin_df = pd.DataFrame({'x':x,'y':y,'bucket':pd.cut(x.to_numpy(),bins, duplicates='drop')})  # 等距分箱
            bin_df = bin_df.groupby('bucket', dropna=False, as_index=True)  # sort by key。空值单独为一箱
        except:
            print("binning feature: {0} failed!!!".format(x))
        woe_df = pd.DataFrame(bin_df.size())
        woe_df.columns=['total']
        woe_df['bin_low'] = bin_df.x.min()
        woe_df['bin_up'] = bin_df.x.max()
        woe_df['bad'] = bin_df.y.sum()
        woe_df['good'] = woe_df['total']-woe_df['bad']
        woe_df['bad_rate'] = woe_df['bad']/woe_df['total']

        woe_df['cum_total'] = woe_df['total'].cumsum()
        woe_df['cum_bad'] = woe_df['bad'].cumsum()
        # woe_df['lift'] = woe_df['cum_bad']/woe_df['cum_total']/total_bad_rate
        woe_df['lift'] = woe_df['bad_rate']/total_bad_rate

        woe_df['bad%'] = woe_df['bad']/total_bad
        woe_df['bad%'] = woe_df['bad%'].apply(lambda x:1e-5 if x==0 else x)
        woe_df['good%'] = woe_df['good']/total_good
        woe_df['good%'] = woe_df['good%'].apply(lambda x:1e-5 if x==0 else x)
        woe_df['woe'] = np.log(woe_df['good%']/woe_df['bad%'])
        woe_df['iv'] = (woe_df['good%']-woe_df['bad%'])*woe_df['woe']
        return woe_df

    def _disc_feature_bin(self,x,y):
        """离散变量分箱"""
        total_size = y.count()
        total_bad = y.sum()
        total_good = total_size-total_bad
        total_bad_rate = total_bad/total_size
        bin_df = pd.DataFrame({'x':x,'y':y})
        bin_df = bin_df.groupby('x', dropna=False)  # default: sort by key。空值单独一箱
        woe_df = pd.DataFrame(bin_df.size())
        woe_df.columns=['total']
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
        return woe_df

    def bins_by_dataframe(self, bin_type=''):
        """各分箱上的描述统计"""
        # 单个分箱上的样本数不低于5%；单个分箱上的正负样本数不为0；
        # woe值应该按照分箱顺序单调；缺失值单独为一箱；
        self.total_woe_df = pd.DataFrame()
        for feat in self.features:
            if self.df[feat].dtype.kind in 'ifc':
                woe_df = self._cont_feature_bin(self.df[feat], self.df[self.target])
            else:
                woe_df = self._disc_feature_bin(self.df[feat], self.df[self.target])
            woe_df = woe_df.reset_index().rename(columns={woe_df.index.name:'bin'})
            woe_df['feature'] = feat
            cols = ['feature','bin','total','bad','bad_rate','lift','woe','iv']
            self.total_woe_df = pd.concat([self.total_woe_df,woe_df[cols]],axis=0)
        return
    
    def generate_single_rules(self):
        """
        根据self.total_woe_df
        寻找lift高, bad_rate及lift排序性好的变量, 根据分箱边界生成单维度策略阈值。
        """
        self.total_woe_df.reset_index(inplace=True)
        self.total_woe_df['rank'] = self.total_woe_df.groupby('feature',as_index=True)['index'].rank(ascending=False)-1
        features_size =  self.total_woe_df.groupby('feature',as_index=True)['index'].size().to_frame(name='size')
        self.total_woe_df = self.total_woe_df.merge(features_size, how='left', on='feature')
        self.total_woe_df['rank'] = self.total_woe_df['size'] - np.abs(self.total_woe_df['rank']-self.total_woe_df['index'])
        # 头尾两箱存在lift>=3的变量，纳入单维度策略。
        # ringle_rule = self.total_woe_df[(self.total_woe_df['lift']>=3) & (self.total_woe_df['rank']<5)]['feature']
        ringle_rule_feat = self.total_woe_df[(self.total_woe_df['lift']>=3) & (self.total_woe_df['rank']<5)]['feature'].unique() # 有效变量

        self.total_woe_df.drop(['rank','size'], axis=1, inplace=True)
        # return ringle_rule
        return self.total_woe_df[self.total_woe_df['feature'].isin(ringle_rule_feat)]

    def generate_multi_rules(self, tree, feature_names, class_names, threshold=0.32):
        """
        根据决策树模型（基于较高lift或bad_rate的U型变量，所构造的决策树）
        抽取多维规则。
        """
        from sklearn.tree import _tree
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature]

        paths = []
        path = []
        def recurse(node, path, paths):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node], tree_.impurity[node])] # 样本分布、样本个数、gini值
                paths += [path]
                
        recurse(0, path, paths)
        
        # 从 tree_model.tree_.n_leaves条规则中，抽取m条gini指数小于0.32（对应80%的坏客户概率）的规则。
        paths = [p for p in paths if p[-1][-1]<threshold]
        samples_count = [p[-1][1] for p in paths]      # 样本个数
        sorted_index = list(np.argsort(samples_count)) # 样本个数排序后的索引
        paths = [paths[i] for i in reversed(sorted_index)]
        # print(paths)
        rules = []
        for path in paths: # N条规则
            rule = "if "
            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: "+str(np.round(path[-1][0][0][0],3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}, sample_cnt: {path[-1][1]})"
            # rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]
        return rules


        