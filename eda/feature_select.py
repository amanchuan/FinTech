import pandas as pd
import numpy as np
from .desc_stats import show_desc
from .bin_iv_woe import cont_feature_bin,disc_feature_bin,woe_transform
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split


class feature_select(object):
    """
    特征选择:缺失值与最大单一值占比,相关系数,IV,PSI,VIF,bad_rate单调性,逐步回归,模型系数符号与显著性.
    前序步骤:数据预处理
    """
    def __init__(self,df,feats,target,random=3017,verbose=True,bins=10):
        self.df = df
        self.feats = feats
        self.target = target
        self.random = random
        self.verbose = verbose
        self.bins = bins
        
        self.df.reset_index(drop=True, inplace=True)
        self.train_idx,self.test_idx = train_test_split(self.df.index, test_size=0.3, random_state=self.random)
        
    def bins_and_iv(self, cont_bin='cut', dist_bin='cut'):
        """训练集上，各分箱上的描述统计"""
        # 单个分箱上的样本数不低于5%；单个分箱上的正负样本数不为0；
        # woe值应该按照分箱顺序单调；缺失值单独为一箱；
        self.bin_iv_woe_df = pd.DataFrame()
        for feat in self.feats:
            if self.df.loc[self.train_idx, feat].dtype.kind in 'ifc':
                woe_df = cont_feature_bin(self.df.loc[self.train_idx, [feat, self.target]], bin_type=cont_bin, bins=10)
            else:
                woe_df = disc_feature_bin(self.df.loc[self.train_idx, [feat, self.target]], bin_type=dist_bin, bins=10)
            woe_df = woe_df.reset_index().rename(columns={woe_df.index.name:'bin'})
            woe_df['feature'] = feat
            # cols = ['feature','bin', 'total', 'bad', 'bad_rate', 'lift', 'woe', 'iv']
            cols = ['feature','bin', 'bin_low', 'bin_up', 'total', 'bad', 'good', 'bad_rate', 'cum_total', 'cum_bad', 'lift', 'bad%', 'good%', 'woe', 'iv']
            self.bin_iv_woe_df = pd.concat([self.bin_iv_woe_df,woe_df[cols]],axis=0)

    def _filter_by_desc(self, missing=0.8, freq=0.9):
        """总体数据上选择"""
        self.desc_df = show_desc(self.df[self.feats], summary=False, detail=True)
        filter_feats = self.desc_df[(self.desc_df['null%']<missing)&(self.desc_df['freq%']<freq)].index.to_list()
        drop_feats = [feat for feat in self.feats if feat not in filter_feats]
        
        self.feats = filter_feats
        if self.verbose:
            print("选择缺失值占比小于{0}且最大单一值占比小于{1}的特征：{2}个。剔除变量{3}个。".format(missing, freq, len(self.feats), len(drop_feats)))

    def _filter_by_bins_iv(self, iv_filter=0.02):
        """"训练集上,iv筛选"""
        self.iv_df = self.bin_iv_woe_df[['feature', 'iv']].groupby('feature').agg('sum').squeeze() # IV求和
        filter_feats = self.iv_df>=iv_filter
        filter_feats = self.iv_df[filter_feats].index.values
        drop_feats = [feat for feat in self.feats if feat not in filter_feats]
        
        self.feats = filter_feats
        # self.iv_df = self.iv_df[filter_feats]
        if self.verbose:
            print("选择分箱后IV大于等于{0}的特征：{1}个。剔除变量{2}个。".format(iv_filter, len(self.feats), len(drop_feats)))


    def _filter_by_corr(self, corr_filter=0.8):
        """训练集上,相关性筛选,提出相关性较高的变量中,iv较低的"""
        corr = self.df.loc[self.train_idx, self.feats].corr() # 皮尔逊相关系数
        features = corr.columns
        feature_num = corr.shape[0]
        feat_mask = np.full((feature_num,), True, dtype=bool)  # 保留变量

        for i in range(feature_num):
            if not feat_mask[i]: # 如果第i个feature已经被剔除，则继续看其他feature
                continue
            for j in range(i+1, feature_num):
                if feat_mask[j] and np.abs(corr.iloc[i,j]) > corr_filter:
                    if self.iv_df[features[i]]>=self.iv_df[features[j]]:
                        feat_mask[j] = False
                        continue
                    else:
                        feat_mask[i] = False
                        break
                else:
                    continue
        self.feats = features[feat_mask].values
        drop_feats = features[~feat_mask].values
        if self.verbose:
            print("选择相关系数小于等于{0}的特征：{1}个。剔除变量{2}个。".format(corr_filter, len(self.feats), len(drop_feats)))


    def _filter_by_psi(self, psi_filter=0.1):
        """
        训练集、测试集进行woe转换后,psi筛选
        """
        self.train = woe_transform(self.df.loc[self.train_idx, self.feats], self.bin_iv_woe_df)
        self.test = woe_transform(self.df.loc[self.test_idx, self.feats], self.bin_iv_woe_df)

        total_psi_df = pd.DataFrame()
        for feat in self.feats:
            train_psi_df = self.train.groupby(feat, as_index=True)
            train_psi_df = train_psi_df.size().to_frame('train_cnt')
            train_sum = train_psi_df['train_cnt'].sum()
            train_psi_df['train_cnt'] = train_psi_df['train_cnt']/train_sum

            test_psi_df = self.test.groupby(feat, as_index=True)
            test_psi_df = test_psi_df.size().to_frame('test_cnt')
            test_sum = test_psi_df['test_cnt'].sum()
            test_psi_df['test_cnt'] = test_psi_df['test_cnt']/test_sum

            # psi_df = pd.concat([train_psi_df,test_psi_df], axis=1, join='left')
            psi_df = train_psi_df.merge(test_psi_df, how='left', left_index=True, right_index=True)
            psi_df.index.name='value'
            psi_df.reset_index(inplace=True)
            psi_df['feature'] = feat
            psi_df['psi'] = (psi_df['train_cnt']-psi_df['test_cnt'])*np.log(psi_df['train_cnt']/psi_df['test_cnt'])
            total_psi_df = pd.concat([total_psi_df, psi_df],axis=0)
        # return total_psi_df[['feature', 'value', 'train_cnt', 'test_cnt', 'psi']]
        self.features_psi = total_psi_df[['feature','psi']].groupby('feature').agg('sum').squeeze()
        filter_feats = self.features_psi[self.features_psi<=psi_filter].index.values
        drop_feats = self.features_psi[self.features_psi>psi_filter].index.values
        self.feats = filter_feats
        if self.verbose:
            print("选择PSI小于等于{0}的特征：{1}个。剔除变量{2}个。".format(psi_filter, len(filter_feats), len(drop_feats)))


    def _filter_by_vif(self, vif_filter=10):
        """训练集上,vif筛选"""
        assert len(self.feats)==self.train.shape[1],"woe转换后的特征数应该与最终筛选的特征数一致!"
        vif = [variance_inflation_factor(self.train, i) for i in range(len(self.feats))]
        self.vif = pd.Series(vif, index=self.feats)
        filter_feats = self.vif[self.vif<=vif_filter].index.values
        drop_feats = self.vif[self.vif>vif_filter].index.values
        self.feats = filter_feats
        if self.verbose:
            print("选择VIF小于等于{0}的特征：{1}个。剔除变量{2}个。".format(vif_filter, len(filter_feats), len(drop_feats)))