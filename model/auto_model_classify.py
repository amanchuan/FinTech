import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV

class AutoMachineLearning(object):
    """
    对给定的数据集，从候选模型集中搜索最佳模型
    """
    def __init__(self
                ,base_data:pd.DataFrame()
                ,features:list
                ,target:str
                ,cand_models:list
                ):
        # super(AutoMachineLearning, self).__init__()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            base_data[features], base_data[target], test_size=0.2, random_state=2022)
        self.features = features
        self.target = target
        self.cand_models = cand_models

    def _search_tree_model(self,model):
        super_param = {
            'max_depth': np.arange(2,5,1),
            # 'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}],
            'class_weight': ['balanced'],
            'min_samples_leaf': [0.05],
            }
        tree_search = GridSearchCV(
            estimator=model(),
            param_grid=super_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
        tree_search.fit(self.x_train, self.y_train)
        print('决策树模型最优得分 {0},\n最优参数{1}'.format(tree_search.best_score_,tree_search.best_params_))
        tree_model = model(max_depth = tree_search.best_params_['max_depth'],
            class_weight=tree_search.best_params_['class_weight'])
        tree_model.fit(self.x_train, self.y_train)

        return tree_model
    
    def main_search(self):
        for model in self.cand_models:
            ret_model = self._search_tree_model(model)

        return ret_model
