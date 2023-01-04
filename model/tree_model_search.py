import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
# from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
# from lightgbm import LGBMClassifier,LGBMRegressor
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform



class TreeModelSearch(object):
    """
    树模型的通用搜索，包括:
    单颗决策树: decision_tree
    gbdt分类与回归树：gbdt_clf,gbdt_reg
    xgb;
    lgbm;
    """
    def __init__(self, X_y_df, super_parm=None, set_parm=None, model_type='gbdt_clf'):
        self.data = X_y_df
        self.model_type = model_type
        # self.super_parm = super_parm
        # self.set_parm = set_parm

    @property
    def template(self):
        if self.model_type == "dt_clf":
            self.set_parm = {

            }
            self.super_parm = {

            }

        elif self.model_type == "dt_reg":
            self.set_parm = {

            }
            self.super_parm = {
            
            }

        elif self.model_type == "gbdt_clf":
            self.set_parm = {
                loss="log_loss",
                learning_rate=0.1,
                n_estimators=100,
                subsample=1.0,
                criterion="friedman_mse",
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_depth=3,
                min_impurity_decrease=0.0,
                init=None,
                random_state=None,
                max_features=None,
                verbose=0,
                max_leaf_nodes=None,
                warm_start=False,
                validation_fraction=0.1,
                n_iter_no_change=None,
                tol=1e-4,
                ccp_alpha=0.0,
            }            
            self.super_parm = {
                'max_depth': sp_randint(3, 5),
                'max_leaf_nodes': sp_randint(16, 2048),
                'min_samples_leaf': [0.05, 0.02, 0.01],
                'learning_rate': sp_uniform(loc=0.05, scale=0.05),
                'n_estimators': sp_randint(100, 500),
                'n_iter_no_change': sp_randint(10, 30),

            }
        
        elif self.model_type == "gbdt_reg":
            self.set_parm = {

            }
            self.super_parm = {
            
            }

        elif self.model_type == "lgmb_clf":
            self.set_parm = {

            }
            self.super_parm = {
            
            }

        elif self.model_type == "lgmb_reg":
            self.set_parm = {

            }
            self.super_parm = {
            
            }
        return self.set_parm,self.super_parm


    def grid_search(self):
        if self.model_type == "dt_clf":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(self.set_parm)
            
        elif self.model_type == "dt_reg":
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(self.set_parm)

        elif self.model_type == "gbdt_clf":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(self.set_parm)

        elif self.model_type == "gbdt_reg":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(self.set_parm)

        elif self.model_type == "lgmb_clf":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(self.set_parm)

        elif self.model_type == "lgmb_reg":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(self.set_parm)

        gs = RandomizedSearchCV(
                estimator=model, 
                param_distributions=self.super_parm,
                scoring='roc_auc',
                cv=5
                )

        return model



