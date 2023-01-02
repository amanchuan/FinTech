import pandas as pd
import numpy as np


def generate_single_rules(bin_iv_woe_df, lift=3):
    """
    寻找lift高, bad_rate及lift排序性好的变量, 根据分箱边界生成单维度策略阈值。
    """
    bin_iv_woe_df.reset_index(inplace=True)
    bin_iv_woe_df['rank'] = bin_iv_woe_df.groupby('feature',as_index=True)['index'].rank(ascending=False)-1
    features_size =  bin_iv_woe_df.groupby('feature',as_index=True)['index'].size().to_frame(name='size')
    bin_iv_woe_df = bin_iv_woe_df.merge(features_size, how='left', on='feature')
    bin_iv_woe_df['rank'] = bin_iv_woe_df['size'] - np.abs(bin_iv_woe_df['rank']-bin_iv_woe_df['index'])
    # 头尾两箱存在lift>=3的变量，纳入单维度策略。
    # ringle_rule = bin_iv_woe_df[(bin_iv_woe_df['lift']>=3) & (bin_iv_woe_df['rank']<5)]['feature']
    ringle_rule_feat = bin_iv_woe_df[(bin_iv_woe_df['lift']>=lift) & (bin_iv_woe_df['rank']<5)]['feature'].unique() # 有效变量

    bin_iv_woe_df.drop(['rank','size'], axis=1, inplace=True)
    return bin_iv_woe_df[bin_iv_woe_df['feature'].isin(ringle_rule_feat)]



def generate_multi_rules(tree, feature_names, class_names, threshold=0.32):
    """
    抽取gini值0.32以下(对应bad_rate大于80%)、样本数量较多的叶子节点，输出组合规则。
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
    
    print(paths)
    # 从 tree_model.tree_.n_leaves条规则中，抽取m条gini指数小于0.32（对应80%的浓度）的规则。
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