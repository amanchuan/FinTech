构造automl工作流。

--data
    --input
    --output
--eda
    --describe: parse,desc
--preprocess
    --data_preprocess: encoding(label-encoding and OneHot-coding,对于Y标签，分类任务需要对训练数据和测试数据同时进行encoder，回归任务则不需要对标签进行encoder),missing value,normalization
    --feature_preprocess: feature selection,feature space transformation(i.e. PCA)
    --visualization
--model
    --model select: train_test_split,
    --param search
    --metrics
    --ensemble model
        --ensamble_size,ensamble_nbest,max_models_on_disc
--pipeline
    --get result of each step by callable attributes. add turn-off on steps. 

#### auto-sklearn

1. 并行计算时，不论单机多核并行还是多机并行，都需要用Dask来管理worker集合。单机并行：在AutoSklearnClassifier方法中通过n_job调用单机上的多核进行计算。
2. 