import os
import pandas as pd
import numpy as np
from pandas.io.pytables import dropna_doc
import vaex as vx
from impala.dbapi import connect
from impala.util import as_pandas
# from ..utils.func import timer
import gc


def get_by_sql(sql_str):
	"""根据sql语句查询数据，返回dataframe"""
	conn = connect(host='10.1.15.13',
					port=10000,
					auth_mechanism='PLAIN',
					user='name',
					password='fjadskfjag')
	cur = conn.cursor()
	cur.execute(sql_str)
	data = as_pandas(cur)
	cur.close()
	conn.close()
	return data

def load_data(prefix,data_file):
	"""
	param:
		prefix: 文件路径名(不包含文件名)
		data_file: 包含后缀的文件名
	return:
		加载后的dataframe.
	"""
	file_dir = prefix+data_file
	suffix = file_dir.strip().split('.')
	if not os.path.exists(file_dir):
		print('not exist {0} file in dir.'.format(data_file))
		return
	elif suffix[0]=='' and len(suffix)==2:
		# with timer('load {0} data '.format(data_file)):
		with open(file_dir, 'r') as file:
			data = file.readlines()
	elif suffix[-1]=='csv':
		# with timer('load {0} data '.format(data_file)):
		data = pd.read_csv(file_dir)
	elif suffix[-1] in ('xlsx','xls'):
		# with timer('load {0} data '.format(data_file)):
		data = pd.read_excel(file_dir)
	elif suffix[-1]=='hdf5':
		# with timer('load {0} data '.format(data_file)):
		data = vx.open(file_dir)
	else: 
		print('please config optimal load way for {0} file'.format(data_file))
		return
	return data

def load_data_generate(data_file):
    with open(data_file, 'r') as file:
        # data = file.readlines()
        while True:
            one_line = file.readline().strip()
            if not one_line:
                return
            yield one_line

def prase_single_file(file):
    file_lines = load_data_generate(file)
    # data_list = [line.split(',') for line in file_lines]
    # return pd.DataFrame(data_list)
    data_list = [line.split('\x01') for line in file_lines]
    key_id_df = pd.DataFrame([v[:2] for v in data_list], columns=['id','time'])
    all_feat_series = [pd.Series(eval(v[2])) for v in data_list]
    filter_feat_df = pd.concat(all_feat_series, join='outer', axis=1).T
    del data_list,file_lines,all_feat_series
    gc.collect()
    return pd.concat([key_id_df,filter_feat_df],axis=1)

def reduce_mem_usage(df):
	start_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

	for col in df.columns:
		col_type = df[col].dtype
		if col_type != object:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
		else:
			df[col] = df[col].astype('category')
	
	end_mem = df.memory_usage().sum() / 1024**2
	print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
	print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
	return df