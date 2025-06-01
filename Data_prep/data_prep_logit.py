import numpy as np
import pandas as pd
from loguru import logger
import datetime
import sys
sys.path.append('../utilities')
from basic_utilities import *
from model_utilities import *
from data_config import *
sys.path.append('../../../../infrastructure/utilities')
from ml_tools import feature_engineer

def main():

    logger.add("feature_engineering.log", level="INFO")

    logger.info(f"Execution time stamp: {datetime.datetime.now()}")

    logger.info('creating 1d knot features, dummy features and binning features  for train/test sample')

    train_test_data = pd.DataFrame()

    input_train_test_data = pd.read_csv(f'{intermediate_data_path}/train_test_sample.csv')

    yml_file = read_yaml_file(cap_n_floor)


    logger.info('creating missing label and capping for train/test sample')

    floors = yml_file['floors']

    for floor in floors:
        temp = floor.split(',')
        input_train_test_data[f'{temp[0]}_missing'] = input_train_test_data[temp[0]].apply(lambda x: True if x < float(temp[1]) else False)
        input_train_test_data.loc[input_train_test_data[f'{temp[0]}_missing']==True, temp[0]] = 0

    caps = yml_file['caps']

    for cap in caps:
        temp = cap.split(',')
        input_train_test_data[f'{temp[0]}_outlier'] = input_train_test_data[temp[0]].apply(lambda x: True if x > float(temp[1]) else False)
        input_train_test_data.loc[input_train_test_data[f'{temp[0]}_outlier']==True,f'{temp[0]}'] = float(temp[1])

    for floor in floors:
        temp = floor.split(',')
        input_train_test_data[f'{temp[0]}'] = input_train_test_data.loc[
            input_train_test_data[f'{temp[0]}_missing'] == True, temp[0]] = input_train_test_data[f'{temp[0]}'].mean()


    for key in segments:

        train_test_data = pd.concat([train_test_data, create_data(input_train_test_data.loc[input_train_test_data['segment']==key], cleanup_intermediate=False,
                                                                  dummy_grouping=dummy_grouping[f'segment{key}'], knots_1d=knots_1d[f'segment{key}'], binning_features=binning_feature[f'segment{key}'])],axis=0, ignore_index=True)

    train_test_data.to_csv(f'{processed_data_path}/train_test_sample.csv', index=False)

    logger.info(f'data saved to {processed_data_path}/train_test_sample.csv')

    logger.info('creating 1d knot features, dummy features and binning features for out of time sample')

    out_of_time_data = pd.DataFrame()

    input_out_of_time_data = pd.read_csv(f'{intermediate_data_path}/out_of_time_sample.csv')

    yml_file = read_yaml_file(cap_n_floor)

    #rule based goes here

    logger.info('creating data issue lable for out of time sample')

    floors = yml_file['floors']

    for floor in floors:
        temp = floor.split(',')
        input_out_of_time_data[f'{temp[0]}_issue'] = input_out_of_time_data[temp[0]].apply(lambda x: 'missing' if x < int(temp[1]) else 'N/A')

    caps = yml_file['caps']

    for cap in caps:
        temp = cap.split(',')
        input_out_of_time_data[f'{temp[0]}_issue'] = input_out_of_time_data.apply(lambda x: x[f'{temp[0]}_issue'] + '|Outlier' if x[temp[0]]  > int(temp[1]) else x[f'{temp[0]}_issue'], axis=1)

    for key in segments:

        out_of_time_data = pd.concat([out_of_time_data, create_data(input_out_of_time_data.loc[input_out_of_time_data['segment']==key], cleanup_intermediate=False,
                                                                  dummy_grouping=dummy_grouping[f'segment{key}'], knots_1d=knots_1d[f'segment{key}'], binning_features=binning_feature[f'segment{key}'])],axis=0, ignore_index=True)

    out_of_time_data.to_csv(f'{processed_data_path}/out_of_time_data.csv', index=False)

    logger.info(f'saving data to {processed_data_path}/out_of_time_sample.csv')


def create_data(df:pd.DataFrame, cleanup_intermediate:bool=False, dummy_grouping:str='', knots_1d:str='', binning_features:str=''):

    df1 = df.copy()

    yml_file = read_yaml_file(dummy_grouping)

    df1 = f_get_dummies(df1, yml_file['get_dummies'], drop_first=False)

    if cleanup_intermediate:

        vars_to_drop = list(set(df1.columns) - set(df.columns))

    for var in yml_file['get_dummies']:

        var_grouping = yml_file[var]

        for level in var_grouping:

            items = level.split(',')

            if var ==  'device_distinct_emails_8w':
                print('stop')

            df1[f'{var}_{items[0]}'] = 0

            for item in items[1:]:
                df1[f'{var}_{items[0]}'] += df1[item.strip()]

    yml_file = read_yaml_file(knots_1d)

    varlist = {}
    for key in yml_file['get_knots']:
        temp = yml_file[key].split(',')
        varlist[key] = [float(t.replace("−", "-")) for t in temp]

    df1 = f_get_1d_knots(df1, varlist.keys(), varlist)


    yml_file = read_yaml_file(binning_features)

    features = yml_file['binning_features']

    varlist = {}
    for f in features:
        temp = f.split(',')
        cutpoints = feature_engineer.auto_binning(df1, temp[0],'fraud_bool', int(temp[1]), mtype='classification', class_weight={0:1,1:(1/sample_down_rate)})
        varlist[temp[0]] = cutpoints

    df1, cps = binning_c(df1, varlist)

    pd.DataFrame.from_dict(cps, orient='index').to_csv(binning_features.replace('yaml','csv'), index=False)


    if cleanup_intermediate:

        df1.drop(vars_to_drop, axis=1, inplace=True)

    return df1




if __name__ == "__main__":
    main()