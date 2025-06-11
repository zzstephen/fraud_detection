import numpy as np
import pandas as pd
from loguru import logger
import datetime
import sys
sys.path.append('../utilities')
from basic_utilities import *
from model_utilities import *
from data_config import *
sys.path.append('../../../../infrastructure/tools')
from feature_engineering import feature_engineering
from utilities import utilities

def main():

    logger.add("feature_engineering.log", level="INFO")

    logger.info(f"Execution time stamp: {datetime.datetime.now()}")

    logger.info('creating 1d knot features, dummy features and binning features  for train/test sample')


    input_train_test_data = pd.read_csv(f'{intermediate_data_path}/train_test_sample.csv')

    yml_file = read_yaml_file(cap_n_floor)

    logger.info('capping features')

    caps = yml_file['caps']

    for cap in caps:
        temp = cap.split(',')
        if temp[0] in ['prev_address_months_count','intended_balcon_amount','bank_months_count']:
            continue
        else:
            input_train_test_data[f'{temp[0]}_capped'] = input_train_test_data[temp[0]].apply(lambda x: True if x > float(temp[1]) else False)
            input_train_test_data.loc[input_train_test_data[f'{temp[0]}_capped']==True,f'{temp[0]}'] = float(temp[1])

    for key in segments:
        created_data = create_data(input_train_test_data.loc[input_train_test_data['segment']==key], cleanup_intermediate=False,
                                                                  dummy_grouping=dummy_grouping[f'segment{key}'], knots_1d=knots_1d[f'segment{key}'], binning_features=binning_feature[f'segment{key}'])

        created_data.to_csv(f'{processed_data_path}/train_test_sample_segment{key}_logit.csv', index=False)

        logger.info(f'Segment {key} data saved to {processed_data_path}/train_test_sample_segment{key}_logit.csv')





    logger.info('creating 1d knot features, dummy features and binning features for out of time sample')

    input_out_of_time_data = pd.read_csv(f'{intermediate_data_path}/out_of_time_sample.csv')

    # yml_file = read_yaml_file(cap_n_floor)

    #rule based goes here

    logger.info('creating data cap lable for out of time sample')

    for cap in caps:
        temp = cap.split(',')
        if temp[0] in ['prev_address_months_count','intended_balcon_amount','bank_months_count']:
            continue
        else:
            input_out_of_time_data[f'{temp[0]}_capped'] = input_out_of_time_data[temp[0]].apply(lambda x: True if x > float(temp[1]) else False)
            input_out_of_time_data.loc[input_out_of_time_data[f'{temp[0]}_capped']==True,f'{temp[0]}'] = float(temp[1])


    for key in segments:

        created_data=create_data(input_out_of_time_data.loc[input_out_of_time_data['segment']==key], cleanup_intermediate=False,
                        dummy_grouping=dummy_grouping[f'segment{key}'], knots_1d=knots_1d[f'segment{key}'], binning_features=binning_feature[f'segment{key}'])


        created_data.to_csv(f'{processed_data_path}/out_of_time_sample_segment{key}_logit.csv', index=False)

        logger.info(f'saving segment {key} data to {processed_data_path}/out_of_time_sample_segment{key}_logit.csv')


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

            df1[f'{var}_{items[0]}'] = 0

            for item in items[1:]:
                if item.strip() in df1.columns:
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
        cutpoints = feature_engineering.auto_binning(df1, temp[0],'fraud_bool', int(temp[1]), mtype='classification', class_weight={0:1,1:(1/sample_down_rate)})
        varlist[temp[0]] = cutpoints

    df1, cps = utilities.binning_c(df1, varlist, labels=False)

    pd.DataFrame.from_dict(cps, orient='index').to_csv(binning_features.replace('yaml','csv'))


    if cleanup_intermediate:

        df1.drop(vars_to_drop, axis=1, inplace=True)

    return df1




if __name__ == "__main__":
    main()