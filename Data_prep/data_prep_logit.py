import numpy as np
import pandas as pd
from loguru import logger
import datetime
import sys
sys.path.append('../utilities')
from basic_utilities import *
from model_utilities import *
from data_config import *


def main():

    logger.add("feature_engineering.log", level="INFO")

    logger.info(f"Execution time stamp: {datetime.datetime.now()}")

    logger.info('creating features from intermediate datasets')

    train_test_data = pd.DataFrame()

    input_train_test_data = pd.read_csv(f'{intermediate_data_path}/train_test_sample.csv')


    for key in segments:

        train_test_data = pd.concat([train_test_data, create_data(input_train_test_data.loc[input_train_test_data['segment']==key], cleanup_intermediate=False,
                                                                  dummy_grouping=f'dummy_grouping_segment{key}', knots_1d=f'1d_knots_segment{key}')],axis=0, ignore_index=True)

    train_test_data.to_csv(f'{processed_data_path}/train_test_sample.csv', index=False)

    logger.info(f'data saved to {processed_data_path}/train_test_sample.csv')

    logger.info('creating features for out of time sample')

    out_of_time_data = pd.DataFrame()

    input_out_of_time_data = create_data(pd.read_csv(f'{intermediate_data_path}/out_of_time_sample.csv'))

    for key in knots_1d.keys():

        out_of_time_data = pd.concat([out_of_time_data, create_data(input_out_of_time_data.loc[input_out_of_time_data['segment']==key], cleanup_intermediate=False,
                                                                  dummy_grouping=f'dummy_grouping_segment{key}', knots_1d=f'1d_knots_segment{key}')],axis=0, ignore_index=True)

    out_of_time_data.to_csv(f'{processed_data_path}/out_of_time_data.csv', index=False)

    logger.info(f'saving data to {processed_data_path}/out_of_time_sample.csv')


def create_data(df:pd.DataFrame, cleanup_intermediate:bool=False, dummy_grouping:str='', knots_1d:str=''):

    df1 = df.copy()

    yml_file = read_yaml_file(dummy_grouping)

    df1 = f_get_dummies(df1, yml_file['get_dummies'], drop_first=False)

    if cleanup_intermediate:

        vars_to_drop = list(set(df1.columns) - set(df.columns))

    for var in yml_file['get_dummies']:

        var_grouping = yml_file[var]

        for item in var_grouping:

            for key in item.keys():

                df1[f'{var}_{key}'] = 0

                for level in item[key].split(','):

                    df1[f'{var}_{key}'] += df1[level.strip()]

    yml_file = read_yaml_file(knots_1d)

    varlist = {}
    for key in yml_file['get_knots']:
        temp = yml_file[key].split(',')
        varlist[key] = [float(t) for t in temp]

    df1 = f_get_1d_knots(df1, varlist.keys(), varlist)

    if cleanup_intermediate:

        df1.drop(vars_to_drop, axis=1, inplace=True)

    return df1




if __name__ == "__main__":
    main()