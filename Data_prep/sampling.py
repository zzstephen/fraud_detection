import numpy as np
import pandas as pd
from loguru import logger
import datetime
import pickle

sample_down_rate = 0.1

testing = 0.3

raw_data_path = '../../../data'

intermediate_data_path = '../../../data/intermediate'

segment_model_path = '../../../model_objects'


def main():

    logger.add("raw_data_sampling.log", level="INFO")

    logger.info(f"Execution time stamp: {datetime.datetime.now()}")

    logger.info(f'sample_down_rate: {sample_down_rate}')

    logger.info(f'testing rate: {testing}')

    logger.info(f'raw_data_path, {raw_data_path}')

    logger.info('Reading in raw data...')

    raw_data = pd.read_csv(f'{raw_data_path}/Base.csv')

    with open(f'{segment_model_path}/segment_model.pkl', 'rb') as file:

        seg_model = pickle.load(file)

    raw_data['segment'] = seg_model.predict(raw_data[['credit_risk_score','name_email_similarity']])

    logger.info(f'segmentation created...')

    logger.info(f'Raw data size: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns')

    logger.info(f'Creating out of time sample')

    out_of_time_sample = raw_data.loc[(raw_data['month']==6)|(raw_data['month']==7)].copy()

    out_of_time_sample['sample_weight'] = 1

    out_of_time_sample.to_csv(f'{intermediate_data_path}/out_of_time_sample.csv', index=False)

    logger.info(f'Out of time dataset saved as {intermediate_data_path}/out_of_time_sample.csv')

    logger.info(f'Done:Creating out of time sample')

    logger.info(f'Creating training/testing sample')

    train_test_sample = raw_data.loc[raw_data['month']<6].copy()

    train_test_sample_fraud_1 = train_test_sample.loc[train_test_sample['fraud_bool'] == 1].copy()
    train_test_sample_fraud_1['sample_weight'] = 1

    train_test_sample_fraud_0 = train_test_sample.loc[train_test_sample['fraud_bool'] == 0].copy()
    train_test_sample_fraud_0 = train_test_sample_fraud_0.sample(frac = sample_down_rate, random_state= 123)
    train_test_sample_fraud_0['sample_weight'] = 1/sample_down_rate

    train_test_data = pd.concat([train_test_sample_fraud_1, train_test_sample_fraud_0], axis=0, ignore_index=True)

    train_test_data.to_csv(f'{intermediate_data_path}/train_test_sample.csv', index=False)

    logger.info(f'Training/testing dataset saved as {intermediate_data_path}/train_test_sample.csv')

    logger.info(f'Done:Creating training/testing sample')



if __name__ == "__main__":
    main()