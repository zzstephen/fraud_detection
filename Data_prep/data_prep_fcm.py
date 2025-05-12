import numpy as np
import pandas as pd
from loguru import logger
import datetime

intermediate_data_path = '../../../data/intermediate'

processed_data_path = '../../../data/processed_data'

dummy_grouping = '../EDA/summary/dummy_feature_importance.xlsx'

def main():

    logger.add("feature_engineering.log", level="INFO")

    logger.info(f"Execution time stamp: {datetime.datetime.now()}")

    logger.info('creating features from intermediate datasets')

    train_test_data = create_data(pd.read_csv(f'{intermediate_data_path}/train_test_sample.csv'))

    logger.info(f'saving data to {processed_data_path}/train_test_sample.csv')

    logger.info('creating features for out of time sample')

    out_of_time_data = create_data(pd.read_csv(f'{intermediate_data_path}/out_of_time_sample.csv'))

    logger.info(f'saving data to {processed_data_path}/train_test_sample.csv')




def create_data(df):



    




    return df




if __name__ == "__main__":
    main()