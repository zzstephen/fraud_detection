
import numpy as np
import pandas as pd
from loguru import logger
import datetime
from data_config import *
from sklearn.model_selection import train_test_split

def main():

    logger.add("train_test_split.log", level="INFO")

    logger.info('loading train_test_sample')

    input_train_test_data = pd.read_csv(f'{intermediate_data_path}/train_test_sample.csv')

    logger.info(f'Creating train/test split. test_size = {testing_rate}')

    X_train, X_test, y_train, y_test = train_test_split(input_train_test_data.loc[:,input_train_test_data.columns != 'fraud_bool'],
                                                        input_train_test_data['fraud_bool'], test_size=testing_rate, random_state=42)

    X_train['fraud_bool'] = y_train.copy()

    X_test['fraud_bool'] = y_test.copy()

    X_train.to_csv(f'{intermediate_data_path}/training.csv', index=False)

    X_test.to_csv(f'{intermediate_data_path}/testing.csv', index=False)

    logger.info(f'training sample saved as {intermediate_data_path}/training.csv')

    logger.info(f'testing sample saved as {intermediate_data_path}/testing.csv')

if __name__ == "__main__":
    main()