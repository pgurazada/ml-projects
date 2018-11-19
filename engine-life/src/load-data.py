import pandas as pd
import os
import logging


def read_csv_file(data_dir, file_name, ops=3, sensors=26):

    operational_settings = ['operational_setting_{}'.format(i+1)
                            for i in range(ops)]

    sensor_columns = ['sensor_measurement_{}'.format(i+1) for i in range(sensors)]

    cols = ['engine_no', 'time_in_cycles'] + operational_settings + \
            sensor_columns

    data_df = pd.read_csv(os.path.join(data_dir, file_name),
                          sep=' ',
                          header=-1,
                          names=cols)

    logging.info('data from {} imported with shape: {}'.format(file_name,
                                                               data_df.shape))

    return data_df


if __name__ == '__main__':
    logging.basicConfig(filename='io.log', level=logging.INFO)
    
    data_dfs = []

    data_dir_str = 'input'
    processed_data_dir = 'processed-input'
    processed_filename = 'engine-data.feather'

    for file_name_str in ['train_FD001.txt',
                          'train_FD002.txt',
                          'train_FD003.txt',
                          'train_FD004.txt']:
        
        data = read_csv_file(data_dir=data_dir_str, file_name=file_name_str)

        data_dfs.append(data)

    final_data_df = pd.concat(data_dfs)
    final_data_df.reset_index(inplace=True)

    logging.info('Writing the aggregated data to {}'.format(os.path.join(processed_data_dir,
                                                                         processed_filename)))

    final_data_df.to_feather(os.path.join(processed_data_dir,
                                          processed_filename))
