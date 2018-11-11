import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
"""
We convert the  data to TF records format and then create a queue and read them in a later code!!
these files will always be read off of disk in a standardized way and never all at once. In other words, 
your dataset size is only bounded by hard drive space.
Here, train, validation, and test data are spread in 10 different files each.
"""

FLAGS = None
data_files_dict = dict()
data_files_dict['TRAIN_FILE_LIST'] = ['train_'+'%d.csv' % i for i in range(1, 11)]
data_files_dict['VALIDATION_FILE_LIST'] = ['validation_'+'%d.csv' % i for i in range(1, 11)]
data_files_dict['TEST_FILE_LIST'] = ['test_'+'%d.csv' % i for i in range(1, 11)]
N_FEATURES = 1500


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    y = data_set['Label'].values
    x = data_set.drop('Label', axis=1)
    x = x.values.astype(np.int64)
    num_examples = x.shape[0]
    filename = os.path.join(FLAGS.dest_directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        # Construct Python bytes containing the raw data bytes in the array.
        x_raw = x[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(y[index])),
            'x_raw': _bytes_feature(x_raw)
        }
        ))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    for item in data_files_dict:
        for filename in data_files_dict[item]:
            data_set = pd.read_csv(FLAGS.source_directory + filename, index_col='UserObjectId')
            name = filename.split('.')[0]
            # Convert to Examples and write the result to TFRecords.
            convert_to(data_set, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_directory',
        type=str,
        default='C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/ExcelUsageData/data_chunks/',
        help='source data directory'
    )
    parser.add_argument(
        '--dest_directory',
        type=str,
        default='C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/ExcelUsageData/tfrecords/',
        help='Directory to write the converted result'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)