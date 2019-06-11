import yaml
import pandas as pd
from util.indicators import add_indicators


def getConfiguration(configFile="config.yaml"):
    with open(configFile, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print("> Using the following parameters:")
    for attr, value in sorted(params.items()):
        print("\t{}={}".format(attr.upper(), value))

    return params


def getDatasets(inputfile, testPercentage=15, valPercentage=15, percentageToUse=100):
    df = pd.read_csv(inputfile)
    df = df.drop(['Symbol'], axis=1)
    df = df.sort_values(['Date'])
    df = add_indicators(df.reset_index())

    # Just to speed things up
    print('> Using {}% of the dataset'.format(percentageToUse))
    df = df[:int(len(df) * percentageToUse/100)]

    test_len = int(len(df) * testPercentage/100)
    val_len = int(len(df) * valPercentage/100)
    train_len = int(len(df)) - test_len - val_len

    train_df = df[:train_len]
    val_df = df[train_len:train_len+val_len]
    test_df = df[train_len+val_len:]

    return train_df, val_df, test_df
