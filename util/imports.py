import yaml
import pandas as pd
from util.indicators import add_indicators
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common.policies import MlpLnLstmPolicy, MlpPolicy


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


def getDatasets(inputfile, testPercentage=20, percentageToUse=100, withBug=True):
    df = pd.read_csv(inputfile)
    df = df.drop(['Symbol'], axis=1)
    if not withBug:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %I-%p')    # parse date from string to the correct format
    df = df.sort_values(['Date'])
    df = add_indicators(df.reset_index())

    # Just to speed things up
    print('> Using {}% of the dataset'.format(percentageToUse))
    df = df[:int(len(df) * percentageToUse/100)]

    test_len = int(len(df) * testPercentage/100)
    train_len = int(len(df)) - test_len

    train_df = df[:train_len]
    test_df = df[train_len:]

    return train_df, test_df

def selectFunctionAccordingToParams(type, param):
    if type == 'model':
        if param == 'ppo2':
            """
            PPO2:
                The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers) and TRPO 
                (it uses a trust region to improve the actor).
           """
            return PPO2
        elif param == 'a2c':
            return A2C
        else:
            print('> Error, unsupported params')
            return None

    elif type == 'policy':
        if param == 'mlp':
            return MlpPolicy

        if param == 'lstm':
            """ MlpLnLstmPolicy: 
                Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction.
            """
            return MlpLnLstmPolicy
    else:
        print('> Error, unsupported  type')


