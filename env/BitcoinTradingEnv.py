import gym
import pandas as pd
import numpy as np
from numpy import inf
from sklearn import preprocessing
from gym import spaces
from util.stationarization import stationarize
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
from pprint import pprint

HOURSINAYEAR = 365*24
"""
    Future Work:
        - Stop when certain profit/loss achieved or is that useless for when training ?
"""
class BitcoinTradingEnv(gym.Env):

    def __init__(self, df, reward_func='sortino', initial_balance=10000, commission=0.0025, forecast_len=10):
        """
        :param df:
        :param reward_func:   Risk-adjusted return metric to be used.   # Obligatory argument for gym.Env
        :param initial_balance: Initial resources the agent holds in USD.
        :param commission: Commission the "middleman" takes per transaction.
        :param forecast_len:
        :param confidence_interval:
        :param scaler: Function to rescale data, default=minMaxScaler to normalize
        """

        # Set constant values, and functions to use throughout
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_func = reward_func    # ['sortino', 'calmar', 'omega', 'dummy']
        self.forecast_len = forecast_len
        self.confidence_interval = 0.95
        self.scaler = preprocessing.MinMaxScaler()
        self.strategies = {
            'sortino': sortino_ratio,
            'calmar': calmar_ratio,
            'omega':  omega_ratio
        }

        # Prepare datasets
        self.df = df.fillna(method='bfill').reset_index()   # Treat NaN (fill with next !NaN val) and reset index accordingly
        self.stationaryDf = stationarize(self.df,
                               ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD'])
        self.feature_cols = self.df[self.df.columns.difference(['index', 'Date'])].columns
        self.features = self._preprocessDataset(self.stationaryDf)
        self.accountHistory = np.array([
            [self.initial_balance], # balance
            [0],    # BTC bought
            [0],    # BTC bought * price
            [0],    # BTC sold
            [0]     # BTC sold * price
        ])

        # Agent spaces definition
        self.action_space = spaces.MultiDiscrete([3, 10])  
        self.obsShape = (1, len(self.accountHistory) + len(self.feature_cols) + (self.forecast_len * 3))   # forecast_len * 3 because it's for the mean and the bound of the confidence interval
        # print(self.obsShape)

        # Observes the OHCLV values, net worth, and trade history.
        # A box in R^n with an identical bound for each dimension
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obsShape, dtype=np.float16)

        # print('> Finished init ')
    
    def _preprocessDataset(self, stationaryDf):
        # Isolate important features
        features_df = self.stationaryDf[self.feature_cols]
        features = features_df.values 
        features[abs(features) == inf] = 0
        return features
        
    def reset(self):
        """
                Reset gym environment to base values
                :return: first observation
        """
        # Initialize empty variables
        self.wallet = self.initial_balance                      # Agent's nº of USD
        self.btcWallet = 0                                   # Agent's nº of Bitcoin
        self.iterator = self.forecast_len                  # Timestamp, we cannot start our ts at t=0 because of forecast
        self.worthHistory = [self.wallet+self.btcWallet*self._getCurrentPrice()]     # History
        

        self.tradeHistory = []
        """
            Array of: {
                'step': iterator value (timestamp)
                'type': 'sell' || 'buy'
                'amount': nº of bitcoin 
                'total': nº of bitcoin * amount,
                }
        """

        self.accountHistory = np.array([
            [self.wallet],
            [0],    # BTC bought
            [0],    # BTC bought * price
            [0],    # BTC sold
            [0]     # BTC sold * price
        ])

        # print('> Finished reset ')

        return self._getNextObs()
    """
        Run a step in the environment
    """

    def step(self, action):
        """
        :param action[actionID, amount]
        :return: new observation, reward of action took, whether the agent's finished, <nothing>
        """
        balance_threshold = 0.1
        self._takeNewAction(action)
        self.iterator += 1
        obs = self._getNextObs()
        reward = self._getReward()
        finished = self.worthHistory[-1] < self.initial_balance * balance_threshold or self.iterator == len(self.df) - self.forecast_len-1
        return obs, reward, finished, {"worthHistory": self.worthHistory}

    def render(self):
        pass

    def close(self):
        pass

    def _getPrediction(self, step, prediction_horizon, confidence_interval):
        # Predict next values
        # Maybe the close prices should be taken from the normalized array?
        pastDf = self.stationaryDf['Close'][:step]
        forecast_model = SARIMAX(pastDf.values, simple_differencing=True)
        model_fit = forecast_model.fit(method='bfgs', disp=False)
        forecast = model_fit.get_forecast(
            steps=prediction_horizon, alpha=(1 - confidence_interval))
        return forecast
        

    def _getNextObs(self):
        """
            :return:    ????
        """
        # selected what we're gonna use
        t_step = self.iterator + 1
        # select
        feats_sofar = self.features[:t_step]

        # Normalize obs
        norm_sofar = self.scaler.fit_transform(feats_sofar.astype('float32'))
        # Normalize movement history
        scaled_history = self.scaler.fit_transform(
            self.accountHistory.astype('float32'))
        # to Df
        #norm_sofar = pd.DataFrame(norm_sofar, columns=self.feature_cols)
        forecast = self._getPrediction(self.iterator + 1, self.forecast_len, self.confidence_interval)

        obs = norm_sofar[-1]                                                 # len 44 
        obs = np.append(obs, forecast.predicted_mean, axis=0)             # Appends forecastlen
        obs = np.append(obs, forecast.conf_int().flatten(), axis=0)       # Appends forecastlen*2
        obs = np.append(obs, scaled_history[:, -1], axis=0)
        obs = np.reshape(obs.astype('float16'), self.obsShape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0
        return obs

    def _takeNewAction(self, action):
        """
        :param action[actionID, amount]
        """

        actionType = action[0]      # 0=Buy, 1=Sell, 2=Hold
        amount = action[1] / 10
        currentPrice = self._getCurrentPrice()
        
        position = {}
        btc_diff = 0
        usd_diff = 0
        
        if actionType == 0:
            # BUY
            price = currentPrice * (1 + self.commission)
            btc_diff = min(self.wallet * amount /
                             price, self.wallet / price)      # We can't buy the entire amount if we're out of funds

        if actionType == 1:
            # SELL
            price = currentPrice * (1 - self.commission)
            btc_diff = -1 * self.btcWallet * amount
        
        if actionType != 2:
            usd_diff = btc_diff * price
            self.btcWallet += btc_diff
            self.wallet -= usd_diff
        # Document transactions if that's the case:
        if np.abs(btc_diff) > 0:
            self.tradeHistory.append(
                {
                    'step': self.iterator,
                    'amount': btc_diff,
                    'total': usd_diff,
                    'type': 'sell' if btc_diff < 0 else 'buy'
                })

        self.worthHistory.append(self.wallet + self.btcWallet * currentPrice)
        # print(self.worthHistory)

        self.accountHistory = np.append(self.accountHistory, [
            [self.wallet],
            [btc_diff if actionType == 0 else 0],
            [usd_diff if actionType == 0 else 0],
            [btc_diff if actionType == 1 else 0],
            [usd_diff if actionType == 1 else 0]
        ], axis=1)

        # print('> Finished _takeNewAction ')

    def _getReward(self):
        """
            Calculate the reward value based on the reward strategy which we have chosen which can be one of the following:
            ['sortino', 'calmar', 'omega', 'dummy']

            :return: Reward value
        """
        # print('> In _getReward ')
        #reward = 0
        length = (self.iterator-self.forecast_len if self.iterator < self.forecast_len else self.forecast_len)

        # Get the differences in worth value (USD+BTC) from the last "length" positions
        worthEvolution = np.diff(self.worthHistory[-length:])
        # Avoid errors for when there's no variation in worth value during the last "length" timestamps
        if np.count_nonzero(worthEvolution) < 1:
            print("all zero diff")
            return 0
        # Calculate rewards according to the chosen strategy.
        if self.reward_func in ['sortino', 'calmar', 'omega']:
            reward = self.strategies[self.reward_func](worthEvolution, annualization=365*24)
            print("in fancy strats reqard: " + str(reward))
        if self.reward_func == 'profit':
            # Simple strategy where the reward is the last variation in worth.
            reward = worthEvolution[-1]
            #print("in profit worth ev: ")
            #pprint(reward)
        if self.reward_func == 'networth':
            # Simple strategy where the reward is .
            reward = self.worthHistory[-1]
            print("in networth reward")
    
        # print('> Finished _getReward ')
        # Avoid errors
        return reward if np.isfinite(reward) else 0

    def _getCurrentPrice(self):
        return self.df['Close'].values[self.iterator] + 0.0001        # +0.01 ??
