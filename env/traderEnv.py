import gym
import pandas as pd
import numpy as np
from numpy import inf
from sklearn import preprocessing
from gym import spaces
from util.stationarization import log_and_difference
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio


"""
    Future Work:
        - Stop when certain profit/loss achieved or is that useless for when training?
"""
class BitcoinTradingEnv(gym.Env):

    def __init__(self, df, initialFunds=10000, commission=0.0025, rewardType='sortino', forecastLength=10,
                 confidenceInterval=0.95, scaler=preprocessing.MinMaxScaler()):
        """
        :param df:
        :param initialwallet: Initial resources the agent holds in USD.
        :param commission: Commission the "middleman" takes per transaction.
        :param rewardType:   Risk-adjusted return metric to be used.
        :param forecastLength:
        :param confidenceInterval:
        :param scaler: Function to rescale data, default=minMaxScaler to normalize
        """

        # Set constant values, and functions to use throughout
        self.initialFunds = initialFunds
        self.commission = commission
        self.rewardType = rewardType    # ['sortino', 'calmar', 'omega', 'dummy']
        self.forecastLength = forecastLength
        self.confidenceInterval = confidenceInterval
        self.scaler = scaler
        self.strategies = {
            'sortino': sortino_ratio,
            'calmar': calmar_ratio,
            'omega':  omega_ratio
        }

        # Prepare datasets
        self.df = df.fillna(method='bfill').reset_index()   # Treat NaN (fill with next !NaN val) and reset index accordingly
        self.stacionaryDf = log_and_difference(self.df,
                               ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD'])


        # Agent spaces definition
        self.action_space = spaces.MultiDiscrete([3, 10])    # spaces.Discrete(12)         # fixme - not being done by the current model on github
        self.obsShape = (1, 5 + len(self.df.columns) - 2 + (self.forecastLength * 3))     # fixme - why the 5 -2 ...
        # print(self.obsShape)

        # Observes the OHCLV values, net worth, and trade history.
        # A box in R^n with an identical bound for each dimension
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obsShape, dtype=np.float16)

        print('> Finished init ')

    def reset(self):
        """
                    Reset gym environment to base values
                :return: first observation
        """

        print('> In reset')

        # Initialize empty variables
        self.wallet = self.initialFunds                      # Agent's nº of USD
        self.btcWallet = 0                                   # Agent's nº of Bitcoin
        self.worthHistory = [self.wallet+self.btcWallet]     # History
        
        self.iterator = self.forecastLength        # Timestamp, we cannot start our ts at t=0 because of forecast
        
        self.tradeHistory = []
        """
            Array of: {
                'step': iterator value (timestamp)
                'amount': nº of bitcoin 
                'total': nº of bitcoin * amount,
                'type': 'sell' || 'buy'
                }
        """

        self.accountHistory = np.array([
            [self.wallet],
            [0],    # BTC bought
            [0],    # BTC bought * price
            [0],    # BTC sold
            [0]     # BTC sold * price
        ])
        print('> Finished reset ')

        return self._getNextObs()
    """
        Run a step in the environment
    """

    def step(self, action):
        """
        :param action[actionID, amount]
        :return: new observation, reward of action took, whether the agent's finished, <nothing>
        """
        print('> In step')
        self._takeNewAction(action)
        self.iterator += 1

        obs = self._getNextObs()
        reward = self._getReward()

        finished = self.worthHistory[-1] < self.initialFunds / 10 or self.iterator == len(self.df) - self.forecastLength-1

        print('Finished step')

        return obs, reward, finished, {}    # fixme - {} ??

    def render(self):
        pass

    def close(self):
        pass

    def _getNextObs(self):
        """
            :return:    ????
        """

        print('> In _getNextObs ')

        # Isolate important features
        features = self.stacionaryDf[self.stacionaryDf.columns.difference(['index', 'Date'])]

        # selected what we're gonna use
        scaled = features[:self.iterator + 1].values
        # remove infinites
        scaled[abs(scaled) == inf] = 0
        # Normalize
        scaled = self.scaler.fit_transform(scaled.astype('float32'))
        # to Df
        scaled = pd.DataFrame(scaled, columns=features.columns)

        # Predict next values
        pastDf = self.stacionaryDf['Close'][:self.iterator + 1]
        forecast_model = SARIMAX(pastDf.values, enforce_stationarity=False, simple_differencing=True)
        model_fit = forecast_model.fit(method='bfgs', disp=False)
        forecast = model_fit.get_forecast(
            steps=self.forecastLength, alpha=(1 - self.confidenceInterval))

        # ??? ??? ???
        obs = scaled.values[-1]                                                     # len 44
        obs = np.insert(obs, len(obs), forecast.predicted_mean, axis=0)             # Appends 10
        obs = np.insert(obs, len(obs), forecast.conf_int().flatten(), axis=0)       # Appends 20

        scaled_history = self.scaler.fit_transform(
            self.accountHistory.astype('float32'))

        obs = np.insert(obs, len(obs), scaled_history[:, -1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obsShape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        print('> Finished getNextObs ')

        return obs

    def _takeNewAction(self, action):
        """
        :param action[actionID, amount]
        """

        print('> In _takeNewAction ')

        actionType = action[0]      # 0=Buy, 1=Sell, 2=Hold
        amount = action[1] / 10
        currentPrice = self._getCurrentPrice()
        
        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0
        
        if actionType == 0:
            # BUY
            price = currentPrice * (1 + self.commission)
            btc_bought = min(self.wallet * amount /
                             price, self.wallet / price)      # We can't buy the entire amount if we're out of funds
            cost = btc_bought * price

            self.btcWallet += btc_bought
            self.wallet -= cost
            
        if actionType == 0:
            # SELL
            price = currentPrice * (1 - self.commission)
            btc_sold = self.btcWallet * amount
            sales = btc_sold * price

            self.btcWallet -= btc_sold
            self.wallet += sales

        # Document transactions if that's the case:
        if btc_sold > 0 or btc_bought > 0:
            self.tradeHistory.append(
                {
                    'step': self.iterator,
                    'amount': btc_sold if btc_sold > 0 else btc_bought,
                    'total': sales if btc_sold > 0 else cost,
                    'type': 'sell' if btc_sold > 0 else 'buy'
                })

        self.worthHistory.append(self.wallet + self.btcWallet * currentPrice)

        self.accountHistory = np.append(self.accountHistory, [
            [self.wallet],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

        print('> Finished _takeNewAction ')

    def _getReward(self):
        """
            Calculate the reward value based on the reward strategy which we have chosen which can be one of the following:
            ['sortino', 'calmar', 'omega', 'dummy']

            :return: Reward value
        """

        print('> In _getReward ')

        hoursInAYear = 365*24
        length = (self.iterator-self.forecastLength if self.iterator < self.forecastLength else self.forecastLength)

        # Get the differences in worth value (USD+BTC) from the last "length" positions
        worthEvolution = np.diff(self.worthHistory[-length:])

        # Avoid errors for when there's no variation in worth value during the last "length" timestamps
        if np.count_nonzero(worthEvolution) < 1:
            return 0

        # Calculate rewards according to the chosen strategy.
        if self.rewardType in ['sortino', 'calmar', 'omega']:
            reward = self.strategies[self.rewardType](worthEvolution, annualization=hoursInAYear)
        elif self.rewardType is 'dummy':
            # Simple strategy where the reward is the last variation in worth.
            reward = worthEvolution[-1]

        print('> Finished _getReward ')

        # Avoid errors
        return reward if np.isfinite(reward) else 0

    def _getCurrentPrice(self):
        return self.df['Close'].values[self.iterator] + 0.01        # +0.01 ??
