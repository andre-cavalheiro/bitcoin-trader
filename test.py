from env.BitcoinTradingEnv import BitcoinTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2
from util.imports import getDatasets, getConfiguration, selectFunctionAccordingToParams
from os.path import join, exists
from os import makedirs
from util.plots import makePlots

# ====== IMPORT CONFIGURATION PARAMETERS ======
mainparams = getConfiguration()
testDirs = [join(mainparams.get('testMainDir'), test) for test in mainparams.get('testSubDirs')]

# ====== DATA IMPORT AND SPLIT ======
_, test_df = getDatasets(mainparams.get('input_data_file'), percentageToUse=mainparams.get('dataset_percentage'))

worthHistory = []
tradesHistoryBuy = []
tradesHistorySell = []
print('Testing for {} instances'.format(len(test_df)))

for td in testDirs:

    params = getConfiguration(join(td, 'config.yaml'))

    # ====== ENVIRONMENT SETUP =======
    # https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#dummyvecenv
    testEnv = DummyVecEnv([lambda: BitcoinTradingEnv(
        test_df, reward_func=params.get('reward_strategy'), forecast_len=params.get('forecast_len'),
        confidence_interval=0.81)])

    # ====== IMPORT MODEL ======
    # fixme - should be able to import a previous model.
    modelToUse = selectFunctionAccordingToParams('model', params.get('model'))
    polictyToUse = selectFunctionAccordingToParams('policy', params.get('policy'))
    agentsDir = join(td, 'agents')

    model = modelToUse.load(join(agentsDir, 'agentFinal.pkl'), env=testEnv)

    # ===== TEST MODEL ======
    obs, done = testEnv.reset(), False
    rewards = []
    while not done:
        action, _states = model.predict(obs)

        worthHistory = testEnv.get_attr('net_worths')
        tradeHistory = testEnv.get_attr('trades')

        obs, reward, done, _ = testEnv.step(action)
        # testEnv.render(mode="human")
        rewards.append(reward)

    print(' Total reward: {}'.format(sum(rewards)/len(rewards)))

    # ===== PLOTS =====
    worthHistory = worthHistory[0]  # We only use one environment -> understand better why we have vectorized
    tradeHistory = tradeHistory[0]
    print('Size of worth history: ' + str(len(worthHistory)))
    print('Size of trade history: ' + str(len(tradeHistory)))
    bitcoinPrice = test_df['Close'].values[params.get('forecast_len'):]
    """
    print('Bitcoin price ' + str(len(bitcoinPrice)))
    print(bitcoinPrice.shape)
    """
    makePlots(worthHistory, tradeHistory, bitcoinPrice, rewards, td)
