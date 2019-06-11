from env.traderEnv import BitcoinTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2
from util.imports import getDatasets, getConfiguration, selectFunctionAccordingToParams
from os.path import join, exists
from os import makedirs
from util.plots import plotWorthGraph

# ====== IMPORT CONFIGURATION PARAMETERS ======
mainparams = getConfiguration()
testDirs = [join(mainparams.get('testMainDir'), test) for test in mainparams.get('testSubDirs')]

# ====== DATA IMPORT AND SPLIT ======
_, _, test_df = getDatasets(mainparams.get('input_data_file'), percentageToUse=1)

# ====== ENVIRONMENT SETUP =======
# Dummy environment for development purposes.
testEnv = DummyVecEnv([lambda: BitcoinTradingEnv(test_df)])
print('Testing for {} instances'.format(len(test_df)))

for td in testDirs:

    params = getConfiguration(join(td, 'config.yaml'))

    # ====== IMPORT MODEL ======
    # fixme - should be able to import a previous model.
    modelToUse = selectFunctionAccordingToParams('model', params.get('model'))
    polictyToUse = selectFunctionAccordingToParams('policy', params.get('policy'))
    agentsDir = join(td, 'agents')

    model = modelToUse.load(join(agentsDir, 'agent.pkl'), env=testEnv)

    # ===== TEST MODEL ======
    obs, done = testEnv.reset(), False
    reward_sum = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, worthHistory = testEnv.step(action)
        worthHistory = worthHistory[0]
        reward_sum += reward

    print(' Total reward: {}'.format(reward_sum))
    plotWorthGraph(worthHistory, join(td, 'net-worth-history-test'))
