from env.traderEnv import BitcoinTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2
from util.imports import getDatasets, getConfiguration

# ====== IMPORT CONFIGURATION PARAMETERS ======
params = getConfiguration()

# ====== DATA IMPORT AND SPLIT ======
_, _, test_df = getDatasets(params.get('input_data_file'), percentageToUse=1)

# ====== ENVIRONMENT SETUP =======
# Dummy environment for development purposes.
testEnv = DummyVecEnv([lambda: BitcoinTradingEnv(test_df)])

# ====== IMPORT MODEL ======
model = PPO2.load('./agents/ppo2_{}.pkl'.format(params.get('reward_strategy')), env=testEnv)

# ===== TEST MODEL ======
obs, done = testEnv.reset(), False
reward_sum = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = testEnv.step(action)
    reward_sum += reward

print(' Total reward: {}'.format(reward_sum))
