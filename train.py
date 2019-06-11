from env.traderEnv import BitcoinTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import A2C, ACKTR, PPO2
from util.imports import getDatasets, getConfiguration

# ====== IMPORT CONFIGURATION PARAMETERS ======
params = getConfiguration()

# ====== DATA IMPORT AND SPLIT ======
train_df, val_df, _ = getDatasets(params.get('input_data_file'), percentageToUse=1)

# ====== ENVIRONMENT SETUP =======
# Dummy environment for development purposes.
trainEnv = DummyVecEnv([lambda: BitcoinTradingEnv(train_df)])
valEnv = DummyVecEnv([lambda: BitcoinTradingEnv(val_df)])

# ====== IMPORT MODEL ======
"""
PPO2:
    The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers) and TRPO 
    (it uses a trust region to improve the actor).
MlpLnLstmPolicy: 
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction.
"""
# fixme - should be able to import a previous model.
# fixme - model and policy should be abstracted
model = PPO2(MlpLnLstmPolicy, trainEnv, verbose=0, nminibatches=params.get('numMiniBatchesPerUpdate'),
             tensorboard_log=params.get('tensorLog'), **params.get('model_params'))

# ====== TRAIN THE MODEL ======
for it in range(params.get('numTrainingIterations')):
    # Training
    print('[', it, '] Training for: ', len(train_df), ' time steps')
    model.learn(total_timesteps=len(train_df))

    # Validation
    # fixme - maybe this should only be done after a few training iterations and not in all
    print('[', it, '] Validating for: ', len(val_df), ' time steps')
    obs = valEnv.reset()
    done, reward_sum = False, 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = valEnv.step(action)
        print('Current reward {}'.format(reward))
        reward_sum += reward

    print('[', it, '] Total reward: ', reward_sum, ' (' + params.get('reward_strategy') + ')')

model.save('./agents/ppo2_{}.pkl'.format(params.get('reward_strategy')))



