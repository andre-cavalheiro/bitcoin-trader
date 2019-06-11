from env.traderEnv import BitcoinTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import A2C, ACKTR, PPO2
from util.imports import getDatasets, getConfiguration, selectFunctionAccordingToParams
from util.plots import plotWorthGraph
from os.path import join, exists
from os import makedirs

# ====== IMPORT CONFIGURATION PARAMETERS ======
mainparams = getConfiguration()
testDirs = [join(mainparams.get('testMainDir'),test) for test in mainparams.get('testSubDirs')]

# ====== DATA IMPORT AND SPLIT ======
train_df, val_df, _ = getDatasets(mainparams.get('input_data_file'), percentageToUse=1)

# ====== ENVIRONMENT SETUP =======
# Dummy environment for development purposes.
trainEnv = DummyVecEnv([lambda: BitcoinTradingEnv(train_df)])
valEnv = DummyVecEnv([lambda: BitcoinTradingEnv(val_df)])

for td in testDirs:

    params = getConfiguration(join(td, 'config.yaml'))

    # ====== IMPORT MODEL ======
    # fixme - should be able to import a previous model.
    modelToUse = selectFunctionAccordingToParams('model', params.get('model'))
    polictyToUse = selectFunctionAccordingToParams('policy', params.get('policy'))

    boardDir = join(td, 'tensorboard')
    if not exists(boardDir):
        makedirs(boardDir)

    model = modelToUse(polictyToUse, trainEnv, verbose=0, nminibatches=params.get('numMiniBatchesPerUpdate'),
                 tensorboard_log=boardDir, **params.get('model_params'))

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
            obs, reward, done, worthHistory = valEnv.step(action)
            print('Current reward {}'.format(reward))
            reward_sum += reward

        print('[', it, '] Total reward: ', reward_sum, ' (' + params.get('reward_strategy') + ')')

    agentsDir = join(td, 'agents')
    if not exists(agentsDir):
        makedirs(agentsDir)
    model.save(join(agentsDir, 'agent.pkl'))

    plotWorthGraph(worthHistory,  join(td, 'net-worth-history-train'))
