from env.BitcoinTradingEnv import BitcoinTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import A2C, ACKTR, PPO2
from util.imports import getDatasets, getConfiguration, selectFunctionAccordingToParams
from util.plots import plotEveryReward
from os.path import join, exists
from os import makedirs
import pickle as pkl



# ====== IMPORT CONFIGURATION PARAMETERS ======
mainparams = getConfiguration()
testDirs = [join(mainparams.get('testMainDir'), test) for test in mainparams.get('testSubDirs')]

# ====== DATA IMPORT AND SPLIT ======
train_df, test_df = getDatasets(mainparams.get('input_data_file'), percentageToUse=mainparams.get('dataset_percentage'))


for td in testDirs:

    params = getConfiguration(join(td, 'config.yaml'))

    # ====== IMPORT MODEL ======
    modelToUse = selectFunctionAccordingToParams('model', params.get('model'))
    polictyToUse = selectFunctionAccordingToParams('policy', params.get('policy'))

    # ====== ENVIRONMENT SETUP =======
    trainEnv = DummyVecEnv([lambda: BitcoinTradingEnv(
        train_df, reward_func=params.get('reward_strategy'), forecast_len=params.get('forecast_len'),
        confidence_interval=params.get('confidence_interval'))])

    testEnv = DummyVecEnv([lambda: BitcoinTradingEnv(
        test_df, reward_func=params.get('reward_strategy'), forecast_len=params.get('forecast_len'),
        confidence_interval=params.get('confidence_interval'))])


    boardDir = join(td, 'tensorboard')
    if not exists(boardDir):
        makedirs(boardDir)

    # tensorboard_log=boardDir,
    print(params.get('model'))
    if params.get('model') == 'a2c':
        print('> A2C')
        model = modelToUse(polictyToUse, trainEnv, verbose=0, **params.get('model_params'))
    else:
        print('> PPO')
        model = modelToUse(polictyToUse, trainEnv, verbose=0, nminibatches=1, **params.get('model_params'))

    # ====== TRAIN THE MODEL ======
    agentsDir = join(td, 'agents')
    if not exists(agentsDir):
        makedirs(agentsDir)

    savedAgent = 0
    rewards = [[] for i in range(params.get('numTrainingIterations'))]

    for it in range(params.get('numTrainingIterations')):
        # Training
        print('[', it, '] Training for: ', len(train_df), ' time steps')
        model.learn(total_timesteps=len(train_df))

        obs = testEnv.reset()
        done, reward_sum = False, 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = testEnv.step(action)
            reward_sum += reward
            rewards[it].append(reward_sum[0])

        print('[', it, '] Total reward: ', reward_sum, ' (' + params.get('reward_strategy') + ')')

        if it % mainparams.get('savingModelInterval'):
            model.save(join(agentsDir, 'agent{}.pkl').format(savedAgent))
            savedAgent += 1

    with open(join(td, 'rewards.pkl'), 'wb') as f:
        pkl.dump(rewards, f)
    model.save(join(agentsDir, 'agentFinal.pkl'))
    plotEveryReward(rewards, join(td, 'reward-plot-train'), label='Cumulative Rewards')

    print('> Finished ' + td)
