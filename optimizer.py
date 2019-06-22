import optuna
from env.BitcoinTradingEnv import BitcoinTradingEnv
from stable_baselines.common.policies import MlpLnLstmPolicy
from util.imports import getDatasets, getConfiguration, selectFunctionAccordingToParams
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import json
import numpy as np

def objective(trial):
    # Define what to optimize in environment
    envParams = {
        'reward_func': reward_strategy,
        'forecast_len': int(trial.suggest_loguniform('forecast_len', 1, 200)),
        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
    }
    train_df, test_df = getDatasets(params.get('input_data_file'), percentageToUse=100)
    trainEnv = DummyVecEnv([lambda: BitcoinTradingEnv(train_df, **envParams)])
    testEnv = DummyVecEnv([lambda: BitcoinTradingEnv(test_df, **envParams)])

    # Define what to optimize in agent
    agentParams = {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }

    model = PPO2(MlpLnLstmPolicy, trainEnv, verbose=0, nminibatches=1, **agentParams)

    # Run optimizer
    last_reward = -np.finfo(np.float16).max
    evaluation_interval = int(len(train_df) / params.get('n_test_episodes'))

    for eval_idx in range(params.get('n_evaluations')):
        try:
            model.learn(evaluation_interval)
        except AssertionError:
            raise

        rewards = []
        n_episodes, reward_sum = 0, 0.0

        obs = testEnv.reset()
        while n_episodes < params.get('n_test_episodes'):
            action, _ = model.predict(obs)
            obs, reward, done, _ = testEnv.step(action)
            reward_sum += reward

            if done:
                rewards.append(reward_sum)
                reward_sum = 0.0
                n_episodes += 1
                obs = testEnv.reset()

        last_reward = np.mean(rewards)
        trial.report(-1 * last_reward, eval_idx)

        if trial.should_prune(eval_idx):
            raise optuna.structs.TrialPruned()

    return -1 * last_reward


params = getConfiguration('configOptimizer.yaml')
study_name = params.get('model') + params.get('reward_strategy')
output_file = study_name + '-output.json'
reward_strategy = params.get('reward_strategy')

study = optuna.create_study(study_name=study_name, load_if_exists=True)

try:
    study.optimize(objective, n_trials=params.get('n_trials'), n_jobs=params.get('n_jobs'))
except KeyboardInterrupt:
    pass


print('> Finished {} trials'.format(len(study.trials)))
print('> Outputting best trial params to {}'.format(output_file))

bestBoy = study.best_trial

with open(output_file, 'w') as outfile:
    json.dump(bestBoy, outfile)

print('Best value: ', bestBoy.value)

