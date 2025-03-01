import time

import optuna

from optuna.trial import FrozenTrial

count = 0
result = {}


def objective(trial):
    global count
    count += 1
    x = trial.suggest_float('x', -10, 10)
    if isinstance(trial, FrozenTrial):
        print(f'FrozenTrial: {trial.number}')
    res = (x - 2) ** 2  # minimize
    result[trial.number] = res
    return res


study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                            storage='sqlite:///storage.sqlite', study_name=time.strftime('%Y%m%d%H%M%S'),
                            direction='minimize')
study.optimize(objective, n_trials=100, n_jobs=10)

print(count, len(result))

print(study.best_params)  # E.g. {'x': 2.002108042}

best_trial = study.best_trial
print(objective(trial=best_trial))


load_study = optuna.load_study(study_name='20240507200337', storage='sqlite:///storage.sqlite')
print(load_study.best_params)
print(load_study.best_value)
