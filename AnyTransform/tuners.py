import itertools
import logging
from math import ceil

import optuna
from optuna import study
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from AnyTransform.parser import seed


class BaseTuner:

    def ask(self):
        raise NotImplementedError

    def tell(self, param_dict, score):
        raise NotImplementedError


class MyGridTuner(BaseTuner):

    def __init__(self, params_space):
        self.params_space = params_space
        params_list = itertools.product(*[params_space[key]['values'] for key in params_space.keys()])
        self.param_dict_list = [dict(zip(params_space.keys(), params)) for params in params_list]
        self.finished_num = 0

    def ask(self):
        if self.finished_num < len(self.param_dict_list):
            return self.param_dict_list[self.finished_num]
        raise Exception('All params have been tried!')

    def tell(self, param_dict, score):
        self.finished_num += 1


class MyBatchTuner(BaseTuner):

    def __init__(self, params_space, param_dict_list):
        self.params_space = params_space
        self.param_dict_list = param_dict_list
        self.finished_num = 0

    def ask(self):
        if self.finished_num < len(self.param_dict_list):
            return self.param_dict_list[self.finished_num]
        raise Exception('All params have been tried!')

    def tell(self, param_dict, score):
        self.finished_num += 1

    def report(self, param_dict, intermediate_value, step):
        return None

    def should_prune(self, param_dict):
        return False


class OptunaTuner(BaseTuner):

    def __init__(self, params_space, direction, tuner_name, pruner_name, pruner_kwargs, enqueue_param_dicts):
        # 将params_space转换成optuna的dimensions
        # 转换成Optuna的distributions
        self.distributions = {}
        for key, value in params_space.items():
            if value['type'] == 'float':
                self.distributions[key] = FloatDistribution(min(value['values']), max(value['values']))
            elif value['type'] == 'int':
                step = 1 if len(value['values']) == 1 else ceil(value['values'][1] - value['values'][0])
                self.distributions[key] = IntDistribution(min(value['values']), max(value['values']), step=step)
            elif value['type'] == 'str':
                self.distributions[key] = CategoricalDistribution(value['values'])
            else:
                raise ValueError(f"Unknown type: {value['type']}")
        if tuner_name == 'TPESampler':
            sampler = optuna.samplers.TPESampler(seed=seed)  # FIXME: seed很重要
        # elif tuner_name == 'GridSampler':
        #     sampler = optuna.samplers.GridSampler()
        elif tuner_name == 'RandomSampler':
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"Unknown tuner_name: {tuner_name}")

        if pruner_name == 'MedianPruner':
            # FIXME：这里的参数需要调整
            # Ok：interval_steps=1 -> 有batch之后max_step本身就小
            # Ok：n_warmup_steps=0 -> 有batch之后max_step本身就小
            # 两者有点类似。。。
            # n_startup_trials=1 -> 有org打底，能保证最后有n个跑完...
            # n_min_trials=1 -> 能保证最后有n个跑完...
            # pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0,
            #                                      interval_steps=1, n_min_trials=10)
            needed_kwargs_keys = ['n_startup_trials', 'n_warmup_steps', 'interval_steps', 'n_min_trials']
            needed_kwargs = {key: pruner_kwargs[key] for key in needed_kwargs_keys}
            pruner = optuna.pruners.MedianPruner(**needed_kwargs)
        elif pruner_name == 'PercentilePruner':
            # percentile: float=50
            needed_kwargs_keys = ['percentile', 'n_startup_trials', 'n_warmup_steps', 'interval_steps', 'n_min_trials']
            needed_kwargs = {key: pruner_kwargs[key] for key in needed_kwargs_keys}
            pruner = optuna.pruners.PercentilePruner(**needed_kwargs)
        elif pruner_name == 'NoPruner':
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner_name: {pruner_name}")

        self.study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
        self.random_study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=seed))
        self.param_dict_trial_dict = {}
        self.max_repeat = 10000  # FIXME：重复的后果很严重！，影响排序计数

        if enqueue_param_dicts is not None:
            for param_dict in enqueue_param_dicts:
                self.study.enqueue_trial(param_dict)

    def ask(self):
        trial = self.study.ask(self.distributions)
        param_dict = trial.params
        # FIXME: 如果重复了
        repeat = self.max_repeat
        while str(param_dict) in self.param_dict_trial_dict and repeat > 0:
            param_dict = self.random_study.ask(self.distributions).params
            self.study.enqueue_trial(param_dict)
            trial = self.study.ask(self.distributions)
            repeat -= 1
        if repeat != self.max_repeat:
            logging.warning(f"Randomly choose param_dict {self.max_repeat - repeat} times!")
        if repeat == 0:
            raise Exception('All params have been tried!')

        self.param_dict_trial_dict[str(param_dict)] = trial
        return param_dict

    def tell(self, param_dict, score):
        trial = self.param_dict_trial_dict[str(param_dict)]
        self.study.tell(trial, score)

    def report(self, param_dict, intermediate_value, step):
        trial = self.param_dict_trial_dict[str(param_dict)]
        trial.report(intermediate_value, step)

    def should_prune(self, param_dict):
        trial = self.param_dict_trial_dict[str(param_dict)]
        return trial.should_prune()
