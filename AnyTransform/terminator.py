import logging
import sys
import time
from collections import deque
from abc import ABC, abstractmethod


class Terminator(ABC):
    @abstractmethod
    def update(self, current_score):
        pass

    @abstractmethod
    def check_termination(self):
        pass


class TimeLimitTerminator(Terminator):
    def __init__(self, max_time):
        self.max_time = max_time
        self.start_time = time.time()
        logging.info(f'TimeLimitTerminator: max_time={max_time}')

    def update(self, current_score):
        pass

    def check_termination(self):
        flag = time.time() - self.start_time > self.max_time
        logging.info(f'{self.__class__.__name__}: '
                     f'Time elapsed: {time.time() - self.start_time}, max time: {self.max_time}')
        return flag


class MaxIterationsTerminator(Terminator):
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.iteration_count = 0
        logging.info(f'{self.__class__.__name__}: max_iterations={max_iterations}')

    def update(self, current_score):
        self.iteration_count += 1

    def check_termination(self):
        flag = self.iteration_count > self.max_iterations
        logging.info(f'{self.__class__.__name__}: '
                     f'Iteration count: {self.iteration_count}, max iterations: {self.max_iterations}')
        return flag


class NoImprovementTerminator(Terminator):
    def __init__(self, mode, min_improvement, patience):
        assert mode in ['maximize', 'minimize']
        self.mode = mode
        self.min_improvement = min_improvement
        self.patience = patience
        self.best_score = None
        self.no_improvement_count = 0
        logging.info(f'{self.__class__.__name__}: mode={mode}, min_improvement={min_improvement}, patience={patience}')

    def update(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return
        improvement = (current_score - self.best_score) / ((self.best_score + current_score) / 2) \
            if self.mode == 'maximize' else (self.best_score - current_score) / ((self.best_score + current_score) / 2)
        if improvement > self.min_improvement:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        logging.info(f'{self.__class__.__name__}: Improvement: {improvement}, '
                     f'best_score: {self.best_score}, current_score: {current_score}')

    def check_termination(self):
        flag = self.no_improvement_count > self.patience
        logging.info(f'{self.__class__.__name__}: No improvement count: {self.no_improvement_count}, '
                     f'patience: {self.patience}, best_score: {self.best_score}')
        return flag


class RelativeImprovementTerminator(Terminator):
    def __init__(self, mode, min_improvement, check_interval, maxlen):
        assert mode in ['maximize', 'minimize']
        assert maxlen > 2
        self.mode = mode
        self.min_improvement = min_improvement
        self.check_interval = check_interval
        self.latest_scores = deque(maxlen=maxlen)
        self.iteration_count = 0
        logging.info(f'{self.__class__.__name__}: mode={mode}, min_improvement={min_improvement}, '
                     f'check_interval={check_interval}, maxlen={maxlen}')

    def update(self, current_score):
        self.iteration_count += 1
        self.latest_scores.append(current_score)

    def check_termination(self):
        # 希望current不要比latest_best差太多
        if len(self.latest_scores) == self.latest_scores.maxlen and self.iteration_count % self.check_interval == 0:
            half_maxlen = len(self.latest_scores) // 2
            prev_best = max(self.latest_scores[:half_maxlen]) if self.mode == 'maximize' else min(self.latest_scores)
            latest_best = max(self.latest_scores[half_maxlen:]) if self.mode == 'maximize' else min(self.latest_scores)
            improvement = (latest_best - prev_best) / ((latest_best + prev_best) / 2) \
                if self.mode == 'maximize' else (prev_best - latest_best) / ((latest_best + prev_best) / 2)
            flag = improvement < self.min_improvement
            logging.info(f'{self.__class__.__name__}: Relative improvement: {improvement}, '
                         f'latest best: {prev_best}, current score: {latest_best}')
            return flag
        logging.info(f'Not checking termination at iteration {self.iteration_count}')
        return False


class TerminatorManager:
    def __init__(self, terminators, min_num_valid_params, min_num_better_params, mode='minimize'):
        self.terminators = terminators
        self.min_num_valid_params = min_num_valid_params
        self.org_value = None
        self.cur_valid_num_params = 0
        self.min_num_better_params = min_num_better_params
        self.cur_num_better_params = 0
        self.mode = mode
        assert mode in ['maximize', 'minimize']
        logging.info(f'TerminatorManager: min_num_valid_params={min_num_valid_params}, '
                     f'min_num_better_params={min_num_better_params}')

    def update(self, current_score, pruned):
        for terminator in self.terminators:
            terminator.update(current_score)

        if not pruned:  # 只有非prune到的才算达到better的入门要求
            self.cur_valid_num_params += 1
            self.org_value = current_score if self.org_value is None else self.org_value
            if current_score < self.org_value and self.mode == 'minimize':
                self.cur_num_better_params += 1
        else:
            pass

    def check_termination(self):
        flag = [terminator.check_termination() for terminator in self.terminators]
        if flag and self.cur_valid_num_params < self.min_num_valid_params:
            logging.info(f'Not terminating because cur_num_params={self.cur_valid_num_params} '
                         f'< min_num_params={self.min_num_valid_params}')
            return False
        if flag and self.cur_num_better_params < self.min_num_better_params:
            logging.info(f'Not terminating because cur_num_better_params={self.cur_num_better_params} '
                         f'< min_num_better_params={self.min_num_better_params}')
            return False
        return any(flag)

    def update_and_check(self, current_score, pruned):
        self.update(current_score, pruned)
        return self.check_termination()


if __name__ == '__main__':
    # log_file = os.path.join(res_dir, 'exp_single.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Example usage
    manager = TerminatorManager([  # FIXME
        TimeLimitTerminator(60),
        MaxIterationsTerminator(300),
        # FIXME: 应该判断的是什么时候不只用100个！！！
        NoImprovementTerminator('minimize', -1, 30),  # 25->40就停
        RelativeImprovementTerminator('minimize', -0.1, 20, 40)  # 希望current不要比latest_best差太多 # FIXME：探索难说
    ], 75, 10)  # FIXME: 100->67min 600->260min # 保护：至少要选出10个更好的！！！

    # Simulating the training process
    import random

    scores = []
    for i in range(10000):
        print()
        current_score = random.uniform(0, 1) - i / 1000
        scores.append(current_score)
        manager.update(current_score)
        if manager.check_termination():
            logging.info(f'Terminating at iteration {i}')
            logging.info(f'Scores: {scores}')
            break
