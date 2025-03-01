import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

def process_task(task_config_tuple):
    # Your processing code here
    # Example plotting code:
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig(f'plot_{task_config_tuple[0]}.png')
    plt.close()

if __name__ == "__main__":
    # Use 'spawn' start method
    mp.set_start_method('spawn')

    max_concurrency = 4  # Set according to your requirement
    task_config_tuple_list = [(i,) for i in range(10)]  # Example tasks

    with Pool(processes=max_concurrency) as pool:
        results = list(tqdm(pool.imap(process_task, task_config_tuple_list), total=len(task_config_tuple_list)))
