import sys

import psutil
from utils import *


def run_command(command, log_file):
    """启动一个独立进程并执行给定的命令，同时将输出重定向到指定的文件"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    cmd = f"{command} > {log_file} 2>&1 & echo $!"
    pid = os.popen(cmd).read().strip()
    return int(pid)


def check_process(pid):
    """检查进程是否还在运行"""
    return psutil.pid_exists(pid)


def kill_process(pid):
    """杀死指定的进程"""
    try:
        proc = psutil.Process(pid)
        proc.kill()
    except psutil.NoSuchProcess:
        logging.warning(f"Process with pid {pid} does not exist.")
    except Exception as e:
        logging.error(f"Error killing process {pid}: {e}")


# get return_code

class MyFuture:
    def __init__(self, pid, res_dir):
        self.pid = pid
        self.res_dir = res_dir

    def done(self):
        # 以return_code文件是否存在作为任务是否完成的标志
        if os.path.exists(os.path.join(self.res_dir, 'return_code.txt')):
            return True
        else:
            if not check_process(self.pid):
                raise ValueError(f"Process is not running but return_code file does not exist, check log files in {self.res_dir}")
            return False

    def running(self):
        return check_process(self.pid)

    def result(self):
        return int(open(os.path.join(self.res_dir, 'return_code.txt')).read().strip())


class MyExecutor:

    def __init__(self):
        pass

    def submit(self, func, *args):
        command, res_log_file = args
        res_dir = os.path.dirname(res_log_file)
        return MyFuture(func(command, res_log_file), res_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def signal_handler(signum, frame):
    """信号处理函数，用于捕获子进程终止信号并避免僵尸进程"""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            # running_pids.pop(pid, None)
        except ChildProcessError:
            break


def atexit_handler(futures):
    for future in futures:
        if future.running():
            kill_process(future.pid)
    futures.clear()
