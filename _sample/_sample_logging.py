import os
import sys
import logging


def setup_logging(log_file):
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    #
    # # Remove all existing handlers
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    #
    # # File handler
    # file_handler = logging.FileHandler(log_file, 'w', 'utf-8')
    # file_handler.setLevel(logging.INFO)
    # file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # file_handler.setFormatter(file_formatter)
    # logger.addHandler(file_handler)
    #
    # # Stream handler
    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setLevel(logging.INFO)
    # stream_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # stream_handler.setFormatter(stream_formatter)
    # logger.addHandler(stream_handler)

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.info(f"log_file={log_file}")


# Test the logging setup
def test_multiple_logging_setups():
    res_root_dir = "."

    # First log file setup
    log_file1 = os.path.join(res_root_dir, '_manage_exp1.log')
    setup_logging(log_file1)
    logging.info("This is a log message for the first setup.")

    # Second log file setup
    log_file2 = os.path.join(res_root_dir, '_manage_exp2.log')
    setup_logging(log_file2)
    logging.info("This is a log message for the second setup.")


if __name__ == "__main__":
    test_multiple_logging_setups()
