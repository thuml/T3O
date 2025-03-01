import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Example script that receives arguments')
    parser.add_argument('--arg1', type=str, help='First argument')
    parser.add_argument('--arg2', type=str, help='Second argument')
    args = parser.parse_args()

    arg1 = args.arg1
    arg2 = args.arg2

    parser2 = argparse.ArgumentParser(description='Example script that receives arguments')
    parser2.add_argument('--arg1', type=str, help='First argument')
    parser2.add_argument('--arg2', type=str, help='Second argument')
    _args = parser2.parse_args()
    print('_args', _args)

    if arg1 and arg2:
        print(f"Received arguments: arg1 = {arg1}, arg2 = {arg2}")
        t = arg1.split("e")[-1]
        os.system(f"sleep {t}")
        print("Finished processing arguments.")
    else:
        print("Please provide both --arg1 and --arg2 arguments.")


if __name__ == "__main__":
    main()
