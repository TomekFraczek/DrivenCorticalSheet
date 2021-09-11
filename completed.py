import os
import argparse


def search(start_dir):
    total_runs = 0
    finished_runs = 0

    for root, dirs, files in os.walk(start_dir, topdown=False):
        for name in dirs:
            if os.path.exists(os.path.join(root, name, 'config.json')):
                total_runs += 1
                if os.path.exists(os.path.join(root, name, 'completion.txt')):
                    finished_runs += 1

    print(f" {finished_runs} out of {total_runs} have been completed")


def main():
    """Function to run the sweep from the commandline"""
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')

    parser.add_argument('--dir', metavar='output directory',
                        type=str, nargs='?',
                        help='directory to count completed runs in',
                        default='plots')

    args = parser.parse_args()
    search(args.dir)

if __name__ == '__main__':
    main()
