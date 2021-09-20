import os
import argparse


def completed(start_dir):
    total_runs = 0
    finished_runs = 0

    for root, dirs, files in os.walk(start_dir, topdown=False):
        for name in dirs:
            if os.path.exists(os.path.join(root, name, 'config.json')):
                total_runs += 1
                if os.path.exists(os.path.join(root, name, 'completion.txt')):
                    finished_runs += 1

    print(f" {finished_runs} out of {total_runs} have been completed")


def remove_file(start_dir, filename):
    for root, dirs, files in os.walk(start_dir, topdown=False):
        for file in files:
            if filename in file:
                filepath = os.path.join(root, file)
                print(f'Removing:  {filepath}')
                os.remove(filepath)


def main():
    """Function to run the sweep from the commandline"""
    parser = argparse.ArgumentParser(description='Select model_config scenario & path (optional):')

    parser.add_argument('--dir', metavar='output directory',
                        type=str, nargs='?',
                        help='directory to count completed runs in')
    parser.add_argument('--completed', action='store_true',
                        help='Count the number of completed simulations')
    parser.add_argument('--remove', action='store_true',
                        help='Remove all files matching filename')
    parser.add_argument('--filename', metavar='filename to use',
                        type=str, nargs='?')

    args = parser.parse_args()

    if args.completed:
        completed(args.dir)
    elif args.remove:
        remove_file(args.dir, args.filename)


if __name__ == '__main__':
    main()
