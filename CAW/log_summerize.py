import os
import re
import numpy as np

def parse_log_file(file_name):
    with open(file_name, 'r') as f:
        content = f.readlines()

    # Initialize the stats dict
    stats = {}
    
    for line in content:
        match = re.search(r'Test statistics: (Old nodes|New nodes) -- (\w+): (\d+\.\d+)', line)
        if match:
            node_type = match.group(1)
            metric = match.group(2)
            value = float(match.group(3))

            if node_type not in stats:
                stats[node_type] = {}
            if metric not in stats[node_type]:
                stats[node_type][metric] = []

            stats[node_type][metric].append(value)

    return stats

def calculate_stats(stats):
    result = {}
    for node_type in stats:
        result[node_type] = {}
        for metric in stats[node_type]:
            values = np.array(stats[node_type][metric])
            mean = np.mean(values) * 100  # Convert to percentage
            std = np.std(values) * 100  # Convert to percentage
            result[node_type][metric] = (mean, std)

    return result

def append_to_log_file(file_name, stats):
    with open(file_name, 'a') as f:
        for node_type in stats:
            f.write(f'\n*** {node_type} Average +/- Std. Dev. ***\n')
            print(f'\n*** {node_type} Average +/- Std. Dev. ***\n')
            for metric in stats[node_type]:
                mean, std = stats[node_type][metric]
                f.write(f'{metric}: {mean:.2f} +/- {std:.2f}\n')
                print(f'{metric}: {mean:.2f} +/- {std:.2f}\n')

def main(directory):
    
    for filename in os.listdir(directory):
        if filename.endswith(".log"):  # if the file is a log file
            file_path = os.path.join(directory, filename)
            stats = parse_log_file(file_path)
            result = calculate_stats(stats)
            append_to_log_file(file_path, result)

def summerize(file_path):
    stats = parse_log_file(file_path)
    result = calculate_stats(stats)
    append_to_log_file(file_path, result)


def remove_added_content(directory, start_marker="*** Old nodes Average +/- Std. Dev. ***"):
    for filename in os.listdir(directory):
        if filename.endswith(".log"):  # if the file is a log file
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Find the start_marker and delete everything after it
            for i, line in enumerate(lines):
                if start_marker in line:
                    # Delete everything from here to the end
                    lines = lines[:i]
                    break

            # Write the remaining lines back to the file
            with open(file_path, 'w') as f:
                f.writelines(lines)


if __name__ == '__main__':
    # directory = './tgn/log/neg_sample/'  # replace this with your log directory path
    directory = './tgn/log/'  # replace this with your log directory path
    main(directory)
    # remove_added_content(directory)

