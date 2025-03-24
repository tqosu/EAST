import csv
import os
import argparse

def read_best_row(file_path):
    """Read the CSV file and return the row with the highest average_mAP."""
    best_row = None
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if best_row is None or float(row['Edit']) > float(best_row['Edit']):
                best_row = row
    return best_row

def main(base_path, num_splits):
    # Define the file paths based on the number of splits
    file_paths = [
        os.path.join(base_path, f'split{i+1}/gpu2_id0/{args.path}') for i in range(num_splits)
    ]

    # Initialize a dictionary to hold the sum of the best metrics
    metrics_sum = {
        'F1@0.1': 0,
        'F1@0.25': 0,
        'F1@0.5': 0,
        'Edit': 0,
        'Acc':0
    }

    # Read each CSV file, find the best row based on average_mAP, and accumulate the metrics
    for file_path in file_paths:
        best_row = read_best_row(file_path)
        print(best_row)
        # Accumulate the metrics (excluding 'epoch' and 'average_mAP')
        for key in metrics_sum.keys():
            metrics_sum[key] += float(best_row[key])

    # Calculate the average metrics over the splits
    metrics_avg = {key: value / num_splits for key, value in metrics_sum.items()}

    # Print the average metrics
    print("Average metrics over the splits:")
    for key, value in metrics_avg.items():
        print(f"{key}: {value:.6f}")
    for key, value in metrics_avg.items():
        print(f"{value*100:.1f}&",end='')
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate metrics from multiple CSV files.')
    parser.add_argument('--base_path', type=str, help='The base path to the directory containing the CSV files.')
    parser.add_argument('--num_splits', type=int, help='The number of splits (CSV files) to process.')
    parser.add_argument('--path', type=str)
    
    args = parser.parse_args()
    main(args.base_path, args.num_splits)
