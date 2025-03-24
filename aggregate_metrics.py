import csv
import os
import argparse

def read_best_row(file_path):
    """Read the CSV file and return the row with the highest average_mAP."""
    best_row = None
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if best_row is None or float(row['average_mAP']) > float(best_row['average_mAP']):
                best_row = row
    return best_row

def main(base_path, num_splits):
    # Define the file paths based on the number of splits
    file_paths = [
        os.path.join(base_path, f'split{i+1}/gpu2_id0/epoch_metrics_mAP.csv') for i in range(num_splits)
    ]

    # Initialize a dictionary to hold the sum of the best metrics
    metrics_sum = {
        'average_mAP': 0,
        'mAP@0.3': 0,
        'mAP@0.4': 0,
        'mAP@0.5': 0,
        'mAP@0.6': 0,
        'mAP@0.7': 0
    }

    # Read each CSV file, find the best row based on average_mAP, and accumulate the metrics
    for file_path in file_paths:
        best_row = read_best_row(file_path)
        print(best_row)
        # Accumulate the metrics (excluding 'epoch' and 'average_mAP')
        for key in metrics_sum.keys():
            metrics_sum[key] += float(best_row[key])

    # Calculate the average metrics over the splits
    metrics_avg = {key: value / num_splits*100 for key, value in metrics_sum.items()}

    # Print the average metrics
    # average_mAP: 0.870641
    # mAP@0.3: 0.947249
    # mAP@0.4: 0.929403
    # mAP@0.5: 0.882117
    # mAP@0.6: 0.830628
    # mAP@0.7: 0.763809

    print("Average metrics over the splits:")
    for key, value in metrics_avg.items():
        print(f"{key}: {value:.6f}")
    print('{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}&{:.1f}'.format( metrics_avg['mAP@0.3'], metrics_avg['mAP@0.4'], metrics_avg['mAP@0.5'], metrics_avg['mAP@0.6'], metrics_avg['mAP@0.7'], metrics_avg['average_mAP']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate metrics from multiple CSV files.')
    parser.add_argument('base_path', type=str, help='The base path to the directory containing the CSV files.')
    parser.add_argument('num_splits', type=int, help='The number of splits (CSV files) to process.')
    
    args = parser.parse_args()
    main(args.base_path, args.num_splits)
