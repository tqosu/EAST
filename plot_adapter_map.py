import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def extract_number(value):
    # Extract numbers from strings like '[4]'
    if isinstance(value, str):
        numbers = re.findall(r'\d+', value)
        if numbers:
            return int(numbers[0])
    return value

def main():
    parser = argparse.ArgumentParser(description="Process CSV data and plot adapter index vs average mAP")
    parser.add_argument("csv_file_path", help="Path to the CSV file to process")
    parser.add_argument("--output", "-o", default="adapter_map_plot.png", 
                        help="Output filename for the plot (default: adapter_map_plot.png)")
    parser.add_argument("--title", "-t", default="Average mAP by Adapter Index",
                        help="Title for the plot (default: Average mAP by Adapter Index)")
    args = parser.parse_args()
    
    # Read the CSV file
    df = pd.read_csv(args.csv_file_path)
    
    # Apply the extract_number function to adapter_index column if needed
    if 'adapter_index1' in df.columns:
        df['adapter_index'] = df['adapter_index1'].apply(extract_number)
    print(df.columns)
    print(df.head())
    
    # Sort by adapter_index
    df_sorted = df.sort_values(by='adapter_index')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted['adapter_index'], df_sorted['average_mAP'], marker='o', linestyle='-')
    plt.xlabel('Adapter Index')
    plt.ylabel('Average mAP')
    plt.title(args.title)  # Use the title from command-line argument
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, 24))
    
    # Add data points labels if desired
    for i, txt in enumerate(df_sorted['average_mAP']):
        plt.annotate(f"{txt:.3f}", 
                    (df_sorted['adapter_index'].iloc[i], df_sorted['average_mAP'].iloc[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Save the plot with the user-specified filename
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as {args.output}")

if __name__ == "__main__":
    main()
