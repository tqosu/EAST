import pandas as pd
import argparse

def generate_max_and_avg_scores_per_dataset(csv_file):
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Create the 'FEA' column by summing over the specified columns
    # df['FEA'] = df[['F1@10', 'F1@25', 'F1@50', 'Edit', 'f_Acc']].sum(axis=1)

    # Find the rows with the maximum 'FEA' for each group of 'dataset' and 'split'
    # print(df.groupby(['dataset', 'split'])
    max_fea_df = df.loc[df.groupby(['dataset', 'split'])['FEA'].idxmax()]
    print(max_fea_df)

    # Group by 'dataset' and calculate the mean for each specified column
    avg_df = max_fea_df.groupby('dataset', as_index=False).agg({
        'F1@10': 'mean', 'F1@25': 'mean', 'F1@50': 'mean',
        'Edit': 'mean', 'f_Acc': 'mean', 'o_Acc': 'mean'
    })

    return avg_df

# Format the selected columns with one decimal and join them with '&'
def format_avg_scores(df):
    formatted_scores = df.apply(
        lambda row: ' & '.join(f"{value:.1f}" for value in [
            row['F1@10'], row['F1@25'], row['F1@50'], row['Edit'], row['f_Acc']
        ]), axis=1
    )
    return formatted_scores

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process CSV file to generate scores.')
    parser.add_argument('--csv_file', 
                       type=str, 
                       default='results.csv',
                       help='Path to the CSV file (default: results.csv)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate scores using the provided CSV file
    avg_scores_df = generate_max_and_avg_scores_per_dataset(args.csv_file)

    # Display the new dataframe
    print(avg_scores_df)

    # Get the formatted column values
    formatted_scores = format_avg_scores(avg_scores_df)

    # Display the formatted scores
    print(formatted_scores)