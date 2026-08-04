import pandas as pd
import sys

def filter_csv(data_csv, filter_csv, output_csv):
	# Read the CSV files
	df1 = pd.read_csv(data_csv)
	df2 = pd.read_csv(filter_csv)

	# Get unique values from the second column (assuming it's the first column)
	values_to_remove = df2.iloc[:, 0].unique()

	# Filter the first dataframe
	filtered_df = df1[~df1["Body_Site"].isin(values_to_remove)]

	# Save the filtered result
	filtered_df.to_csv(output_csv, index=False)
	print(f"Filtered data saved to {output_csv}")


def main():
	if len(sys.argv) > 2:
		indata   = sys.argv[1]
		infilter = sys.argv[2]
		outfile  = sys.argv[1] + ".output"
		filter_csv(indata, infilter, outfile);

main()
