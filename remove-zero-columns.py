import pandas as pd
import sys
 
def remove_zero_columns(input_csv, output_csv):
	# Read the CSV file 
	dataframe = pd.read_csv(input_csv) 

	# Identify columns with all zeros (numeric columns only)
	zero_columns = [] 
	for col in dataframe.columns:
		if pd.api.types.is_numeric_dtype(dataframe[col]):
			if (dataframe[col] == 0).all():
				zero_columns.append(col) 
 
	# Drop the zero columns 
	df_cleaned = dataframe.drop(columns=zero_columns)

	# Save the cleaned data to a new CSV file
	df_cleaned.to_csv(output_csv, index=False)
	print(f"Columns with all zeros removed. Result saved to {output_csv}")

def main():
	if len(sys.argv) > 1:
		infile = sys.argv[1]
		outfile = sys.argv[1] + ".output"
		remove_zero_columns(infile, outfile);

main()

