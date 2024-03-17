import json
import statistics

# Open the JSON file
with open('results_1.json') as file:
    # Load the JSON data
    data = json.load(file)

# Print the content of the JSON file
#print(data)

# Extract the numbers from the data
numbers = [item[1] for item in data]

# Calculate the average
average = statistics.mean(numbers)

# Calculate the standard deviation
std_dev = statistics.stdev(numbers)

# Calculate the variance
variance = statistics.variance(numbers)

# Print the average, standard deviation and variance
print("Average:", average)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# Sort the data based on the results in descending order
sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

# Get the names of the files with the highest results
top_files = [item[0] for item in sorted_data[:5]]

# Print the names of the files
print("Files with the highest results:")
for file_name in top_files:
    print(file_name)