import csv


'''
Reads a CSV file and returns a dictionary 
with the first column as keys and the rest of the columns as values
'''
def read_csv_file(file_path):
    result = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            key = row[0]
            value = row[1:]

            result[key] = value

    return result



'''
Writes a dictionary to a CSV file
'''
def write_to_file(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in data.items():
            writer.writerow([key] + value)