import openpyxl
from utilitaries import write_to_file


XLSX_FILE_PATH = 'data/All Calcite 1.0.0-1.15.0 Doc2Vec+LSI.xlsx'
BUGS_SHEET_NAME = 'All Bugs Doc2Vec+LSI '
ALL_SHEET_NAME = 'All Doc2Vec+LSI'


'''
Reads an Excel file and returns a dictionary
keeping the Version-ID column as keys and columns > 'H' as values
'''
def read_excel_file(file_path, sheet_name):
    data_to_cluster = {}

    workbook = openpyxl.load_workbook(file_path)
    worksheet = workbook[sheet_name]

    for row in worksheet.iter_rows(min_row=7, min_col=5):
        
        id = None
        cells_list = []

        for cell in row:
            column_number = openpyxl.utils.column_index_from_string(cell.column_letter)

            if column_number == 5:
                id = cell.value

            if column_number > 7:
                cells_list.append(cell.value)
        
        data_to_cluster[id] = cells_list
    
    return data_to_cluster


'''
Reads both Excel sheets and writes them to CSV files
with only the important features
'''
if __name__ == '__main__':
    all = read_excel_file(XLSX_FILE_PATH, ALL_SHEET_NAME)
    write_to_file("parsed_data/all_to_cluster.csv", all)

    bugs = read_excel_file(XLSX_FILE_PATH, BUGS_SHEET_NAME)
    write_to_file("parsed_data/bugs_to_cluster.csv", bugs)



