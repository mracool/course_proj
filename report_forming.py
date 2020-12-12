#  forming data types info into report

from pathlib import Path
from analyse import data_dict


Path('C:/Users/Ulyana/PycharmProjects/data_anal/file.txt').touch()
file_name = 'file.txt'
file = open(file_name, 'w')
file.write(str(data_dict))
file.close()

