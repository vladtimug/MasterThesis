import csv

class CSVlogger():
    def __init__(self, logname, header_names):
        self.header_names = header_names
        self.logname = logname
        with open(logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(header_names)
    
    def write(self, inputs):
        with open(self.logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)