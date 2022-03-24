import csv
import codecs

def data_write_csv(file_name, datas): # file_name is the path to write the CSV file, datas is the list of data to be written
    file_csv = codecs.open(file_name,'w+','utf-8') # append
    writer = csv.writer(file_csv)
    for data in datas:
        writer.writerow(data)
    print("csv saved")

if __name__ == "__main__":
    epoch = 3
    loss = [[]]
    for i in range(epoch):
        loss.append([i+1, 1.23, 3.45, 5.67])
        data_write_csv("intermid_results_revised/csv/results_{}.csv".format(i+1), loss)
    print("finished.")