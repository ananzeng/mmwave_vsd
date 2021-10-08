import os
import csv

dataset_csv_path = "dataset/test/1.8"

temp_array = []
for filename in os.listdir(dataset_csv_path):
    if filename[0] != "R":
        with open(os.path.join(dataset_csv_path, filename), newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                print(row[7])
                temp_array.append(row[7])
            #寫入txt
            temp_array = temp_array[1:]

            f = open(os.path.join(dataset_csv_path, filename[:-4] + ".txt"), "w")
            for i in temp_array:
                f.write(i)
                f.write("\n")
            f.close()