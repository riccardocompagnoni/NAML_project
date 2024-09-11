import json
import os

if __name__ == '__main__':
    mapping = {}

    labels_path = "C://Users//Riccardo//Downloads//labels//personal"
    dir_list = os.listdir(labels_path)
    file = []
    labels = []
    curr_genre = 0
    for file_id in dir_list:
        file_name = labels_path + '//' + file_id
        with open(file_name) as file:
            while line := file.readline():
                md5 = line.rstrip()
                mapping.update({md5: curr_genre})
        curr_genre += 1

        print(file_id, ':', curr_genre)

    with open("labels.json", "w") as outfile:
        json.dump(mapping, outfile)
