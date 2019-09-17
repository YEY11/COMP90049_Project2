## This program is used to extract test labels from the result csv file and 
## write it into a new txt file with tweet ids for the further topic analysis.

import csv

# Get label list from the CSV test result file.   
def get_labels(result_file_csv):
    with open(result_file_csv,'r') as labels_result:
        reader = csv.DictReader(labels_result)
        labels = [row["'predicted sentiment'"] for row in reader]
        return labels

# Get id list from the TXT  file.  
def get_ids(tweets_file_txt):
    with open(tweets_file_txt) as tweets:
        lines = tweets.read().splitlines()
        ids = []
        for i in range(len(lines)):
            id_position = lines[i].find('\t')
            tweet_id = lines[i][:id_position]
            ids.append(tweet_id)
        return ids
            
id_list = get_ids("../Data/test-tweets.txt")
label_list = get_labels("../Models/test_RF.csv")

# Write the ids and labels into a new txt file.
fileObject = open('../Models/test-labels-RF.txt', 'w')
for i in range(len(id_list)):
	fileObject.write(id_list[i] + '\t' + label_list[i])
	fileObject.write('\n')
fileObject.close()
