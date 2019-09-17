## This program is used for tweets sentiment analysis based on tweets text and a given topic.

import re

# Get a dictionary of tweets id and text.    
def get_tweetdict(tweet_file_txt):
    with open(tweet_file_txt) as tweets:
        lines = tweets.read().splitlines()
        tweet_dict = {}
        for line in lines:
            id_position = line.find('\t')
            tweet_id = line[:id_position]
            line = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', line, flags=re.MULTILINE)
            line = re.sub(r'[0-9]+\t', '', line, flags=re.MULTILINE)
            line = re.sub(r'@[\w]*', '', line, flags=re.MULTILINE)
            line = re.sub(r'#', ' ', line, flags=re.MULTILINE)
            tweet_dict[tweet_id] = line.lstrip().lower()
        return tweet_dict

# Get a dictionary of tweets id and label.
def get_labeldict(label_file_txt):
    with open(label_file_txt) as labels:
        lines = labels.read().splitlines()
        label_dict = {}
        for line in lines:
            id_position = line.find('\t')
            tweet_id = line[:id_position]
            line = re.sub(r'[0-9]+\t', '', line, flags=re.MULTILINE)
            label_dict[tweet_id] = line.lstrip()
        return label_dict

tweet_dict = get_tweetdict("../Data/test-tweets.txt")
label_dict = get_labeldict("../Models/test-labels-MLR.txt")

count = 0
negative = 0
positive = 0
neutral = 0
key_word = 'love'

# Print the tweet id, label and text according to the key word.
for key in tweet_dict:
    words = tweet_dict[key].split()
    if key_word in words:
        print(key,'|', label_dict[key],'\t', tweet_dict[key])
        count += 1
        if label_dict[key] == 'negative':
            negative += 1
        elif label_dict[key] == 'positive':
            positive += 1
        else: neutral += 1

print('\nKey-Word:',key_word)
print('Count:',count)
print('Negative:',negative,'\t({:.2%})'.format(negative/count))
print('Positive:',positive,'\t({:.2%})'.format(positive/count))
print('Neutral:',neutral,'\t({:.2%})'.format(neutral/count))

