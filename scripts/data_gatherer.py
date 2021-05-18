from twitter import get_tweet_text
from time import sleep
import datetime

"""
This script was used to retrieve data from Twitter. Due to rate limit, script works periodically. It should be noted, that this script was continuously modified through datasets retrieval process, since different datasets stored tweet_ids differently. Therefore it was NOT meant that this script works for every dataset, but only for the last one, which we retrieved. Moreover, in order to run it, user needs to have right dataset files stored locally. 

"""

path = "..\data\\racist\\"
rate_limit = 300
delimeter = ','

end = False

while not end:
    
    tweet_ids_file = open(path+"NAACL_SRW_2016.csv", "r")
    tweet_texts = open(path+"texts.csv", "r", encoding="utf-8")

    print("[",datetime.datetime.now(),"] ", "Starting with retrieving tweets")

    last_tweet_index = 0

    for line in tweet_texts:
        last_tweet_index += 1

    tweet_texts.close()
    tweet_texts = open(path+"texts.csv", "a+", encoding="utf-8")

    tweet_ids = []
    labels = []
    for line in tweet_ids_file:
        
        splitted_line = line.split(delimeter)
        tweet_ids.append(splitted_line[0].strip())
        labels.append(splitted_line[1].strip())

    for i in range(0,rate_limit):
        if i + last_tweet_index >= len(tweet_ids):
            end = True
            print("Successfully retrieved every tweet!")
            break
        text = get_tweet_text(tweet_ids[last_tweet_index+i])
        text = text.replace("\n"," ")
        tweet_texts.write("{},{},{}\n".format(tweet_ids[last_tweet_index+i],text,labels[last_tweet_index+i]))

    tweet_ids_file.close()
    tweet_texts.close()

    if not end:
        print("Tweets retrieved, going to sleep...")
        sleep(15*60 + 30)