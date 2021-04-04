from twitter import get_tweet_text
from time import sleep
import datetime

path = "..\data\sexists\\"
data_type = "hostile"
rate_limit = 300

end = False

while not end:
    
    tweet_ids_file = open(path+data_type+"_sexist.tsv", "r")
    tweet_texts = open(path+data_type+"_texts.csv", "r", encoding="utf-8")

    print("[",datetime.datetime.now(),"] ", "Starting with retrieving tweets")

    last_tweet_index = 0

    for line in tweet_texts:
        last_tweet_index += 1

    tweet_texts.close()
    tweet_texts = open(path+data_type+"_texts.csv", "a+", encoding="utf-8")

    tweet_ids = []
    for line in tweet_ids_file:
        tweet_ids.append(line.strip())

    for i in range(0,rate_limit):
        if i + last_tweet_index >= len(tweet_ids):
            end = True
            print("Successfully retrieved every tweet!")
            break
        text = get_tweet_text(tweet_ids[last_tweet_index+i])
        text = text.replace("\n"," ")
        tweet_texts.write("{},{}\n".format(tweet_ids[last_tweet_index+i],text))

    tweet_ids_file.close()
    tweet_texts.close()

    if not end:
        print("Tweets retrieved, going to sleep...")
        sleep(15*60 + 30)