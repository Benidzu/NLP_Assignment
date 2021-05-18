import requests
import os
import json
"""
Script was modified for our use from the original source: https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/master/Tweet-Lookup/get_tweets_with_bearer_token.py

In order for this script to work, one should replace <your_bearer_token> in auth function with their own private Twitter API bearer token.
"""

def auth():
    return "<your_bearer_token>"


def create_url(tweet_id):
    url = "https://api.twitter.com/2/tweets?ids={}&tweet.fields=text".format(tweet_id)
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def get_tweet_text(tweet_id):
    bearer_token = auth()
    url = create_url(tweet_id)
    headers = create_headers(bearer_token)
    json_response = connect_to_endpoint(url, headers)
    if "data" not in json_response:
        return "ERROR: no tweet" 
    return (json_response["data"][0]["text"])

if __name__ == "__main__":
    #get_tweet_text("1369748710930931718")839880162586071040
    print(get_tweet_text(839630739335495681))