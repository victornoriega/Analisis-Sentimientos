import json
import os


sin_repetidos = []
rt_s = 'retweeted_status'
f = open("no_clasificados.txt", "a", encoding="utf-8")
#falta 20181025-14 posiblemente
for j in range(41, 47):
    with open('20181029-' + str(j) + '.json', encoding="utf-8") as tweet_data:
        json_data = json.load(tweet_data)

    for i in range(100):
        tweet = json_data['results'][i]

        if rt_s in tweet.keys():
            if 'extended_tweet' in tweet[rt_s].keys():
                if tweet[rt_s]['extended_tweet']['full_text'] in sin_repetidos:
                    continue
                sin_repetidos.append(tweet[rt_s]['extended_tweet']['full_text'])

                f.write(str(tweet[rt_s]['extended_tweet']['full_text']))
                f.write('\n')
                f.write('<\\s>\n')
            else:
                if tweet[rt_s]['text'] in sin_repetidos:
                    continue
                sin_repetidos.append(tweet[rt_s]['text'])

                f.write(str(tweet[rt_s]['text']))

                f.write('\n')
                f.write('<\\s>\n')
        elif 'extended_tweet' in tweet.keys():
            if tweet['extended_tweet']['full_text'] in sin_repetidos:
                continue
            sin_repetidos.append(tweet['extended_tweet']['full_text'])

            f.write(str(tweet['extended_tweet']['full_text']))

            f.write('\n')
            f.write('<\\s>\n')
        else:
            if tweet['text'] in sin_repetidos:
                continue
            sin_repetidos.append(tweet['text'])

            f.write(str(tweet['text']))

            f.write('\n')
            f.write('<\\s>\n')
