import pandas as pd
import os
import csv
import time
from geopy.geocoders import Nominatim


def map_coordinate(county_data):
    stat = {}
    loc_la = []
    loc_long = []
    with open(county_data, 'r') as f:
        data = f.readlines()
        index = 0
        for line in data:
            if line == '\n':
                continue
            else:
                line = line.split(',')
                county = line[0]
                stat[index] = [county,0]
                loc_la.append(float(line[1][2:]))
                loc_long.append(float(line[2][:-3]))
                index += 1
    return stat,loc_la,loc_long



def findline(tweetfile,matchtext,stat=None,loc_la=None,loc_long=None,geolocator= None):
    def cloest(x,y):
        min_distance = float('inf')
        index = None
        #print(len(loc_la))
        for i in range(len(loc_la)):
            dis = (loc_la[i]-x)**2 + (loc_long[i]-y)**2
            if dis <= min_distance:
                min_distance = dis
                #print(min_distance)
                index = i
        return index

    with open(matchtext,'w',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        dic = ['storm','cold','freez','outage']
        #stat = {}
        data = pd.read_csv(tweetfile, encoding= '')
        min_la,max_la = min(loc_la),max(loc_la)
        min_long,max_long = min(loc_long),max(loc_long)

        for i in range(len(data)):
            tweet = data.iloc[i,0]
            for key in dic:
                if key in tweet:
                    #stat[data.iloc[i,1]] = stat.get(data.iloc[i,1],0) + 1
                    loc = data.iloc[i,1].split(',')
                    for lc in loc:
                        if lc == 'Texas' or lc == 'TEXAS' or lc == 'texas':
                            continue
                        else:
                            location = geolocator.geocode(lc)
                            try:
                                x,y = location.latitude, location.longitude
                                if x>= min_la and x <= max_la and y >= min_long and y < max_long:
                                    index = cloest(x,y)
                                    #print(index)
                                #else:
                                #    time.sleep(1)
                                    stat[index][1] += 1
                                    print(stat)
                            except AttributeError:
                                continue
                            break

                    break
    print(stat)

if __name__ == "__main__":
    geolocator = Nominatim(user_agent="yuhan-application")
    stat,loc_la,loc_long=map_coordinate('Texas_county_2.txt')
    #print(len(loc_la))
    #print(loc_long)
    #print(stat)
    findline("Copy of geo_tweet.csv", 'textdata.csv',stat,loc_la,loc_long,geolocator)