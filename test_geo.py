import pandas as pd
import os
import spacy
import csv
nlp = spacy.load("en_core_web_sm")

def readfile(address,geotext):
    with open(geotext,'w',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for file in os.listdir(address):
            info = os.path.join(address,file)
            data = pd.read_csv(info)
            #print(data.loc[1,:])
            for i in range(1,len(data)):
                tw_text = data.loc[i,'tweet']
                doc = nlp(tw_text)
                geoloc = []
                for ent in doc.ents:
                    if ent.label_ == 'GPE' or ent.label_ == 'LOC':
                        print(ent.text, ent.label_)
                        geoloc.append(ent.text)
                if len(geoloc) > 0:
                    #print(tw_text)
                    writer.writerow([tw_text,','.join(geoloc)])



if __name__ == "__main__":
    readfile("twitter",'geo_tweet.csv')