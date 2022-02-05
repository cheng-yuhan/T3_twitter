from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="yuhan-application")
import csv

with open("Texas_county.txt", 'w', encoding='utf-8') as w:
    #writer = csv.writer(csvfile)
    with open('Texas_county_2.txt','r') as f:
        data = f.readlines()
        for line in data:
            line = line.split(',')
            location = geolocator.geocode(line[0])
            print((location.latitude, location.longitude))
            #county = line[1][:-7]
            #print(location)
            #break
            w.write(line[0]+','+str(location.latitude)+','+str(location.longitude))


#address1 = ''
#address2 = 'Brazos County, Texas'
#location = geolocator.geocode(address1)
#print((location.latitude, location.longitude))
#location = geolocator.geocode(address2)
#print((location.latitude, location.longitude))
