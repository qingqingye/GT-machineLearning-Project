import os
import csv
from geopy import distance
import requests
from select import select
import time
from email import utils as eutils
import datetime
from datetime import datetime as dt
from posixpath import split
import pandas as pd

class DataCleaning():
    def __init__(self):
        pass

    def get_county_location(self, data_df, output_path):
        addresses = data_df[ "RegionName"].tolist()
        result_latitude = []
        result_longitude = []
        api_key = "AIzaSyC13hek9jE7jbyEGT53j_PoNCQ4mvMsQTc"
        for adr in addresses:
            geocode_url = "https://maps.googleapis.com/maps/api/geocode/json?address={}".format(adr)
            if api_key is not None:
                geocode_url = geocode_url + "&key={}".format(api_key)
                r = requests.get(geocode_url)
                location = r.json()['results'][0]['geometry']['location']
                result_latitude.append(location['lat'])
                result_longitude.append(location['lng'])
                print(location['lat'], location['lng'])
        data_df["latitude"] = result_latitude
        data_df["longitude"] = result_longitude
        if not os.path.exists(output_path):
            data_df.to_csv(output_path)
        return data_df    
          


    def coordination_refine(self, origin_filepath, lat_long_filepath):
        data_df = None
        if os.path.exists(lat_long_filepath):
            data_df = pd.read_csv(lat_long_filepath)
        else:
            data_df = self.get_county_location(pd.read_csv(origin_filepath), lat_long_filepath)
         
      
        latitudes = data_df["latitude"].tolist()
        longitudes = data_df["longitude"].tolist()

        min_lat = min(latitudes)
        max_lat = max(latitudes)
        min_lon = max(longitudes)
        max_lon = min(longitudes)
        print(min_lat,min_lon)

        origin = (min_lat, min_lon) # origin point
        x_dist_list = []
        y_dist_list = []
 
        for i in range(len(latitudes)):
            coordinate = (latitudes[i], longitudes[i])
            x_dist = distance.distance(origin, (origin[0], coordinate[1])).km
            y_dist = distance.distance(origin, (coordinate[0], origin[1])).km
            x_dist_list.append(x_dist)
            y_dist_list.append(y_dist)
        data_df["x_dist"] = x_dist_list
        data_df["y_dist"] = y_dist_list

        data_df.to_csv("../data_collection_clean/df_after_FINAL_with_xyAxis.csv")
        return data_df
      

        
    def timestamp_refine(self, data_df):
        times = data_df["Date"].tolist()
        modified_times = []
        
        for time in times:
            splited = time.split("-")  # [year, month]
            mod_time = (int(splited[0])-2020) * 12 + int(splited[1])
            modified_times.append(mod_time)
     
        data_df["timenum"] = modified_times
        return data_df


if __name__ == "__main__":
    oper = DataCleaning()
    selected_path = "../data_collection_clean/df_after_FINAL.csv"
    lat_log_output_path = "../data_collection_clean/df_after_FINAL_withgeoinfo.csv"
    # get latitude and longitude then turn to xy axis
    xy_df = oper.coordination_refine(selected_path, lat_log_output_path)
    xy_time_df =   oper.timestamp_refine(xy_df)

    refined_data_path = "../data_collection_clean/df_after_FINAL_refined.csv"
    xy_time_df.to_csv(refined_data_path)
  
# 纬度， 经度