__author__ = 'chhavi21'

from spyre import server

import pandas as pd
import urllib2
import json
# from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

class StockExample(server.App):
    title = "User Choices"

    inputs = [{     "type":'dropdown',
                    "label": 'User Name',
                    "options" : [ {"label": "Michael", "value":"90a6z--_CUrl84aCzZyPsg"},
                                  {"label": "Lindsey", "value":"4ozupHULqGyO42s3zNUzOQ"},
                                  {"label": "Albert", "value":"joIzw_aUiNvBTuGoytrH7g"},
                                  {"label": "Aileen", "value":"0bNXP9quoJEgyVZu9ipGgQ"}],
                    "key": 'user',
                    "action_id": "update_data"},
              {     "type":'dropdown',
                    "label": 'Category',
                    "options" : [ {"label": "Restaurants", "value":"Restaurants"},
                                  {"label": "Active Life", "value":"Active Life"},
                                  {"label": "Shopping", "value":"Shopping"}],
                    "key": 'category',
                    "action_id": "update_data"},

              {     "type":'dropdown',
                    "label": 'Zipcode',
                    "options" : [ {"label": "85003", "value":"85003"},
                                  {"label": "85004", "value":"85004"},
                                  {"label": "85006", "value":"85006"},
                                  # {"label": "85007", "value":"85007"},
                                  {"label": "85008", "value":"85008"}],
                    "key": 'zipcode',
                    "action_id": "update_data"}]

    controls = [{   "type" : "hidden",
                    "id" : "update_data"}]

    tabs = ["Map"]

    outputs = [{"type" : "html",
                    "id" : "custom_html",
                    "tab" : "Map"},
                { "type" : "table",
                    "id" : "table_id",
                    "control_id" : "update_data",
                    "tab" : "Table",
                    "on_page_load" : True }]



    def getHTML(self, params):
        # return "<img src=https://maps.googleapis.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=13&size=600x300&maptype=roadmap&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318&markers=color:red%7Clabel:C%7C40.718217,-73.998284&key=AIzaSyDEsD1LRBqWa2u9IZMHa3B4JOGaAG7Qp20>"
        cat = params['category']
        usr = params['user']
        zip = params['zipcode']
        df = pd.read_json('data/for_demo.json')

        new_df = df[df.open == True] # get all the business that are open

        new_df_bool = new_df.categories.map(lambda x: cat in x)
        new_df = new_df.loc[new_df_bool,]

        new_df["zip_diff"] = abs(new_df.zip_code - int(zip))

        if usr in list(new_df.user_id.unique()):
            # if user has reviewed a business of category
            temp = new_df.loc[new_df.user_id == usr,]
            if 0 in temp.zip_diff.unique():
                new_df = temp

        new_df = new_df.sort(['zip_diff',
                                  'r_stars', 'b_stars'], ascending=[1, 0, 0])

        place = new_df.reset_index().drop('index', axis=1).loc[0,]

        name = place.biz_name
        zip = place.zip_code
        city = place.city

        name =  '+'.join(name.replace('&', '%26').split(' '))

        return """<iframe
                  width="900"
                  height="650"
                  frameborder="0" style="border:0"
                  src="https://www.google.com/maps/embed/v1/place?key=AIzaSyAIEDh33h9Ibmzsgry2xO7WXA5HMFDvewQ
                    &q=%s,%s+AZ" allowfullscreen>
                </iframe>""" %(name,city)

app = StockExample()
app.launch(port=9093)