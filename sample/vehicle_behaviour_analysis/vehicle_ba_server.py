import threading
import socket
import re
import time
import datetime
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template

ba_df = pd.DataFrame(columns=["time", "channel", "ba_label", "involve_targets", "properties_of_involve_targets", "involve_region", "ba_image", "ba_video"])
ba_df['time'] = pd.to_datetime(ba_df['time'])

# live stream for channel 0, change to right url first
live_stream_0 = "http://192.168.77.60/flv?app=live&stream=vehicle_ba_sample_0" 
# live stream for channel 1, change to right url first
live_stream_1 = "http://192.168.77.60/flv?app=live&stream=vehicle_ba_sample_1" 

'''
receive ba data from socket and save to dataframe of pandas.
'''
def ba_receiver():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bind_ip = '192.168.77.68'
    bind_port = 7777
    udp_socket.bind((bind_ip, bind_port))

    while True:
        data, addr = udp_socket.recvfrom(1024 * 64)
        s = data.decode()
        print('############################################')
        print(f'Received data: \n{s}')
        print(f'From address: {addr}')

        li = re.findall("\<\-\-.*?\-\-\>", s, re.DOTALL)
        print(f"total {len(li)} records parsed! going to append to data frame...")
        for l in li:
            items = str(l).split('\n')
            ''' 
            items[0] <--
            items[1] time (y:m:d h:m:s.mili)
            items[2] channel index, frame index
            items[3] ba type, ba label
            items[4] ids of targets
            items[5] vertexs of region
            items[6] properties of targets
            items[7] record image path
            items[8] record video path
            items[9] -->
            '''
            assert(len(items) == 10)
            # append items to dataframe
            ba_df.loc[len(ba_df)] = [items[1].split('.')[0], items[2].split(',')[0], items[3].split(',')[1], items[4], items[6], items[5], items[7], items[8]]
            print(f"now total {len(ba_df)} records in ba_df!")
        print('############################################')

    udp_socket.close()


'''
create flask app
'''
app = Flask(__name__)

# home page
@app.route("/", methods=["GET", "POST"])
def home():
    '''
    filter properties
    '''
    select_channel = "all" 
    if 'select_channel' in request.form:
        select_channel = request.form['select_channel']

    select_ba_type = "all"
    if 'select_ba_type' in request.form:    
        select_ba_type = request.form['select_ba_type']
    
    # 1 hour by default
    now = time.time()
    past_1_hour = now - 3600
    select_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(past_1_hour))
    select_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
    
    if 'select_start_time' in request.form and request.form['select_start_time'] != '': 
        select_start_time = datetime.datetime.strptime(request.form['select_start_time'], '%Y-%m-%dT%H:%M:%S')
    
    if 'select_end_time' in request.form and request.form['select_end_time'] != '': 
        select_end_time = datetime.datetime.strptime(request.form['select_end_time'], '%Y-%m-%dT%H:%M:%S')

    sql = "ba_label != ''"
    if select_channel != 'all':
        sql += f" & channel == '{select_channel}'"
    if select_ba_type != 'all':
        sql += f" & ba_label == '{select_ba_type}'"
    sql += f" & time >= '{select_start_time}'"
    sql += f" & time < '{select_end_time}'"

    print(sql)
    filter_df = ba_df.query(sql)
    print(filter_df)
    ba_results = []
    for index, row in filter_df.iterrows():
        ba_results.append({'uid':index, \
                        'time':row['time'], \
                        'channel':row['channel'], \
                        'ba_label':row['ba_label'], \
                        'involve_targets':row['involve_targets'], \
                        'properties_of_involve_targets': row['properties_of_involve_targets'], \
                        'involve_region': row['involve_region'], \
                        'ba_image': './static/record_images/' + row['ba_image'] + '.jpg' if row['ba_image'] != '' else '', \
                        'ba_video': './static/record_videos/' + row['ba_video'] + '.mp4' if row['ba_video'] != '' else ''})
    return render_template("home.html", \
                            ba_results=ba_results, \
                            total_records=len(filter_df), \
                            select_channel = select_channel, \
                            select_ba_type = select_ba_type, \
                            select_start_time = select_start_time, \
                            select_end_time = select_end_time, \
                            live_stream_0=live_stream_0, \
                            live_stream_1=live_stream_1)

if __name__ == "__main__":
    # start receiving thread
    receiver = threading.Thread(target = ba_receiver)
    receiver.start()
    # run server
    app.run(host="192.168.77.68", port=7878)