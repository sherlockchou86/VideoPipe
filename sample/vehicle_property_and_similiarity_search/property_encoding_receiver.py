import socket
import re
import pandas as pd

'''
receive embedding data AND properties of vehicles from socket and save to csv using pandas periodically.
'''
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
bind_ip = '192.168.77.68'
bind_port = 8989
udp_socket.bind((bind_ip, bind_port))

embedding_properties_df = pd.DataFrame(columns=["path", "color", "type", "plate", "embedding"])
count = 0
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
        items[1] path
        items[2] properties (color, type)
        items[3] plate (empty is possible)
        items[4] embedding
        items[5] -->
        '''
        assert(len(items) == 6)
        color = items[2].split(',')[0]
        type = items[2].split(',')[1]
        # append path, color, type, plate, embedding to dataframe
        embedding_properties_df.loc[len(embedding_properties_df)] = [items[1], color, type, items[3], items[4]]
        count += 1
    print(embedding_properties_df)
    print(f"now total {len(embedding_properties_df)} records in data frame.")
    # save to file every 100 records
    if count > 100:
        count = 0
        embedding_properties_df.to_csv('embeddings_properties.csv', sep='|', header=True, index=True)
        print("new snapshot generated, saved to csv!")
    print('############################################')

udp_socket.close()