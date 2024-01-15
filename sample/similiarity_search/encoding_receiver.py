import socket
import re
import pandas as pd

'''
receive embedding data from socket and save to csv using pandas periodically.
'''
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
bind_ip = '192.168.77.68'
bind_port = 8888
udp_socket.bind((bind_ip, bind_port))

embedding_df = pd.DataFrame(columns=["path", "embedding"])
count = 0
while True:
    data, addr = udp_socket.recvfrom(1024 * 64)
    s = data.decode()
    print('############################################')
    print(f'Received data: \n{s}')
    print(f'From address: {addr}')

    li = re.findall("\<\-\-.*?\-\-\>", s, re.DOTALL)
    print(f"total {len(li)} embeddings parsed! going to append to data frame...")
    for l in li:
        items = str(l).split('\n')
        ''' 
        items[0] <--
        items[1] path
        items[2] embedding
        items[3] -->
        '''
        assert(len(items) == 4)
        # append path and embedding to dataframe
        embedding_df.loc[len(embedding_df)] = [items[1], items[2]]
        count += 1
    print(embedding_df)
    print(f"now total {len(embedding_df)} embeddings in data frame.")
    # save to file every 100 embeddings
    if count > 100:
        count = 0
        embedding_df.to_csv('embeddings.csv', sep='|', header=True, index=True)
        print("new snapshot generated, saved to csv!")
    print('############################################')

udp_socket.close()