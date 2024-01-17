# property and similiarity search #

Demonstrate property and similiarity search using VideoPipe.

1. `vehicle_encoding_classify_pipeline.cpp` run a pipeline to generate embedding data AND classification labels for vehicles and send them via socket (using message broker node in VideoPipe)
2. `property_encoding_receiver.py` receive embedding data AND properties via socket, and save them into csv file using `pandas` library (note, it over writes to disk every about 100 new records received).
3. `vehicle_search_server.py` load csv file and create Index Object of `faiss`. the script will start a web server powered by `flask` to demonstrate how to search images using embedding similiarity AND property. this script would also generate embedding for the query image uploaded by user.
4. `clean.sh` help to clean all cache data in current sample directory, you can regenerate embedding data AND classification labels again.

```
### how to use ###
step1. compile `vehicle_encoding_classify_pipeline.cpp` and run binary file, it will generate embedding data and classification labels. (change des_ip first)
step2. run `property_encoding_receiver.py` script to receive embedding data and properties. (change bind_ip first)
step3. run `vehicle_search_server.py` script to start a web server, then open url in your browser to search vehicles.
```

```
### data flow ###
1.[pipeline -> embedding data and classification labels -> send via socket] -> 2.[receive embedding data and labels -> serialize to disk] -> 3.[deserialize from disk -> create Index Object of faiss -> search images from web]
```

```
please run `sh clean.sh` if you want to regenerate embeddings and classification labels, for example restart sample from scratch.
```

# what is faiss #
A library for similiarity search created by Facebook, the fullname is `Facebook AI Similiarity Search`.

# how to install faiss #
```
pip3 install faiss-cpu 
# pip3 install faiss-gpu
```

# screenshot #
**select query image AND input property filter**

![](../../doc/p46.png)

**search results for vehicle**

![](../../doc/p47.png)