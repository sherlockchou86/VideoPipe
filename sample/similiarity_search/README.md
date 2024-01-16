# similiarity search #

Demonstrate similiarity search using VideoPipe.

1. `vehicle_encoding_pipeline.cpp` run a pipeline to generate embedding data for vehicles and send them via socket (using message broker node in VideoPipe)
1. `face_encoding_pipeline.cpp` run a pipeline to generate embedding data for faces and send them via socket (using message broker node in VideoPipe)
2. `encoding_receiver.py` receive embedding data via socket, and save them into csv file using `pandas` library (note, it over writes to disk every about 100 new embeddings received).
3. `search_server.py` load csv file and create Index Object of `faiss`. the script will start a web server powered by `flask` to demonstrate how to search images using embedding similiarity. this script would also generate embedding for the query image uploaded by user.
4. `clean.sh` help to clean all cache data in current sample directory, you can regenerate embedding data again.

```
### how to use ###
step1. compile `vehicle_encoding_pipeline.cpp` OR `face_encoding_pipeline.cpp` and run a binary file (ONLY 1 pipeline one time), it will generate embedding data. (change des_ip first)
step2. run `encoding_receiver.py` script to receive embedding data. (change bind_ip first)
step3. run `search_server.py vehicle/face` script to start a web server, then open url in your browser to search vehicles or faces.
```

```
### data flow ###
1.[pipeline -> embedding data -> send via socket] -> 2.[receive embedding data -> serialize to disk] -> 3.[deserialize from disk -> create Index Object of faiss -> search images from web]
```

```
please run `sh clean.sh` if you want to regenerate embeddings, for example, switch vehicle similiarity search to face similiarity search.
```

# what is faiss #
A library for similiarity search created by Facebook, the fullname is `Facebook AI Similiarity Search`.

# how to install faiss #
```
pip3 install faiss-cpu 
# pip3 install faiss-gpu
```

# screenshot #
**select query image**

![](../../doc/p44.png)

**search results for vehicle**

![](../../doc/p43.png)

**search results for face**

![](../../doc/p45.png)