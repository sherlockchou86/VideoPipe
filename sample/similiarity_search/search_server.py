import faiss
import cv2
import time
import sys
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template

### search for vehicle or face
if len(sys.argv) != 2:
    print("search for vehicle or face?")
    print("'python3 search_server.py vehicle' OR 'python3 search_server.py face'")
    quit()
search_for_vehicle = True if sys.argv[1] == "vehicle" else False
###


'''
create Index Object of faiss
'''
# load csv from disk
embedding_df = pd.read_csv("embeddings.csv", sep='|', index_col=0)
print(f"load csv snapshot, contains {len(embedding_df)} embeddings!")
print(embedding_df)

# create Index Object
embeddings_list = [[e for e in str(embedding).split(',')] for embedding in embedding_df["embedding"].to_list()]
embeddings = np.array(embeddings_list)
print(f"load embeddings successfully, shape is {embeddings.shape}")
dimension = embeddings.shape[1]
# you can try different Index type
search_index = faiss.IndexFlatIP(dimension)
# add all embeddings into Index, use it in search action later
search_index.add(embeddings)


'''
generate embedding data for query image
'''
# vehicle embedding model keep as same as vehicle_encoding_pipeline.cpp
vehicle_embedding_model_path = "../../third_party/trt_vehicle/data/model/vehicle/dim256_batch=1_0916_norm.onnx"
# face embedding model keep as same as face_encoding_pipeline.cpp
face_embedding_model_path = "../models/face/face_recognition_sface_2021dec.onnx"

embedding_model_path =  vehicle_embedding_model_path if search_for_vehicle else face_embedding_model_path
input_size = (256, 256) if search_for_vehicle else (112, 112)
net = cv2.dnn.readNetFromONNX(embedding_model_path)
output_layer_names = net.getUnconnectedOutLayersNames()
def generate_embedding(query_image):
    input_image = cv2.resize(query_image, input_size)
    blob = cv2.dnn.blobFromImage(input_image)
    net.setInput(blob)
    out = net.forward(output_layer_names)
    embedding = np.array(out[0])
    return embedding


'''
create flask app
'''
app = Flask(__name__)

# home page
@app.route("/")
def home():
    return render_template("home.html", total_images = search_index.ntotal)

# search action
@app.route("/search", methods=["POST"])
def search():
    # no exception catch
    query_image = request.files['query_image']
    topK = int(request.form['topK'])
    
    # convert to cv2.mat and save to disk
    img_array = np.asarray(bytearray(query_image.read()), dtype=np.uint8)
    query_mat = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    query_image_path = "static/query_images/" + str(time.time()) + ".jpg"
    cv2.imwrite(query_image_path, query_mat)

    # generate embedding for query image and do search action
    query_embedding = generate_embedding(query_mat)
    D, I = search_index.search(query_embedding, topK)

    # print search info
    print("query embedding:")
    print(query_embedding)
    print("similiarity distance:")
    print(D)
    print("similiarity uid:")
    print(I)

    # shoot data
    shoot_df = embedding_df.iloc[I[0]]
    shoot_path = shoot_df['path']
    shoot_distance = D[0]
    shoot_uid = I[0]

    # render search results
    search_results = []
    for i,p,d in zip(shoot_uid, shoot_path, shoot_distance):
        search_results.append({"uid": i, "image_url": p, "distance":d})
    return render_template("search_result.html", topK=topK, search_results=search_results, query_image_url=query_image_path)


if __name__ == "__main__":
    app.run(host="192.168.77.68", port=9999)