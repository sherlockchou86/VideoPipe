import faiss
import cv2
import time
import sys
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template


'''
create Index Object of faiss
'''
# load csv from disk
embeddings_properties_df = pd.read_csv("embeddings_properties.csv", sep='|', index_col=0)
print(f"load csv snapshot, contains {len(embeddings_properties_df)} records!")
print(embeddings_properties_df)

# create Index Object
embeddings_list = [[e for e in str(embedding).split(',')] for embedding in embeddings_properties_df["embedding"].to_list()]
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
input_size = (256, 256)
net = cv2.dnn.readNetFromONNX(vehicle_embedding_model_path)
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
    vehicle_types = ["all", "bus", "business_car", "construction_truck", "large_truck", "sedan", "small_truck", "suv", "tanker", "van"]
    vehicle_colors = ["all", "black", "blue", "grey", "red", "white", "yellow", "other"]
    return render_template("home.html", total_records=search_index.ntotal, vehicle_types=vehicle_types, vehicle_colors=vehicle_colors)

# search action
@app.route("/search", methods=["POST"])
def search():
    # no exception catch
    query_image = request.files['query_image']
    topK = int(request.form['topK'])
    vehicle_type = str(request.form['vehicle_type'])
    vehicle_color = str(request.form['vehicle_color'])
    vehicle_plate = str(request.form['vehicle_plate'])
    
    # convert to cv2.mat and save to disk
    img_array = np.asarray(bytearray(query_image.read()), dtype=np.uint8)
    query_mat = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    query_image_path = "static/query_images/" + str(time.time()) + ".jpg"
    cv2.imwrite(query_image_path, query_mat)

    # generate embedding for query image and do search action
    query_embedding = generate_embedding(query_mat)
    D, I = search_index.search(query_embedding, topK)

    # print similiarity search info
    print("query embedding:")
    print(query_embedding)
    print("similiarity distance:")
    print(D)
    print("similiarity uid:")
    print(I)

    # need properties filter
    sql = "path != ''"
    if vehicle_color != "all" or vehicle_type != "all" or vehicle_plate != "":
        if vehicle_color != "all":
            sql += f" & color == '{vehicle_color}'"
        if vehicle_type != "all":
            sql += f" & type == '{vehicle_type}'"
        if vehicle_plate != "":
            sql += f" & plate.str.contains('{vehicle_plate}')"
    filter_df = embeddings_properties_df.query(sql)
    filter_uids = filter_df.index.to_list()

    # filter action
    paired_lists = list(zip(I[0], D[0]))
    filter_result = [elem for elem in paired_lists if elem[0] in filter_uids]
    shoot_I = [e[0] for e in filter_result]
    shoot_D = [e[1] for e in filter_result]

    # shoot data
    print(shoot_I)
    print(shoot_D)

    shoot_df = embeddings_properties_df.iloc[shoot_I]
    shoot_path = shoot_df['path']
    shoot_color = shoot_df['color']
    shoot_type = shoot_df['type']
    shoot_plate = shoot_df['plate']
    shoot_distance = shoot_D
    shoot_uid = shoot_I

    # render search results
    search_results = []
    for i,p,d,c,t,pl in zip(shoot_uid, shoot_path, shoot_distance, shoot_color, shoot_type, shoot_plate):
        search_results.append({"uid": i, "image_url": p, "distance":d, 'vehicle_color':c, 'vehicle_type':t, 'vehicle_plate':pl})
    return render_template("search_result.html", topK=topK, \
        search_results=search_results, query_image_url=query_image_path, \
            search_color=vehicle_color, search_type=vehicle_type, search_plate=vehicle_plate)


if __name__ == "__main__":
    app.run(host="192.168.77.68", port=9898)