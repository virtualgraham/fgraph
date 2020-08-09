from scipy import spatial
import hnswlib

a = [1,2,3,4,5,6,7,8,9,10]
b = [0,2,0,4,0,6,0,8,0,10]

index = hnswlib.Index(space='cosine', dim=10)
index.init_index(max_elements=2, ef_construction=100, M=16)
index.set_ef(10)  

index.add_items([a, b], [0, 1])

r = index.knn_query(a, 2) 

d = spatial.distance.cosine(a, b)

print(r, d)