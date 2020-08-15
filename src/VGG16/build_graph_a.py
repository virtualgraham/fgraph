stride = 16
window_size = 32
playback_random_walk_length = 10

video_path = "../../media/Tabletop Objects/videos/001_apple.mp4"
db_path = "../../data/test2_001_apple.db"
patch_dir = "../../patches"

from vgg16_window_walker_lib_a import build_graph

build_graph(db_path, video_path, patch_dir, stride=stride, window_size=window_size)

# from vgg16_window_walker_lib_b import build_sample_vectors
# build_sample_vectors("../../media/Tabletop Objects/videos", "../../data/samples.npy", 30, window_size)

# import faiss
# import numpy as np

# print("Starting")
# quantizer = faiss.IndexFlatL2( 512 )
# index = faiss.IndexIVFFlat( quantizer, 512, 100 )

# assert not index.is_trained
# print("Loading Training Data")
# samples = np.load( "../../data/samples.npy")
# print("Training")
# index.train(samples)
# assert index.is_trained

# import ngtpy
# ngtpy.create(path=b'test_index', dimension=512, distance_type="Cosine")
# index = ngtpy.Index(b'index')

# index.batch_insert(features)

# index.save()
# index.close(