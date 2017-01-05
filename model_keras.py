
import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers.core import Merge


def mul

model = Sequential()
model.add(Embedding(1000, 64, input_length=2))
mode.add(Merge(

input_array = np.random.randint
