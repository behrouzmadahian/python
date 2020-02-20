"""
Best Practice:
Deferring weight creation until the shape of the inputs is known.
In many cases, you may not know in advance the size of your inputs, and you would like to lazily create weights when
that value becomes known, sometimes after instantiating the layer
"""
import tensorflow as tf
from tensorflow import keras
