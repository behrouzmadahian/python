"""
1. Use KerasLayer:
    If transfer learning is desired and change of final layer, make sure you get the model
    without head from tensorflow hub:
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = (224, 224)
print(IMAGE_SHAPE + (3,))
classifier = keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))])

2.
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

3. Does not give model.summary() but it can be treated as a later!
Used for serving purposes!!

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector_model = hub.load(module_handle).signatures['default']


"""
