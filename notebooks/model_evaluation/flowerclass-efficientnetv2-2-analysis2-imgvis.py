# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.030579, "end_time": "2022-03-23T14:34:09.159023", "exception": false, "start_time": "2022-03-23T14:34:09.128444", "status": "completed"} tags=[]
# # Analysis of Model flowerclass-efficientnetv2-2 2: with Image Visualizations
#
# ### Goals
#
# * Analysis of the top 8 worst performing classes
# * Leverage simple image visualizations to gain insight into algorithm
#

# %% papermill={"duration": 9.192275, "end_time": "2022-03-23T14:34:18.382832", "exception": false, "start_time": "2022-03-23T14:34:09.190557", "status": "completed"} tags=[]
import math, re, os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
print(tf.__version__)
print(tfa.__version__)

from flowerclass_read_tf_ds import get_datasets, display_batch_by_class, display_batch_of_images #, load_dataset, display_batch_of_images, batch_to_numpy_images_and_labels, display_one_flower
import tensorflow_hub as hub
import pandas as pd
import math
import plotly_express as px
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# %% papermill={"duration": 2.327232, "end_time": "2022-03-23T14:34:20.740782", "exception": false, "start_time": "2022-03-23T14:34:18.413550", "status": "completed"} tags=[]
tf.test.gpu_device_name()

# %% [markdown] _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.062845, "end_time": "2022-03-23T14:34:20.859477", "exception": false, "start_time": "2022-03-23T14:34:20.796632", "status": "completed"} tags=[]
# # I. Data Loading

# %% papermill={"duration": 0.064539, "end_time": "2022-03-23T14:34:20.986778", "exception": false, "start_time": "2022-03-23T14:34:20.922239", "status": "completed"} tags=[]
image_size = 224
batch_size = 64

# %% papermill={"duration": 0.071997, "end_time": "2022-03-23T14:34:21.109546", "exception": false, "start_time": "2022-03-23T14:34:21.037549", "status": "completed"} tags=[]
class_names = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
len(class_names)

# %% [markdown] papermill={"duration": 0.049729, "end_time": "2022-03-23T14:34:21.209799", "exception": false, "start_time": "2022-03-23T14:34:21.160070", "status": "completed"} tags=[]
# # II. Model Loading and Predictions: EfficientNetV2

# %% papermill={"duration": 0.043137, "end_time": "2022-03-23T14:34:21.303444", "exception": false, "start_time": "2022-03-23T14:34:21.260307", "status": "completed"} tags=[]
effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2"

    # %% papermill={"duration": 13.110289, "end_time": "2022-03-23T14:34:34.445188", "exception": false, "start_time": "2022-03-23T14:34:21.334899", "status": "completed"} tags=[]
    effnet2_tfhub = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=(image_size, image_size,3)),
    hub.KerasLayer(effnet2_base, trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(104, activation='softmax')
])
effnet2_tfhub.build((None, image_size, image_size,3,)) #This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.


effnet2_tfhub.summary()

# %% papermill={"duration": 2.489696, "end_time": "2022-03-23T14:34:36.966175", "exception": false, "start_time": "2022-03-23T14:34:34.476479", "status": "completed"} tags=[]
best_phase = 12
effnet2_tfhub.load_weights("../input/flowerclass-efficientnetv2-2/training/"+"cp-"+f"{best_phase}".rjust(4, '0')+".ckpt")

# %% papermill={"duration": 0.067387, "end_time": "2022-03-23T14:34:37.098965", "exception": false, "start_time": "2022-03-23T14:34:37.031578", "status": "completed"} tags=[]
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown] papermill={"duration": 0.051904, "end_time": "2022-03-23T14:34:37.203202", "exception": false, "start_time": "2022-03-23T14:34:37.151298", "status": "completed"} tags=[]
# Ensure that validation data loader returns fixed order of elements.

# %% papermill={"duration": 41.49374, "end_time": "2022-03-23T14:35:18.749124", "exception": false, "start_time": "2022-03-23T14:34:37.255384", "status": "completed"} tags=[]
ds_train, ds_valid, ds_test = get_datasets(BATCH_SIZE=batch_size, IMAGE_SIZE=(image_size, image_size),
                                           RESIZE=None, tpu=False, with_id=True)

img_preds = []
img_labels = []
img_ids = []
for imgs, label, imgs_id in tqdm(ds_valid):
    img_preds.append(effnet2_tfhub.predict(imgs, batch_size=batch_size))
    img_labels.append(label.numpy())
    img_ids.append(imgs_id.numpy())

img_preds = np.concatenate([img_pred.argmax(1) for img_pred in img_preds])
img_labels = np.concatenate([img_label.argmax(1) for img_label in img_labels])
img_ids = np.concatenate([img_id for img_id in img_ids])


# %% papermill={"duration": 0.071121, "end_time": "2022-03-23T14:35:18.872142", "exception": false, "start_time": "2022-03-23T14:35:18.801021", "status": "completed"} tags=[]
val_results = pd.DataFrame({'pred': img_preds, "label":img_labels, "id": img_ids})
val_results['id'] = val_results['id'].apply(lambda txt: txt.decode())

# %% papermill={"duration": 0.065199, "end_time": "2022-03-23T14:35:18.988644", "exception": false, "start_time": "2022-03-23T14:35:18.923445", "status": "completed"} tags=[]
val_results.head()

# %% [markdown] papermill={"duration": 0.051825, "end_time": "2022-03-23T14:35:19.091765", "exception": false, "start_time": "2022-03-23T14:35:19.039940", "status": "completed"} tags=[]
# # III. Analysis of low-performant classes

# %% papermill={"duration": 0.059149, "end_time": "2022-03-23T14:35:19.201755", "exception": false, "start_time": "2022-03-23T14:35:19.142606", "status": "completed"} tags=[]
worst_classes = pd.DataFrame({'class':['globe-flower', 'clematis', 'canterbury bells', 'mexican petunia',
                'black-eyed susan', 'peruvian lily']})

# %% papermill={"duration": 0.058289, "end_time": "2022-03-23T14:35:19.311164", "exception": false, "start_time": "2022-03-23T14:35:19.252875", "status": "completed"} tags=[]
class_names_mapping = {value:key for key, value in  enumerate(class_names)}

# %% papermill={"duration": 0.068794, "end_time": "2022-03-23T14:35:19.431771", "exception": false, "start_time": "2022-03-23T14:35:19.362977", "status": "completed"} tags=[]
worst_classes['idx'] = worst_classes['class'].map(class_names_mapping)
worst_classes

# %% papermill={"duration": 0.065148, "end_time": "2022-03-23T14:35:19.548383", "exception": false, "start_time": "2022-03-23T14:35:19.483235", "status": "completed"} tags=[]
conf_matrix = confusion_matrix(val_results['label'], val_results['pred'])

# %% papermill={"duration": 0.063496, "end_time": "2022-03-23T14:35:19.663455", "exception": false, "start_time": "2022-03-23T14:35:19.599959", "status": "completed"} tags=[]
val_results_classes = val_results[(val_results['pred'].isin(worst_classes['idx'])) | (val_results['label'].isin(worst_classes['idx']))]
val_results_classes.shape

# %% papermill={"duration": 0.063998, "end_time": "2022-03-23T14:35:19.779765", "exception": false, "start_time": "2022-03-23T14:35:19.715767", "status": "completed"} tags=[]
val_results_classes.head()

# %% [markdown] papermill={"duration": 0.052277, "end_time": "2022-03-23T14:35:19.885441", "exception": false, "start_time": "2022-03-23T14:35:19.833164", "status": "completed"} tags=[]
# # IIIa). globe-flower

# %% papermill={"duration": 0.058621, "end_time": "2022-03-23T14:35:19.996776", "exception": false, "start_time": "2022-03-23T14:35:19.938155", "status": "completed"} tags=[]
class_name = 'globe-flower'

# %% papermill={"duration": 0.065818, "end_time": "2022-03-23T14:35:20.115326", "exception": false, "start_time": "2022-03-23T14:35:20.049508", "status": "completed"} tags=[]
val_results_class = val_results[(val_results['pred'] == class_names_mapping[class_name]) | (val_results['label'] == class_names_mapping[class_name])].copy()

class_names_mapping_inv = {class_names_mapping[name]:name for name in class_names_mapping}
for el in ['pred', 'label']:
    val_results_class.loc[:, f"{el}_class"] = val_results_class[el].map(class_names_mapping_inv)

# %% papermill={"duration": 0.065145, "end_time": "2022-03-23T14:35:20.233287", "exception": false, "start_time": "2022-03-23T14:35:20.168142", "status": "completed"} tags=[]
val_results_class

# %% papermill={"duration": 6.408096, "end_time": "2022-03-23T14:35:26.694214", "exception": false, "start_time": "2022-03-23T14:35:20.286118", "status": "completed"} tags=[]
data_root = "../input/tpu-getting-started"

data_path = data_root + '/tfrecords-jpeg-224x224'
val_224 = tf.io.gfile.glob(data_path + '/val/*.tfrec')
train_224 = tf.io.gfile.glob(data_path + '/train/*.tfrec')

display_batch_by_class(val_224, name = class_name, top_n= 10)

# %% papermill={"duration": 0.160987, "end_time": "2022-03-23T14:35:27.015613", "exception": false, "start_time": "2022-03-23T14:35:26.854626", "status": "completed"} tags=[]
vis_imgs = val_results_class.loc[val_results_class.id.isin(['ed3a59a35', '4a6f8b3ad'])]
vis_imgs


# %% papermill={"duration": 0.137651, "end_time": "2022-03-23T14:35:27.294346", "exception": false, "start_time": "2022-03-23T14:35:27.156695", "status": "completed"} tags=[]
def get_images_by_ids(image_ids_search):
    ds_train, ds_valid, ds_test = get_datasets(BATCH_SIZE=batch_size, IMAGE_SIZE=(image_size, image_size),
                                               RESIZE=None, tpu=False, with_id=True)

    imgs_found = []
    imgage_ids_found = []
    labels_found = []
    for imgs, labels, imgs_id in tqdm(ds_valid):
        for img, img_id, label in zip(imgs, imgs_id, labels) :
            if img_id in image_ids_search:
                imgage_ids_found.append(img_id)
                imgs_found.append(img)
                labels_found.append(tf.argmax(label))

    return (tf.stack(imgs_found, 0), tf.cast(tf.concat(labels_found, 0), tf.int64)), imgage_ids_found


# %% papermill={"duration": 6.519297, "end_time": "2022-03-23T14:35:33.900248", "exception": false, "start_time": "2022-03-23T14:35:27.380951", "status": "completed"} tags=[]
batch_found,  imgage_ids_found= get_images_by_ids(vis_imgs['id'].values)

# %% papermill={"duration": 0.55734, "end_time": "2022-03-23T14:35:34.552471", "exception": false, "start_time": "2022-03-23T14:35:33.995131", "status": "completed"} tags=[]
display_batch_of_images(batch_found, predictions=vis_imgs['pred'].values, FIGSIZE=16, image_ids= vis_imgs['id'].values)

# %% [markdown] papermill={"duration": 0.104174, "end_time": "2022-03-23T14:35:34.761044", "exception": false, "start_time": "2022-03-23T14:35:34.656870", "status": "completed"} tags=[]
# > * ed3a59a35 image: Flower shot from the side, and flower seem not to have opened yet. No such type of image exists in the val set. But the training set?
# > * 4a6f8b3ad image: the flower seems close to the other globe-flower flowers, in terms of flower and stem leaves. Is buttercup very similar?

# %% papermill={"duration": 25.879739, "end_time": "2022-03-23T14:36:00.744975", "exception": false, "start_time": "2022-03-23T14:35:34.865236", "status": "completed"} tags=[]
display_batch_by_class(train_224, name = class_name, top_n= 10)

# %% [markdown] papermill={"duration": 0.200656, "end_time": "2022-03-23T14:36:01.148161", "exception": false, "start_time": "2022-03-23T14:36:00.947505", "status": "completed"} tags=[]
# > ed3a59a35 image: Training set does not include does not include such an image. On what is the network focusing on?

# %% papermill={"duration": 22.828404, "end_time": "2022-03-23T14:36:24.177674", "exception": false, "start_time": "2022-03-23T14:36:01.349270", "status": "completed"} tags=[]
display_batch_by_class(train_224, name = "lotus", top_n= 25)

# %% [markdown] papermill={"duration": 0.291861, "end_time": "2022-03-23T14:36:24.758841", "exception": false, "start_time": "2022-03-23T14:36:24.466980", "status": "completed"} tags=[]
# > Given the form of hte flower in ed3a59a35 image with some of the lotus flowers, it is reasonable to assume it belongs to the class

# %% papermill={"duration": 21.786826, "end_time": "2022-03-23T14:36:46.838183", "exception": false, "start_time": "2022-03-23T14:36:25.051357", "status": "completed"} tags=[]
display_batch_by_class(train_224, name = "buttercup", top_n= 25)

# %% [markdown] papermill={"duration": 0.385139, "end_time": "2022-03-23T14:36:47.601716", "exception": false, "start_time": "2022-03-23T14:36:47.216577", "status": "completed"} tags=[]
# > In its closed flower-closed form, buttercup flowers resemble the flower in image 4a6f8b3ad.

# %% papermill={"duration": 0.380278, "end_time": "2022-03-23T14:36:48.369054", "exception": false, "start_time": "2022-03-23T14:36:47.988776", "status": "completed"} tags=[]
