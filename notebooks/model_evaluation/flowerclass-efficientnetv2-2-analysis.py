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

# %% [markdown] papermill={"duration": 0.045341, "end_time": "2022-03-21T13:58:29.000943", "exception": false, "start_time": "2022-03-21T13:58:28.955602", "status": "completed"} tags=[]
# # Analysis of Model flowerclass-efficientnetv2-2
#

# %% papermill={"duration": 9.06998, "end_time": "2022-03-21T13:58:38.115967", "exception": false, "start_time": "2022-03-21T13:58:29.045987", "status": "completed"} tags=[]
import math, re, os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
print(tf.__version__)
print(tfa.__version__)

from flowerclass_read_tf_ds import get_datasets
import tensorflow_hub as hub
import pandas as pd
import math
import plotly_express as px
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# %% papermill={"duration": 2.174804, "end_time": "2022-03-21T13:58:40.334579", "exception": false, "start_time": "2022-03-21T13:58:38.159775", "status": "completed"} tags=[]
tf.test.gpu_device_name()

# %% [markdown] _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.043585, "end_time": "2022-03-21T13:58:40.422129", "exception": false, "start_time": "2022-03-21T13:58:40.378544", "status": "completed"} tags=[]
# # I. Data Loading

# %% papermill={"duration": 0.049473, "end_time": "2022-03-21T13:58:40.515679", "exception": false, "start_time": "2022-03-21T13:58:40.466206", "status": "completed"} tags=[]
image_size = 224
batch_size = 64

# %% papermill={"duration": 0.050989, "end_time": "2022-03-21T13:58:40.610388", "exception": false, "start_time": "2022-03-21T13:58:40.559399", "status": "completed"} tags=[]
# #%%debug (50, 480)


# %% papermill={"duration": 0.057086, "end_time": "2022-03-21T13:58:40.711186", "exception": false, "start_time": "2022-03-21T13:58:40.654100", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.043604, "end_time": "2022-03-21T13:58:40.798775", "exception": false, "start_time": "2022-03-21T13:58:40.755171", "status": "completed"} tags=[]
# # II. Model Loading: EfficientNetV2

# %% papermill={"duration": 0.049642, "end_time": "2022-03-21T13:58:40.892500", "exception": false, "start_time": "2022-03-21T13:58:40.842858", "status": "completed"} tags=[]
effnet2_base = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2"

    # %% papermill={"duration": 12.927122, "end_time": "2022-03-21T13:58:53.863536", "exception": false, "start_time": "2022-03-21T13:58:40.936414", "status": "completed"} tags=[]
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

# %% papermill={"duration": 1.95434, "end_time": "2022-03-21T13:58:55.867674", "exception": false, "start_time": "2022-03-21T13:58:53.913334", "status": "completed"} tags=[]
best_phase = 12
effnet2_tfhub.load_weights("../input/flowerclass-efficientnetv2-2/training/"+"cp-"+f"{best_phase}".rjust(4, '0')+".ckpt")

# %% [markdown] papermill={"duration": 0.044016, "end_time": "2022-03-21T13:58:55.956921", "exception": false, "start_time": "2022-03-21T13:58:55.912905", "status": "completed"} tags=[]
# # III. Model Analysis

# %% papermill={"duration": 0.051439, "end_time": "2022-03-21T13:58:56.052858", "exception": false, "start_time": "2022-03-21T13:58:56.001419", "status": "completed"} tags=[]
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown] papermill={"duration": 0.044776, "end_time": "2022-03-21T13:58:56.142398", "exception": false, "start_time": "2022-03-21T13:58:56.097622", "status": "completed"} tags=[]
# Ensure that validation data loader returns fixed order of elements.

# %% papermill={"duration": 25.387595, "end_time": "2022-03-21T13:59:21.574564", "exception": false, "start_time": "2022-03-21T13:58:56.186969", "status": "completed"} tags=[]
ds_train, ds_valid, ds_test = get_datasets(BATCH_SIZE=batch_size, IMAGE_SIZE=(image_size, image_size),
                                           RESIZE=None, tpu=False)

img_preds = []
img_labels = []
for imgs, label in tqdm(ds_valid):
    img_preds.append(effnet2_tfhub.predict(imgs, batch_size=batch_size))
    img_labels.append(label.numpy())

img_preds = np.concatenate([img_pred.argmax(1) for img_pred in img_preds])
img_labels = np.concatenate([img_label.argmax(1) for img_label in img_labels])


# %% papermill={"duration": 0.073315, "end_time": "2022-03-21T13:59:21.712094", "exception": false, "start_time": "2022-03-21T13:59:21.638779", "status": "completed"} tags=[]
val_results = pd.DataFrame({'pred': img_preds, "label":img_labels})

# %% papermill={"duration": 0.079708, "end_time": "2022-03-21T13:59:21.855862", "exception": false, "start_time": "2022-03-21T13:59:21.776154", "status": "completed"} tags=[]
val_results.head()

# %% [markdown] papermill={"duration": 0.063954, "end_time": "2022-03-21T13:59:21.984052", "exception": false, "start_time": "2022-03-21T13:59:21.920098", "status": "completed"} tags=[]
# # IIIa) Overall Evaluation

# %% papermill={"duration": 0.081601, "end_time": "2022-03-21T13:59:22.130101", "exception": false, "start_time": "2022-03-21T13:59:22.048500", "status": "completed"} tags=[]
confusion_matrix(val_results['label'], val_results['pred'])

# %% papermill={"duration": 0.090292, "end_time": "2022-03-21T13:59:22.285939", "exception": false, "start_time": "2022-03-21T13:59:22.195647", "status": "completed"} tags=[]
print(classification_report(val_results['label'], val_results['pred'], target_names=class_names))

# %% papermill={"duration": 0.108118, "end_time": "2022-03-21T13:59:22.459146", "exception": false, "start_time": "2022-03-21T13:59:22.351028", "status": "completed"} tags=[]
class_report = pd.DataFrame.from_dict(classification_report(val_results['label'], val_results['pred'], target_names=class_names, output_dict=True)).T

class_report['class'] = class_report.index
class_report= class_report.reset_index(drop=True)

class_report.head()

# %% [markdown] papermill={"duration": 0.07051, "end_time": "2022-03-21T13:59:22.596498", "exception": false, "start_time": "2022-03-21T13:59:22.525988", "status": "completed"} tags=[]
# MOst problematic classes with f1 below 90:
#
#
# > How would improving these classes raise the macro f1 score?

# %% papermill={"duration": 0.071745, "end_time": "2022-03-21T13:59:22.737158", "exception": false, "start_time": "2022-03-21T13:59:22.665413", "status": "completed"} tags=[]
class_report = class_report.loc[:103] # remove the summary statistics, e.g. accuracy

# %% papermill={"duration": 0.074572, "end_time": "2022-03-21T13:59:22.877256", "exception": false, "start_time": "2022-03-21T13:59:22.802684", "status": "completed"} tags=[]
class_report = class_report.sort_values("f1-score").reset_index(drop=True)

# %% papermill={"duration": 0.078934, "end_time": "2022-03-21T13:59:23.021806", "exception": false, "start_time": "2022-03-21T13:59:22.942872", "status": "completed"} tags=[]
class_report.head(9)

# %% [markdown] papermill={"duration": 0.066872, "end_time": "2022-03-21T13:59:23.155991", "exception": false, "start_time": "2022-03-21T13:59:23.089119", "status": "completed"} tags=[]
# > What is wrong with the rose class? bad performance despite many images

# %% [markdown] papermill={"duration": 0.067806, "end_time": "2022-03-21T13:59:23.290814", "exception": false, "start_time": "2022-03-21T13:59:23.223008", "status": "completed"} tags=[]
# > * If we would improve all 8 worst-performing classes to f1 score of 1, it would still only raise performance by 1%! See below.
# > *
#

# %% papermill={"duration": 0.077227, "end_time": "2022-03-21T13:59:23.434699", "exception": false, "start_time": "2022-03-21T13:59:23.357472", "status": "completed"} tags=[]
class_report_test = class_report.copy()
class_report_test.loc[:7, 'f1-score'] = 1
class_report_test['f1-score'].mean()

# %% papermill={"duration": 0.076769, "end_time": "2022-03-21T13:59:23.578916", "exception": false, "start_time": "2022-03-21T13:59:23.502147", "status": "completed"} tags=[]
class_report_test.loc[:20, 'f1-score'] = 1
class_report_test['f1-score'].mean()

# %% [markdown] papermill={"duration": 0.067107, "end_time": "2022-03-21T13:59:23.713082", "exception": false, "start_time": "2022-03-21T13:59:23.645975", "status": "completed"} tags=[]
# > * Improve first 20 classes would raise by another 1%.
# > * It might be better to improve the overall performance of the model then trying to improve individual classes
#
# > * Nevertheless continue with error analysis

# %% papermill={"duration": 0.080406, "end_time": "2022-03-21T13:59:23.860405", "exception": false, "start_time": "2022-03-21T13:59:23.779999", "status": "completed"} tags=[]
class_report.head()

# %% papermill={"duration": 0.365979, "end_time": "2022-03-21T13:59:24.293360", "exception": false, "start_time": "2022-03-21T13:59:23.927381", "status": "completed"} tags=[]
sns.displot(class_report['f1-score'], kde=False)

# %% papermill={"duration": 0.085229, "end_time": "2022-03-21T13:59:24.452111", "exception": false, "start_time": "2022-03-21T13:59:24.366882", "status": "completed"} tags=[]
class_report['f1-score'].describe().to_frame().T

# %% [markdown] papermill={"duration": 0.069222, "end_time": "2022-03-21T13:59:24.591051", "exception": false, "start_time": "2022-03-21T13:59:24.521829", "status": "completed"} tags=[]
# Group classes into a easy category (good performance) and bad performance.

# %% papermill={"duration": 0.076699, "end_time": "2022-03-21T13:59:24.736429", "exception": false, "start_time": "2022-03-21T13:59:24.659730", "status": "completed"} tags=[]
class_report['difficulty'] = 'hard'
class_report.loc[8:, 'difficulty'] = class_report.loc[8:, 'f1-score'].apply(lambda x: 'easy' if x>0.969 else 'medium')

# %% papermill={"duration": 0.096093, "end_time": "2022-03-21T13:59:24.901379", "exception": false, "start_time": "2022-03-21T13:59:24.805286", "status": "completed"} tags=[]
class_report.groupby("difficulty").agg(['mean', 'median'])

# %% [markdown] papermill={"duration": 0.070379, "end_time": "2022-03-21T13:59:25.040550", "exception": false, "start_time": "2022-03-21T13:59:24.970171", "status": "completed"} tags=[]
# Hypothesis test with nonparametric Mann-Whitney U test to compare the samples with label easy and hard above:

# %% papermill={"duration": 0.080003, "end_time": "2022-03-21T13:59:25.190475", "exception": false, "start_time": "2022-03-21T13:59:25.110472", "status": "completed"} tags=[]
import scipy
scipy.stats.mannwhitneyu(class_report.loc[class_report['difficulty'] == 'easy', 'support'],
                        class_report.loc[class_report['difficulty'] == 'hard', 'support'])


# %% [markdown] papermill={"duration": 0.076731, "end_time": "2022-03-21T13:59:25.337681", "exception": false, "start_time": "2022-03-21T13:59:25.260950", "status": "completed"} tags=[]
# > We cannot reject the null hypothesis that both samples, easy and hard, come from the same distribution. This means there is no evidence to reject the null hypothesis at the 5% level that the number of data points are a reason for the difference between the easy and and hard classes.

# %% [markdown] papermill={"duration": 0.120114, "end_time": "2022-03-21T13:59:25.591274", "exception": false, "start_time": "2022-03-21T13:59:25.471160", "status": "completed"} tags=[]
# ### Common Errors

# %% papermill={"duration": 0.147379, "end_time": "2022-03-21T13:59:25.852262", "exception": false, "start_time": "2022-03-21T13:59:25.704883", "status": "completed"} tags=[]
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %% papermill={"duration": 0.08377, "end_time": "2022-03-21T13:59:26.042347", "exception": false, "start_time": "2022-03-21T13:59:25.958577", "status": "completed"} tags=[]
conf_matrix = confusion_matrix(val_results['label'], val_results['pred'])

# %% papermill={"duration": 0.074673, "end_time": "2022-03-21T13:59:26.186736", "exception": false, "start_time": "2022-03-21T13:59:26.112063", "status": "completed"} tags=[]
#plot_confusion_matrix(confusion_matrix(val_results['label'], val_results['pred']), class_names)

# %% papermill={"duration": 0.07678, "end_time": "2022-03-21T13:59:26.332707", "exception": false, "start_time": "2022-03-21T13:59:26.255927", "status": "completed"} tags=[]
conf_matrix.shape

# %% [markdown] papermill={"duration": 0.071117, "end_time": "2022-03-21T13:59:26.473763", "exception": false, "start_time": "2022-03-21T13:59:26.402646", "status": "completed"} tags=[]
# ### Confusion (matrix) of top 7 worst performing classes

# %% papermill={"duration": 0.077392, "end_time": "2022-03-21T13:59:26.622143", "exception": false, "start_time": "2022-03-21T13:59:26.544751", "status": "completed"} tags=[]
class_names[:3]

# %% papermill={"duration": 0.078938, "end_time": "2022-03-21T13:59:26.771487", "exception": false, "start_time": "2022-03-21T13:59:26.692549", "status": "completed"} tags=[]
conf_matrix

# %% papermill={"duration": 0.077091, "end_time": "2022-03-21T13:59:26.919428", "exception": false, "start_time": "2022-03-21T13:59:26.842337", "status": "completed"} tags=[]
class_names_mapping = {value:key for key, value in  enumerate(class_names)}

# %% papermill={"duration": 0.077881, "end_time": "2022-03-21T13:59:27.068162", "exception": false, "start_time": "2022-03-21T13:59:26.990281", "status": "completed"} tags=[]
class_names_mapping
class_report['idx'] = class_report['class'].map(class_names_mapping)

# %% papermill={"duration": 0.084075, "end_time": "2022-03-21T13:59:27.223982", "exception": false, "start_time": "2022-03-21T13:59:27.139907", "status": "completed"} tags=[]
class_report.head(7)

# %% papermill={"duration": 0.078613, "end_time": "2022-03-21T13:59:27.374419", "exception": false, "start_time": "2022-03-21T13:59:27.295806", "status": "completed"} tags=[]
worst_classes_FN = conf_matrix[class_report.loc[:7, "idx"]]
worst_classes_FN.shape

# %% papermill={"duration": 0.078399, "end_time": "2022-03-21T13:59:27.523285", "exception": false, "start_time": "2022-03-21T13:59:27.444886", "status": "completed"} tags=[]
worst_classes_FN_sub = worst_classes_FN[:, worst_classes_FN.sum(0) > 0]
worst_classes_FN_sub.shape

# %% papermill={"duration": 0.148592, "end_time": "2022-03-21T13:59:27.743966", "exception": false, "start_time": "2022-03-21T13:59:27.595374", "status": "completed"} tags=[]
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_confusion_matrix(cm, xclasses, yclasses, title_prefix, figsize=(16,8)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=figsize)
    plt.title(title_prefix+f" top {len(yclasses)} classes by f1 score")
    ax = plt.gca()
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks_y = np.arange(len(yclasses))
    tick_marks_x = np.arange(len(xclasses))
    plt.xticks(tick_marks_x, xclasses, rotation=45)
    plt.yticks(tick_marks_y, yclasses)

    for (j,i),label in np.ndenumerate(cm):
        ax.text(i,j,label,ha='center',va='center')

    plt.tight_layout()
    if not title_prefix=='FP':
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    else:
        plt.xlabel('True label')
        plt.ylabel('Predicted label')


# %% papermill={"duration": 1.19054, "end_time": "2022-03-21T13:59:29.007098", "exception": false, "start_time": "2022-03-21T13:59:27.816558", "status": "completed"} tags=[]
plot_confusion_matrix(worst_classes_FN_sub, xclasses= np.array(class_names)[worst_classes_FN.sum(0) > 0],
                     yclasses=class_report.loc[:7, "class"], title_prefix='FN' )

# %% [markdown] papermill={"duration": 0.074068, "end_time": "2022-03-21T13:59:29.155165", "exception": false, "start_time": "2022-03-21T13:59:29.081097", "status": "completed"} tags=[]
# ### FN Results:
# * globe-flower (true): only 1 confused with buttercup
# * clematis: 2 confused with windflower and columbine
# * canterbury bells: no FN
# * mexican petunia: 1 confused with petunia, maybe label error?
# * black-eyed susan: 5 confused with sunflower.
# *  peruvian lily: 1 with lenten rose, one with rose. both are of type rose, by chance?
# * rose: 2 with sunflower, 3 with commun tulip, 1 confused with baberton daisy, daisy, 2 sunflower, 1 lotus: mix ups spread among classes
# * gazania: one tiger lily, 1 baberton daisy, 1 rose, 1 blanket flower.

# %% papermill={"duration": 0.085364, "end_time": "2022-03-21T13:59:29.314595", "exception": false, "start_time": "2022-03-21T13:59:29.229231", "status": "completed"} tags=[]
worst_classes_FP = conf_matrix[:, class_report.loc[:7, "idx"]]
print(worst_classes_FP.shape)

worst_classes_conf_FP = worst_classes_FP[worst_classes_FP.sum(1) > 0, :]
worst_classes_conf_FP.shape

# %% papermill={"duration": 1.172325, "end_time": "2022-03-21T13:59:30.561537", "exception": false, "start_time": "2022-03-21T13:59:29.389212", "status": "completed"} tags=[]
plot_confusion_matrix(worst_classes_conf_FP.T, xclasses= np.array(class_names)[worst_classes_FP.sum(1) > 0],
                     yclasses=class_report.loc[:7, "class"], title_prefix='FP' )

# %% [markdown] papermill={"duration": 0.075685, "end_time": "2022-03-21T13:59:30.714991", "exception": false, "start_time": "2022-03-21T13:59:30.639306", "status": "completed"} tags=[]
# ### FP Results
# * globe-flower: 1 confused with (true) lotus
# * dematis: no wrong detection in other classes
# * caterbury bells: confused with true balloon flower
# * mexican petunia: confusesd iwth 1 true petunia and 1 true desert rose
# * black-eyed susan: confused with 1 true daisy
# * peruvian lily: confused with 1 true tiger lily,
# * rose: confused with 1 true snapdragon, 1 true peruvian lily, 2  camation and other classes. algo thinks everything is a rose which could be due to the relatively larger amount of images for this class.
# * gazania: confused with 2 true marigold,
#
# > the differences between FN and FP confused classes for the top 8 indicates that the type of confusion of the algorithm might be of different nature.

# %% papermill={"duration": 0.07563, "end_time": "2022-03-21T13:59:30.866210", "exception": false, "start_time": "2022-03-21T13:59:30.790580", "status": "completed"} tags=[]
