# %% [markdown]
# # Lab2: CNN Visualization
# by Lawrence (Jake) Klinkert and Hongjin (Tony) Yu

# %% [markdown]
# Submission Details: Turn in the rendered jupyter notebook (exported as HTML) to canvas. Only one notebook per team is required, but team names must be on the assignment.

# %%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from IPython.display import Image, display
from PIL import Image as PIL_Image
from keras import backend as K
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from pprint import pprint

# %% [markdown]
# In this lab you will find and analyze a circuit in a common neural network.  A reference figure is also shown to help clarify the process of finding and analyzing deep circuits. https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

# %%

PIL_Image.open("Data/CircuitsFigure.png")

# Source: From Canvas

# %% [markdown]
# ## 1. Model Selection and Task

# %% [markdown]
# [3 Points] In groups, you should select a convolutional neural network model that has been pre-trained on a large dataset (preferably, ImageNet). These already trained models are readily available online through many mechanisms, including the keras.application package (Inception, Xception, VGG etc.) https://keras.io/api/applications/Links to an external site. Simplicity in architecture is helpful (such as VGG or other network with relatively few layers and without complex feed forward operations in the overall flow).

# %% [markdown]
# Explain the model you chose and why.  Classify a few images with pre-trained network to verify that it is working properly.

# %% [markdown]
# We've choosen the VGG19 model, a pre-trianed convolutional neural network created by Simonyan and Zisserman from Visual Geometry Group (VGG) at University of Oxford in 2014. Trained on ImageNet ILSVRC data set of 1000 image classes. We choose this model because we wish to learn more about it's simplistic design, as well as it's reputation within the Machine Learning and Computer Vision compunity.
#


# %% [markdown]
# ### 1.1 Model Overview

# %%
PIL_Image.open("Data/VGG19.png")

# Source: https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0

# %%
model = VGG19(weights='imagenet', include_top=True)
model.summary()

# %% [markdown]
# ### 1.2 Test Images of Normal Classification

# %%
# Code based on Francois Chollet book, "Deep learning with python"


def get_img_array(img_path, target_size):
    img = keras.utils.load_img(
        img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


# %%
# Get Image from website
img_path = keras.utils.get_file(
    fname="monkey.jpg",
    origin="https://upload.wikimedia.org/wikipedia/commons/8/87/Chimpanzee-Head.jpg")

# Transform Image to tensor
img_tensor = get_img_array(img_path, target_size=(224, 224))

plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

x = preprocess_input(img_tensor)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# %%
# Get Image from website
img_path = keras.utils.get_file(
    fname="racecar.jpg",
    origin="https://media.wired.com/photos/5bb7b096ffac9b2ce1d57958/master/pass/Racecar.jpg")

# Transform Image to tensor
img_tensor = get_img_array(img_path, target_size=(224, 224))

plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

x = preprocess_input(img_tensor)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# %%
# Get Image from website
img_path = keras.utils.get_file(
    fname="coffee.jpg",
    origin="https://upload.wikimedia.org/wikipedia/commons/a/a9/Espresso_shot.jpg")

# Transform Image to tensor
img_tensor = get_img_array(img_path, target_size=(224, 224))

plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

x = preprocess_input(img_tensor)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])


# %% [markdown]
# ## Filter Selection and Explanation
# [4 Points] Select a multi-channel filter (i.e., a feature) in a layer in which to analyze as part of a circuit. This should be a multi-channel filter in a "mid-level" portion of the network (that is, there are a few convolutional layers before and after this chosen layer). You might find using OpenAI microscope a helpful tool for selecting a filter to analyze without writing too much code: https://microscope.openai.com/models/

# %% [markdown]
# We select the filter VGG 19, conv3_3, unit 195 (https://microscope.openai.com/models/vgg19_caffe/conv3_3_conv3_3_0/195)

# %% [markdown]
# Using image gradient techniques, find an input image that maximally excites this chosen multi-channel filter. General techniques are available from f. Chollet: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter09_part03_interpreting-what-convnets-learn.ipynbLinks to an external site.

# %%
# Code based on Francois Chollet's blog post (https://keras.io/examples/vision/visualizing_what_convnets_learn/)

img_width = 224
img_height = 224

# Based on OpenAI Miecroscope, the layer that is most interesting to us is conv3_3/conv3_3 unit 195
layer_name = "block3_conv3"

# Build a VGG19 model loaded with pre-trained ImageNet weights
# Need to set include_top to False, since we are no longer predicting images
model = VGG19(weights='imagenet', include_top=False)

# Set VGG19 model to return the activation values of our layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

# %%


def compute_loss(input_image, filter_index, feature_extractor):
    # get the activation values for our target layer
    activation = feature_extractor(input_image)

    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, feature_extractor):
    # Gradient tap bookkeeps the changes of an input. Given an image, we want to determine the
    # loss produced by the image and the specific filter from the network
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image(img_w, img_h):
    img = tf.random.uniform((1, img_w, img_h, 3))
    return (img - 0.5) * 0.25


def visualize_filter(filter_index, img_w, img_h, feature_extractor, iterations=30, learning_rate=10.0):
    # Given an image with random pixle values, can we iterate over the image and iterested filter
    # such that each iteration increases the pixle values of the image based on the activation of interesed filter
    img = initialize_image(img_w, img_h)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(
            img, filter_index, learning_rate, feature_extractor)

    # transfrom the final image into a visual picture
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


# %%
loss, img = visualize_filter(195, img_width, img_height, feature_extractor)
keras.preprocessing.image.save_img("195.png", img)
display(Image("195.png"))

# %% [markdown]
# Give a hypothesis for what this multi-channel filter might be extracting. That is, what do you think its function is in the network?

# %%
img_path = keras.utils.get_file(
    fname="unit_195.jpg",
    origin="https://i.ibb.co/Z6NgBGk/layer-195.png")

# Transform Image to tensor
img_tensor = get_img_array(img_path, target_size=(526, 1306))

plt.axis("off")
plt.figure(figsize=(20, 20))
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

# %% [markdown]
# According to OpenAI Microscope, VGG19 ath conv3_3, unit 195 these are the images activate this unit the most (https://microscope.openai.com/models/vgg19_caffe/conv3_3_conv3_3_0/195). The images look to be spherical and either have a glowing or bluring of the edges. This makes sense, because the filter has several circles with a bluring around each circle. Each circle also has a rainbow of diffrent colors, allowing for the circle itself to be diffrent colors. Other filters such as unit 8 and unit 11 focus on blue or red colors, respectively. This unit seems to allow any others, but must be a circular shape.

# %% [markdown]
# If using code from another source, you must heavily document the code so that I can grade your understanding of the code used.

# %% [markdown]
# [4 Points] Analyze each channel of the multi-channel filter to this feature that might form a circuit. That is, visualize the convolutional filter (one channel) between the input activations and the current activation to understand which inputs make up a circuit. One method of doing this is given below:

# %% [markdown]
# ## Extract Input Filters
# Extract the filter coefficients for each input activation to that multi-channel filter. Note: If the multi-channel filter is 5x5 with an input channel size of 64, then this extraction will result in 64 different input filters, each of size 5x5.


# get all filters that feed into block3_conv3 unit 195 (have to remember index / name so we know which they are)
def get_weights_and_bias_for_unit(layer_weights, unit_idx: int):
    # Note: the shape of the layer weights is [3 : 3 : input_channels : output_channels]
    return layer_weights[0][:, :, :, unit_idx], layer_weights[1][unit_idx]


layer_name = 'block3_conv3'
unit_idx = 195
layer = model.get_layer(name=layer_name)
layer_weights = layer.get_weights()

unit_weights, unit_bias = get_weights_and_bias_for_unit(
    layer_weights, unit_idx)

print(
    f'Layer:{layer_name}, unit:{unit_idx}, weights shape:{unit_weights.shape}, bias:{unit_bias}')


# %% [markdown]
# ## Top Six Filters
# Keep the top six sets of inputs with the "strongest" weights. For now, you can use the L2 norm of each input filter as a measure of strength. Visualize these top six filters.


# Compute L2 Norms
unit_weights_flattened = unit_weights.reshape(9, unit_weights.shape[2])
print(unit_weights_flattened.shape)
norms = np.linalg.norm(unit_weights_flattened, ord=2, axis=0)
print(f'shape of norms {norms.shape}')


fig, ax = plt.subplots()
fig.suptitle(
    f'Histogram of L2 Norms for Layer: {layer_name}, unit: {unit_idx}')
hist = ax.hist(norms)

# %% [markdown]

# - From the histogram, we can see that a large amount of filters have very "weak" weights.
# - Only a few filters have "strong" weights.

# %%

# Get "strongest" weights

TOP_N = 6


def get_indices_of_top_n(a, n):
    sorted_args = np.argsort(-a)
    return sorted_args[:n]


top_indices = get_indices_of_top_n(norms, TOP_N)
print(top_indices)
print(f'"Strongest" L2 norms {norms[top_indices]}')

top_filters = unit_weights[:, :, top_indices]
print(top_filters.shape)
print(f'"Strongest" 3x3 Filters:')
for i in range(top_filters.shape[2]):
    pprint(top_filters[:, :, i])


def plot_filters(filters):
    for i in range(filters.shape[2]):
        filter = filters[:, :, i]
        plt.imshow(filter,
                   norm=matplotlib.colors.Normalize(-0.12, 0.12),
                   cmap=plt.get_cmap('bwr'))
        plt.title(f'Filter for channel {top_indices[i]} of previous layer')
        plt.show()


plot_filters(top_filters)

# %% [markdown]
# For these six strongest input filters, categorize each as "mostly inhibitory" or "mostly excitatory." That is, does each filter consist of mostly negative or mostly positive coefficients?
# - Note: red is positive, and blue is negative.
# - From the above images, we can see that the filters for channels (of previous layer) 244, 113, 169, 219, 104 are "mostly excitatory", filter for channel 124 is "mostly inhibitory"

# %% [markdown]
# ## Top Six Filters Visualized
# [4 Points] For each of the six input filters that are strongest, use image gradient techniques to visualize what each of these filters is most excited by (that is, what image maximally excites each of these filters?).


prev_layer_name = "block3_conv2"

# Set VGG19 model to return the activation values of our layer
prev_layer = model.get_layer(name=prev_layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=prev_layer.output)

for idx, channel in enumerate(top_indices):
    iterations = 30
    learning_rate = 10
    # if channel == 124:
    #     iterations = 30
    # else:
    #     continue
    loss, img = visualize_filter(
        channel, img_width, img_height, feature_extractor, iterations, learning_rate)
    image_path = f"prev_{channel}.png"
    keras.preprocessing.image.save_img(image_path, img)
    with PIL_Image.open(image_path) as pil_img:
        plt.imshow(pil_img)
        plt.title(
            f'Image that "activates" channel {channel} of previous layer most')
        plt.show()

    filter = top_filters[:, :, idx]
    plt.imshow(filter,
               norm=matplotlib.colors.Normalize(-0.12, 0.12),
               cmap=plt.get_cmap('bwr'))
    plt.title(
        f'Filter for channel {channel} of previous layer')
    plt.show()


# %% [markdown]
# Use these visualizations, along with the circuit weights you just discovered to try and explain how this particular circuit works. An example of this visualization style can be seen here: https://storage.googleapis.com/distill-circuits/inceptionv1-weight-explorer/mixed3b_379.htmlLinks to an external site.


# %%
with PIL_Image.open('explanation.png') as pil_img:
    display(pil_img)

# %% [markdown]
# ## Hypothesis of how Circuit Works
# Note: The image that 'excites' channel 124 looks like random noise / did not converge and it does not match the image found at https://microscope.openai.com/models/vgg19_caffe/conv3_2_conv3_2_0/124 . We tried increasing the number of iterations and adjusting the learning rate, both which did not result in improvement. This issues was not found in other channels.
#
# How this circuit might work:
# From the histogram in the previous part, we know that the filters for the top three channels have much higher L2 norms than the other filters so we will focus on them.
# From observing the generated image that excites the channels the most, and from observing actual images that excite the channels we hypothesize that:
# - Channel 244 from previous layer is excited by small bright spots. It is highly excitatory to channel 195.
# - Channel 113 from previous layer is excited by medium sized circles of light, and also text to some degree (possibly due to the circles inside 'O' and '6' and '9' etc). It is highly excitatory to channel 195.
# - Channel 124 from previous layer is excited by text. It is highly inhibitory to channel 195. Therefore channel 195 will ignore the text somewhat excites channel 113.
# - The end result is that channel 195 is excited by medium sized circles with small highlights within, for example smooth spherical objects with reflected highlights, and glowing lights with highlight and bloom.


# %% [markdown]
# Try to define the properties of this circuit using vocabulary from https://distill.pub/2020/circuits/zoom-in/Links to an external site. (such as determining if this is polysemantic, pose-invariant, etc.)

# - This circuit does show properties of pose-invariance / rotational invariance as it is focused on circular shapes which have rotational symmetry.
# - This circuit does not show too much scale invariance. It is focused on small bright circles inside larger circles. It does not combine several filters that are excited by the same features of different scales.
# - The circuit is somewhat polysemantic as it seems to be excited by two visually similar but different phenomenon: reflected highlights on a spherical object and 'internal' highlights from a glowing object. 'Polysemantic Neurons' usually refer to channels that responds to multiple unrelated inputs, the two phenomenon are visually similar, thus it is not really polysemantic in the sense that a channel is being used for two completely unrelated functions, but it does seem to be a circuit that is general enough to be used in several different ways deeper down the network.
# - The circuit removes some of the polysemantic characteristics found in previous channels (ignores text from channel 113 that detects circular lights as well as text)

# %%
