# %% [markdown]
# ## Universal Style Transfer
# The models above are trained to work for a single style. Using these methods, in order to create a new style transfer model, you have to train the model with a wide variety of content images.
#
# Recent work by Yijun Li et al. shows that it is possible to create a model that generalizes to unseen style images, while maintaining the quality of output images.
#
# Their method works by treating style transfer as an image reconstruction task. They use the output of a VGG19 ReLU layer to encode features of various content images and traing a decoder to reconstruct these images. Then, with these two networks fixed, they feed the content and the style image into the encoder and use a whitening and coloring transform so that the covarience matrix of the features matches the covarience matrix of the style.
#
# This process can then be expanded to the remaining ReLU layers of VGG19 to create a style transfer pipeline that can apply to all spatial scales.
#
# Since only content images were used to train the encoder and decoder, additional training is not needed when generalizing this to new styles.
#
# <img src="images/universal-style-transfer.png" style="width: 600px;"/>
# (Yijun Li et al., Universal Style Transfer)
#
# <img src="images/doge_the_scream.jpg" style="width: 300px;"/>
# <img src="images/doge_mosaic.jpg" style="width: 300px;"/>
#
# The results are pretty impressive, but there are some patches of blurriness, most likely as a result of the transforms.
#
# ### Whitening Transform
#
# The whitening transform removes the style from the content image, keeping the global content structure.
#
# The features of the content image, $f_c$, are transformed to obtain $\hat{f}_c$, such that the feature maps
# are uncorrelated ($\hat{f}_c \hat{f}_c^T = I$),
#
# $$
#     \hat{f}_c = E_c D_c^{- \frac{1}{2}} E_c^T f_c
# $$
#
# where $D_c$ is a diagonal matrix with the eigenvalues of the covariance matrix $f_c f_c^T \in R^{C \times C}$,
# and $E_c$ is the corresponding orthogonal matrix of eigenvectors, satisfying $f_c f_c^T = E_c D_c E_c^T$.
#
# <img src="images/whitening.png" style="width: 300px;"/>
# (Yijun Li et al., Universal Style Transfer)
#
#
# ### Coloring Transform
#
# The coloring transform adds the style from the style image onto the content image.
#
# The whitening transformed features of the content image, $\hat{f}_c$, are transformed to obtain $\hat{f}_{cs}$, such that the feature maps have that desired correlations ($\hat{f}_{cs} \hat{f}_{cs}^T = f_s f_s^T$),
#
# $$
#     \hat{f}_{cs} = E_s D_s^{\frac{1}{2}} E_s^T \hat{f}_c
# $$
#
# where $D_s$ is a diagonal matrix with the eigenvalues of the covariance matrix $f_s f_s^T \in R^{C \times C}$,
# and $E_s$ is the corresponding orthogonal matrix of eigenvectors, satisfying $f_c f_c^T = E_c D_c E_c^T$.
#
# In practice, we also take a weighted sum of the colored and original activations such that:
#
# $$ f_{blend} = \alpha\hat{f}_{cs} + (1-\alpha)\hat{f}_c $$
#
# Before each transform step, the mean of the corresponding feature maps are subtracted, and the mean of the style features are added back to the final transformed features.

# %%
# workaround for multiple OpenMP on Mac
from PIL import Image
from skimage.transform import resize
import functools
import time
import PIL.Image
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import IPython.display as display
from pathlib import PurePath
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


# %%

def pass_through(a):
    return a


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, channels=3):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def squeeze_axis(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    return image


def add_axis(image):
    image = image[tf.newaxis, :]
    return image


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    # if title==None:
    #     title = str(image.shape)
    # else:
    #     title += ' '+str(image.shape)
    plt.title(title)

# %% [markdown]
# # Using a pre-trained AutoEncoder
# For this assignment, i will be using an auto encoder created with Yihao Wang, a PhD student in the UbiComp lab here at SMU. The original code used to created this encoder is available for SMU students.
#
# The model that was trained can be downloaded from:
# https://www.dropbox.com/sh/2djb2c0ohxtvy2t/AAAxA2dnoFBcHGqfP0zLx-Oua?dl=0

# %%


class VGG19AutoEncoder(tf.keras.Model):
    def __init__(self, files_path):
        super(VGG19AutoEncoder, self).__init__()
        # Load Full Model with every trained decoder

        # Get Each SubModel
        # Each model has an encoder, a decoder, and an extra output convolution
        # that converts the upsampled activations into output images

        # DO NOT load models four and five because they are not great auto encoders
        # and therefore will cause weird artifacts when used for style transfer

        ModelBlock3 = tf.keras.models.load_model(
            str(PurePath(files_path, 'Block3_Model')), compile=False)
        self.E3 = ModelBlock3.layers[0]  # VGG encoder
        self.D3 = ModelBlock3.layers[1]  # Trained decoder from VGG
        # Conv layer to get to three channels, RGB image
        self.O3 = ModelBlock3.layers[2]

        ModelBlock2 = tf.keras.models.load_model(
            str(PurePath(files_path, 'Block2_Model')), compile=False)
        self.E2 = ModelBlock2.layers[0]  # VGG encoder
        self.D2 = ModelBlock2.layers[1]  # Trained decoder from VGG
        # Conv layer to get to three channels, RGB image
        self.O2 = ModelBlock2.layers[2]

        # no special decoder for this one becasue VGG first layer has
        # no downsampling. So the decoder is just a convolution
        ModelBlock1 = tf.keras.models.load_model(
            str(PurePath(files_path, 'Block1_Model')), compile=False)
        self.E1 = ModelBlock1.layers[0]  # VGG encoder, one layer
        # Conv layer to get to three channels, RGB image
        self.O1 = ModelBlock1.layers[1]

    def show_reconstructions(self, image_paths):
        model_ids = [1, 2, 3]

        for image_path in image_paths:

            for model_id in model_ids:
                model_name = f'Block{model_id}_Model'
                encoder = getattr(self, f'E{model_id}')
                if model_id == 1:
                    def decoder(a): return a  # pass through
                else:
                    decoder = getattr(self, f'D{model_id}')
                final_layer = getattr(self, f'O{model_id}')

                print(image_path)
                plt.figure(figsize=(20, 20))
                plt.subplot(1, 2, 1)
                original_image = load_img(image_path)
                imshow(original_image, 'Original')
                plt.subplot(1, 2, 2)

                encoded = encoder(tf.constant(original_image))
                decoded = decoder(encoded)
                reconstructed = final_layer(decoded)
                imshow(reconstructed, f'Reconstructed, Model: {model_name}')

    def call_style_blend(self, content_image, style_a_image, style_b_image, alpha_a, alpha_b, alpha_content):

        model_ids = [3, 2, 1]
        x = content_image

        for model_id in model_ids:
            encoder = getattr(self, f'E{model_id}')
            if model_id == 1:
                decoder = pass_through  # pass through
            else:
                decoder = getattr(self, f'D{model_id}')
            final_layer = getattr(self, f'O{model_id}')

            activation_content = encoder(tf.constant(x))
            activation_style_a = encoder(tf.constant(style_a_image))
            activation_style_b = encoder(tf.constant(style_b_image))
            blended_activations = VGG19AutoEncoder.style_blend(activation_content,
                                                               activation_style_a,
                                                               activation_style_b,
                                                               alpha_content,
                                                               alpha_a,
                                                               alpha_b)
            blended_image = final_layer(decoder(blended_activations))
            blended_image = self.enhance_contrast(blended_image)
            x = blended_image

        blended_image = tf.clip_by_value(tf.squeeze(x), 0, 1)
        return blended_image

    def call_one_style(self, content_image, style_image,  alpha_style=0.8):
        model_ids = [3, 2, 1]
        x = content_image

        for model_id in model_ids:
            encoder = getattr(self, f'E{model_id}')
            if model_id == 1:
                decoder = pass_through  # pass through
            else:
                decoder = getattr(self, f'D{model_id}')
            final_layer = getattr(self, f'O{model_id}')

            activation_content = encoder(tf.constant(x))
            activation_style = encoder(tf.constant(style_image))
            colored_activations = VGG19AutoEncoder.wct_from_cov(activation_content,
                                                                activation_style,
                                                                alpha_style)
            colored_image = final_layer(decoder(colored_activations))
            colored_image = self.enhance_contrast(colored_image)
            x = colored_image

        colored_image = tf.clip_by_value(tf.squeeze(x), 0, 1)
        return colored_image

    def call(self, image, alphas=None, training=False):
        # Input should be dictionary with 'style' and 'content' keys
        # {'style':style_image, 'content':content_image}
        # value in each should be a 4D Tensor,: (batch, i,j, channel)

        style_image = image['style']
        content_image = image['content']

        output_dict = dict()
        # this will be the output, where each value is a styled
        # version of the image at layer 1, 2, and 3. So each key in the
        # dictionary corresponds to layer1, layer2, and layer3.
        # we also give back the reconstructed image from the auto encoder
        # so each value in the dict is a tuple (styled, reconstructed)

        x = content_image
        # choose covariance function
        # covariance is more stable, but signal will work for very small images
        wct = self.wct_from_cov

        if alphas == None:
            alphas = {'layer3': 0.6,
                      'layer2': 0.6,
                      'layer1': 0.6}

        # ------Layer 3----------
        # apply whiten/color on layer 3 from the original image
        # get activations
        a_c = self.E3(tf.constant(x))
        a_s = self.E3(tf.constant(style_image))
        # swap grammian of activations, blended with original
        x = wct(a_c.numpy(), a_s.numpy(), alpha=alphas['layer3'])
        # decode the new style
        x = self.O3(self.D3(x))
        x = self.enhance_contrast(x)
        # get reconstruction
        reconst3 = self.O3(self.D3(self.E3(tf.constant(content_image))))
        # save off the styled and reconstructed images for display
        blended3 = tf.clip_by_value(tf.squeeze(x), 0, 1)
        reconst3 = tf.clip_by_value(tf.squeeze(reconst3), 0, 1)
        output_dict['layer3'] = (blended3, reconst3)

        # ------Layer 2----------
        # apply whiten/color on layer 2 from the already blended image
        # get activations
        a_c = self.E2(tf.constant(x))
        a_s = self.E2(tf.constant(style_image))
        # swap grammian of activations, blended with original
        x = wct(a_c.numpy(), a_s.numpy(), alpha=alphas['layer2'])
        # decode the new style
        x = self.O2(self.D2(x))
        x = self.enhance_contrast(x, 1.3)
        # get reconstruction
        reconst2 = self.O2(self.D2(self.E2(tf.constant(content_image))))
        # save off the styled and reconstructed images for display
        blended2 = tf.clip_by_value(tf.squeeze(x), 0, 1)
        reconst2 = tf.clip_by_value(tf.squeeze(reconst2), 0, 1)
        output_dict['layer2'] = (blended2, reconst2)

        # ------Layer 1----------
        # apply whiten/color on layer 1 from the already blended image
        # get activations
        a_c = self.E1(tf.constant(x))
        a_s = self.E1(tf.constant(style_image))
        # swap grammian of activations, blended with original
        x = wct(a_c.numpy(), a_s.numpy(), alpha=alphas['layer1'])
        # decode the new style
        x = self.O1(x)
        x = self.enhance_contrast(x, 1.2)
        # get reconstruction
        reconst1 = self.O1(self.E1(tf.constant(content_image)))
        # save off the styled and reconstructed images for display
        blended1 = tf.clip_by_value(tf.squeeze(x), 0, 1)
        reconst1 = tf.clip_by_value(tf.squeeze(reconst1), 0, 1)
        output_dict['layer1'] = (blended1, reconst1)

        return output_dict

    @staticmethod
    def enhance_contrast(image, factor=1.25):
        return tf.image.adjust_contrast(image, factor)

    @staticmethod
    def decompose(activations):
        '''
        Get covariance matrix of encoded image
        Decompose covariance matrix into U, sigma_diag_values
        Flattened sigma makes some operations easier latter
        '''
        eps = 1e-5
        # 1xHxWxC -> CxHxW
        activations_t = np.transpose(np.squeeze(activations), (2, 0, 1))
        shape_C_H_W = activations_t.shape
        # CxHxW -> CxH*W
        activations_flat = activations_t.reshape(-1,
                                                 activations_t.shape[1]*activations_t.shape[2])
        channel_means = activations_flat.mean(axis=1, keepdims=True)
        # Zero mean
        activations_flat_zero_mean = activations_flat - channel_means
        covariance_mat = np.dot(activations_flat_zero_mean,
                                activations_flat_zero_mean.T) / \
            (activations_t.shape[1]*activations_t.shape[2] - 1)
        # SVD
        U, Sigma, _ = np.linalg.svd(covariance_mat)

        # discard small values
        greater_than_eps_idxs = (Sigma > eps).sum()
        sigma_diag_values = Sigma[:greater_than_eps_idxs]
        U = U[:, :greater_than_eps_idxs]

        return (
            U,
            sigma_diag_values,
            activations_flat_zero_mean,
            channel_means,
            shape_C_H_W
        )

    @staticmethod
    def style_blend(content, style_a, style_b, alpha_content, alpha_a, alpha_b):
        content = content.numpy()
        style_a = style_a.numpy()
        style_b = style_b.numpy()

        (content_u,
         content_sigma_diag_values,
         content_activations_flat_zero_mean,
         _, shape_C_H_W) = VGG19AutoEncoder.decompose(content)

        content_d = np.diag(1/np.sqrt(content_sigma_diag_values))
        content_whitened = (content_u @
                            content_d @
                            content_u.T) @ content_activations_flat_zero_mean

        (style_a_u,
         style_a_sigma_diag_values,
         style_a_activations_flat_zero_mean,
         style_a_channel_means, _) = VGG19AutoEncoder.decompose(style_a)

        style_a_d = np.diag(np.sqrt(style_a_sigma_diag_values))

        content_colored_a = (style_a_u @ style_a_d @
                             style_a_u.T) @ content_whitened
        # add style mean back to each channel
        content_colored_a = content_colored_a + style_a_channel_means
        # # CxH*W -> CxHxW
        content_colored_a = content_colored_a.reshape(shape_C_H_W)
        # # CxHxW -> 1xHxWxC
        content_colored_a = np.expand_dims(
            np.transpose(content_colored_a, (1, 2, 0)), 0)

        (style_b_u,
         style_b_sigma_diag_values,
         style_b_activations_flat_zero_mean,
         style_b_channel_means, _) = VGG19AutoEncoder.decompose(style_b)

        style_b_d = np.diag(np.sqrt(style_b_sigma_diag_values))

        content_colored_b = (style_b_u @ style_b_d @
                             style_b_u.T) @ content_whitened
        # add style mean back to each channel
        content_colored_b = content_colored_b + style_b_channel_means
        # # CxH*W -> CxHxW
        content_colored_b = content_colored_b.reshape(shape_C_H_W)
        # # CxHxW -> 1xHxWxC
        content_colored_b = np.expand_dims(
            np.transpose(content_colored_b, (1, 2, 0)), 0)

        blended = alpha_a*content_colored_a + \
            alpha_b * content_colored_b + \
            alpha_content * content

        return np.float32(blended)

    @staticmethod
    def wct_from_cov(content, style, alpha=0.6):
        '''
        https://github.com/eridgd/WCT-TF/blob/master/ops.py
           Perform Whiten-Color Transform on feature maps using numpy
           See p.4 of the Universal Style Transfer paper for equations:
           https://arxiv.org/pdf/1705.08086.pdf
        '''

        (content_u,
         content_sigma_diag_values,
         content_activations_flat_zero_mean,
         _, shape_C_H_W) = VGG19AutoEncoder.decompose(content)

        content_d = np.diag(1/np.sqrt(content_sigma_diag_values))
        content_whitened = (content_u @
                            content_d @
                            content_u.T) @ content_activations_flat_zero_mean

        (style_u,
         style_sigma_diag_values,
         _, style_channel_means, _) = VGG19AutoEncoder.decompose(style)

        style_d = np.diag(np.sqrt(style_sigma_diag_values))

        content_colored = (style_u @ style_d @ style_u.T) @ content_whitened
        # add style mean back to each channel
        content_colored = content_colored + style_channel_means
        # # CxH*W -> CxHxW
        content_colored = content_colored.reshape(shape_C_H_W)
        # # CxHxW -> 1xHxWxC
        content_colored = np.expand_dims(
            np.transpose(content_colored, (1, 2, 0)), 0)

        blended = alpha*content_colored + (1 - alpha)*(content)

        return np.float32(blended)


AE = VGG19AutoEncoder('models/vgg_decoder/')
IMAGES_TO_RECONSTRUCT = [
    'images/dallas_hall.jpg',
    'images/dog.jpg',
    # 'images/doge.jpg',
    # 'images/newton.jpg',
    # 'images/python.jpg',
    # 'images/anakin.png',
    'images/starry_style.png',
    # 'images/wave_style.png',
]


# %%
AE.show_reconstructions(IMAGES_TO_RECONSTRUCT)


# %% [markdown]
# ## Reconstructions using different decoders
#
# - The decoders Block1_Model and Block2_Model perform very well and the reconstructed images are hard to tell apart from the original.
# - The decoder Block3_Model shows slight decolorization and show some grid like artifacts.
# - The earlier layers have gone through less convolutions, their information is less 'distilled' and closer to the original image, thus the decoders have an easier time reconstructing the image.
# - The artifacts are probably the result of the way the decoder upsamples.
#     - 'Deconvolution' can cause uneven overlaps, to prevent this the kernel size should be a multiple of the stride.
#

# %%


# %%


content_path = 'images/dallas_hall.jpg'
style_path = 'images/Vincent_van_Gogh_Sunflowers.jpg'

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style')

tmp = {'style': style_image,
       'content': content_image}

alphas = {'layer3': 0.8, 'layer2': 0.6, 'layer1': 0.6}
decoded_images = AE(tmp, alphas=alphas)

# %% [markdown]

# ## Multi style blend

# %%
imshow(style_image, 'Style')
for layer in decoded_images.keys():
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    imshow(decoded_images[layer][0], 'Styled')
    plt.subplot(1, 2, 2)
    imshow(decoded_images[layer][1], 'Reconstructed')

# %%
content_path = 'images/dallas_hall.jpg'
style_a_path = 'images/Vincent_van_Gogh_Sunflowers.jpg'
style_b_path = 'images/mosaic_style.png'

content_image = load_img(content_path)
style_a_image = load_img(style_a_path)
style_b_image = load_img(style_b_path)

plt.subplot(1, 3, 1)
imshow(content_image, 'Content')

plt.subplot(1, 3, 2)
imshow(style_a_image, 'Style A')

plt.subplot(1, 3, 3)
imshow(style_b_image, 'Style B')


alphas = [
    {'alpha_a': 0.8, 'alpha_b': 0.0, 'alpha_content': 0.2},
    {'alpha_a': 0.6, 'alpha_b': 0.2, 'alpha_content': 0.2},
    {'alpha_a': 0.4, 'alpha_b': 0.4, 'alpha_content': 0.2},
    {'alpha_a': 0.2, 'alpha_b': 0.6, 'alpha_content': 0.2},
    {'alpha_a': 0.0, 'alpha_b': 0.8, 'alpha_content': 0.2},
]

plt.figure(figsize=(20, 20))

for i, alpha_set in enumerate(alphas):

    plt.subplot(1, 5, i+1)
    decoded_image = AE.call_style_blend(content_image,
                                        style_a_image,
                                        style_b_image,
                                        **alpha_set)

    imshow(decoded_image,
           f'A:{alpha_set["alpha_a"]}  B:{alpha_set["alpha_b"]}')

# %% [markdown]
# ## Masked blend

# %%


def to_pillow_image(image):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    image_np = image.numpy()*255
    image_np = image_np.astype(np.uint8)
    return Image.fromarray(image_np)


def masked_blend(content_path, mask_path, style_a_path, style_b_path):

    content_image = load_img(content_path)
    content_image_pil = to_pillow_image(content_image)

    img_style_a = AE.call_one_style(content_image, style_a_image)
    img_style_b = AE.call_one_style(content_image, style_b_image)
    img_style_a = to_pillow_image(img_style_a)
    img_style_b = to_pillow_image(img_style_b)
    mask_image_pil = Image.open(mask_path).convert(
        'L').resize(content_image_pil.size)
    im = Image.composite(
        img_style_a, img_style_b, mask_image_pil)

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 4, 1)
    imshow(style_a_image, 'Style A')
    plt.subplot(1, 4, 2)
    imshow(style_b_image, 'Style B')
    plt.subplot(1, 4, 3)
    imshow(content_image, 'Original')
    plt.subplot(1, 4, 4)
    plt.imshow(mask_image_pil, cmap='gray')
    plt.title('Mask')

    plt.figure(figsize=(20, 20))
    plt.imshow(im)
    plt.title('Masked Styles')


content_path = 'images/dallas_hall.jpg'
mask_path = 'images/dallas_hall_mask.jpg'
masked_blend(content_path, mask_path, style_a_path, style_b_path)
# %%
content_path = 'images/newton.jpg'
mask_path = 'images/newton_mask.jpg'
masked_blend(content_path, mask_path, style_a_path, style_b_path)

# %% [markdown]
# ## HSV color preservation
# %%

hsv_cylinder = Image.open('images/HSV.png')
plt.figure(figsize=(10, 10))
plt.imshow(hsv_cylinder)
plt.title('hsv cylinder')

# %%


def color_preserving_transfer(content_path, style_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    hsv_original = mpl.colors.rgb_to_hsv(squeeze_axis(content_image))
    plt.figure(figsize=(20, 20))

    plt.subplot(1, 6, 1)
    imshow(content_image, 'original')

    plt.subplot(1, 6, 2)
    plt.imshow(hsv_original[:, :, 0], cmap='hsv')
    plt.title('hue')

    plt.subplot(1, 6, 3)
    plt.imshow(hsv_original[:, :, 1])
    plt.title('saturation')

    plt.subplot(1, 6, 4)
    plt.imshow(hsv_original[:, :, 2])
    plt.title('value')

    rgb = mpl.colors.hsv_to_rgb(hsv_original)
    plt.subplot(1, 6, 5)
    plt.imshow(rgb)
    plt.title('hsv_to_rgb')

    plt.subplot(1, 6, 6)
    imshow(style_image)
    plt.title('style image')

    stylized_image = AE.call_one_style(content_image, style_image)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    imshow(stylized_image, 'stylized without color preservation')

    hsv_stylized = mpl.colors.rgb_to_hsv(squeeze_axis(stylized_image))
    hsv_color_restored = hsv_original
    hsv_color_restored[:, :, 2] = hsv_stylized[:, :, 2]
    rgb_color_restored = mpl.colors.hsv_to_rgb(hsv_original)
    plt.subplot(1, 3, 2)
    plt.imshow(rgb_color_restored)
    plt.title('stylized with color restored')

    plt.subplot(1, 3, 3)
    plt.imshow(hsv_stylized[:, :, 2])
    plt.title('value stylized')


color_preserving_transfer(content_path, style_path)
# %% [markdown]

# ### Some explanation on the hue channel
# - At first glance the hue channel looks wrong but it is in fact correct.
# - Yes, brown is in fact orange, see: https://www.youtube.com/watch?v=wh4aWZRtTwU
# - The sky is red because white can be anywhere on the hue channel (low saturation and high value makes white), same for the floor
# - The sky has red/green blocks probably due to jpeg compression artifacts, since both red and green can show up as white, these artifacts are not apparent in rgb space.

# %% [markdown]
# ## Resources
#
# ### Images:
# - Piet_Mondrian.jpg
# - https://www.artsy.net/artwork/piet-mondrian-composition-with-large-red-plane-yellow-black-grey-and-blue
#
# - Vincent_van_Gogh_Sunflowers.jpg
# - https://commons.wikimedia.org/wiki/File:Vincent_van_Gogh_-_Sunflowers_-_VGM_F458.jpg
#
# - Andy_Warhol_Marilyn.jpg
# - https://musartboutique.com/andy-warhol-the-pop-art-king/
#
# - HSV.png
# - https://en.wikipedia.org/wiki/HSL_and_HSV
# %%
