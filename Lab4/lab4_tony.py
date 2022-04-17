# %% [markdown]
# Compare this implimentation to the one from the official Torch tutorial:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# %%
from pathlib import Path


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from IPython.display import HTML

# Seed control, for better reproducibility
# NOTE: this does not guarantee results are always the same
seed = 22
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# %%

EPOCHS = 1000
# might try to use large batches (we will discuss why later when we talk about BigGAN)
batch_size = 1250
# NOTE: the batch_size should be an integer divisor of the data set size  or torch
# will give you an error regarding batch sizes of "0" when the data loader tries to
# load in the final batch

# ims has been used, and only works when save file or load file has been call.
# Thus it is now a global variable. 
ims = None


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # work on a single GPU or CPU
    cudnn.benchmark = True
    # generator.cuda()
    # discriminator.cuda()
    # adversarial_loss.cuda()
    Tensor = torch.cuda.FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    cudnn.benchmark = False
    Tensor = torch.FloatTensor

print(device)
print(Tensor)


# %%
def imshow(img):
    # custom show in order to display
    # torch tensors as numpy
    npimg = img.numpy() / 2 + 0.5  # from tensor to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataset = dset.CIFAR10(root='data/cifar/', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))

# frogs are the sixth class in the dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
frog = 6
frog_index = [i for i, x in enumerate(dataset.targets) if x == 6]
print("number of frog imgs: ", len(frog_index))

frog_set = torch.utils.data.Subset(dataset, frog_index)

dataloader = torch.utils.data.DataLoader(frog_set, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


# %%
# get some random training images
dataiter = iter(dataloader)
real_image_examples, _ = dataiter.next()

# show images
plt.figure(figsize=(10, 10))
imshow(torchvision.utils.make_grid(
    real_image_examples, nrow=int(np.sqrt(batch_size))))
imshow(torchvision.utils.make_grid(
    real_image_examples, nrow=int(np.sqrt(batch_size))))
print("Image shape: ", real_image_examples[0].size())

# %%
# create a tensor dataset than can be moved entirely to the gpu in one go

dataloader = torch.utils.data.DataLoader(frog_set, batch_size=5000,
                                         shuffle=False, num_workers=0)
gpu_frogs = None
for i, (imgs, _) in enumerate(dataloader):
    print(i)
    print(imgs.device)
    gpu_frogs = imgs.to(device)

# print(gpu_frogs.shape)
print(gpu_frogs.shape)
frog_set_gpu = torch.utils.data.TensorDataset(gpu_frogs)

dataloader = torch.utils.data.DataLoader(frog_set_gpu, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

# %%
# create Utils functions for saving and loading


def save_checkpoint(new_img_list, loaded_ims, g_model, d_model, file_prefix):
    # save off a checkpoint of the current models and images

    # convert to numpy images for saving
    ims = np.array([np.transpose(np.hstack((i, real_image_numpy)), (2, 1, 0))
                   for i in new_img_list])

    # if we have saved images from another run, concatenate and save them here
    if len(loaded_ims) > 0:
        # concatenate these images with other runs (if needed)
        ims = np.concatenate((loaded_ims, ims))

    Path('models/gan_models').mkdir(parents=True, exist_ok=True)

    np.save(f'models/gan_models/{file_prefix}_images.npy', ims)

    # save the state of the models (will need to recreate upon reloading)
    torch.save({'state_dict': g_model.state_dict()},
               f'models/gan_models/{file_prefix}_gen.pth')
    torch.save({'state_dict': d_model.state_dict()},
               f'models/gan_models/{file_prefix}_dis.pth')


def load_checkpoint(file_prefix, gen_func, disc_func):
    # load up checkpoint images from previous runs
    ims = np.load(f'models/gan_models/{file_prefix}_images.npy')

    generator = gen_func()  # create generator (no weights)
    discriminator = disc_func()  # create disciminator (no weights)

    # now populate the weights from a previous training
    checkpoint = torch.load(f'models/gan_models/{file_prefix}_gen.pth')
    generator.load_state_dict(checkpoint['state_dict'])

    checkpoint = torch.load(f'models/gan_models/{file_prefix}_dis.pth')
    discriminator.load_state_dict(checkpoint['state_dict'])

    return ims, generator, discriminator

# %%
# create Utils functions to graph Generator and Discriminator loss
# function based on Quentin Garrido work (https://github.com/garridoq/gan-guide/blob/master/Virtual%20batch%20normalization.ipynb)
# 
def plot_losses(d_real_losses, d_fake_losses, g_losses, a_real_hist = None, a_fake_hist = None, file_prefix = None):
    # plot loss
	#plt.subplot(2, 1, 1)
    plt.plot(d_real_losses, label='d-real')
    plt.plot(d_fake_losses, label='d-fake')
    plt.plot(g_losses, label='gen')
    plt.legend()
    plt.show()
    plt.savefig(f'models/gan_models/{file_prefix}_images.png')

	# plot discriminator accuracy
	#plt.subplot(2, 1, 2)
	#plt.plot(a_real_hist, label='acc-real')
	#plt.plot(a_fake_hist, label='acc-fake')
	#plt.legend()
	# save plot to file
	#plt.close()
    


# %% [markdown]
# # Vanilla Generative Adversarial Networks
# In this implementation of GANS, we will use a few of the tricks from F. Chollet and from Salimans et al. In particular, we will add some noise to the labels.

# %%
latent_dim = 32
height = 32
width = 32
channels = 3

# Note: according to Radford (2016), is there anything done here
# that potentially could have been different?
# Anything wrong here based on Radford paper???
# NOTE: Dr. Larson Fixed most errors here for understanding


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # save these two functions

        # First, transform the input into a 8x8 128-channels feature map
        self.init_size = width // 4  # one quarter the image size
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2))
        # there is no reshape layer, this will be done in forward function
        # alternately we could us only the functional API
        # and bypass sequential altogether

        # we will use the sequential API
        # in order to create some blocks
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),  # 16x16
            nn.Conv2d(128, 128, 3, padding=1),  # 16x16

            # Then, add a convolution layer
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Upsample to 32x32
            # Transpose is not causing problems, but is slowing down because stride default 1
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),  # 32x32
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=1),
            #nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Produce a 32x32xRGB-channel feature map
            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # call the functions from earlier:

        # expand the sampled z to 8x8
        out = self.l1(z)
        out = torch.reshape(
            out, (out.shape[0], 128, self.init_size, self.init_size))
        # use the view function to reshape the layer output
        #  old way for earlier Torch versions: out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# %%


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # dropout layer - important, or just slowing down the optimization?
            # nn.Dropout2d(0.25),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.3, inplace=True),
            # nn.Dropout2d(0.25),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            nn.BatchNorm2d(128),
        )

        # The height and width of downsampled image
        ds_size = width // 2 ** 4
        # Classification layer
        self.classification_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1),
                                                  nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        # use the view function to flatten the layer output
        #    old way for earlier Torch versions: out = out.view(out.shape[0], -1)
        out = torch.flatten(out, start_dim=1)  # don't flatten over batch size
        validity = self.classification_layer(out)
        return validity

# %%
# custom weights initialization called on netG and netD
# this function from PyTorch's officail DCGAN example:
# https://github.com/pytorch/examples/blob/master/dcgan/main.py#L112


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # filters are zero mean, small STDev
    elif classname.find('BatchNorm') != -1:
        # batch norm is unit mean, small STDev
        m.weight.data.normal_(1.0, 0.02)
        # gamma starts around 1
        m.bias.data.fill_(0)  # like normal, biases start at zero
        # beta starts around zero
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# %%


def norm_grid(im):
    # first half should be normalized and second half also, separately
    im = im.astype(float)
    rows, cols, chan = im.shape
    cols_over2 = int(cols/2)
    tmp = im[:, :cols_over2, :]
    im[:, :cols_over2, :] = (tmp-tmp.min())/(tmp.max()-tmp.min())
    tmp = im[:, cols_over2:, :]
    im[:, cols_over2:, :] = (tmp-tmp.min())/(tmp.max()-tmp.min())
    return im


# %% [markdown]
# We are getting something that is similar to a frog, but also we are seeing a bit of mode collapse. The global properties of a greenish or gray blob surrounded by various background is starting to comes across. However, the finer structure is not doing too well. That is, the legs and details in the background are not present yet.
#
# To improve this result, there are a number of things we might try such as:
# - Using the Radford guided methods (implemented already in 2022)
# - Adding more randomization to the optimizer
# - Running the discriminator multiple times for each generator update
# - Changing the objective function (let's try this one)
#
# ____
# # Least Squares GAN
# Actually, the only thing we need to do here is replace the adversarial loss function. Note that we are NOT going to make additions to the architecture where the one hot encoding of the classes (and random classes) are used in both the generator and discriminator. This means that we might see a bit more mode collapse in our implementation.
# %%
generator = Generator()
discriminator = Discriminator()


# LSGAN paper says they use ADAM, but follow up papers say RMSProp is slightly better
#lr = 0.0002
#betas = (0.5, 0.999)

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
clip_value = 1.0  # This value will use in the future training process since
# PyTorch didn't has the feature to set clipvalue for
# RMSprop optimizer.

# set discriminator learning higher than generator
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(),
                                              lr=0.0008, weight_decay=1e-8)

gan_optimizer = torch.optim.RMSprop(generator.parameters(),
                                    lr=0.0004, weight_decay=1e-8)

# used to be: adversarial_loss = torch.nn.BCELoss() # binary cross entropy
adversarial_loss = torch.nn.MSELoss()  # mean squared error loss

generator.apply(weights_init)
discriminator.apply(weights_init)

generator.to(device)
discriminator.to(device)

# %%
iterations = EPOCHS  # defined above

# Sample random points in the latent space
plot_num_examples = 25
fixed_random_latent_vectors = torch.randn(
    plot_num_examples, latent_dim, device=device)
img_list = []
total_steps = 0

real_image_numpy = np.transpose(torchvision.utils.make_grid(
    real_image_examples[:plot_num_examples, :, :, :], padding=2, normalize=False, nrow=5), (0, 1, 2))


# %%

# code is the exact same as above, no need to change it
# because we have changed the adversarial loss function
# Start training loop

# saving generator and discriminator losses for each epoch
g_losses = []
d_real_losses = []
d_fake_losses = []
a_real_losses = []
a_fake_losses = []

# Because not much is changing, an interesting update would
# be to write a "train step" function and use it here.
# Something like: train_step(g, d, imgs, loss_select=MSE, num_d_steps=1)

run_from_checkpoint = False
if not run_from_checkpoint:
    loaded_ims = []
else:
    loaded_ims, generator, discriminator = load_checkpoint(f'ls_{EPOCHS}e_{batch_size}b',
                                                           Generator,
                                                           Discriminator)
    # can get previous steps based on saved checkpoints
    total_steps = loaded_ims.shape[0]*10

# For each EPOCH
for step in range(iterations):
    total_steps = total_steps+1
    generator.train()
    discriminator.train()

    running_g_loss = 0.0
    running_d_real_loss = 0.0
    running_d_fake_loss = 0.0
    correct = 0
    total = 0

    # For each batch size
    for i, imgs in enumerate(dataloader):

        imgs = imgs[0]
        # ===================================
        # GENERATOR OPTIMIZE AND GET LABELS

        # Zero out any previous calculated gradients
        gan_optimizer.zero_grad()

        # Sample random points in the latent space
        random_latent_vectors = Variable(
            Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Decode them to fake images, through the generator
        generated_images = generator(random_latent_vectors)

        # Assemble labels that say "all real images"
        # misleading target, c=1
        misleading_targets = Variable(
            Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

        # Get MSE Loss function
        # want generator output to generate images that are "close" to all "ones"
        g_loss = adversarial_loss(discriminator(
            generated_images), misleading_targets)
        
        # Saving g_loss to graph later
        running_g_loss += g_loss.item()

        # TODO: Determine the "accuracy" of the generator

        # now back propagate to get derivatives
        g_loss.backward()

        # use gan optimizer to only update the parameters of the generator
        # this was setup above to only use the params of generator
        gan_optimizer.step()

        # ===================================
        # DISCRIMINATOR OPTIMIZE AND GET LABELS

        # Zero out any previous calculated gradients
        discriminator_optimizer.zero_grad()
        

        # Combine real images with some generator images
        real_images = Variable(imgs.type(Tensor))
        combined_images = torch.cat([real_images, generated_images.detach()])
        # in the above line, we "detach" the generated images from the generator
        # this is to ensure that no needless gradients are calculated
        # those parameters wouldn't be updated (becasue we already defined the optimized parameters)
        # but they would be calculated here, which wastes time.

        # Assemble labels discriminating real from fake images
        # real label, a=1 and fake label, b=0
        labels = torch.cat((
            Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False),
            Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        ))
        # Add random noise to the labels - important trick!
        labels += 0.05 * torch.rand(labels.shape)

        # Setup Discriminator loss
        # this takes the average of MSE(real images labeled as real) + MSE(fake images labeled as fake)
        d_real_loss = adversarial_loss(discriminator(combined_images[:batch_size]), labels[:batch_size])
        d_fake_loss = adversarial_loss(discriminator(combined_images[batch_size:]), labels[batch_size:])
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # Saving the loss from both real and fake images to graph later
        running_d_real_loss += d_real_loss.item()
        running_d_fake_loss += d_fake_loss.item()

        # TODO: Determine the "accuracy" of the discriminator

        # get gradients according to loss above
        d_loss.backward()
        # optimize the discriminator parameters to better classify images
        discriminator_optimizer.step()

        # Now Clip weights of discriminator (manually)
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # ===================================

    # Calculate this epoch's loss and accuracy
    train_g_loss = running_g_loss/len(dataloader)
    train_d_real_loss = running_d_real_loss/len(dataloader)
    train_d_fake_loss = running_d_fake_loss/len(dataloader)
    train_d = (train_d_real_loss + train_d_fake_loss) / 2.0

    g_losses.append(train_g_loss)
    d_real_losses.append(train_d_real_loss)
    d_fake_losses.append(train_d_fake_loss)
    #accuracy = correct/total

    # Occasionally save / plot
    if step % 10 == 0:
        generator.eval()
        discriminator.eval()

        # Print metrics
        # TODO: d_loss and g_loss are the loss for the very last image inthe batch, not the epoch
        print('Loss at step %s: D(z_c)=%s, D(G(z_mis))=%s' %
              (total_steps, train_d, train_g_loss))
        # save images in a list for display later
        with torch.no_grad():
            fake_output = generator(fixed_random_latent_vectors).detach().cpu()
        img_list.append(torchvision.utils.make_grid(
            fake_output, padding=2, normalize=True, nrow=5))

        save_checkpoint(img_list, loaded_ims, generator, discriminator, f'ls_{EPOCHS}e_{batch_size}b')

# %%
# According to Dr. Jason Browniee (https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)
#   "Discriminator loss on real and fake images is expected to sit around 0.5"
#   "Generator loss on fake images is expected to sit between 0.5 and perhaps 2.0"
#   "Discriminator accuracy on real and fake images is expected to sit around 80%"
#   "Variance of generator and discriminator loss is expected to remain modest."
#   "The generator is expected to produce its highest quality images during a period of stability."
#   "Training stability may degenerate into periods of high-variance loss and corresponding lower quality generated images."
#
# Dr. Jason Browniee also suggests that mode collapse occurse when:
#   "The loss for the generator, and probably the discriminator, is expected to oscillate over time."
#   "The generator model is expected to generate identical output images from different points in the latent space."
#
# Finaly, Dr. Jason Browniee notes the properties of a convergence failer as:
#   "The loss for the discriminator is expected to rapidly decrease to a value close to zero where it remains during training."
#   The loss for the generator is expected to either decrease to zero or continually decrease during training."
#   The generator is expected to produce extremely low-quality images that are easily identified as fake by the discriminator.""
plot_losses(d_real_losses, d_fake_losses, g_losses, f'ls_{EPOCHS}e_{batch_size}b')

# %%
save_checkpoint(img_list, loaded_ims, generator, discriminator, f'ls_{EPOCHS}e_{batch_size}b')

# %%
# Load up a run, if you want
#ims, generator, discriminator = load_checkpoint('ls', Generator, Discriminator)

# %%
fig = plt.figure(figsize=(12, 4))
plt.axis("off")
pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
ani = animation.ArtistAnimation(
    fig, pls, interval=500, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())

# %% [markdown]
# Well, these results are not exactly a great improvement. Mode collapse is more apparent here as well, but the fine structure of the frogs is also not quite the improvement that we wanted. Looking back through the iterations, there was some indication of more successful generations. Subjectively, the frogs started to show up, but then generation became slightly worse. We could run this code for many more iterations, and that might work in terms of getting the optimizers to create better distributions. But it is not guaranteed.
#
# Instead, now let's try using a Wasserstein GAN, where we use the gradient penalty as a method of making the discriminator 1-lipschitz (and therefore a valid critic to approximate the earth mover distance).
#
# ___
# # Wasserstein GAN with Gradient Penalty
# For this implementation, we need to add functionality to the gradient of the Discriminator to make it a critic. For the most part, we need to add the gradient loss function calculations to match the WGAN-GP.

# %%


class WGCritic(nn.Module):

    def __init__(self):
        super(WGCritic, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            # nn.LayerNorm(32),
            nn.GroupNorm(1, 32),  # group==1 is same as 2d layer norm

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            # nn.LayerNorm(64),
            nn.GroupNorm(1, 64),  # group==1 is same as 2d layer norm

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            # nn.LayerNorm(128),
            nn.GroupNorm(1, 128),  # group==1 is same as 2d layer norm
        )

        # The height and width of downsampled image
        ds_size = width // 2 ** 4
        # Classification layer (just linear for the WGAN, critic)
        self.classification_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        # use the view function to flatten the layer output
        #    old way for earlier Torch versions: out = out.view(out.shape[0], -1)
        out = torch.flatten(out, start_dim=1)  # don't flatten over batch size
        validity = self.classification_layer(out)
        return validity


# %%
# Initialize generator and discriminator
generator = Generator()  # same generator, with new discriminator
discriminator = WGCritic()

# params from WGAN-GP paper
# learning rate
lr = 0.0001
beta1 = 0
beta2 = 0.9
# number of training steps for discriminator per iter for WGANGP
n_critic = 5
# Loss weight for gradient penalty
lambda_gp = 10

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
clip_value = 1

# Optimizers, no loss function defined here as
# will use torch.mean as loss function for WGAN.


# discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
# gan_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)

# Use ADAM
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                           lr=lr, betas=(beta1, beta2))
gan_optimizer = torch.optim.Adam(generator.parameters(),
                                 lr=lr, betas=(beta1, beta2))


# History: This worked okay with RMSProp and Batch norm/dropout in the critic
# . Attempt 1 to improve: took out batch norm, dropout, and started using Adam (bad results)
# . Attempt 2 to improve: took out batch norm, dropout, and started using RMSProp (working from previous gen/critic, awful results)
# . Attempt 3: Mirrored more from the WGAN-GP paper (LayerNorm and Adam, w/ beta1=0)

# %% [markdown]
# This compute_gradient_penalty function for WGAN-GP comes from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py#L119.

# %%
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    # use norm approx equal to one, as stated in paper. Rather than <1.
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# %%
iterations = EPOCHS  # defined above

# Sample random points in the latent space
plot_num_examples = 25
fixed_random_latent_vectors = torch.randn(
    plot_num_examples, latent_dim, device=device)
img_list = []
total_steps = 0

real_image_numpy = np.transpose(torchvision.utils.make_grid(
    real_image_examples[:plot_num_examples, :, :, :], padding=2, normalize=False, nrow=5), (0, 1, 2))


# %%

# saving generator and discriminator losses for each epoch
g_losses = []
d_real_losses = []
d_fake_losses = []
a_real_losses = []
a_fake_losses = []


#   we can continue a longer training run.
run_from_checkpoint = False
if not run_from_checkpoint:
    loaded_ims = []
    total_steps = 0
else:
    loaded_ims, generator, discriminator = load_checkpoint(f'wgan_{EPOCHS}e_{batch_size}b',
                                                           Generator,
                                                           WGCritic)
    # can get previous steps based on saved checkpoints
    total_steps = loaded_ims.shape[0]*10
    # Use ADAM
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                               lr=lr, betas=(beta1, beta2))
    gan_optimizer = torch.optim.Adam(generator.parameters(),
                                     lr=lr, betas=(beta1, beta2))

for step in range(iterations):
    total_steps = total_steps+1
    generator.train()
    discriminator.train()

    running_g_loss = 0.0
    running_d_real_loss = 0.0
    running_d_fake_loss = 0.0
    correct = 0
    total = 0
    
    for i, imgs in enumerate(dataloader):

        imgs = imgs[0]
        # ===================================
        # DISCRIMINATOR OPTIMIZE AND GET LABELS

        # Zero out any previous calculated gradients
        discriminator_optimizer.zero_grad()

        # Combine real images with some generator images
        real_images = Variable(imgs.type(Tensor))

        # Sample random points in the latent space
        random_latent_vectors = Variable(
            Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Decode them to fake images
        generated_images = generator(random_latent_vectors)

        # Compute gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_images.data, generated_images.data)

        # minimize this,
        d_real_loss = torch.mean(discriminator(real_images))
        d_fake_loss = torch.mean(discriminator(generated_images))
        d_loss = -d_real_loss + d_fake_loss + lambda_gp * gradient_penalty

        # Saving the loss from both real and fake images to graph later
        running_d_real_loss += d_real_loss.item()
        running_d_fake_loss += d_fake_loss.item()

        # get gradients according to loss above
        d_loss.backward()
        # optimize the discriminator parameters to better classify images
        discriminator_optimizer.step()

        # ===================================

        # ===================================
        # GENERATOR OPTIMIZE AND GET LABELS

        # Zero out any previous calculated gradients
        gan_optimizer.zero_grad()

        # Train the generator for every n_critic iterations
        if i % n_critic == 0:
            # Decode them to fake images, through the generator
            generated_images = generator(random_latent_vectors)

            # Adversarial loss from critic
            g_loss = -torch.mean(discriminator(generated_images))
            
            # Saving g_loss to graph later
            running_g_loss += g_loss.item()

            # now back propagate to get derivatives
            g_loss.backward()

            # use gan optimizer to only update the parameters of the generator
            # this was setup above to only use the params of generator
            gan_optimizer.step()
        else:
            g_loss = -torch.mean(discriminator(generated_images))
            running_g_loss += g_loss.item()


    # Calculate this epoch's loss and accuracy
    train_g_loss = running_g_loss/len(dataloader)
    train_d_real_loss = running_d_real_loss/len(dataloader)
    train_d_fake_loss = running_d_fake_loss/len(dataloader)
    train_d = -train_d_real_loss + train_d_fake_loss + lambda_gp * gradient_penalty

    g_losses.append(train_g_loss)
    d_real_losses.append(train_d_real_loss)
    d_fake_losses.append(train_d_fake_loss)
    #accuracy = correct/total

    # Occasionally save / plot
    if step % 10 == 0:
        generator.eval()
        discriminator.eval()

        # Print metrics
        print('Loss at step %s: D(z_c)=%s, D(G(z_mis))=%s' %
              (total_steps, train_d.item(), train_g_loss))
        # save images in a list for display later
        with torch.no_grad():
            fake_output = generator(fixed_random_latent_vectors).detach().cpu()
        img_list.append(torchvision.utils.make_grid(
            fake_output, padding=2, normalize=True, nrow=5))

        save_checkpoint(img_list, loaded_ims, generator, discriminator, f'wgan_{EPOCHS}e_{batch_size}b')

# %%
plot_losses(d_real_losses, d_fake_losses, g_losses, f'wgan_{EPOCHS}e_{batch_size}b')

# %%
#ims, generator, discriminator = load_checkpoint('wgan', Generator, WGCritic)

# %%
fig = plt.figure(figsize=(12, 4))
plt.axis("off")
pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
ani = animation.ArtistAnimation(
    fig, pls, interval=500, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())

# %%
