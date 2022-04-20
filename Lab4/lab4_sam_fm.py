# %% [markdown]
#  Compare this implimentation to the one from the official Torch tutorial:
#  https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

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
from IPython.display import HTML

# Seed control, for better reproducibility
# NOTE: this does not guarantee results are always the same
seed = 22
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

run_from_checkpoint = False


# %%

EPOCHS = 500
# might try to use large batches (we will discuss why later when we talk about BigGAN)
batch_size = 250
# NOTE: the batch_size should be an integer divisor of the data set size  or torch
# will give you an error regarding batch sizes of "0" when the data loader tries to
# load in the final batch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # work on a single GPU or CPU
    cudnn.benchmark = True
    # generator.cuda()
    # discriminator.cuda()
    # adversarial_loss.cuda()
    Tensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    cudnn.benchmark = False
    Tensor = torch.FloatTensor
    IntTensor = torch.IntTensor


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


def label_to_text(label):
    return classes[label]


selected_classes = [6, 8, 9]  # frog, ship, truck

reindex_class_label_dict = {6: 0, 8: 1, 9: 2}


def reindex_class_label(original_class_label):
    return reindex_class_label_dict[original_class_label]


selected_indices = [i for i, x in enumerate(
    dataset.targets) if x in selected_classes]
print("number of selected imgs: ", len(selected_indices))

subset = torch.utils.data.Subset(dataset, selected_indices)


dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)



# %%
# get some random training images
dataiter = iter(dataloader)
real_image_examples, real_label_examples = dataiter.next()

# print first 10 labels for sanity check
print('Some sample images:')
print('First 10 labels for sanity check:')
for label in real_label_examples[:10]:
    print(label_to_text(label))

# show images
plt.figure(figsize=(10, 10))
imshow(torchvision.utils.make_grid(
    real_image_examples, nrow=int(np.sqrt(batch_size))))

print("Image shape: ", real_image_examples[0].size())


# %%
# create a tensor dataset than can be moved entirely to the gpu in one go

dataloader = torch.utils.data.DataLoader(subset, batch_size=15000,
                                         shuffle=False, num_workers=0)
gpu_dataset = None
gpu_targets = None
for i, (imgs, targets) in enumerate(dataloader):
    print(i)
    print(imgs.device)
    targets_reindexed = torch.tensor(
        [reindex_class_label(x.item()) for x in targets])
    print(targets_reindexed)
    gpu_dataset = imgs.to(device)
    gpu_targets = targets_reindexed.to(device)


print(gpu_dataset.device)
# print(gpu_frogs.shape)
print(gpu_dataset.shape)
print(gpu_targets.shape)

tensor_dataset = torch.utils.data.TensorDataset(gpu_dataset, gpu_targets)

dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size,
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



# %% [markdown]
#  # Vanilla Generative Adversarial Networks
#  In this implementation of GANS, we will use a few of the tricks from F. Chollet and from Salimans et al. In particular, we will add some noise to the labels.

# %%
latent_dim = 32
height = 32
width = 32
channels = 3

num_classes = 3
one_hot_embedding_dim = 3


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

        self.label_embedding = nn.Embedding(
            num_classes, one_hot_embedding_dim)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + one_hot_embedding_dim, 128 * self.init_size ** 2))
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

    def forward(self, z, class_labels):
        # call the functions from earlier:

        # expand the sampled z to 8x8

        embedded_labels = self.label_embedding(class_labels)

        extended_z = torch.cat((z, embedded_labels), dim=1)
        out = self.l1(extended_z)

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

        self.label_embedding = nn.Embedding(
            num_classes, one_hot_embedding_dim)

        # The height and width of downsampled image
        ds_size = width // 2 ** 4
        # Classification layer
        self.classification_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2 + 3, 1),
                                                  nn.Sigmoid())

    def forward(self, img, class_labels, matching=False):
        out = self.model(img)
        # use the view function to flatten the layer output
        #    old way for earlier Torch versions: out = out.view(out.shape[0], -1)
        out = torch.flatten(out, start_dim=1)  # don't flatten over batch size
        embedded_labels = self.label_embedding(class_labels)

        extended_flattened = torch.cat((out, embedded_labels), dim=1)

        validity = self.classification_layer(extended_flattened)

        if matching:
            return validity, out
        else:
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
#  We are getting something that is similar to a frog, but also we are seeing a bit of mode collapse. The global properties of a greenish or gray blob surrounded by various background is starting to comes across. However, the finer structure is not doing too well. That is, the legs and details in the background are not present yet.
# 
#  To improve this result, there are a number of things we might try such as:
#  - Using the Radford guided methods (implemented already in 2022)
#  - Adding more randomization to the optimizer
#  - Running the discriminator multiple times for each generator update
#  - Changing the objective function (let's try this one)
# 
#  ____
#  # Least Squares GAN
#  Actually, the only thing we need to do here is replace the adversarial loss function. Note that we are NOT going to make additions to the architecture where the one hot encoding of the classes (and random classes) are used in both the generator and discriminator. This means that we might see a bit more mode collapse in our implementation.

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
iterations = EPOCHS*2  # defined above

# Sample random points in the latent space
num_examples_per_class = 20
plot_num_examples = num_examples_per_class * num_classes
fixed_random_latent_vectors = torch.randn(
    plot_num_examples, latent_dim, device=device)
_labels = []
for i in range(num_classes):
    _labels += [i] * num_examples_per_class
fixed_class_labels = IntTensor(_labels)

img_list = []
total_steps = 0


example_real_indices = []
for i in range(num_classes):
    class_label = selected_classes[i]
    class_indices = [i for i, label in enumerate(
        real_label_examples) if label == class_label]
    class_indices = class_indices[:num_examples_per_class]
    example_real_indices += class_indices

real_image_numpy = np.transpose(torchvision.utils.make_grid(
    real_image_examples[example_real_indices, :, :, :], padding=2, normalize=False, nrow=10), (0, 1, 2))


# %%


# %%


if not run_from_checkpoint:
    loaded_ims = []
else:
    loaded_ims, generator, discriminator = load_checkpoint('ls',
                                                           Generator,
                                                           Discriminator)
    # can get previous steps based on saved checkpoints
    total_steps = loaded_ims.shape[0]*10

for step in range(iterations):
    total_steps = total_steps+1
    generator.train()
    discriminator.train()

    for i, (imgs, real_class_labels) in enumerate(dataloader):

        # ===================================
        # GENERATOR OPTIMIZE AND GET LABELS

        # Zero out any previous calculated gradients
        gan_optimizer.zero_grad()

        real_images = Variable(imgs.type(Tensor))

        this_batch_size = imgs.shape[0]
        # Sample random points in the latent space
        random_latent_vectors = Variable(
            Tensor(np.random.normal(0, 1, (this_batch_size, latent_dim))))

        random_class_label = Variable(
            IntTensor(np.random.randint(0, num_classes, this_batch_size)))

        # Decode them to fake images, through the generator
        generated_images = generator(random_latent_vectors, random_class_label)

        # Assemble labels that say "all real images"
        # misleading target, c=1
        misleading_targets = Variable(
            Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

        # Get MSE Loss function
        # want generator output to generate images that are "close" to all "ones"

        # _, features_real = discriminator(
        #     real_images, random_class_label, matching=True)

        output, features_fake = discriminator(
            generated_images, random_class_label, matching=True)

        features_real = torch.mean(features_real, 0)
        features_fake = torch.mean(features_fake, 0)

        t_loss = adversarial_loss(output, misleading_targets)
        f_loss = adversarial_loss(features_fake, features_real)
        g_loss = 0.1* t_loss + f_loss

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
        combined_images = torch.cat([real_images, generated_images.detach()])

        combined_class_labels = torch.cat(
            [real_class_labels, random_class_label])
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
        d_loss = (
            adversarial_loss(discriminator(combined_images[:batch_size], real_class_labels), labels[:batch_size]) +
            adversarial_loss(discriminator(
                combined_images[batch_size:], random_class_label), labels[batch_size:])
        ) / 2

        # get gradients according to loss above
        d_loss.backward()
        # optimize the discriminator parameters to better classify images
        discriminator_optimizer.step()

        # Now Clip weights of discriminator (manually)
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # ===================================

    # Occasionally save / plot
    if step % 10 == 0:
        generator.eval()
        discriminator.eval()

        # Print metrics
        print('Loss at step %s: D(z_c)=%s, D(G(z_mis))=%s' %
              (total_steps, d_loss.item(), g_loss.item()))
        # save images in a list for display later
        with torch.no_grad():
            fake_output = generator(
                fixed_random_latent_vectors, fixed_class_labels).detach().cpu()
        img_list.append(torchvision.utils.make_grid(
            fake_output, padding=2, normalize=True, nrow=10))

        save_checkpoint(img_list, loaded_ims, generator, discriminator, 'tl_1000')



# %%
save_checkpoint(img_list, loaded_ims, generator, discriminator, 'suml')


# %% [markdown]
# tloss 1000

# %%
# Load up a run, if you want
ims, generator, discriminator = load_checkpoint('tl_1000', Generator, Discriminator)


# %%
fig = plt.figure(figsize=(20, 20))
plt.axis("off")
pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
ani = animation.ArtistAnimation(
    fig, pls, interval=500, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())

# %% [markdown]
# floss 1000

# %%
# Load up a run, if you want
ims, generator, discriminator = load_checkpoint('fm_1000', Generator, Discriminator)


# %%
fig = plt.figure(figsize=(20, 20))
plt.axis("off")
pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
ani = animation.ArtistAnimation(
    fig, pls, interval=500, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())

# %% [markdown]
# 0.1 t_loss + f_loss

# %%
# Load up a run, if you want
ims, generator, discriminator = load_checkpoint('suml', Generator, Discriminator)


# %%
fig = plt.figure(figsize=(20, 20))
plt.axis("off")
pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
ani = animation.ArtistAnimation(
    fig, pls, interval=500, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())

# %% [markdown]
# t_loss

# %%
# Load up a run, if you want
ims, generator, discriminator = load_checkpoint('tl', Generator, Discriminator)

# %%
fig = plt.figure(figsize=(20, 20))
plt.axis("off")
pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
ani = animation.ArtistAnimation(
    fig, pls, interval=500, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())

# %% [markdown]
# f_loss

# %%
# Load up a run, if you want
ims, generator, discriminator = load_checkpoint('fm', Generator, Discriminator)

# %%
fig = plt.figure(figsize=(20, 20))
plt.axis("off")
pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
ani = animation.ArtistAnimation(
    fig, pls, interval=500, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())


# %% [markdown]
#  Well, these results are not exactly a great improvement. Mode collapse is more apparent here as well, but the fine structure of the frogs is also not quite the improvement that we wanted. Looking back through the iterations, there was some indication of more successful generations. Subjectively, the frogs started to show up, but then generation became slightly worse. We could run this code for many more iterations, and that might work in terms of getting the optimizers to create better distributions. But it is not guaranteed.
# 
#  Instead, now let's try using a Wasserstein GAN, where we use the gradient penalty as a method of making the discriminator 1-lipschitz (and therefore a valid critic to approximate the earth mover distance).
# 
#  ___
#  # Wasserstein GAN with Gradient Penalty
#  For this implementation, we need to add functionality to the gradient of the Discriminator to make it a critic. For the most part, we need to add the gradient loss function calculations to match the WGAN-GP.

# %%



