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
from IPython.display import HTML

# Seed control, for better reproducibility
seed = 22
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

run_from_checkpoint = False
# [historical_averaging]
do_historical_averaging = False

# %%

EPOCHS = 500
batch_size = 250
# NOTE: the batch_size should be an integer divisor of the data set size  or torch
# will give you an error regarding batch sizes of "0" when the data loader tries to
# load in the final batch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # work on a single GPU or CPU
    cudnn.benchmark = True
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
    npimg = img.numpy() / 2 + 0.5 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataset = dset.CIFAR10(root='data/cifar/', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def label_to_text(label):
    return classes[label]

# [one_hot_encoding]
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
def save_checkpoint(new_img_list, loaded_ims, g_model, d_model, file_prefix):
    ims = np.array([np.transpose(np.hstack((i, real_image_numpy)), (2, 1, 0))
                   for i in new_img_list])
    # if we have saved images from another run, concatenate and save them here
    if len(loaded_ims) > 0:
        # concatenate these images with other runs (if needed)
        ims = np.concatenate((loaded_ims, ims))

    Path('models/gan_models').mkdir(parents=True, exist_ok=True)
    np.save(f'models/gan_models/{file_prefix}_images.npy', ims)
    torch.save({'state_dict': g_model.state_dict()},
               f'models/gan_models/{file_prefix}_gen.pth')
    torch.save({'state_dict': d_model.state_dict()},
               f'models/gan_models/{file_prefix}_dis.pth')

def load_checkpoint(file_prefix, gen_func, disc_func):
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

# %%
latent_dim = 32
height = 32
width = 32
channels = 3

num_classes = 3
one_hot_embedding_dim = 3

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # First, transform the input into a 8x8 128-channels feature map
        self.init_size = width // 4  # one quarter the image size

        # [one_hot_encoding]
        # embedding layer for one hot encoding
        self.label_embedding = nn.Embedding(
            num_classes, one_hot_embedding_dim)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + one_hot_embedding_dim, 128 * self.init_size ** 2))

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

        # [one_hot_encoding]
        embedded_labels = self.label_embedding(class_labels)
        # concatenate the one hot embedding to the latent space vector z 
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

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.3, inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
        )

        # [one_hot_encoding]
        self.label_embedding = nn.Embedding(
            num_classes, one_hot_embedding_dim)

        # The height and width of downsampled image
        ds_size = width // 2 ** 4
        # Classification layer
        self.classification_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2 + 3, 1),
                                                  nn.Sigmoid())

    def forward(self, img, class_labels):
        out = self.model(img)
        # use the view function to flatten the layer output
        #    old way for earlier Torch versions: out = out.view(out.shape[0], -1)
        out = torch.flatten(out, start_dim=1)  # don't flatten over batch size

        # [one_hot_encoding]
        embedded_labels = self.label_embedding(class_labels)
        # concatenate the one hot embedding to the flattened layer of discriminator, just before the classification layer
        extended_flattened = torch.cat((out, embedded_labels), dim=1)

        validity = self.classification_layer(extended_flattened)
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
    im = np.rot90(im, 3)
    return im


# %% [markdown]
#
# ____
# # Least Squares GAN
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

adversarial_loss = torch.nn.MSELoss()  # mean squared error loss

# [historical_averaging]
exploration_loss = torch.nn.MSELoss()

generator.apply(weights_init)
discriminator.apply(weights_init)

generator.to(device)
discriminator.to(device)

# %%
iterations = EPOCHS  # defined above

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
%%time

def train_and_save(model_name):
    if not run_from_checkpoint:
        loaded_ims = []
    else:
        loaded_ims, generator, discriminator = load_checkpoint(model_name,
                                                            Generator,
                                                            Discriminator)
        # can get previous steps based on saved checkpoints
        total_steps = loaded_ims.shape[0]*10

    # [historical_averaging]
    exploration_lambda = 30000
    exploration_lambda_decay = 0.8
    # at this point skip exploration to speed up calculations
    exploration_lambda_eps = 0.01
    historical_parameters = None
    historical_parameter_decay = 0.9

    for step in range(iterations):

        # [historical_averaging]
        if do_historical_averaging:
            exploration_lambda *= exploration_lambda_decay
            print(f'epoch = {step}, exploration_lambda = {exploration_lambda}')

        total_steps = total_steps+1
        generator.train()
        discriminator.train()

        for i, (imgs, real_class_labels) in enumerate(dataloader):

            # ===================================
            # GENERATOR OPTIMIZE AND GET LABELS

            # Zero out any previous calculated gradients
            gan_optimizer.zero_grad()

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
            g_loss = adversarial_loss(discriminator(
                generated_images, random_class_label), misleading_targets)

            # [historical_averaging]
            if do_historical_averaging and exploration_lambda > exploration_lambda_eps:
                # if exploration_lambda > exploration_lambda_eps:
                all_params = torch.nn.utils.parameters_to_vector(
                    generator.parameters())
                # print(all_params.shape)
                # print(all_params.requires_grad) # note: all_params does and should require params since we will backprop through them
                detached_parameters = all_params.detach()
                if historical_parameters == None:
                    historical_parameters = detached_parameters

                e_loss = exploration_loss(all_params, historical_parameters)
                # print(f'original_loss: {g_loss}')
                # print(
                #     f'exploration_loss (bigger means more exploration): {e_loss}')

                g_loss = g_loss - e_loss * exploration_lambda

                hp_decay = historical_parameter_decay
                # update historical_parameters using exponential moving average
                historical_parameters = (historical_parameters * hp_decay) + \
                    (detached_parameters * (1-hp_decay))

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

            save_checkpoint(img_list, loaded_ims, generator, discriminator, model_name)


    save_checkpoint(img_list, loaded_ims, generator, discriminator, model_name)

# %%

# Load up a run, if you want

def load_and_plot(model_name):
    ims, generator, discriminator = load_checkpoint(model_name, Generator, Discriminator)
    fig = plt.figure(figsize=(20, 20))
    plt.axis("off")
    pls = [[plt.imshow(norm_grid(im), animated=True)] for im in ims]
    ani = animation.ArtistAnimation(
        fig, pls, interval=500, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


# %% [markdown]
#### [3 points] First, look at the code for the generator and discriminator/critic. Adjust the architecture to also sample from one hot encodings and use embeddings of class encodings (like in the LS-GAN paper). Use this same one hot encoding(both sampled and from the actual classes) in the discriminator/critic.  This GAN will be your base implementation--run this for at least 500 epochs (ideally more)
# - We modify the LS-GAN provided by the lab notebook https://github.com/8000net/LectureNotesMaster/blob/master/07c%20GansWithTorch.ipynb
# - We continue to use cifar10 as this dataset is well known, and the images are relatively small resulting in faster training.
# - We pick frogs, ships and trucks as our target classes from the dataset.
# - These three classes have labels 6,8 and 9 in the original dataset. They are reindexed to 0,1,2
# - The embedding is done via nn.Embedding, this takes in integers instead of a sparse one hot vector to be more space efficient.
# - The size of the embedding vector is 3. This does not make much sense in our case since we have so few classes. We are fully connecting a size 3 vector to a size 3 vector. The original paper had thousands of chinese characters, and the embedding vector was 
# - Use ctrl+f and search for [one_hot_encoding] to find relevant code segments.

# %%
