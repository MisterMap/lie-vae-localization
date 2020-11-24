import matplotlib.pyplot as plt
import numpy as np
import torch


def show_reconstruction(image_count, output_data, input_data, **kwargs):
    images = output_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :]
    input_images = input_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :]
    plt.figure(**kwargs)
    for i in range(image_count):
        plt.subplot(image_count, 2, 2 * i + 1)
        plt.imshow(np.clip(input_images[i], 0, 1))
        plt.axis('off')
        plt.subplot(image_count, 2, 2 * i + 2)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.axis('off')


def show_pose_mvae_reconstruction(model, batch, image_count=10, **kwargs):
    figure = plt.figure(**kwargs)
    input_data = batch["image"]
    output_data = model.reconstruct_image_from_image(input_data)
    images = output_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :]
    input_images = input_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :]
    for i in range(image_count):
        plt.subplot(image_count, 2, 2 * i + 1)
        plt.imshow(np.clip(input_images[i], 0, 1))
        plt.axis('off')
        plt.subplot(image_count, 2, 2 * i + 2)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.axis('off')
    return figure


def show_pose_mvae_reconstruction_pose(model, batch, image_count=10, reparametrize=True, **kwargs):
    figure = plt.figure(**kwargs)
    input_data = batch["image"]
    pose = batch["position"]
    output_data = model.reconstruct_image_from_position(pose, reparametrize)
    images = output_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :]
    input_images = input_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :]
    for i in range(image_count):
        plt.subplot(image_count, 2, 2 * i + 1)
        plt.imshow(np.clip(input_images[i], 0, 1))
        plt.axis('off')
        plt.subplot(image_count, 2, 2 * i + 2)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.axis('off')
    return figure


def show_image(batch, index, **kwargs):
    figure = plt.figure(**kwargs)
    image = batch["image"][index].cpu().detach().permute(1, 2, 0).numpy()
    plt.imshow(image)
    return figure


def show_pose_sampling(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    image_hidden = model.image_encoder(batch["image"][index][None])

    image_z_mu = model._image_mu_linear(image_hidden)
    image_z_logvar = model._image_logvar_linear(image_hidden)

    latent_space = image_z_logvar.shape[1]
    batch_size = image_z_logvar.shape[0]

    mu = torch.cat([image_z_mu[None], torch.zeros_like(image_z_mu)[None]], dim=0)
    logvar = torch.cat([image_z_logvar[None], torch.zeros_like(image_z_logvar)[None]], dim=0)

    epsilon = torch.randn((100000, batch_size, latent_space), device=mu.device)
    image_z_mu, image_z_logvar = model.calculate_distribution_product(mu, logvar)

    z = image_z_mu + torch.exp(0.5 * image_z_logvar) * epsilon
    z = z.reshape(-1, latent_space)
    position = model.pose_decoder(z)
    positions = model.pose_distribution.sample(position[0], position[1])
    truth_position = batch["position"][index][0].cpu().detach().numpy()

    figure = plt.figure(**kwargs)
    plt.hist2d(positions[:, 0], positions[:, 1], range=range_lim, bins=(40, 40), cmap=plt.cm.jet)
    plt.scatter(truth_position[None, 0], truth_position[None, 1], s=10, c="black", label="truth")
    mean = np.mean(positions, 0)
    plt.scatter(mean[None, 0], mean[None, 1], s=10, c="white", label="mean")
    if colors is not None and centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c=colors / 255)
    plt.gca().set_aspect("equal")
    plt.xlim(range_lim[0][0], range_lim[0][1])
    plt.ylim(range_lim[1][0], range_lim[1][1])
    plt.legend()
    return figure
