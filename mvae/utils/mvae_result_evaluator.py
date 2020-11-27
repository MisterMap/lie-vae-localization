import cv2
import matplotlib.pyplot as plt

from .math import *


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
    plt.imshow(image, origin="lower")
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
    truth_position = batch["position"][0][index].cpu().detach().numpy()

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


def show_pose_sampling_from_pose(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    pose_hidden = model.pose_encoder([batch["position"][x][index][None] for x in range(2)])

    pose_z_mu = model._pose_mu_linear(pose_hidden)
    pose_z_logvar = model._pose_logvar_linear(pose_hidden)

    latent_space = pose_z_logvar.shape[1]
    batch_size = pose_z_logvar.shape[0]

    mu = torch.cat([pose_z_mu[None], torch.zeros_like(pose_z_mu)[None]], dim=0)
    logvar = torch.cat([pose_z_logvar[None], torch.zeros_like(pose_z_logvar)[None]], dim=0)

    epsilon = torch.randn((100000, batch_size, latent_space), device=mu.device)
    pose_z_mu, pose_z_logvar = model.calculate_distribution_product(mu, logvar)

    z = pose_z_mu + torch.exp(0.5 * pose_z_logvar) * epsilon
    z = z.reshape(-1, latent_space)
    position = model.pose_decoder(z)
    positions = model.pose_distribution.sample(position[0], position[1])
    truth_position = batch["position"][0][index].cpu().detach().numpy()

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


def show_pose_sampling_from_pose_image(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    pose_hidden = model.pose_encoder([batch["position"][x][index][None] for x in range(2)])

    pose_z_mu = model._pose_mu_linear(pose_hidden)
    pose_z_logvar = model._pose_logvar_linear(pose_hidden)

    image_hidden = model.image_encoder(batch["image"][index][None])

    image_z_mu = model._image_mu_linear(image_hidden)
    image_z_logvar = model._image_logvar_linear(image_hidden)

    latent_space = pose_z_logvar.shape[1]
    batch_size = pose_z_logvar.shape[0]

    mu = torch.cat([pose_z_mu[None], image_z_mu[None], torch.zeros_like(pose_z_mu)[None]], dim=0)
    logvar = torch.cat([pose_z_logvar[None], image_z_logvar[None], torch.zeros_like(pose_z_logvar)[None]], dim=0)

    epsilon = torch.randn((100000, batch_size, latent_space), device=mu.device)
    pose_z_mu, pose_z_logvar = model.calculate_distribution_product(mu, logvar)

    z = pose_z_mu + torch.exp(0.5 * pose_z_logvar) * epsilon
    z = z.reshape(-1, latent_space)
    position = model.pose_decoder(z)
    positions = model.pose_distribution.sample(position[0], position[1])
    truth_position = batch["position"][0][index].cpu().detach().numpy()

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


def get_image(centers, colors, image_size, radius, resolution, position):
    mask = np.sum((centers - position[:2]) ** 2, axis=1) < (image_size + radius) ** 2
    image_origin = cvt_local2global(np.array([-image_size / 2, -image_size / 2, 0]), position)
    centers = cvt_global2local(centers[mask], image_origin) / resolution
    image = np.ones((int(image_size // resolution),
                     int(image_size // resolution), 3),
                    dtype=np.uint8) * 255
    for center, color in zip(centers, colors[mask]):
        cv2.circle(image, (int(center[0]), int(center[1])), int(radius // resolution),
                   (int(color[0]), int(color[1]), int(color[2])), thickness=-1)
    image = cv2.blur(image, (int(radius // resolution // 2),
                             int(radius // resolution // 2)))
    return image


def sample_pose_from_image(model, batch, index):
    image_hidden = model.image_encoder(batch["image"][index][None])

    image_z_mu = model._image_mu_linear(image_hidden)
    image_z_logvar = model._image_logvar_linear(image_hidden)

    latent_space = image_z_logvar.shape[1]
    batch_size = image_z_logvar.shape[0]

    mu = torch.cat([image_z_mu[None], torch.zeros_like(image_z_mu)[None]], dim=0)
    logvar = torch.cat([image_z_logvar[None], torch.zeros_like(image_z_logvar)[None]], dim=0)

    epsilon = torch.randn((3, batch_size, latent_space), device=mu.device)
    image_z_mu, image_z_logvar = model.calculate_distribution_product(mu, logvar)

    z = image_z_mu + torch.exp(0.5 * image_z_logvar) * epsilon
    z = z.reshape(-1, latent_space)
    position = model.pose_decoder(z)
    positions = model.pose_distribution.sample_position(position[0], position[1])
    return positions


def show_image_from_pose_sampling(model, batch, indexes, image_size, radius, resolution, centers=None,
                                  colors = None, **kwargs):

    figure, axes = plt.subplots(len(indexes), 4, **kwargs)
    if centers is None:
        return figure
    for i in range(len(indexes)):
        image = batch["image"][indexes[i]].cpu().detach().permute(1, 2, 0).numpy()
        axes[i][0].imshow(image, origin="lower")
        axes[i][0].axis('off')
        positions = sample_pose_from_image(model, batch, indexes[i])
        for j, position in enumerate(positions):
            image = get_image(centers, colors, image_size, radius, resolution, position)
            axes[i][j + 1].imshow(image, origin="lower")
            axes[i][j + 1].axis('off')
    return figure
