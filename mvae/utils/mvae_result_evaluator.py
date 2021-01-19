import cv2
import matplotlib.pyplot as plt

from .math import *
from .math_torch import *


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


def show_pose_mvae_reconstruction_pose(model, batch, image_count=10, **kwargs):
    figure = plt.figure(**kwargs)
    input_data = batch["image"]
    pose = batch["position"]
    output_data = model.reconstruct_image_from_pose(pose)
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


def show_pose_distribution(z_mu, z_logvar, model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    batch_size = z_mu.shape[0]
    latent_space = z_mu.shape[1]
    epsilon = torch.randn((100000, batch_size, latent_space), device=z_mu.device)
    z = z_mu + torch.exp(0.5 * z_logvar) * epsilon
    z = z.reshape(-1, latent_space)
    position = model.pose_vae.decoder(z)
    positions = model.pose_vae.pose_distribution.sample(position[0], position[1])
    truth_position = batch["position"][0][index].cpu().detach().numpy()
    truth_orientation = batch["position"][1][index].cpu().detach().numpy()
    figure = plt.figure(**kwargs)
    plt.hist2d(positions[:, 0], positions[:, 1], range=range_lim, bins=(40, 40), cmap=plt.cm.jet)
    plt.arrow(truth_position[0], truth_position[1], 0.25 * truth_orientation[0], 0.25 * truth_orientation[1],
              color="red", label="truth", width=0.05)
    if colors is not None and centers is not None:
        for color, center in zip(colors, centers):
            circle = plt.Circle((center[0], center[1]), 0.3, alpha=0.8, color=color / 255)
            plt.gca().add_patch(circle)
    plt.title("Pose sampling")
    plt.gca().set_aspect("equal")
    plt.xlim(range_lim[0][0], range_lim[0][1])
    plt.ylim(range_lim[1][0], range_lim[1][1])
    return figure


def show_pose_sampling_from_image(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    z_mu, z_logvar = deregularize_normal_distribution(*model.image_vae.generate_z(batch["image"][index][None]))
    return show_pose_distribution(z_mu, z_logvar, model, batch, index, range_lim, centers, colors, **kwargs)


def show_pose_sampling_from_pose(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    position = batch["position"]
    position = (position[0][index][None], position[1][index][None])
    z_mu, z_logvar = deregularize_normal_distribution(*model.pose_vae.generate_z(position))
    return show_pose_distribution(z_mu, z_logvar, model, batch, index, range_lim, centers, colors, **kwargs)


def show_pose_sampling_from_joint(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    position = batch["position"]
    position = (position[0][index][None], position[1][index][None])
    z_mu, z_logvar = model.generate_z(position, batch["image"][index][None])
    return show_pose_distribution(z_mu, z_logvar, model, batch, index, range_lim, centers, colors, **kwargs)


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


def show_image_from_pose_sampling(model, batch, indexes, image_size, radius, resolution, centers=None,
                                  colors=None, **kwargs):
    figure, axes = plt.subplots(len(indexes), 4, **kwargs)
    if centers is None:
        return figure
    for i in range(len(indexes)):
        image = batch["image"][indexes[i]].cpu().detach().permute(1, 2, 0).numpy()
        axes[i][0].imshow(image, origin="lower")
        axes[i][0].axis('off')
        for j in range(3):
            position = model.sample_pose_from_image(batch["image"][indexes[i]][None])[0]
            image = get_image(centers, colors, image_size, radius, resolution, position)
            axes[i][j + 1].imshow(image, origin="lower")
            axes[i][j + 1].axis('off')
    return figure
