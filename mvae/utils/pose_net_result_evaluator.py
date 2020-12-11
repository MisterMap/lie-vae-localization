import matplotlib.pyplot as plt
import numpy as np
from .mvae_result_evaluator import show_image, show_image_from_pose_sampling


def show_pose_sampling_from_image(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    small_batch = {"image": batch["image"][index][None],
                   "position": (batch["position"][0][index][None], batch["position"][1][index][None])}
    position, loss = model.forward(small_batch)
    position = (position[0].repeat_interleave(10000, dim=0), position[1].repeat_interleave(10000, dim=0))
    positions = model.pose_distribution.sample(position[0], position[1])
    truth_position = batch["position"][0][index].cpu().detach().numpy()
    figure = plt.figure(**kwargs)
    plt.hist2d(positions[:, 0], positions[:, 1], range=range_lim, bins=(40, 40), cmap=plt.cm.jet)
    plt.scatter(truth_position[None, 0], truth_position[None, 1], s=10, c="black", label="truth")
    mean = np.mean(positions, 0)
    plt.scatter(mean[None, 0], mean[None, 1], s=10, c="white", label="mean")
    if colors is not None and centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c=colors / 255)
    plt.title("log prob = {:.3f}".format(loss["loss"]))
    plt.gca().set_aspect("equal")
    plt.xlim(range_lim[0][0], range_lim[0][1])
    plt.ylim(range_lim[1][0], range_lim[1][1])
    plt.legend()
    return figure
