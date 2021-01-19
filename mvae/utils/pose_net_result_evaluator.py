import matplotlib.pyplot as plt
import numpy as np
from .mvae_result_evaluator import show_image, show_image_from_pose_sampling


def show_pose_sampling_from_image(model, batch, index, range_lim, centers=None, colors=None, **kwargs):
    small_batch = {"image": batch["image"][index][None],
                   "position": (batch["position"][0][index][None], batch["position"][1][index][None])}
    position, loss = model.forward(small_batch)
    position = (position[0].repeat_interleave(100000, dim=0), position[1].repeat_interleave(100000, dim=0))
    positions = model.pose_distribution.sample(position[0], position[1])[:, :2]
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
    plt.title("Pose sampling (nll = {:.3f})".format(loss["loss"]))
    plt.gca().set_aspect("equal")
    plt.xlim(range_lim[0][0], range_lim[0][1])
    plt.ylim(range_lim[1][0], range_lim[1][1])
    return figure
