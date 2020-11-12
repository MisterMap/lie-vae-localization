import matplotlib.pyplot as plt
import numpy as np


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


def show_pose_mvae_reconstruction(model, data_module, image_count=10, device="cpu", **kwargs):
    data_module.setup("test")
    batches = data_module.train_dataloader()
    input_data = None
    for batch in batches:
        input_data = batch["image"]
        break
    model.eval()
    output_data = model.reconstruct_image_from_image(input_data.to(device))
    show_reconstruction(image_count, output_data, input_data, **kwargs)


def show_pose_mvae_reconstruction_from_pose(model, data_module, image_count=10, **kwargs):
    data_module.setup("test")
    batches = data_module.train_dataloader()
    input_data = None
    pose = None
    for batch in batches:
        input_data = batch["image"]
        pose = batch["position"]
        break
    model.eval()
    output_data = model.reconstruct_image_from_position(pose)
    show_reconstruction(image_count, output_data, input_data, **kwargs)
