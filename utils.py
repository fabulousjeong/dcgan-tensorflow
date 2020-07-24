import matplotlib.pyplot as plt
import numpy as np

# generate noise vector
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# visualize sample output
def save_samples(title, samples):
    samples = (samples+1.0)*0.5 # normalize to [0,1]
    n_grid = int(np.sqrt(samples.shape[0]))
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(2*n_grid, 2*n_grid))

    samples_grid = np.reshape(samples[:n_grid * n_grid],(n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))

    if samples.shape[3] != 3:
        samples_grid = np.squeeze(samples_grid, 4)

    for i in range(n_grid):
        for j in range(n_grid):
            axes[i][j].set_axis_off()
            axes[i][j].imshow(samples_grid[i][j])

    plt.savefig(title, bbox_inches='tight')
    print('saved %s.' % title)
    plt.close(fig)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig(f'./images/image_at_epoch_{epoch:04d}.png')
    plt.close()

def generate_gif(anim_file):
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('./images/image*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
