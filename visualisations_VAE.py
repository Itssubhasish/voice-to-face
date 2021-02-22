# This part decodes the latent image representation of the dataset.
# Credit for code: https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/

def viz_decoded(encoder, decoder, data):
    """ Visualizes the samples from latent space."""
    num_samples = 10
    figure = np.zeros((image_width * num_samples, image_height * num_samples, num_channels))
    grid_x = np.linspace(-8, 8, num_samples)
    grid_y = np.linspace(-8, 8, num_samples)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
#             z_sample = np.array([np.random.normal(0, 1, latent_dim)])
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(image_width, image_height, num_channels)
            figure[i * image_width: (i + 1) * image_width,
                  j * image_height: (j + 1) * image_height] = digit
    plt.figure(figsize=(10, 10))
    start_range = image_width // 2
    end_range = num_samples * image_width + start_range + 1
    pixel_range = np.arange(start_range, end_range, image_width)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
    # So reshape if necessary
    fig_shape = np.shape(figure)
    if fig_shape[2] == 1:
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
    # Show image
    plt.imshow(figure)
    plt.show()

# Plot results
data = (normalized_test_data, normalized_test_data)
# data = (normalized_train_data, normalized_train_data)

viz_decoded(encoder, decoder, data)