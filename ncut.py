import os
import json
import numpy as np
import skimage as ski
from skimage import segmentation, color
from skimage import graph
from matplotlib import pyplot as plt

# Create a list all files in the directory
horses = os.listdir('archive/weizmann_horse_db/horse/')
IOU_list = []

# Iterate over all the files
for horse in horses:

    # Load the image and mask
    img = ski.io.imread('archive/weizmann_horse_db/horse/' + horse)
    mask = ski.io.imread('archive/weizmann_horse_db/mask/'+ horse)

    ###############################################################################################################

    # Segmenting the image using SLIC to generate the superpixels
    labels1 = segmentation.slic(img, compactness=20, n_segments=400, enforce_connectivity = True, start_label=1)
    out1 = color.label2rgb(labels1, img, kind='avg')

    ###############################################################################################################

    # Compute the Region Adjacency Graph using mean colors
    g = graph.rag_mean_color(img, labels1, mode='similarity')

    # Perform Normalized Graph cut on the Region Adjacency Graph
    labels2 = graph.cut_normalized(labels1, g, max_edge=0.2)
    out2 = color.label2rgb(labels2, img, kind='avg')

    ###############################################################################################################

    # Get the unique elements and their counts before and after multiplying with the mask
    unique_labels_pre, counts_pre = np.unique(labels2, return_counts=True)
    element_wise_mul = np.multiply(labels2, mask)
    unique_labels_post, counts_post = np.unique(element_wise_mul, return_counts=True)

    # Create a new matrix to store wich pixel are considered to be in the horse
    horse_labels = np.zeros_like(labels2)

    # Iterate over each label of the reamining regions
    for label in unique_labels_post:

        # Check if more than half of the pixels in the region belongs to the horse mask
        if label != 0 and counts_post[np.where(unique_labels_post == label)[0][0]] > counts_pre[np.where(unique_labels_pre == label)[0][0]] / 2:
            horse_labels[np.where(labels2 == label)] = 1

    ###############################################################################################################

    # Intersection between the mask and the regions that are considered to be the horse
    intersection = np.logical_and(horse_labels, mask)

    # Union between the mask and the regions that are considered to be the horse
    union = np.logical_or(horse_labels, mask)

    # Compute the Jaccard Similarity (Intersection over Union) and append the result
    jaccard_similarity = np.sum(intersection) / np.sum(union)
    IOU_list.append((horse, jaccard_similarity))
    print("Jaccard Similarity:", jaccard_similarity)

    # Write the list to a JSON file
    with open('results.json', 'w') as json_file:
        json.dump(IOU_list, json_file, indent=4)

    ###############################################################################################################

    # Setting up the plot
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True,
                            sharey=True, figsize=(6, 8))

    # Display the images and the corresponding titles
    axs[0, 0].imshow(out1)
    axs[0, 0].set_title('Superpixels')
    axs[0, 1].imshow(out2)
    axs[0, 1].set_title('Ncut')
    axs[1, 0].imshow(horse_labels)
    axs[1, 0].set_title('Result')
    axs[1, 1].imshow(mask)
    axs[1, 1].set_title('Mask')

    for ax in axs.flat:
        ax.axis('off')

    # Save the figure obtained
    plt.tight_layout()
    os.makedirs('archive/weizmann_horse_db/results/', exist_ok=True)
    plt.savefig('archive/weizmann_horse_db/results/' + horse)
    plt.close()