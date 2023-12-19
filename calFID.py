import torch
from pytorch_fid import fid_score


# By giving the directory paths, the method will contribite to compute the FID-score between the image sets:
real_images_folder = r'results\val'
generated_images_folder = r"results\val_results"
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=128, device=torch.device('cuda'), dims=2048)
print('FID value:', fid_value)

