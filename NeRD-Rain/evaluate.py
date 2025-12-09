import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import utils
import argparse

parser = argparse.ArgumentParser(description="Image Deraining")
parser.add_argument('--result_dir', type=str, default='/kaggle/working/results/Rain100H/', help="Directory of model outputs")
parser.add_argument('--gt_dir', type=str, default='/kaggle/input/rain13kdataset/test/test/Rain100H/target/', help="Directory of ground truth images")
parser.add_argument('--dataset_name', type=str, default='Rain100H', help="Name of the dataset")
parser.add_argument('--save_file', type=str, default='/kaggle/working/results.csv', help="Where to save the combined results")
args = parser.parse_args()

def compute_psnr(img1, img2):
    return utils.numpyPSNR(img1, img2)

def compute_ssim_gray(img1, img2):
    return ssim(img1, img2, data_range=255)

def find_gt_file(gt_dir, pred_name):
    base = os.path.splitext(pred_name)[0]

    for ext in ["png", "jpg", "jpeg"]:
        candidate = os.path.join(gt_dir, base + "." + ext)
        if os.path.exists(candidate):
            return candidate
    
    return None

def evaluate(result_dir, gt_dir):
    result_files = sorted(os.listdir(result_dir))
    psnr_list = []
    ssim_list = []

    for name in result_files:
        pred_path = os.path.join(result_dir, name)
        gt_path = find_gt_file(gt_dir, name)

        if not os.path.exists(gt_path):
            print(f"GT not found for {name}")
            continue

        pred = Image.open(pred_path).convert('L')
        gt   = Image.open(gt_path).convert('L')

        pred_np = np.array(pred, dtype=np.uint8)
        gt_np   = np.array(gt,   dtype=np.uint8)

        psnr_list.append(compute_psnr(pred_np, gt_np))
        ssim_list.append(compute_ssim_gray(pred_np, gt_np))

    return np.mean(psnr_list), np.mean(ssim_list)

def append_results(save_file, dataset_name, psnr, ssim):
    header = "Dataset,PSNR,SSIM\n"

    # Nếu file chưa tồn tại → ghi header trước
    if not os.path.exists(save_file):
        with open(save_file, "w") as f:
            f.write(header)

    with open(save_file, "a") as f:
        f.write(f"{dataset_name},{psnr:.4f},{ssim:.4f}\n")

if __name__ == "__main__":
    psnr, ssim = evaluate(args.result_dir, args.gt_dir)
    print("\n===> RESULTS FOR:", args.dataset_name)
    print("PSNR:", psnr)
    print("SSIM:", ssim)

    append_results(args.save_file, args.dataset_name, psnr, ssim)
    print("\nSaved to:", args.save_file)
