import os
import shutil
import random

def create_sample(src_base, dest_base, sample_ratio=0.10):
    if not os.path.exists(src_base):
        print(f"Source {src_base} does not exist. Skipping.")
        return
        
    for split in ['train', 'val', 'test']:
        src_images = os.path.join(src_base, 'images', split)
        src_labels = os.path.join(src_base, 'labels', split)
        
        if not os.path.exists(src_images):
            continue
            
        dest_images = os.path.join(dest_base, 'images', split)
        dest_labels = os.path.join(dest_base, 'labels', split)
        
        os.makedirs(dest_images, exist_ok=True)
        os.makedirs(dest_labels, exist_ok=True)
        
        images = [f for f in os.listdir(src_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Keep 5-10% (at least 5 files, at most 50 files)
        num_to_keep = max(5, int(len(images) * sample_ratio))
        num_to_keep = min(num_to_keep, 50)
        num_to_keep = min(num_to_keep, len(images))
        
        sampled_images = random.sample(images, num_to_keep)
        
        for img in sampled_images:
            shutil.copy(os.path.join(src_images, img), os.path.join(dest_images, img))
            label = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(os.path.join(src_labels, label)):
                shutil.copy(os.path.join(src_labels, label), os.path.join(dest_labels, label))
                
        print(f"Sampled {num_to_keep} files from {src_images} to {dest_images}")

if __name__ == "__main__":
    # 1. Ensure the backup directory exists
    backup_dir = "full_dataset_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    # 2. Move existing data directories to backup
    for d in ["birdnest_data", "data", "test", "test_data"]:
        if os.path.exists(d):
            dest = os.path.join(backup_dir, d)
            if not os.path.exists(dest):
                shutil.move(d, dest)
                print(f"Moved {d} to {dest}")
            else:
                print(f"{dest} already exists. Left {d} alone or need manual intervention.")
                
    # 3. Create a clean sample dataset
    sample_dir = "dataset"
    print("Creating sample dataset in 'dataset/'...")
    
    # We will sample from birdnest_data and data from the backup
    create_sample(os.path.join(backup_dir, "birdnest_data"), sample_dir)
    create_sample(os.path.join(backup_dir, "data"), sample_dir)
    create_sample(os.path.join(backup_dir, "test"), "test_images_sample")
    
    print("Sampling complete!")
