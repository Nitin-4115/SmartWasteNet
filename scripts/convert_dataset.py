import os
import shutil

# Verify exact paths relative to project root
train_src = "dataset/train_imgEFFICIENTNET"  # or "dataset/images/train_imgEFFICIENTNET"
val_src = "dataset/val_imgEFFICIENTNET"      # or "dataset/images/val_imgEFFICIENTNET"

train_img_dest = "dataset/images/train"
val_img_dest = "dataset/images/val"
train_lbl_dest = "dataset/labels/train"
val_lbl_dest = "dataset/labels/val"

classes = {
    "glass": 0,
    "metal": 1,
    "plastic": 2,
    "other": 3
}

os.makedirs(train_img_dest, exist_ok=True)
os.makedirs(val_img_dest, exist_ok=True)
os.makedirs(train_lbl_dest, exist_ok=True)
os.makedirs(val_lbl_dest, exist_ok=True)

def process_dataset(src_folder, img_dest, label_dest):
    count = 0
    if not os.path.exists(src_folder):
        print(f"❌ Source folder not found: {src_folder}")
        return

    for class_name, class_id in classes.items():
        class_folder = os.path.join(src_folder, class_name)
        if not os.path.exists(class_folder):
            print(f"⚠️ Class folder not found: {class_folder}")
            continue

        for img in os.listdir(class_folder):
            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                src_path = os.path.join(class_folder, img)
                new_name = f"{class_name}_{img}"
                dest_img_path = os.path.join(img_dest, new_name)
                shutil.copy(src_path, dest_img_path)

                label_name = new_name.rsplit(".", 1)[0] + ".txt"
                label_path = os.path.join(label_dest, label_name)
                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1 1\n")
                count += 1
    print(f"✅ Processed {count} images from {src_folder} -> {img_dest}")

process_dataset(train_src, train_img_dest, train_lbl_dest)
process_dataset(val_src, val_img_dest, val_lbl_dest)