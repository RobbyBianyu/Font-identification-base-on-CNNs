import torch
import numpy as np
import os
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import random
import hashlib
from torch.amp import autocast
import shutil
import matplotlib.pyplot as plt

print('number_epochs = 50, len(results) >= total_chunks, without augment, double activations, BFP16, 3 conve')

# ----------------------------------------
# 2. Utility Functions Definition
# ----------------------------------------
print("Step 2: Defining utility functions...")

def ensure_dir(path):
    # Ensure the directory exists, create it if necessary
    os.makedirs(path, exist_ok=True)


def img_hash(img):
    # Compute and return the SHA-256 hash of the image’s raw bytes
    return hashlib.sha256(img.tobytes()).hexdigest()


def remove_duplicates(folder_path, exts=(".png", ".bmp")):
    # Initialize a set to track seen hashes
    seen = set()
    # List to collect filenames of unique images
    unique = []
    
    # Iterate over all files in the given folder
    for fname in os.listdir(folder_path):
        # Skip files without the desired extensions
        if not fname.lower().endswith(exts):
            continue
        
        # Build full file path
        path = os.path.join(folder_path, fname)
        # Open the image, convert to grayscale for consistent hashing
        img = Image.open(path).convert("L")
        
        # Compute the hash of the image
        h = img_hash(img)
        
        # If this hash hasn't been seen yet, record it and keep the filename
        if h not in seen:
            seen.add(h)
            unique.append(fname)
    
    # Return list of unique image filenames
    return unique



def crop_sentences(img):
    # Convert the input PIL Image to a NumPy array
    raw_pix = np.array(img)
    # Prepare list to hold cropped sentence images
    crops = []

    # Split the image into two horizontal strips
    for pg_line in [0, 1]:
        # Extract a 200px-high strip from the image
        pix = raw_pix[pg_line * 200 : pg_line * 200 + 200, :]

        # Find the first non-white column from the left
        topcol = np.argmax((np.argmax(pix != 255, axis=0) > 0))
        # Find the first non-white column from the right by flipping both axes
        botcol = np.argmax((np.argmax(np.flip(pix, axis=(0, 1)) != 255, axis=0) > 0))
        # Find the first non-white row from the top
        toprow = np.argmax((np.argmax(pix != 255, axis=1) > 0))
        # Find the first non-white row from the bottom by flipping rows
        botrow = np.argmax((np.argmax(np.flip(pix, axis=0) != 255, axis=1) > 0))

        # Crop out the white margins using the detected bounds
        pix = pix[toprow : 200 - botrow, topcol : pix.shape[1] - botcol]
        # Convert back to a PIL Image and add to results
        crop_img = Image.fromarray(pix)
        crops.append(crop_img)

    # Return the list of cropped sentence images
    return crops


def random_valid_chunks(image, total_chunks=30, crop_width=18, min_black_ratio=0.05):
    # Get image height and width from the NumPy array shape
    h, width = image.shape
    # List to collect valid flattened chunks
    results = []
    # Counter for attempts to find valid chunks
    tries = 0
    # Maximum number of attempts before giving up
    max_tries = total_chunks * 10

    # If the image is narrower than the desired crop width, return empty list
    if width < crop_width:
        return results

    # Keep sampling until we have enough valid chunks or exceed max tries
    while len(results) < total_chunks and tries < max_tries:
        # Pick a random x-coordinate for the left edge of the crop
        try:
            x = random.randint(0, width - crop_width)
        except ValueError:
            # If randint fails (should not happen with the above check), abort
            return results

        # Extract a vertical slice of width `crop_width`
        chunk = image[:, x : x + crop_width]
        # Check the proportion of “black” pixels (pixel value < 200)
        if np.mean(chunk < 200) >= min_black_ratio:
            # Resize the chunk to 16×16 and flatten into a 1D array
            resized = Image.fromarray(chunk).resize((16, 16))
            results.append(np.array(resized).flatten())

        # Increment the attempt counter
        tries += 1

    # If we collected the required number of chunks, return them; otherwise return empty
    return results if len(results) >= total_chunks else []


def process_folder(folder_path):
    # Remove duplicate images and get a list of unique filenames
    unique_files = remove_duplicates(folder_path)
    
    # Prepare lists for feature vectors and corresponding labels
    images, labels = [], []
    
    # Process each unique image file
    for fname in unique_files:
        # Load image in grayscale mode
        img = Image.open(os.path.join(folder_path, fname)).convert("L")
        
        # Split into two sentence-sized crops
        halves = crop_sentences(img)
        
        # Use filename (without extension) as label
        label = os.path.splitext(fname)[0]
        
        # For each half-image, extract valid random chunks
        for half in halves:
            arr = np.array(half)  # Convert to NumPy array
            blocks = random_valid_chunks(arr)
            
            # Append each chunk's flattened array and its label
            for ch in blocks:
                images.append(ch)
                labels.append(label)
    
    # Return feature matrix and label list
    return images, labels


# ----------------------------------------
# 3. Data Processing and Chunk Extraction
# ----------------------------------------
print("Step 3: Processing folder and extracting image chunks...")

# Path to the directory with generated pangram images
process_path = '/root/autodl-tmp/Font_Images/generated_pangrams'

# Process the folder
images, labels = process_folder(process_path)


# ----------------------------------------
# 4. Label Remapping
# ----------------------------------------
print("Step 4: Mapping string labels to integers...")

# Get a sorted list of unique labels
unique_labels = sorted(set(labels))

# Create a mapping from label to integer index
label2idx = {lab: i for i, lab in enumerate(unique_labels)}

# Convert the original labels to their corresponding indices
new_labels = [label2idx[lab] for lab in labels]

# Number of distinct classes (unique labels)
num_classes = len(unique_labels)

# Total number of image feature vectors extracted
num_images = len(images)

print("After remapping, label range:", min(new_labels), max(new_labels), " Number of classes:", num_classes)
print(f"Paths & labels prepared: {num_images} images, {num_classes} classes")


# ----------------------------------------
# 5. Random Subsampling of Classes
# ----------------------------------------
print("Step 5: Subsampling classes for quick experiment...")

# Set random seed for reproducibility
random.seed(42)

# Determine how many classes to keep (up to 1000 or total classes if fewer)
subset_K = min(1000, num_classes)

# List of all class indices
all_classes = list(range(num_classes))

# Randomly select subset_K distinct classes
selected_classes = random.sample(all_classes, subset_K)

# Keep only images whose label is in the selected subset
subset_images = []
subset_labels = []
for img, lbl in zip(images, new_labels):
    if lbl in selected_classes:
        subset_images.append(img)
        subset_labels.append(lbl)

# 3. Re-index the selected classes to 0..K-1 for consistency
selected_classes.sort()  # Sort the class IDs
old2new = {old: new for new, old in enumerate(selected_classes)}  # Map old IDs to new

# Apply the remapping to the filtered labels
subset_labels = [old2new[lbl] for lbl in subset_labels]

# Overwrite the original variables with the subsetted and re-indexed data
images      = subset_images    # Filtered image feature vectors
new_labels  = subset_labels    # Corresponding new label indices
num_classes = subset_K         # Updated number of classes

print(f"Subsampled dataset: {len(images)} images, {num_classes} classes")

# ----------------------------------------
# 6. Dataset and Transforms
# ----------------------------------------
print("Step 6: Defining transforms and dataset...")

# Compose a series of transformations: convert to tensor and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),                    # Convert PIL Image to torch.Tensor, scale to [0,1]
    transforms.Normalize([0.5], [0.5]),       # Normalize channel mean=0.5, std=0.5 -> range [-1,1]
])

class ChunkDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # Store flattened image arrays and their labels
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve the flattened array and reshape back to 16×16 uint8 image
        arr = self.images[idx].reshape(16, 16).astype(np.uint8)
        # Convert NumPy array back to PIL Image for transforms
        img = transforms.ToPILImage()(arr)
        # Apply the transform pipeline if provided (result: Tensor shape=(1,16,16))
        if self.transform:
            img = self.transform(img)
        # Retrieve the corresponding label
        label = self.labels[idx]
        return img, label

# Initialize the dataset with images, labels, and the transform pipeline
full_dataset = ChunkDataset(images, new_labels, transform=transform)
print(f"Dataset initialized: {len(full_dataset)} samples")


# ----------------------------------------
# 7. Split into train/validation/test
# ----------------------------------------
print("Step 7: Splitting dataset...")

# Compute the total number of samples
total_len = len(full_dataset)

# Decide on split sizes:
# train_len is set to 80% of the data (rounded down).
# val_len is 10% of the data.
# test_len gets whatever remains so that the three sums exactly to total_len.
train_len = int(0.8 * total_len)
val_len   = int(0.1 * total_len)
test_len  = total_len - train_len - val_len

# Randomly partition the dataset
train_ds, val_ds, test_ds = random_split(
    full_dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)      # a torch.Generator with a fixed seed to make the split reproducible.
)
print(f"Split into train/val/test: {train_len}/{val_len}/{test_len}")


# ----------------------------------------
# 8. Device setup
# ----------------------------------------
print("Step 8: Checking CUDA...")

# Set the computation device: use GPU ("cuda") if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ----------------------------------------
# 9. Define model
# ----------------------------------------
print("Step 9: Initializing model class... ")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: input channel 1 → output channels 64, 3×3 conv, padding=1 keeps spatial size
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # Downsample by 2 (H×W → H/2×W/2)

            # Block 2: 64 → 128 channels
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # Further downsample (→ H/4×W/4)

            # Block 3: 128 → 256 channels (newly added deeper block)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             # Downsample to H/8×W/8
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                # Flatten 256×2×2 → 1024 features
            nn.Linear(256 * 2 * 2, 256), # Fully connected layer to 256 units
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),             # Dropout for regularization
            nn.Linear(256, num_classes)  # Final output layer for class logits
        )

    def forward(self, x):
        # Pass input through feature extractor, then classifier to get logits
        x = self.features(x)
        return self.classifier(x)


# ----------------------------------------
# 10. Loss function
# ----------------------------------------
print("Step 10: Setting up loss criterion... ")

# Define the loss function
criterion = nn.CrossEntropyLoss()

def train_and_validate(model, train_ds, val_ds, criterion, optimizer,
                       batch_size, num_epochs, device,
                       patience=5):
    # Prepare data loaders for training and validation
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4)

    # Initialize best validation metrics and early-stopping counter
    best_val_loss = float('inf')
    best_val_acc = 0.0
    counter = 0

    # Loop over epochs
    for epoch in range(1, num_epochs + 1):
        # --- Training phase ---
        model.train()
        for imgs, lbs in train_loader:
            # Move inputs and labels to the target device
            imgs, lbs = imgs.to(device), lbs.to(device)
            # Cast inputs to bfloat16 for mixed-precision training
            imgs = imgs.to(torch.bfloat16)

            optimizer.zero_grad()
            # Use autocast for the forward pass
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, lbs)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs, lbs = imgs.to(device), lbs.to(device)
                # Mixed-precision forward pass in validation
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(imgs)
                    l = criterion(outputs, lbs)

                # Accumulate loss and correct predictions
                val_loss += l.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == lbs).sum().item()
                val_total += lbs.size(0)

        # Compute average validation loss and accuracy
        val_loss /= val_total
        val_acc = val_correct / val_total

        # --- Early stopping check ---
        if val_loss < best_val_loss:
            # Improvement: save best metrics and reset counter
            best_val_loss = val_loss
            best_val_acc = val_acc
            counter = 0
        else:
            # No improvement: increment counter and possibly stop
            counter += 1
            if counter >= patience:
                # Stop training early if no improvement for `patience` epochs
                break

    # Return the best validation accuracy achieved
    return best_val_acc

# ----------------------------------------
# 11. Hyper-parameter grid search
# ----------------------------------------
from sklearn.metrics import roc_curve, auc
from torch.nn.functional import softmax

print("Step 11: Starting hyper-parameter search...")

# Define hyperparameter grids to search over
lrs         = [1e-2, 1e-3, 1e-4]      # Learning rates to try
batch_sizes = [16, 32, 64, 128]       # Batch sizes to try
num_epochs  = 50                      # Max epochs for each run

best_val_acc = 0.0    # Track the best validation accuracy seen so far
best_config  = None   # Store the hyperparameters for the best run
roc_data = {}         # key=(lr,bs) -> (fpr_micro, tpr_micro, auc_micro)


# Loop over all combinations of learning rates and batch sizes
for lr in lrs:
    for bs in batch_sizes:
        print(f"Testing lr={lr}, batch_size={bs}")
        
        # Initialize a fresh model and move it to the chosen device
        model = SimpleCNN(num_classes).to(device)
        # Set up the optimizer with momentum and weight decay
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Train and validate the model with current hyperparameters
        val_acc = train_and_validate(
            model, train_ds, val_ds,
            criterion, optimizer,
            bs, num_epochs, device
        )
        print(f"→ val_acc = {val_acc:.4f}\n")
        
        # If this run is the best so far, save its config and model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config  = (lr, bs)
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Calculate ROC curves and AUC
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)
        all_probs, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for imgs, lbs in test_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(lbs.numpy())
        all_probs  = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # one-hot & micro-average
        labels_onehot = np.eye(num_classes)[all_labels]
        fpr_m, tpr_m, _ = roc_curve(labels_onehot.ravel(), all_probs.ravel())
        auc_m = auc(fpr_m, tpr_m)
        
        # save to dictionary
        roc_data[(lr, bs)] = (fpr_m, tpr_m, auc_m)
        print(f"→ ROC AUC for (lr={lr}, bs={bs}): {auc_m:.3f}\n")

# Report the overall best validation accuracy and corresponding hyperparameters
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best config: learning_rate={best_config[0]}, batch_size={best_config[1]}")

# ----------------------------------------
# 12. Test set evaluation
# ----------------------------------------

print("Step 12: evaluating on test set...")

# Initialize a fresh model and move it to the selected device (GPU or CPU)
model = SimpleCNN(num_classes).to(device)
# Load the best-performing weights saved during hyperparameter tuning
model.load_state_dict(torch.load('best_model.pth'))
# Set the model to evaluation mode (disables dropout/batchnorm updates)
model.eval()

# Unpack the best learning rate and batch size found
best_lr, best_bs = best_config

# Create a DataLoader for the test set using the best batch size
test_loader = DataLoader(test_ds, batch_size=best_bs, shuffle=False, num_workers=4)

# Counters for correct predictions and total samples
test_corr = 0
test_tot = 0

# Disable gradient computation for inference
with torch.no_grad():
    for imgs, lbs in test_loader:
        # Move images and labels to the appropriate device
        imgs, lbs = imgs.to(device), lbs.to(device)
        # Forward pass: get logits from the model
        out = model(imgs)
        # Count how many predictions match the true labels
        test_corr += (out.argmax(dim=1) == lbs).sum().item()
        # Count total number of samples processed
        test_tot  += lbs.size(0)

# Compute and print the overall test accuracy
print(f"Test accuracy = {test_corr/test_tot:.4f}")


os.makedirs('root', exist_ok=True)
plt.figure(figsize=(8,6))
for (lr, bs), (fpr_m, tpr_m, auc_m) in roc_data.items():
    plt.plot(fpr_m, tpr_m,
             label=f"lr={lr}, bs={bs} (AUC={auc_m:.3f})")
plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC for All Hyperparam Configs')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.savefig('root/roc_comparison.png', dpi=300, bbox_inches='tight')
print("Saved ROC comparison to root/roc_comparison.png")
plt.show()
