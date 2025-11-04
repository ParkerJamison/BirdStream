import os
import numpy as np
import torch
import torch.utils.data as td
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as TF
import sklearn.model_selection as skms

# ===== Config =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './data/CUB_200_2011'
OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
num_epochs = 100
num_classes = 200
params = {'batch_size': 24, 'num_workers': 0}

# ===== Utilities from the notebook =====
def get_model_desc(pretrained=False, num_classes=200, use_attention=False):
    desc = []
    desc.append('Transfer' if pretrained else 'Baseline')
    if num_classes == 204:
        desc.append('Multitask')
    if use_attention:
        desc.append('Attention')
    return '-'.join(desc)

def pad(img, fill=0, size_max=500):
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

class DatasetBirds(tv.datasets.ImageFolder):
    """
    Wrapper for CUB_200_2011 using official splits (train_test_split.txt).
    Returns (image, label).  Optionally can return bboxes, but disabled here.
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):
        img_root = os.path.join(root, 'images')
        super(DatasetBirds, self).__init__(root=img_root,
                                           transform=None,
                                           target_transform=None,
                                           loader=loader,
                                           is_valid_file=is_valid_file)
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train

        # Read split file: which image indices are train
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = []
        with open(path_to_splits, 'r') as f:
            for line in f:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # Map indices -> filenames to keep
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as f:
            for line in f:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        # Keep only samples in split
        img_paths_cut = {'/'.join(p.rsplit('/', 2)[-2:]): i for i, (p, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]
        _, targets_to_use = list(zip(*imgs_to_use))
        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        # Bounding boxes optional (off here)
        self.bboxes = None
        if bboxes:
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = []
            with open(path_to_bboxes, 'r') as f:
                for line in f:
                    idx, x, y, w, h = map(float, line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))
            self.bboxes = bounding_boxes

    def __getitem__(self, index):
        sample, target = super(DatasetBirds, self).__getitem__(index)
        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)
        return sample, target

# ===== Transforms (data-driven, same as notebook) =====
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))

transforms_train = tv.transforms.Compose([
   max_padding,
   tv.transforms.RandomOrder([
       tv.transforms.RandomCrop((375, 375)),
       tv.transforms.RandomHorizontalFlip(),
       tv.transforms.RandomVerticalFlip()
   ]),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms_eval = tv.transforms.Compose([
   max_padding,
   tv.transforms.CenterCrop((375, 375)),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===== Datasets & split (StratifiedShuffleSplit) =====
in_dir_data = DATA_DIR
ds_train = DatasetBirds(in_dir_data, transform=transforms_train, train=True)
ds_val   = DatasetBirds(in_dir_data, transform=transforms_eval,   train=True)
ds_test  = DatasetBirds(in_dir_data, transform=transforms_eval,   train=False)

splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

train_loader = td.DataLoader(dataset=ds_train,
                             sampler=td.SubsetRandomSampler(idx_train),
                             **params)
val_loader   = td.DataLoader(dataset=ds_val,
                             sampler=td.SubsetRandomSampler(idx_val),
                             **params)
test_loader  = td.DataLoader(dataset=ds_test, **params)

# ===== Model, optimizer, scheduler (Transfer learning ResNet-50) =====
pretrained = True
model_desc = get_model_desc(num_classes=num_classes, pretrained=pretrained)

# NOTE: mirrors the notebook's constructor (pretrained=True)
# Modern torchvision API (>=0.15)
from torchvision.models import ResNet50_Weights

if pretrained:
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = tv.models.resnet50(weights=weights)
else:
    model = tv.models.resnet50(weights=None)

# Replace final classification layer
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# ===== Training loop with best-snapshot on val acc =====
best_snapshot_path = None
best_val_acc = -1.0
val_acc_history = []
print("starting training")
for epoch in range(num_epochs):
    # -- Train --
    model.train()
    train_loss = []
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    # -- Validate --
    model.eval()
    val_loss = []
    val_accs = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)
            val_loss.append(loss.item())
            # sklearn-like accuracy (no import to keep deps minimal): compute here
            pred_labels = y_pred.argmax(dim=-1)
            val_accs.append((pred_labels == y).float().mean().item())

    cur_val_acc = float(np.mean(val_accs)) if val_accs else 0.0
    val_acc_history.append(cur_val_acc)

    # Save best snapshot
    if cur_val_acc > best_val_acc:
        if best_snapshot_path is not None and os.path.isfile(best_snapshot_path):
            os.remove(best_snapshot_path)
        best_val_acc = cur_val_acc
        best_snapshot_path = os.path.join(OUT_DIR, f'model_{model_desc}_ep={epoch}_acc={best_val_acc:.4f}.pt')
        torch.save(model.state_dict(), best_snapshot_path)

    scheduler.step()

    if (epoch == 0) or ((epoch + 1) % 10 == 0):
        print(f"Epoch {epoch+1} |> Train loss: {np.mean(train_loss):.4f} | Val loss: {np.mean(val_loss):.4f} | Val acc: {cur_val_acc:.4f}")

# ===== Test with best checkpoint =====
if best_snapshot_path is not None:
    model.load_state_dict(torch.load(best_snapshot_path, map_location=DEVICE))

model.eval()
true, pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x).argmax(dim=-1)
        true.extend(y.cpu().tolist())
        pred.extend(y_pred.cpu().tolist())

# Compute accuracy (no sklearn to keep it minimal)
true_arr = np.array(true)
pred_arr = np.array(pred)
test_accuracy = float((true_arr == pred_arr).mean()) if len(true_arr) else 0.0
print(f"Test accuracy: {test_accuracy:.3f}")
print(f"Best snapshot: {best_snapshot_path}")
