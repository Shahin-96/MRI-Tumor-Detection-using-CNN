# MRI Tumor Detector

## Project Overview
In this project, I developed and implemented a Convolutional Neural Network (CNN) model for binary classification of MRI images, aiming to distinguish between tumor and healthy cases. The dataset consisted of 253 real brain MRI images, 155 with tumors and 98 without tumors. The workflow involved preprocessing and resizing images, visualizing samples, building a custom dataset class, constructing a CNN using PyTorch, training and evaluating the initial model, and finally optimizing the model with the Adam optimizer to achieve 100% accuracy.

## Steps
- Step 1: Reading and resizing Tumor and Healthy images
```Python
# Tumor Images
img_tumor = cv2.imread(file)
img_tumor = cv2.resize(img_tumor, (128, 128))
```
→ The output is (155, 128, 128, 3).
```Python
img_healthy = cv2.imread(file)
img_healthy = cv2.resize(img_healthy, (128, 128))
```
→ The output is (98, 128, 128, 3).

- Step 2: Visualizing random images
```Python
  def plot_random (tumor, healthy, num = 5):
    # random.choice(a, size=None, replace=True, p=None)
    tumor_images = tumor[np.random.choice(tumor.shape[0], num, replace=False)]
    healthy_images = healthy[np.random.choice(healthy.shape[0], num, replace=False)]

    plt.figure(figsize=(16, 9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title(f"Tumor no. {i+1}")
        plt.imshow(tumor_images[i])

    plt.figure(figsize=(16, 9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title(f"Healthy no. {i+1}")
        plt.imshow(healthy_images[i])
```
Here is the output:
<img width="1296" height="279" alt="image" src="https://github.com/user-attachments/assets/3af3bc1e-fce4-4591-8ad0-f5155fb9c16a" />
<img width="1296" height="279" alt="image" src="https://github.com/user-attachments/assets/45d75c66-847e-41ce-b9ae-fb83b9c2c017" />

- Step 3: Combining the previous two steps and forming a class MRI
```Python
class MRI():
    def __init__(self):

        # READing All images
        tumor = []
        healthy = []
        path_yes = "/content/MRI-Tumor-Detection-using-CNN/brain_tumor_dataset/yes/*.jpg"
        for file in glob.iglob(path_yes):
            img_tumor = cv2.imread(file)
            img_tumor = cv2.resize(img_tumor, (128, 128))
            b, g, r = cv2.split(img_tumor)
            img_tumor = cv2.merge((r,g,b))
            img_tumor = img_tumor.reshape((img_tumor.shape[2], img_tumor.shape[0], img_tumor.shape[1]))
            tumor.append(img_tumor)
        path_no = "/content/MRI-Tumor-Detection-using-CNN/brain_tumor_dataset/no/*.jpg"
        for file in glob.iglob(path_no):
            img_healthy = cv2.imread(file)
            img_healthy = cv2.resize(img_healthy, (128, 128))
            b, g, r = cv2.split(img_healthy)
            img_healthy = cv2.merge((r,g,b))
            img_healthy = img_healthy.reshape((img_healthy.shape[2], img_healthy.shape[0], img_healthy.shape[1]))
            healthy.append(img_healthy)

        # Our images:
        tumor = np.array(tumor, dtype=np.float32)
        print(tumor.shape)
        healthy = np.array(healthy, dtype=np.float32)
        print(healthy.shape)


        # Our labels:
        tumor_label = np.ones(tumor.shape[0], dtype=np.float32)
        healthy_label = np.zeros(healthy.shape[0], dtype=np.float32)

        # Concatenating tumor and healthy images and lables
        self.images = np.concatenate((tumor, healthy), axis = 0)
        #print(self.images.shape)
        self.labels = np.concatenate((tumor_label, healthy_label))
        #print(self.labels)

    def __getitem__(self, index):
        sample = {"image": self.images[index], "label": self.labels[index]}
        return sample

    def __len__(self):
        return self.images.shape[0]

    def normalize(self):
        self.images = self.images/255
```
- Step 4: Building Convolutional Neural Network (CNN) for Binary Classification

This PyTorch model implements a CNN that extracts spatial features from RGB images:

    1. Uses Tanh activations and average pooling for feature extraction.
    2. The fully connected network reduces features from 256 → 120 → 84 → 1.
    3. The final output is passed through a sigmoid function to produce a probability score (0–1).
    4. Suitable for binary classification tasks such as cat vs. dog detection or tumor vs. non-tumor classification.

- Step 5: Define a dataloader

  Dataloader helps us to:
  1. Wrap dataset to make it iterable.
  2. Automatically handles batching, shuffling, and parallel loading.

  ```dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=False)```

  - ``` batch_size=32```
    - Groups data into mini-batches of size 32.
    - Each iteration of the dataloader will return a tuple (inputs, labels) containing 32 MRI samples.

  - ``` shuffle=False```
    - Keeps the dataset order fixed (no randomization).
    - Useful for validation/testing where you want deterministic results.
    - For training, you usually set shuffle=True to improve generalization.
   
- Step 6: Evaluate the initial model

``` print(f'The accuracy score is: {accuracy_score(y_true, threshold(outputs))}')```

  - ``` outputs``` → Array of all predicted probabilities.
  - ``` y_true``` → Array of all true labels.

Accuracy of the initial model is: ```%38.7```.

Here is the confusion matrix of this model:
<img width="1188" height="778" alt="image" src="https://github.com/user-attachments/assets/8bb9b846-a2cb-4ea1-b3e1-e98f946e5cb1" />
⚠ As we can see only the healthy images are labled correctly.

Based on this image, all the outputs are located below the threshold (0.5), which implies that the model only predicts correctly for the healthy images and fails to detect any tumor cases.
<img width="1309" height="736" alt="image" src="https://github.com/user-attachments/assets/a2ceebd8-8ab4-4146-af94-57eef4c7b0a7" />

- Step 7: Improve the model
```Python
eta = 0.001
EPOCH = 100
optimizer = torch.optim.Adam(model.parameters(), lr=eta)
dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=True)
model.train()
```
This is our optimized model:
```
CNN(
  (cnn_model): Sequential(
    (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): Tanh()
    (2): AvgPool2d(kernel_size=2, stride=5, padding=0)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Tanh()
    (5): AvgPool2d(kernel_size=2, stride=5, padding=0)
  )
  (fc_model): Sequential(
    (0): Linear(in_features=256, out_features=120, bias=True)
    (1): Tanh()
    (2): Linear(in_features=120, out_features=84, bias=True)
    (3): Tanh()
    (4): Linear(in_features=84, out_features=1, bias=True)
  )
)
```

Calcuale the loss function:

``` Python
for epoch in range(1, EPOCH):
    losses = []
    for D in dataloader:
        optimizer.zero_grad()
        data = D['image'].to(device)
        label = D['label'].to(device)
        y_hat = model(data)
        # define loss function
        error = nn.BCELoss()
        loss = torch.sum(error(y_hat.squeeze(), label))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch+1, np.mean(losses)))
```
```
Train Epoch:  10	 Loss: 0.519284
Train Epoch:  20	 Loss: 0.352410
Train Epoch:  30	 Loss: 0.232820
Train Epoch:  40	 Loss: 0.064153
Train Epoch:  50	 Loss: 0.006510
Train Epoch:  60	 Loss: 0.002544
Train Epoch:  70	 Loss: 0.001464
Train Epoch:  80	 Loss: 0.000975
Train Epoch:  90	 Loss: 0.000702
Train Epoch: 100	 Loss: 0.000527
```

- Step 8: Evaluate the optimized model

``` print(f'The accuracy score is: {accuracy_score(y_true, threshold(outputs))}')```

Accuracy of the optimized model is: ```%100```.

Here is the confusion matrix of this model:
<img width="1188" height="778" alt="image" src="https://github.com/user-attachments/assets/4c8efc09-18f0-4323-93cb-6f1503066bbd" />
⚠ As we can see all the images are labled correctly.

This graph implies that the model correctly predicts all the cases.
<img width="1291" height="736" alt="image" src="https://github.com/user-attachments/assets/a8f13f37-8722-4aae-aebd-2ff1987f7d5f" />

  
