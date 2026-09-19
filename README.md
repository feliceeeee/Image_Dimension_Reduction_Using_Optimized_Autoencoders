# Image Dimension Reduction Using Optimized Autoencoders

A deep learning project that compresses and reconstructs grayscale images (ships and helicopters from the Overhead-MNIST dataset) using convolutional autoencoders. The project applies baseline and architecturally optimized autoencoder designs, hyperparameter tuning, and structural similarity (SSIM) evaluation to measure reconstruction quality.

## Highlights

- Image preprocessing: grayscale conversion, resizing to 28x28, and pixel normalization
- Stratified train/validation/test split (80/10/10) preserving class balance
- Baseline convolutional autoencoder using Conv2D, MaxPooling2D, and UpSampling2D
- Architecturally optimized autoencoder with Batch Normalization, He Normal initialization, L2 regularization, increased latent dimension, and Conv2DTranspose-based learned upsampling
- Hyperparameter tuning using Huber Loss, ReduceLROnPlateau, EarlyStopping, and ModelCheckpoint
- Reconstruction quality evaluation using Structural Similarity Index (SSIM)
- Visual and quantitative comparison between baseline and modified architectures

## Data

The dataset consists of grayscale images from two classes drawn from the Overhead-MNIST dataset:
- Ship: 8,012 images
- Helicopter: 5,906 images

Total: 13,918 images, resized to 28x28 pixels and normalized to a [0, 1] pixel range.
Data was split into training (80%), validation (10%), and test (10%) sets using stratified sampling to preserve class proportions across splits.

## Model Experiments
Two convolutional autoencoder architectures were developed and compared:
- Baseline Autoencoder: A simple encoder (Conv2D + MaxPooling2D + Dense latent layer of 128 dimensions) and decoder (Dense + Reshape + UpSampling2D + Conv2D), trained with MSE loss, Adam optimizer, and early stopping.
- Modified Autoencoder: A deeper architecture incorporating:
  - Batch Normalization after each convolutional and dense layer for training stability
  - He Normal weight initialization suited to ReLU activations
  - L2 regularization (1×10⁻⁵) to reduce overfitting
  - An increased latent dimension of 192 for a better compression-information tradeoff
  - Conv2DTranspose layers in the decoder for learned upsampling instead of UpSampling2D + Conv2D
  - Hyperparameter tuning: Adam optimizer with learning rate 3×10⁻⁴, Huber Loss (more robust to large reconstruction errors than MSE), ReduceLROnPlateau, EarlyStopping (patience=20), and ModelCheckpoint to retain the best-performing weights

## Results

The Modified Autoencoder achieved a 24.28% improvement in Mean SSIM over the baseline, along with a higher minimum SSIM and a lower standard deviation, indicating both better and more consistent reconstruction quality across the test set. Visual comparisons confirmed that the modified model preserved object shapes and fine details more accurately than the baseline, which struggled particularly on harder-to-reconstruct samples. These improvements are attributed to the combination of architectural enhancements (batch normalization, learned upsampling via Conv2DTranspose, higher latent capacity) and refined training strategy (Huber loss, adaptive learning rate scheduling).

Modified Autoencoder was selected as the final model based on its superior and more consistent reconstruction performance.

## How to Run

The notebook was developed using Python and Jupyter Notebook (originally run on Google Colab).

1. Clone this repository:

```
git clone https://github.com/feliceeeee/Image_Dimension_Reduction_Using_Optimized_Autoencoders.git
```

2. Install the required libraries:

```
pip install numpy pandas matplotlib pillow scikit-learn tensorflow scikit-image jupyter
```

3. Ensure the dataset (Overhead-MNIST ship and helicopter images) is extracted to: `data/version2/train` dan `data/version2/test`, each containing `ship/` and `helicopter/` subfolders
4. Open the notebook: `notebook/Image Dimension Reduction Using Optimized Autoencoders.ipynb`
5. Run all cells to perform data preprocessing, baseline and modified autoencoder training, and SSIM-based evaluation.
