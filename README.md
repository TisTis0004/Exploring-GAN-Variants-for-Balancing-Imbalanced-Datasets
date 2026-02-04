# Exploring GAN Variants for Balancing Imbalanced Datasets

This project explores the use of **Generative Adversarial Networks (GANs)** to address the **class imbalance problem** in image classification. Multiple GAN variants are implemented and compared based on their ability to generate synthetic minority-class samples and improve downstream classification performance.

The study focuses on evaluating how different GAN architectures affect **recall, F1-score, and overall robustness** when rebalancing a severely imbalanced dataset.

---

## 📌 Problem Statement

In many real-world machine learning applications, datasets are **not evenly distributed across classes**, leading to biased models that perform poorly on underrepresented classes. While overall accuracy may appear high, models trained on imbalanced data often suffer from **low recall and poor generalization** for minority classes.

This project investigates whether **GAN-based synthetic data generation** can be used as an effective data-level solution to improve classification performance in imbalanced settings.

---

## 📂 Dataset & Imbalance Setup

- **Dataset:** FashionMNIST  
- **Image Size:** 28×28 grayscale images  
- **Number of Classes:** 10  

To simulate a real-world imbalance scenario:
- Class **“trousers” (class 1)** was selected as the minority class
- Training samples for this class were reduced to **20% of the original size**
- All other classes retained their full data distribution

Two imbalance strategies were implemented:
1. **Static imbalanced datasets** saved and loaded for controlled experiments  
2. **Dynamic imbalancer utility** allowing flexible class reduction during training

This setup represents an **extreme imbalance case**, where minority-class performance becomes the main challenge.

---

## 🧠 GAN Architectures Implemented

All GAN models were implemented in **PyTorch** and trained for **500 epochs** using CUDA-enabled GPUs.  
A **100-dimensional noise vector** was used as input across all models.

### 1. Vanilla GAN
- Fully connected generator and discriminator
- Binary Cross Entropy loss
- Adam optimizer (lr = 0.0003)
- Batch size: 64
- Prone to training instability and mode collapse

### 2. Deep Convolutional GAN (DCGAN)
- Convolutional generator and discriminator
- Batch normalization and ReLU / LeakyReLU activations
- BCE with logits loss for numerical stability
- Adam optimizer (lr = 0.0002)
- Batch size: 128
- Improved sample quality and training stability

### 3. Wasserstein GAN with Gradient Penalty (WGAN-GP)
- Same convolutional structure as DCGAN
- Critic-based training with Wasserstein loss
- Gradient penalty (λ = 10) to enforce Lipschitz constraint
- Multiple critic updates per generator step
- Most stable training behavior and highest-quality samples

---

## ⚙️ Training Challenges & Design Choices

- **Mode collapse** (especially in Vanilla GAN) mitigated using:
  - Label smoothing
  - Noise injection into real samples
- Learning rates and loss functions were tuned per architecture
- Higher-resolution datasets (e.g., PneumoniaMNIST) were excluded due to computational constraints

---

## 🧪 Classifier Setup

To evaluate the impact of GAN-based balancing:

- **Model:** Pre-trained ResNet-18
- **Approach:** Transfer learning
  - Backbone frozen
  - Final classification layer fine-tuned
- **Optimizer:** Adam  
- **Learning rate:** 0.0001  
- **Weight decay:** 0.00001  

The classifier was trained under **four scenarios**:
1. Original imbalanced dataset (baseline)
2. Dataset balanced using Vanilla GAN
3. Dataset balanced using DCGAN
4. Dataset balanced using WGAN-GP

All evaluations were performed on the **original (unaltered) test set** to ensure fair comparison.

---

## 📊 Evaluation Metrics

Because accuracy alone is misleading in imbalanced settings, multiple metrics were used:
- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Confusion matrices

Special emphasis was placed on **recall and F1-score**, as correctly identifying minority-class samples is critical in imbalanced problems.

---

## 📈 Results Summary

Key findings from the experiments:

- **Baseline (imbalanced)** achieved high accuracy but suffered from low recall
- **Vanilla GAN** slightly improved recall but degraded overall performance
- **DCGAN** significantly improved recall (+7.98%) with moderate trade-offs in precision
- **WGAN-GP** achieved the **best overall balance**, with:
  - +8.92% recall improvement
  - Highest gains in F1-score and accuracy
  - More stable and realistic generated samples

While DCGAN and WGAN-GP improved minority-class sensitivity, both introduced a **precision–recall trade-off** due to increased false positives.

---

## 🧠 Key Observations

- GAN-based augmentation can meaningfully improve minority-class performance
- Model architecture plays a crucial role in synthetic data quality
- Advanced GANs (especially WGAN-GP) outperform simpler variants
- Synthetic augmentation should be evaluated using **downstream task metrics**, not visual quality alone

---

## 🏁 Conclusion

This project demonstrates that **GAN-based dataset balancing** is a viable strategy for improving classifier robustness in imbalanced learning scenarios. Among the tested models, **WGAN-GP** showed the most consistent performance improvements, particularly in recall and F1-score.

However, improvements come with trade-offs, and the choice of GAN architecture should depend on application requirements, especially when balancing sensitivity versus precision.

---

## 🔮 Future Work

Potential extensions include:
- Conditional GANs for label-controlled generation
- Hybrid balancing strategies combining GANs with sampling methods
- Experiments on higher-resolution or real-world datasets

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

---

## 📎 Notes

This project was developed as part of the **Special Topics in Artificial Intelligence** course and focuses on practical experimentation with generative models in imbalanced learning settings.
