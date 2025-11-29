# Deep-Learning-LAB
# Deep Learning Experiments








A comprehensive collection of deep learning experiments from fundamentals to advanced applications

                     View Experiments • Setup • Technologies

📋 Table of Contents

About
Repository Structure
Experiments
Setup & Installation
Technologies Used

🎯 About

This repository contains hands-on implementations of deep learning concepts, covering everything from basic neural network components to advanced transfer learning techniques.
Each experiment is designed to build practical understanding through implementation and analysis.

📁 Repository Structure

📦 Deep-Learning-Experiments


┣ 📂 Exp_1

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images


┣ 📂 Exp_2

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images


┣ 📂 Exp_3

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images


┣ 📂 Exp_4

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images


┣ 📂 Exp_5

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images


┣ 📂 Exp_6

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images


┣ 📂 Exp_7

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images


┣ 📂 Exp_8

┃ ┣ 📓 experiment.ipynb

┃ ┣ 📂 datasets

┃ ┗ 📂 images

┗ 📄 README.md


<!-- BEGIN: Experiments Grid (paste into README.md) -->
<style>
  /* Container */
  .exp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 18px;
    margin: 18px 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  }

  /* Card */
  .exp-card {
    background: #0f1720;             /* dark panel */
    color: #e6eef8;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 6px 14px rgba(2,6,23,0.5);
    display: flex;
    flex-direction: column;
    min-height: 180px;
  }

  .exp-card header {
    display:flex;
    align-items:center;
    gap:10px;
    margin-bottom: 12px;
  }

  .exp-icon {
    font-size: 20px;
    width:36px;
    height:36px;
    display:inline-grid;
    place-items:center;
    border-radius:8px;
    background: linear-gradient(180deg,#1f2937,#111827);
    border: 1px solid rgba(255,255,255,0.04);
  }

  .exp-title {
    font-weight:700;
    font-size:18px;
    margin:0;
  }

  .exp-sub {
    color: #cbd5e1;
    font-size:13px;
    margin: 6px 0 12px 0;
    line-height:1.35;
  }

  /* topics box */
  .topics {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.03);
    padding: 12px;
    border-radius:6px;
    font-family: "Courier New", Courier, monospace;
    font-size:13px;
    color:#cfe6ff;
    margin-bottom:12px;
    white-space:pre-wrap;
  }

  .exp-desc {
    flex:1;
    color:#c9d7e6;
    font-size:14px;
    margin-bottom:12px;
  }

  .exp-actions {
    display:flex;
    gap:8px;
    flex-wrap:wrap;
  }

  .btn {
    display:inline-block;
    text-decoration:none;
    padding:8px 12px;
    font-weight:600;
    border-radius:6px;
    font-size:13px;
    border: none;
  }

  .btn-view { background:#0ea5e9; color:white; }
  .btn-data { background:#fbbf24; color:#111827; }

  /* small responsive tweak */
  @media (max-width:420px){
    .exp-card { padding:14px }
  }
</style>

<div class="exp-grid">

  <!-- Experiment 1 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">📊</div>
      <div>
        <h3 class="exp-title">Experiment 1</h3>
        <div style="font-size:13px;color:#bcd7ee">Comparative Study of Deep Learning Frameworks</div>
      </div>
    </header>

    <div class="topics">
Topics:
├── TensorFlow Implementation
├── Keras Implementation
├── PyTorch Implementation
└── Framework Comparison
    </div>

    <div class="exp-desc">
Compare TensorFlow, Keras, and PyTorch by implementing linear regression. Analyze code verbosity, API design patterns, and debugging capabilities across frameworks.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp1.ipynb" target="_blank" rel="noopener">View Experiment</a>
    </div>
  </article>

  <!-- Experiment 2 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🔧</div>
      <div>
        <h3 class="exp-title">Experiment 2</h3>
        <div style="font-size:13px;color:#bcd7ee">Building Neural Networks from Scratch</div>
      </div>
    </header>

    <div class="topics">
Topics:
├── Single Neuron (AND Gate)
├── Feedforward Network (XOR)
├── MLP with Backpropagation
└── Activation & Loss Functions
    </div>

    <div class="exp-desc">
Implement a single neuron for the AND gate, extend to a feedforward network (XOR/AND), and build a full MLP with forward/backpropagation and training mechanisms.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp2.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1WUZKaKSo0CeBNe_SCKvAUwW_XzS6QIJs?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 3 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🎯</div>
      <div>
        <h3 class="exp-title">Experiment 3</h3>
        <div style="font-size:13px;color:#bcd7ee">Classification with DL Frameworks</div>
      </div>
    </header>

    <div class="topics">
Topics:
├── Dataset: MNIST / Fashion-MNIST
├── Data Preprocessing
├── Model Training & Validation
└── Performance Evaluation
    </div>

    <div class="exp-desc">
Train classification models with deep learning frameworks using MNIST / Fashion-MNIST. Perform preprocessing, training/validation, and evaluate performance.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp3.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1PJSrpHZR95mgXz4_flWS3n98qVxMpPtx?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 4 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🖼️</div>
      <div>
        <h3 class="exp-title">Experiment 4</h3>
        <div style="font-size:13px;color:#bcd7ee">Transfer Learning for Image Classification</div>
      </div>
    </header>

    <div class="topics">
Topics:
├── Pretrained Models
├── Feature Extraction
├── Fine-Tuning Strategies
└── Cats vs Dogs / CIFAR-10
    </div>

    <div class="exp-desc">
Use pretrained models for feature extraction and fine-tuning. Apply strategies on Cats vs Dogs or CIFAR-10 to speed up training and improve accuracy.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp4.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1rD1kYwyt9u_qxmjFCYdbCT3hVCwJKqt4?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 5 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">⚙️</div>
      <div>
        <h3 class="exp-title">Experiment 5</h3>
        <div style="font-size:13px;color:#bcd7ee">Training Deep Networks (Loss, Backprop & Opt)</div>
      </div>
    </header>

    <div class="topics">
a. Activation Functions: Sigmoid, ReLU, Tanh, Softmax
b. Loss Functions: MSE, Cross-Entropy
c. Implement Backpropagation
d. Compare Optimizers: SGD, Momentum, Adam
    </div>

    <div class="exp-desc">
Implement and visualize activations and loss functions, implement backpropagation and compare optimizer convergence on a small dataset.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp5.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1_BzIBlw98-jvg4rrk4yBqYVA6HnubA-x?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 6 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">📚</div>
      <div>
        <h3 class="exp-title">Experiment 6</h3>
        <div style="font-size:13px;color:#bcd7ee">Implementation of MLP</div>
      </div>
    </header>

    <div class="topics">
Implement a Multilayer Perceptron (MLP) architecture for supervised tasks, training and evaluation.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp6.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1madQFK2jC27xGE7thicgqDTD5w97dy-d?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 7 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🧠</div>
      <div>
        <h3 class="exp-title">Experiment 7</h3>
        <div style="font-size:13px;color:#bcd7ee">Implementing CNN</div>
      </div>
    </header>

    <div class="topics">
Topics:
├── Convolution
├── Pooling
└── Feature Maps Visualization
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp7.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1TCT6LMwVNUCvQ3qEK7dFlB3UHfm9sXi6?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 8 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🖌️</div>
      <div>
        <h3 class="exp-title">Experiment 8</h3>
        <div style="font-size:13px;color:#bcd7ee">CNN with Data Augmentation</div>
      </div>
    </header>

    <div class="topics">
Implement CNN with augmentation strategies to improve generalization on image classification tasks.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp8.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/140tB6DKKBjv0W31LRhw61UFPYBMrOA-2?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 9 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🎯</div>
      <div>
        <h3 class="exp-title">Experiment 9</h3>
        <div style="font-size:13px;color:#bcd7ee">CNN Object Detection</div>
      </div>
    </header>

    <div class="topics">
Topics:
├── Object Detection Fundamentals
├── CNN Architecture for Detection
├── Bounding Box Prediction
└── Training & Evaluation
    </div>

    <div class="exp-desc">
Implement CNN-based object detection with bounding box regression & classification; build detection pipelines and evaluate performance.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp9.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1cTrlI4NDTCUkzbrZ2gJLAOgQUOwAT1Xv?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 10 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🔍</div>
      <div>
        <h3 class="exp-title">Experiment 10</h3>
        <div style="font-size:13px;color:#bcd7ee">R-CNN Object Detection</div>
      </div>
    </header>

    <div class="topics">
Introduction to Object Detection and basic implementation using R-CNN approach.
    </div>

    <div class="exp-actions">
      <!-- If you later add a link, replace # with the URL -->
      <a class="btn btn-view" href="#" target="_blank" rel="noopener">View Experiment</a>
    </div>
  </article>

  <!-- Experiment 11 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">✂️</div>
      <div>
        <h3 class="exp-title">Experiment 11</h3>
        <div style="font-size:13px;color:#bcd7ee">Image Segmentation (UNet)</div>
      </div>
    </header>

    <div class="topics">
Introduction to image segmentation and implementation using the UNet model.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="#" target="_blank" rel="noopener">View Experiment</a>
    </div>
  </article>

  <!-- Experiment 12 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🔁</div>
      <div>
        <h3 class="exp-title">Experiment 12</h3>
        <div style="font-size:13px;color:#bcd7ee">Autoencoders (Reconstruction & Compression)</div>
      </div>
    </header>

    <div class="topics">
Objective: Design and evaluate a standard autoencoder for image reconstruction and dimensionality reduction.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="#" target="_blank" rel="noopener">View Experiment</a>
    </div>
  </article>

  <!-- Experiment 13 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🧩</div>
      <div>
        <h3 class="exp-title">Experiment 13</h3>
        <div style="font-size:13px;color:#bcd7ee">Variational Autoencoders (VAE)</div>
      </div>
    </header>

    <div class="topics">
Objective: Implement a VAE, learn latent distributions, and generate images. Analyze class-wise latent spaces.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp13.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1sK69f_Wp1dHU3R8L6BG-PiaMtE0fsl33?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

  <!-- Experiment 14 -->
  <article class="exp-card">
    <header>
      <div class="exp-icon">🎨</div>
      <div>
        <h3 class="exp-title">Experiment 14</h3>
        <div style="font-size:13px;color:#bcd7ee">Generative Adversarial Networks (GANs)</div>
      </div>
    </header>

    <div class="topics">
Objective: Develop and train a GAN for realistic image generation. Compare GANs with VAEs for visual fidelity and diversity.
    </div>

    <div class="exp-actions">
      <a class="btn btn-view" href="https://github.com/Kritikargv17/DL_LAB_500120185_KRITIKA_RAGHAV/blob/main/Exp14.ipynb" target="_blank" rel="noopener">View Experiment</a>
      <a class="btn btn-data" href="https://drive.google.com/drive/folders/1IwShryUGtGadSTJKqYSrm7CASCnius2K?usp=drive_link" target="_blank" rel="noopener">Dataset</a>
    </div>
  </article>

</div>
<!-- END: Experiments Grid -->


Matplotlib	3.x	Data Visualization

Scikit-Learn	1.x	Machine Learning Tools

# Thank You!
