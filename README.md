# mHC-for-ResNet-UNet-Depth-Estimation

Monocular Depth Estimation (MDE) is a fundamental ill-posed problem in computer vision, typically addressed by encoder-decoder architectures like ResNet-UNet. While deeper encoders theoretically capture richer semantic context, in practice, standard Residual Connections suffer from signal propagation instability and feature collapse when scaling beyond 100 layers. Furthermore, the simple concatenation in U-Net skip connections lacks the ability to dynamically regulate the mixing of low-level spatial features and high-level semantic features. Drawing inspiration from the recent success of Manifold-Constrained Hyper-Connections (mHC) in Large Language Models (LLMs) and Graph Neural Networks (GNNs), we propose mHC-ResNet-UNet (Manifold-Constrained Hyper-Connections for ResNet-UNet). Our method introduces two key innovations: (1) An mHC-ResNet Encoder that expands residual streams into n parallel subspaces mixed via doubly stochastic matrices, enabling stable training of extremely deep backbones; (2) mHC-Skip Bridges, a novel mechanism that replaces standard concatenation with manifold-constrained routing, ensuring energy-preserving feature fusion. Theoretical analysis shows that our architecture guarantees non-expansive signal propagation. Experiments on the NYU Depth V2 dataset demonstrate that mHC-ResNet-UNet-152 significantly outperforms standard ResNet-based baselines, reducing Root Mean Squared Error (RMSE) by 12% while maintaining training 
stability where standard Deep ResNet diverge. 

https://www.researchgate.net/publication/400591323_Manifold-Constrained_Hyper-Connections_for_ResNet-UNet_Depth_Estimation

# 开始训练 (20 Epochs)...
正则化策略:
  - 数据增强: ✓
  - Dropout (0.1): ✓
  - Early Stopping: ✓
  - 梯度裁剪: ✓ (max_norm=1.0)
  - Weight Decay: 1e-4
  - 验证集比例: 20.0%
Epoch 20 Summary:
  Train Loss: 0.1619
  Val Loss:   0.1386
  Metrics:    a1=0.904, a2=0.986, RMSE=0.438
  LR:         1.00e-07
<img width="3000" height="1000" alt="image" src="https://github.com/user-attachments/assets/84b34a76-b65f-4e04-bbb0-6f9f35f49b06" />
