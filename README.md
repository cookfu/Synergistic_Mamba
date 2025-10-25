# Abstract
Low-light images suffer from severe degradations including diminished visibility, distorted colors, and loss of detail. While existing methods struggle to holistically address these intertwined issues, we draw inspiration from the Fourier domain, which inherently disentangles an image into global amplitude (illumination) and phase (structure) components. However, current Fourier-based methods are limited by their inability to model long-range dependencies between frequency components. Simultaneously, the emerging State Space Models (SSMs), particularly Mamba, exhibit remarkable prowess in capturing long-range correlations but remain under-explored for vision tasks without content-aware mechanisms. In this paper, we propose Synergistic Mamba, a novel framework that unifies frequency and spatial processing within a collaborative, dual-stage architecture powered by Mamba. Our key insight is to leverage the global modeling strengths of the frequency domain for foundational correction and the local precision of the spatial domain for detail refinement, with Mamba serving as the universal computational engine for both. Specifically, our Frequency-domain coarse enhancement stage employs a Euclidean-scanned Mamba to globally and interactively recover illumination and suppress noise by modeling dependencies across all frequencies. In parallel, the spatial-domain refinement stage utilizes an attention-guided Mamba to meticulously enhance local textures and edges. Crucially, the two stages are not sequential but deeply collaborative, featuring crossdomain interaction modules that allow information to flow synergistically between global frequency guidance and local spatial details. Extensive experiments on multiple benchmarks demonstrate that Synergistic Mamba achieves new state-of-the-art performance, effectively restoring clear, vivid, and detailed images from challenging low-light inputs.
# Train
```shell
bash train.sh
```
# Test
```shell
bash test.sh
```
