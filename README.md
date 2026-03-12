# Spiking Neural Networks for Autonomous Driving Perception

## Project Context

This project explores **Spiking Neural Networks (SNNs)** as energy-efficient, low-latency alternatives to traditional ANNs for **autonomous driving perception** tasks (object detection, segmentation, depth estimation). SNNs process information through discrete spikes (like biological neurons), enabling event-driven computation that is fundamentally more efficient than continuous-valued ANNs.

---

## 1. What Are Spiking Neural Networks?

SNNs are the **third generation of neural networks**. Unlike ANNs that use continuous activations, SNNs communicate via binary spikes over time. A neuron accumulates input until a threshold is reached, then fires a spike and resets.

### Neuron Models (most to least common in literature)

| Model | Description | Trainable? |
|-------|-------------|------------|
| **LIF** (Leaky Integrate-and-Fire) | Standard model: membrane potential leaks over time, fires at threshold | Fixed decay |
| **PLIF** (Parametric LIF) | LIF with learnable leak/threshold parameters | Yes |
| **I-LIF** (Integer-valued LIF) | Integer training + spike-driven inference; used in SpikeYOLO | Yes |
| **ECS-LIF** (Extracellular Space LIF) | Models extracellular dynamics; used in ECSLIF-YOLO | Yes |
| **GLIF** (Generalized LIF) | Multi-compartment LIF with richer dynamics | Yes |

### Key Properties
- **Temporal coding**: Information encoded in spike timing, not just rates
- **Sparse activation**: Only active neurons consume energy (most are silent)
- **Event-driven**: No computation when no input changes
- **Timesteps**: SNNs process input over T discrete timesteps (typically T=1-6 for modern models)

---

## 2. Training Methods

### A. Direct Training with Surrogate Gradients (preferred, SOTA)
Since spikes are non-differentiable, surrogate gradient functions (e.g., arctangent, sigmoid) approximate the gradient during backpropagation. Combined with BPTT (Backpropagation Through Time) for temporal dynamics.
- **Pros**: End-to-end optimization, fewer timesteps needed (T=4-6), better accuracy
- **Cons**: Memory-intensive (stores states across timesteps), requires careful tuning

### B. ANN-to-SNN Conversion
Train a standard ANN, then convert activations to spike rates.
- **Pros**: Leverages mature ANN training, easy to implement
- **Cons**: Historically required thousands of timesteps; 2025 methods reduce this to T=4-8 with ~1% accuracy loss
- **Key methods**: Threshold balancing (CLTB), SlipReLU (zero conversion error), Negative Spikes (1/8 timesteps of prior methods)

### C. Hybrid Approaches
Combine direct training with conversion, or mix SNN and ANN layers.

---

## 3. SOTA SNN Architectures for Perception

### 3.1 Object Detection

| Model | Year/Venue | Architecture | Dataset | Result | Notes |
|-------|-----------|-------------|---------|--------|-------|
| **SpikeYOLO** | ECCV 2024 (Best Paper Candidate) | I-LIF neuron + Meta SNN blocks + simplified YOLOv8 | COCO | **66.2% mAP@50, 48.9% mAP@50:95** | 5.7x energy efficiency vs ANN; 67.2% mAP@50 on Gen1 (beats ANN by 2.5%) |
| **ECSLIF-YOLO** | 2025 | ECS-LIF neuron + YOLO | BDD100K/KITTI | **91.7% mAP** | Near-ANN accuracy, >90% energy reduction |
| **EMS-YOLO** | ICCV 2023 | EMS-ResNet + full-spike residual blocks | COCO | 73.5% mAP@50 | First directly-trained SNN detector |
| **SpikeDet** | 2025 | MDSNet backbone + SMFM neck | Various | SOTA | Addresses local firing saturation |
| **Spiking-YOLO** | AAAI 2020 | ANN-to-SNN converted Tiny YOLO | COCO | Low accuracy | Pioneer work; 280x less energy but thousands of timesteps |
| **Hybrid Spiking ViT** | ICML 2025 | SNN + Transformer | Event data | SOTA | Cutting-edge hybrid for event cameras |
| **Brain-Inspired SNN** | CVPR 2025 | Multi-scale SNN | Various | SOTA | Energy-efficient multi-scale detection |

### 3.2 Classification (backbone quality indicator)

| Model | Year/Venue | Dataset | Accuracy | Params |
|-------|-----------|---------|----------|--------|
| **SGLFormer** | 2025 | ImageNet-1k | **83.73% top-1** | 64M |
| **QSD-Transformer** | ICLR 2025 | ImageNet-1k | **80.3% top-1** | 6.8M |
| **Max-Former** | 2024 | ImageNet-1k | 82.39% top-1 | 64M |
| **SpikFormer** | 2023 | ImageNet-1k | 74.81% top-1 | — |

### 3.3 Segmentation
- **Spike-Driven Transformer v2 / Meta-SpikeFormer**: First SNN backbone for unified dense prediction (classification + detection + segmentation). Uses Spike-Driven Self-Attention (SDSA) — replaces multiplication with mask-and-addition, achieving **87.2x lower computation energy** vs vanilla self-attention.
- **Spike-driven Lane Segmentation** (IEEE 2024): LiDAR point clouds encoded into spikes for lane area segmentation.

### 3.4 Depth Estimation
- **StereoSpike**: SNN-based stereo depth on MVSEC dataset
- **SpikeStereoNet** (2025): Brain-inspired stereo depth from spike streams with synthetic + real-world datasets
- **Self-Supervised Event-Based Monocular Depth** (2025): Event cameras + SNNs for monocular depth

### 3.5 End-to-End Autonomous Driving
- **SAD (Spiking Autonomous Driving)** — NeurIPS 2024: **First unified end-to-end SNN for AD**
  - Perception: Multi-view cameras → spatiotemporal BEV
  - Prediction: Dual-pathway spiking neurons for future state forecasting
  - Planning: Trajectory generation considering occupancy, traffic rules, comfort
  - Result: +7.43% mean IoU over VED on nuScenes
  - GitHub: `ridgerchu/SAD`

---

## 4. Event Cameras + SNNs (Natural Pairing)

Event cameras (Dynamic Vision Sensors / DVS) output asynchronous per-pixel brightness changes — a perfect match for SNNs.

### Event Camera Hardware
| Sensor | Resolution | Key Feature |
|--------|-----------|-------------|
| Prophesee GEN1 | 304×240 | Automotive-grade |
| Prophesee GEN4 / EVK4 | 1280×720 | High-res, 1Mpx |
| Sony-Prophesee IMX636 | 1280×720 | Hybrid (events + frames) |
| iniVation DAVIS346 | 346×260 | Combined DVS + APS frames |

### Event Camera Advantages for Driving
- **Microsecond temporal resolution** (vs 33ms for 30fps cameras)
- **High dynamic range**: 120+ dB (vs 60 dB conventional) — handles tunnels, sunlight, night
- **No motion blur** at any speed
- **Low power**: ~10-30 mW
- **Sparse output**: Only changed pixels generate events

---

## 5. Neuromorphic Driving Datasets

| Dataset | Year | Sensor | Resolution | Size | Task |
|---------|------|--------|-----------|------|------|
| **Gen1** | 2020 | Prophesee GEN1 | 304×240 | 39h, 255k labels | Detection (cars + pedestrians) |
| **1Mpx** | 2020 | Prophesee GEN4 | 1280×720 | 14.65h | Detection (high-res) |
| **DSEC** | 2021 | Prophesee GEN3.1 | 640×480 | Various | Stereo, optical flow, depth |
| **DDD20** | 2020 | DAVIS346 | 346×260 | 51h, 4000km | End-to-end driving |
| **DDD17** | 2017 | DAVIS346 | 346×260 | 12h+ | End-to-end driving |
| **MVSEC** | 2018 | DAVIS346 | 346×260 | Various | Stereo depth, odometry |
| **eTraM** | 2024 | Prophesee EVK4 | 1280×720 | 10h, 2M boxes | Traffic monitoring (8 classes) |
| **N-CARS** | 2018 | ATIS | 304×240 | 12,336 samples | Car classification |
| **OpenEvDET** | CVPR 2025 | Various | Various | Various | Unified detection benchmark |

---

## 6. SNN Frameworks & Libraries

| Framework | Maintainer | Backend | Hardware Support | Best For |
|-----------|-----------|---------|-----------------|----------|
| **SpikingJelly** | Peking University | PyTorch | CuPy acceleration | Full-stack: training + deployment, most papers use this |
| **snnTorch** | — | PyTorch | Limited | Education, prototyping, multiple neuron models |
| **Norse** | — | PyTorch | — | Research, bio-plausible models |
| **Intel Lava** | Intel | Custom | Loihi 2 native | Deployment to Loihi hardware |
| **Nengo** | ABR | Multi | Multiple chips | Cross-platform neuromorphic dev |

---

## 7. Neuromorphic Hardware

| Platform | Developer | Scale | Power | Notes |
|----------|-----------|-------|-------|-------|
| **Loihi 2** | Intel | 1M neurons/chip | <1W | Graded spikes (32-bit), programmable, Lava SDK |
| **Hala Point** | Intel | 1.15B neurons (1152× Loihi 2) | 2600W max | 15 TOPS/W, largest neuromorphic system |
| **TrueNorth** | IBM | 1M neurons | ~70mW | Fixed-function, 20 pJ/synaptic event |
| **Akida** | BrainChip | Configurable | milliwatts | Production-ready, CNN-to-SNN converter, TF compatible |
| **SpiNNaker 2** | U. Manchester | Scalable | — | ARM-based, automotive AI target |

---

## 8. SNN vs ANN: Quantitative Comparison

| Dimension | SNN (best known) | ANN (typical) | SNN Advantage |
|-----------|-----------------|---------------|---------------|
| ImageNet top-1 | ~83.7% | ~90%+ | ANN wins by ~6% |
| COCO mAP@50:95 | ~49% | ~55%+ | ANN wins by ~6% |
| Power (chip-level) | <1W (Loihi) | 100-400W (GPU) | **100-1000x less** |
| Inference latency | 3-5 ms | 30-100 ms | **10-30x faster** |
| Energy per inference | 0.27 mJ | 29.8 mJ | **110x less** |
| Event camera processing | Native | Requires frame conversion | **SNN wins** |
| Training ecosystem | Emerging | Mature | ANN wins |
| Hardware availability | Research/niche | Ubiquitous | ANN wins |
| Driving dataset (Gen1 mAP@50) | 67.2% (SpikeYOLO) | 64.7% (equivalent ANN) | **SNN wins (+2.5%)** |

---

## 9. Industry Adoption

| Company | Activity |
|---------|----------|
| **Volvo** | Neuromorphic LiDAR processing in autonomous trucks (50ms → 3ms latency) |
| **Ford** | Partnership with BrainChip (Akida) |
| **Valeo** | Partnership with BrainChip for automotive edge AI |
| **Prophesee** | Event camera manufacturer, automotive-grade sensors |
| **85%+ European OEMs** | Running neuromorphic pilot programs |

---

## 10. Key Challenges

1. **Accuracy gap**: SNNs still ~6% behind ANNs on ImageNet/COCO; gap widens for complex tasks
2. **Training difficulty**: Surrogate gradients are approximations; BPTT is memory-intensive
3. **Timestep-accuracy tradeoff**: Fewer timesteps = lower latency but lower accuracy (sweet spot: T=4-6)
4. **Hardware scarcity**: No mass-produced automotive neuromorphic chips yet
5. **Immature tooling**: Frameworks less mature than PyTorch/TensorFlow ecosystem
6. **Scale variance**: SNN detectors struggle with objects of varying sizes
7. **Limited pretrained models**: No equivalent of ImageNet-pretrained ResNet ecosystem for SNNs
8. **Simulation gap**: Most energy claims are theoretical, not measured on deployed hardware
9. **Sensor fusion**: Combining event cameras with LiDAR/radar/RGB in SNN pipelines is underexplored
10. **Dataset limitations**: Neuromorphic driving datasets are smaller and fewer than frame-based ones

---

## 11. Key Papers (Chronological)

| Paper | Venue | Year | Contribution |
|-------|-------|------|-------------|
| Spiking-YOLO | AAAI | 2020 | First SNN object detector, 280x energy reduction |
| EMS-YOLO | ICCV | 2023 | First directly-trained SNN detector (surrogate gradients) |
| Tr-Spiking-YOLO | Neural Networks | 2023 | End-to-end trainable, only 5 timesteps |
| SpikeYOLO (I-LIF) | ECCV | 2024 | 66.2% mAP@50 COCO, 5.7x efficiency, Best Paper Candidate |
| SAD | NeurIPS | 2024 | First unified end-to-end SNN for autonomous driving |
| SNN for AD: A Review | Eng. App. of AI | 2024 | Comprehensive survey |
| ECSLIF-YOLO | Scientific Reports | 2025 | 91.7% mAP on BDD100K/KITTI |
| SpikeDet | arXiv | 2025 | Addresses local firing saturation |
| QSD-Transformer | ICLR | 2025 | 80.3% top-1, 6.8M params |
| Spiking Transformer (ST-Attention) | CVPR | 2025 | Spatial-temporal attention for spiking domain |
| Brain-Inspired SNN Detection | CVPR | 2025 | Energy-efficient multi-scale detection |
| Hybrid Spiking ViT | ICML | 2025 | SNN + Transformer for event-camera detection |
| SpikeStereoNet | arXiv | 2025 | Stereo depth from spike streams |
| OpenEvDET | CVPR | 2025 | Unified event-based detection benchmark |
| Decision SpikeFormer | CVPR | 2025 | Spike-driven transformer for decision making |

---

## 12. Key GitHub Repositories

- `BICLab/SpikeYOLO` — SpikeYOLO (ECCV 2024)
- `BICLab/EMS-YOLO` — EMS-YOLO (ICCV 2023)
- `ridgerchu/SAD` — Spiking Autonomous Driving (NeurIPS 2024)
- `fangwei123456/spikingjelly` — SpikingJelly framework
- `jeshraghian/snntorch` — snnTorch framework
- `lava-nc/lava` — Intel Lava framework
- `Event-AHU/OpenEvDET` — Event-based detection benchmark (CVPR 2025)
- `open-neuromorphic` — Community resources and benchmarks

---

## 13. Recommended Research Directions

1. **Spiking YOLO variants** on automotive datasets (BDD100K, KITTI, nuScenes) — most practical near-term impact
2. **Hybrid SNN-ANN architectures** — SNN frontend for event data + ANN backend for complex reasoning
3. **Event camera + frame camera fusion** with SNN processing
4. **ANN-to-SNN conversion** of existing pretrained detectors (YOLOv8/v11, RT-DETR)
5. **End-to-end SNN perception** following the SAD (NeurIPS 2024) paradigm
6. **Deployment benchmarking** on actual neuromorphic hardware (Loihi 2, Akida)
7. **Spike-driven Transformers** for multi-task perception (detection + segmentation + depth)
