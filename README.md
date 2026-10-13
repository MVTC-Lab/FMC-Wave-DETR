FMC-Wave-DETR: Cross-Stage Frequency-Guided Learning for Oriented Tiny Object Detection in Remote Sensing Images
JianHong Zhang, LieJun Wang, and Feng Yan
School of Computer Science and Technology, Xinjiang University, Urumqi, China

FMC-Wave-DETR is an end-to-end detector for tiny, densely distributed, and arbitrarily oriented objects in remote sensing images. It preserves fine-grained structures through detail-preserving downsampling and Haar wavelet modeling, calibrates frequency evidence with higher-level semantics, and shares the calibrated evidence across multiscale feature alignment and decoder query initialization. The decoder uses a fixed budget of 1,000 queries, whose allocation across feature levels adapts to each scene.

Potential application scenarios include:

Detecting densely packed tiny objects in aerial and satellite imagery.
Localizing objects with arbitrary orientations using oriented bounding boxes.
Supporting aerial surveillance, urban management, disaster assessment, and emergency response.
On AI-TOD-R, FMC-Wave-DETR achieves 12.2% AP and 32.1% AP50, improving over RT-DETR-L-OBB by 6.3 and 9.7 percentage points, respectively. On SODA-A, it achieves 36.6% AP and 73.9% AP50, improving both metrics by 5.5 percentage points over the same baseline. These results follow the dataset-specific protocols reported in the paper.

Paper: Cross-Stage Frequency-Guided Learning for Oriented Tiny Object Detection in Remote Sensing Images
Source code: MVTC-Lab/FMC-PRNet, the repository address announced in the manuscript for the future code release.

Overview of FMCWaveDETR

Framework overview from Fig. 2 of the paper. Semantically calibrated frequency masks guide both feature alignment and scene-conditioned allocation of the fixed query budget.

The framework consists of the following components:

DP-S2D: Detail-Preserving Space-to-Depth downsampling rearranges local spatial samples into channels before channel mixing and spatial refinement.
HSG: Haar-Based Spatial-Frequency Guidance decomposes and reconstructs features and produces frequency-mask logits at P2, P3, and P4.
Cross-scale semantic calibration: Bounded, gradient-detached top-down residual calibration introduces higher-level semantics into lower-level frequency masks to improve target-background discrimination.
ACSA-FG: Frequency-guided semantic-geometric alignment uses the calibrated masks to modulate geometric features and deformable-convolution responses while retaining semantic context.
SC-SBFG: Scene-conditioned scale-balanced query initialization prioritizes candidates and redistributes 1,000 queries across P2-P4 according to scene evidence.
Oriented decoder: A six-layer deformable Transformer decoder refines object centers and sizes, while an independent periodic angle branch predicts orientation using doubled-angle components.
Requirement
The experiments reported in the paper use the following environment:

Component	Reported version or hardware
PyTorch	2.6.0
CUDA	12.4
cuDNN	9.1
GPU	NVIDIA A40
These entries describe the reported experimental environment. The manuscript does not specify the Python version, the complete dependency list, installation commands, or implementation-specific training entry points.

Training
1. Dataset preparation and preprocessing
Download the datasets and their oriented annotations from the official project resources:

AI-TOD-R: Eight oriented object categories, with extremely small targets and dense object distributions. Use the trainval split for training and the test split for evaluation, following the paper's protocol.
SODA-A: Nine oriented object categories in high-resolution aerial scenes. Use the official dataset splits and original-image-level evaluation protocol.
AI-TOD-R preprocessing

Use an input resolution of 800 x 800.
Apply horizontal flipping with probability 0.5 during training.
Use no multiscale training or additional augmentations in the reported setting.
SODA-A preprocessing

Crop original images into 800 x 800 patches with a 150-pixel overlap.
Sample positive and empty patches at a ratio of 3:1.
Resize patches to 1200 x 1200 and pad them to 1216 x 1216.
Apply random flipping and scale-translation augmentation during training.
During inference, map patch predictions back to original image coordinates and merge them before official evaluation.
2. Training from scratch
Unless otherwise specified in the paper, locally reproduced models are trained from scratch with AdamW and a fixed random seed of 42.

Setting	AI-TOD-R	SODA-A
Training epochs	40	200
Global batch size	16 across two GPUs	24
Initial learning rate	1e-4	5e-5
Weight decay	1e-5	1e-4
Learning-rate schedule	Step schedule at epochs 24, 32, and 38	Cosine schedule
Warm-up	500 updates, linear, initial ratio 1/3	5 epochs
Training precision	FP32	Automatic mixed precision (AMP)
Gradient-norm clipping	35	Not specified in the manuscript
Both settings use P2-P4 features, 1,000 decoder queries, and a six-layer decoder. SC-SBFG changes the per-level allocation while preserving the total query budget. When scale-wise scene scores are nearly indistinguishable, it uses the deterministic fallback allocation [500, 300, 200] for P2, P3, and P4.

Use the released implementation's dataset paths, configuration files, and training commands when available; the manuscript does not define these interfaces.

3. Joint detection and frequency-mask supervision
Training combines the oriented detection objective with auxiliary frequency-mask supervision:

L = L_det + 0.50 * L_mask

L_det = 1.00 * L_cls + 5.00 * L_L1 + 0.75 * L_ang
      + 2.00 * L_GWD + 0.75 * L_PIoU + 0.25 * L_cor
The detection terms correspond to classification, center-size regression, periodic angle prediction, Gaussian-distribution discrepancy, geometric overlap, and cyclic corner consistency. The same objective is applied to the final decoder output, auxiliary decoder layers, and denoising queries.

Calibrated frequency masks are supervised by OBB-derived rotated anisotropic Gaussian targets. Each level combines 0.7 BCE loss and 0.3 Dice loss, with P2-P4 weights of [0.80, 1.00, 1.00]. Mask logits are evaluated in FP32 for numerical stability.

Evaluation and Results
All AP values below are percentages reported in the paper. AP is averaged over IoU thresholds from 0.50 to 0.95 in increments of 0.05; AP50 and AP75 use fixed IoU thresholds of 0.50 and 0.75.

Main results
Dataset	Method	AP	AP50	AP75	Parameters (M)
AI-TOD-R	RT-DETR-L-OBB	5.9	22.4	0.8	33.8
AI-TOD-R	FMC-Wave-DETR	12.2	32.1	5.1	30.4
SODA-A	RT-DETR-L-OBB	31.1	68.4	23.2	33.8
SODA-A	FMC-Wave-DETR	36.6	73.9	31.6	30.4
Source: Tables II and VIII of the paper. RT-DETR-L-OBB is the authors' oriented adaptation of RT-DETR. The AI-TOD-R rows are controlled scratch reproductions using the same local data split, 800 x 800 inputs, 40-epoch schedule, and official test evaluation. SODA-A results use its dataset-specific protocol. Published benchmark results obtained under other training settings are reported separately in the paper.

Performance by object scale
Dataset	APvt	APt	APs	APm	APeS	APrS	APgS	APN
AI-TOD-R	7.8	12.5	14.4	16.1	-	-	-	-
SODA-A	-	-	-	-	14.1	30.8	48.2	38.8
For AI-TOD-R, APvt, APt, APs, and APm denote very-tiny, tiny, small, and medium objects. For SODA-A, APeS, APrS, APgS, and APN denote extremely small, relatively small, generally small, and normal-size objects under the official evaluator.

Component ablation on AI-TOD-R
Configuration	AP	AP50	AP75	APvt
RT-DETR-L-OBB baseline	5.9	22.4	0.8	2.2
+ DP-S2D	8.0	29.6	1.3	3.4
+ DP-S2D + HSG	8.4	31.2	1.4	3.4
+ DP-S2D + HSG + SC-SBFG	10.2	31.4	3.2	5.3
Full FMC-Wave-DETR (+ ACSA-FG)	12.2	32.1	5.1	7.8
Source: Table III. All configurations follow the same 40-epoch training protocol.

Query coverage and computational cost
With the same 1,000-query budget, GT coverage on AI-TOD-R increases from 90.7% for RT-DETR-L-OBB to 93.9% for FMC-Wave-DETR. A ground-truth object is covered when at least one selected query center falls inside its oriented box (Table VI).

Method	Parameters (M)	FLOPs (G)	FPS
RT-DETR-L-OBB	33.8	185.2	28.94
FMC-Wave-DETR	30.4	338.0	22.33
Source: Table X. FLOPs use 800 x 800 inputs and the convention FLOPs = 2 x MACs, excluding preprocessing and postprocessing. FPS is measured on an NVIDIA A40 with batch size 1 in FP32.

FMC-Wave-DETR reduces the parameter count and improves detection accuracy, while frequency-aware multiscale processing increases computational cost and reduces throughput relative to the baseline.

Models
The paper reports FMC-Wave-DETR results on both AI-TOD-R and SODA-A but does not provide checkpoint download links. Model weights, full configurations, and executable training and evaluation instructions should be obtained from the source-code repository announced in the manuscript when released.

Contact
For questions about the work, contact the corresponding author, Feng Yan, at yf@xju.edu.cn.

BibTeX
If this work is useful for your research, please cite the manuscript. Final publication metadata is not provided in the supplied paper, so the entry below uses a manuscript citation.

@misc{zhang_fmc_wave_detr,
  title  = {Cross-Stage Frequency-Guided Learning for Oriented Tiny Object Detection in Remote Sensing Images},
  author = {Zhang, JianHong and Wang, LieJun and Yan, Feng},
  note   = {Manuscript}
}
