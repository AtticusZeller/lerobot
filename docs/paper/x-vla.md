---
tags:
  - paper-note
  - inbox
title: "X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model"
authors: "Zheng, Jinliang; Li, Jianxiong; Wang, Zhihao; Liu, Dongxiu; Kang, Xirui; Feng, Yuchun; Zheng, Yinan; Zou, Jiayin; Chen, Yilun; Zeng, Jia; Zhang, Ya-Qin; Pang, Jiangmiao; Liu, Jingjing; Wang, Tai; Zhan, Xianyuan"
year: 2025
arxiv: "2510.10274"
zotero_key: "VT9EFWEE"
status: unread
compiled_from: "http://arxiv.org/abs/2510.10274"
url: "http://arxiv.org/abs/2510.10274"
---

## X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model

Jinliang Zheng1,2∗, Jianxiong $\mathbf { L i } ^ { 1 , * }$ , Zhihao Wang1,3, Dongxiu Liu1, Xirui Kang1, Yuchun Feng1, Yinan Zheng1, Jiayin ${ \bf Z o u ^ { 1 } }$ , Yilun Chen2, Jia Zeng2, Ya-Qin Zhang1, Jiangmiao Pang2, Jingjing Liu1, Tai Wang2†, Xianyuan Zhan1,2† 1Institute for AI Industry Research (AIR), Tsinghua University, 2Shanghai AI Lab, 3Peking University

∗Project Co-lead & Equal Contribution. †Corresponding author: taiwang.me@gmail.com & zhanxianyuan@air.tsinghua.edu.cn

Project website: https://thu-air-dream.github.io/X-VLA/

Successful generalist Vision-Language-Action (VLA) models rely on effective training across diverse robotic platforms with large-scale, cross-embodiment, heterogeneous datasets. To facilitate and leverage the heterogeneity in rich, diverse robotic data sources, we propose a novel Soft Prompt approach with minimally added parameters, by infusing prompt learning concepts into cross-embodiment robot learning and introducing separate sets of learnable embeddings for each distinct data source. These embeddings serve as embodiment-specific prompts, which in unity empower VLA models with effective exploitation of varying cross-embodiment features. Our new X-VLA, a neat flow-matching-based VLA architecture, relies exclusively on soft-prompted standard Transformer encoders, enjoying both scalability and simplicity. Evaluated across 6 simulations as well as 3 real-world robots, our 0.9B instantiation-X-VLA-0.9B simultaneously achieves SOTA performance over a sweep of benchmarks, demonstrating superior results on a wide axes of capabilities, from flexible dexterity to quick adaptation across embodiments, environments, and tasks.

![[458433babbce2d78475abc7bd903bc7def23e74395db89065a058ee1a43fecf5.jpg]]
X-VLA A simple yet scalable cross-embodiment foundation model

![[d97b3dfe961880fa7e0a6ee66ef3e31960acb1bbaa30c975dd38d3be840a3195.jpg]]

![[3efb72ce328dc873f641e163cd27e4ba95e3add09e3413681543ff760873b898.jpg]]

![[fd6b685274b1f77014eb6fbe2e7893e6d5cf4eb88af9b16057e83497fd4e6219.jpg]]
Figure 1 | X-VLA employs distinctive learnable embeddings, referred to as soft prompt, to effectively address the heterogeneity present in cross-embodiment datasets. This approach, combined with stacking simple self-attention transformer blocks, provides a scalable solution for integrating diverse pretraining datasets and finetuning for a variety of domain-specific applications. Evaluated across 6 simulation benchmark including one autonomous driving bench and 3 real-world robots, X-VLA achieves SOTA performance over most benchmark suites and real-world robotic tasks.

## 1. Introduction

It has long been a central ambition in the robotics community (Brohan et al., 2023, 2022) to build autonomous agents that are capable of: (1) flexibly following arbitrary human instructions, and (2) dexterously operating across diverse environments as well as disparate embodiments. In light of recent success of Large Language Models (LLMs) (Achiam et al., 2023; Bai et al., 2023) and Vision-Language Models (VLMs) (Li et al., 2024a), one promising direction is to extend these advanced architectures to robotics through the incorporation of precise action modalities, thereby giving rise to Vision-Language-Action (VLA) models. The inherent expectation is that such large VLA models can marry out-of-the-box generalization with robust manipulation capabilities, from simple pick-and-place operations to complex dexterous tasks (Black et al., 2024a; Team et al., 2025; Bjorck et al., 2025).

The success of VLA models, particularly their ability to rapidly adapt to out-of-distribution (OOD) domains, hinges on pretraining with large and diverse robotics datasets that encompass a broad spectrum of robotic system architectures and a wide range of task scenarios (O’Neill et al., 2024; Lin et al., 2025; Tan et al., 2024). A key challenge here is that VLA models face substantial heterogeneity from hardware configurations to data collection strategies (Wang et al., 2024c). Such heterogeneity manifests not only in embodiment-specific action spaces (Liu et al., 2025b), but also in setup variations such as camera settings, visual domains, and task distributions (Doshi et al., 2024b; Shi et al., 2025b; Zhen et al., 2024). These various dimensions of diversity induce severe distributional shifts as well as significant semantic misalignments across embodiments, confusing the model and ultimately leading to unsatisfactory pretraining and adaptation performance (Zheng et al., 2025; Liu et al., 2025b; Doshi et al., 2024b). Existing VLA methods primarily assign distinct action decoder heads to accommodate embodiment-specific action spaces as their main focus (Physical Intelligence et al., 2025; Bjorck et al., 2025), with other critical sources of heterogeneity ineluctably overlooked. Reconciliation among these disparate configurations, however, is crucial for proprioceptive-aware reasoning and for distilling shared knowledge from heterogeneous, mixed-domain datasets, which persistently remains an unsolved problem due to: (1) the inconsistency across hardware platforms, (2) the absence of standardized data collection protocols, and (3) the inherent domain shifts that arise across embodiment and environment barriers.

We demonstrate that these obstacles can be effectively overcome with minimal human effort involved, by allowing VLA models to learn domain-specific hardware configurations through a simple Soft Prompt mechanism (Lester et al., 2021). Inspired by insights from meta-learning and multi-task learning, we recast diverse hardware configurations and data types in the robotics domain into the mold of task-specific features, which can then be effectively captured through prompt-learning techniques (Wang et al., 2023; Liu et al., 2023c; Khattak et al., 2023; Liu et al., 2023b; Wu & Shi, 2022). Specifically, to model the varying dimensions of heterogeneity as aforementioned, we assign a set of learnable embeddings to each data source as Soft Prompts. These embeddings provide heterogeneity-aware guidance for structuring the VLA representation space from early stages of feature fusion, which endows the VLA model with an enhanced capacity to exploit and consolidate cross-embodiment variations, improving generalization across different hardware and task configurations.

Formally, we introduce Soft-prompted Transformer, a generalist flow-matching–based VLA framework operating across heterogeneous platforms, called X-VLA. Through Soft Prompts, X-VLA can be guided by explicitly learned individual hardware setups to accommodate various structures of system and data. With a versatile architecture well-equipped for simultaneously encoding multi-view images, language prompts, and proprioceptive features, X-VLA allows scalable VLA training, by simply stacking standard Transformer encoders for multimodal feature fusion and precise action generation. Extensive experiments

demonstrate that Soft Prompts outperform other state-of-the-art (SOTA) methods in handling various heterogeneity dimensions. X-VLA exhibits a stable learning process and superior asymptotic performance, offering favorable scaling capabilities towards larger model size and mixed-robot datasets.

We implement X-VLA-0.9B, a 0.9B instantiation of X-VLA, trained with a carefully designed data processing and learning recipe. The overall training pipeline consists of two phases: Phase I: Pretraining. We pretrain X-VLA-0.9B on a curated heterogeneous data mixture comprising 290K episodes from Droid (Khazatsky et al., 2024), Robomind (Wu et al., 2025) and Agibot (Bu et al., 2025), spanning seven platforms across five types of robotic arms, ranging from single-arm to bi-manual setups. By leveraging soft prompts to absorb embodiment-specific variations, the model learns an embodiment-agnostic generalist policy. Phase II: Domain adaptation. X-VLA-0.9B is adapted to a deployable policy for a target domain. A new set of soft prompts is introduced and optimized to encode the hardware configuration of the novel domain, while the pretrained backbone remains frozen. With these prompts in place, the policy is then effectively specialized to the new embodiment through finetuning.

Extensive evaluations demonstrate strong adaptation across embodiments, environments and tasks, achieving new SOTA results on 6 simulation benchmarks, including one autonomous driving benchmark, and 3 real-world robot platforms. Moreover, with only 1,200 demonstrations, it excels at a dexterous cloth-folding manipulation task in the real world, achieving an average throughput of folding 1 cloth under two minutes. Remarkably, with the aid of previously learned prompts, Phase II adaptation can be efficiently realized through parameter-efficient finetuning (Hu et al., 2022) with minimal training cost. By tuning only $1 \%$ of the model parameters (9M), X-VLA-0.9B achieves $9 3 \%$ success rate on LIBERO and $5 4 \%$ on Simpler-WidowX, which is comparable to $\pi _ { 0 }$ (Physical Intelligence et al., 2025), despite requiring $3 0 0 \times$ fewer parameters (3B vs. 9M).

## 2. Preliminary

VLA models. VLA models are a class of models that unify multi-modal understanding and action generation for robotic control (Physical Intelligence et al., 2025; NVIDIA et al., 2025). Typically, VLA models are initialized from VLMs pretrained on large-scale image-text corpora, and then finetuned on robotics dataset containing expert trajectories: $\mathcal { D } = \{ \tau _ { j } \} _ { j = 1 } ^ { M }$ , ${ \tau _ { j } } = \{ ( o _ { n } , a _ { n } ) \} _ { n = 1 } ^ { N _ { j } }$ , where $o _ { n }$ denotes multimodal observation at step ?? (e.g., visual input, language instruction, proprioceptive states), and $a _ { n }$ is its corresponding expert action. The training objective is typically framed as behavior cloning, where the policy $\pi _ { \theta } ( o _ { n } )$ parameterized by $\theta$ is optimized to predict the demonstrated action chunk $A _ { n } : = [ a _ { n } , a _ { n + 1 } , … , a _ { n + T } ] ^ { T }$ where $T$ denotes the chunk size (Zhao et al., 2023; Chi et al.; Physical Intelligence et al., 2025), by minimizing a suitable supervised loss $\ell ( \cdot )$ as: $\mathcal { L } _ { \mathrm { B C } } ( \theta ) = \mathbb { E } _ { ( o _ { n } , A _ { n } ) \sim \mathcal { D } } \left[ \ell \big ( \pi _ { \theta } ( o _ { n } ) , A _ { n } \big ) \right]$ .

Flow-matching policy. Instead of directly predicting the expert action chunk ?? from observation ??, flow-matching policies commonly learns a velocity field (Lipman et al., 2023; Physical Intelligence et al., 2025; Black et al., 2025) that transports a noise sample to the target action chunk. For instance, one can generate an action ?? by starting from an Gaussian noise $A ^ { 0 } \sim N ( 0 , I )$ and iteratively refining it through a velocity field $\nu _ { \theta } ( A ^ { t } , o , t )$ parameterized by a neural network using ODE solvers such as Euler-Maruyama method: $A ^ { t + \Delta t } = A ^ { t } + \nu _ { \theta } ( A ^ { t } , o , t ) \Delta t$ . Here, $t \in [ 0 , 1 ]$ is a continuous time variable. To train the velocity field, we use the OT (optimal transport) path (Lipman et al., 2024, 2023), which aligns the velocity with the linear interpolated path between noise and expert data:

$$
\mathcal {L} _ {\mathrm {B C}} ^ {\mathrm {F M}} (\theta) = \mathbb {E} _ {t \sim \mathcal {U} (0, 1), (o, A) \sim \mathcal {D}} \Big [ \left\| \nu_ {\theta} (A ^ {t}, o, t) - (A - A ^ {0}) \right\| ^ {2} \Big ],
$$

![[5bbc35ece7eb2c026b2a1c09f37ff06eb026f42da917fc4a9496cc2d5ef443c3.jpg]]
(a) Domain-specific Action Projection

![[83af67035ed6f9ebea33766b95e321c810fd76d2f8ce6c208508a0f7840f4908.jpg]]
(b) HPT-style Projection

![[7b153f129bed1974096df51a95f0af09e0ba1ea0bb016b47df2d0baf3c445641.jpg]]
(c) Language Prompt

![[f4ba068dc0487a6298b3a93f77cdfdfb43ead023e59fcc12724c7e645c8af51b.jpg]]
(d) Soft Prompt (Ours)
Figure 2 | Comparison among four methods in handling heterogeneity in cross-embodiment training.

![[16f5b3e066c7c8e0cd9471e77810f8fabf47d50b90145bff63054e2c21829477.jpg]]

<table><tr><td></td><td>Embodiment</td><td>Freq</td><td>Source</td><td>Camera Setup</td></tr><tr><td>■</td><td>AGIBOT(48.8%)</td><td>30Hz</td><td>AGIBOT-Beta</td><td>Head/Wrist</td></tr><tr><td>■</td><td>Franka(15.8%)</td><td>15Hz</td><td>Droid</td><td>Left/Wrist</td></tr><tr><td>■</td><td>Franka(15.8%)</td><td>15Hz</td><td>Droid</td><td>Right/Wrist</td></tr><tr><td>■</td><td>Franka(6.7%)</td><td>30Hz</td><td>RoboMind</td><td>Top</td></tr><tr><td>■</td><td>Dual-Franka(0.8%)</td><td>30Hz</td><td>RoboMind</td><td>Front/Wrist</td></tr><tr><td>■</td><td>UR-5(8.7%)</td><td>30Hz</td><td>RoboMind</td><td>Top</td></tr><tr><td>■</td><td>Agilex(3.7%)</td><td>30Hz</td><td>RoboMind</td><td>Front/Wrist</td></tr></table>

![[e92fd4b35c97b99c7e0f2693ebcfefa8f910018ea10d88f89b7b38375412ed0b.jpg]]
Figure 3 | The recipe for mixed data used in pretraining experiments.
Figure 4 | Training curves for various methods of handling heterogeneity.

where $A ^ { t } = ( 1 - t ) A ^ { 0 } + t A$ , $\mathcal { U }$ is uniform distribution. By minimizing $\mathcal { L } _ { \mathrm { B C } } ^ { \mathrm { F M } }$ , the policy learns to progressively transport random noise toward expert chunks conditioned on observations.

Heterogeneity in cross-embodiment training. Training on mixed data recipes composed of $H$ heterogeneous datasets, $\mathcal { D } ^ { H } = \{ \mathcal { D } _ { i } \} _ { i = 1 } ^ { H }$ , is essential for developing generalist VLA models (Doshi et al., 2024a; O’Neill et al., 2024). Each dataset, $\mathcal { D } _ { i }$ , is collected under a specific hardware configuration, $h _ { i } \in { \mathcal { H } }$ , where $\mathcal { H }$ represents the space of possible hardware setups, such as arm kinetics, control interfaces, camera configurations, and deployment scenarios. These introduce significant heterogeneity, not only in low-level action signals and distributions, but also in high-level visual understanding, which can result in poor pretraining and adaptation if not effectively addressed (Wang et al., 2024c; Zheng et al., 2025).

## 3. Heterogeneous Soft Prompt Learning

To address heterogeneity, we conduct a comprehensive empirical study to explore potential design choices, as shown in Fig. 2. We follow Reuss et al. (2025); Bjorck et al. (2025) to establish a standard dual-system architecture as our starting point, which leverages VLMs for multimodal perception and a DiT-style decoder for action generation. In Fig. 3, we construct a heterogeneous data mixture from recent high-quality sources, including AGIBOT-beta (Bu et al., 2025), RoboMind (Wu et al., 2025), and Droid (Khazatsky et al., 2024). This dataset spans seven hardware setups across five robots, ranging from single-arm to bi-manual setups, providing sufficient scale and diversity necessary for generalist policy training. We evaluate all methods using a fully aligned training recipe to ensure a fair comparison.

See Appendix I for more training details.

(a) Domain-specific action projection. This strategy addresses heterogeneity by assigning separate projection heads at the model output to map action tokens into embodiment-specific action spaces. While this approach is widely used in prior embodied foundation models (Physical Intelligence et al., 2025; Bjorck et al., 2025; Team et al., 2025; Zheng et al., 2025; Liu et al., 2025b), its effect is limited to the final action generation stage. Consequently, it fails to encourage embodiment-aware reasoning earlier in the pipeline and overlooks other critical sources of heterogeneity, such as variations in different camera setups and task distributions. To circumvent these limitations, we identify three representative strategies that improve pretraining stability on heterogeneous datasets, as summarized in Fig. 2. We analyze these strategies in the following discussion, with additional experimental attempts reported in Appendix E.
(b) HPT-style projection. Inspired by Wang et al. (2024c), this approach aims to mitigate domain discrepancies in observation inputs and promote generalist reasoning by mapping observations from distinct domains into a shared representation space. Specifically, domain-specific projection layers are also applied on top of multi-modal inputs to align them before being fed into the backbone.
(c) Language prompts. Another strategy leverages the language reasoning capabilities of pretrained VLMs (Li et al., 2024a; Li et al.). In this case, natural language descriptions of hardware configurations $h _ { i }$ are provided as additional inputs, enabling the model to attend to embodiment-specific variations through textual descriptions explicitly.
(d) Soft prompts. Finally, we investigate a soft-prompt method that follows the meta-learning and multi-task learning philosophy (Finn et al., 2017; Liu et al., 2023c) by introducing domain-specific learnable parameters ${ P ^ { H } = \{ p _ { i } \} _ { i = 1 } ^ { H } }$ to absorb heterogeneity across data sources. $p _ { i }$ is expected to encode the underlying hardware configuration: $p _ { i } \approx \Phi ( h _ { i } )$ , where $\Phi : \mathcal H \to \mathbb R ^ { k }$ denotes a latent mapping from hardware configurations to the prompt space. Notably, $\Phi$ is not predefined by hard templates as in language prompts (c) but is randomly initialized and then implicitly optimized through end-toend training. These soft prompts are injected into the model at the early stage of action generation, automatically guiding the backbone toward embodiment-aware learning.

While (b) HPT-style projection and (c) Language prompts are conceptually appealing, they suffer from notable limitations. HPT-style projection introduces different projection layers in the middle of the observation processing, which frequently alter feature distributions and are prone to corrupting pretrained VLM representations, often resulting in unstable training dynamics. Language prompts, on the other hand, rely on carefully scripted textual descriptions of hardware configurations, which greatly hinder adaptability and scalability in practice. In contrast, soft prompts offer a flexible and scalable solution for encoding domain-specific hardware configurations. They marry the advantages of both (b) and (c), integrating smoothly with the backbone while preserving pretrained representations and eliminating the need for handcrafted annotations. The empirical results in Fig. 4 confirm that Soft Prompts consistently achieve much more robust and stable training across heterogeneous datasets.

## 4. X-VLA: Soft-Prompted Transformer Enhanced VLA Model

Building on Soft Prompts, we introduce X-VLA, a neat VLA architecture designed for stable pretraining on heterogeneous datasets and efficient adaptation on new domains. We first present the overall architectural design, followed by several key techniques for large-scale pretraining. The complete ablation path is provided in Tab. 1, which highlights the contributions of the components introduced in this section.

Table 1 | The ablation path for each components in Section 4. We evaluate the pretraining (PT) validation error and adaptation (AD) success rates on Simpler-WidowX (Li et al., 2025). Green, Red and Gray denote positive, negative, moderate effects, respectively. Bold scores are SOTA results. We can see that naively training on heterogeneous data leads to degradation. Also, as the validation error decreases during PT, the AD success rate increases progressively, demonstrating a strong correlation between the two. Therefore, we use the validation error as a proxy for PT performance throughout this paper. It is evident that every component in Section 4 contributes to positive improvements for PT.

<table><tr><td>Type</td><td>Improvements</td><td>Val Error (PT)</td><td>Acc (AD)</td></tr><tr><td>Baseline Model (w/o PT)</td><td>Florence-base + Standard DiT-base</td><td>-</td><td>4.1</td></tr><tr><td rowspan="2">Pretraining Technique (Section 4.2.1)</td><td>+Custom LR (w/o PT)</td><td>-</td><td>39.6 (+35.5)</td></tr><tr><td>+Heterogeneous PT</td><td>0.11</td><td>25.0 (-14.6)</td></tr><tr><td rowspan="3">Data Processing (Section 4.2.2)</td><td>+Action alignment</td><td rowspan="3">0.077 (-0.033)</td><td rowspan="3">50.0 (+25.0)</td></tr><tr><td>+Intension abstraction</td></tr><tr><td>+Balanced data sampling</td></tr><tr><td rowspan="3">Architecture Design (Section 4.1)</td><td>+Replace DiT with Transformer encoder</td><td>0.071 (-0.006)</td><td>47.9 (-2.1)</td></tr><tr><td>+Encoding pipeline</td><td>0.053 (-0.018)</td><td>64.6 (+16.7)</td></tr><tr><td>+Soft-prompt</td><td>0.041 (-0.012)</td><td>73.8 (+9.2)</td></tr><tr><td></td><td>+Scaling up</td><td>0.032 (-0.009)</td><td>89.6 (+15.8)</td></tr><tr><td>Finetuning Technique (Section 4.2.1)</td><td>+Two-step adaptation</td><td>0.032</td><td>95.8 (+6.2)</td></tr></table>

## 4.1. Architecture

The core idea of our design is to build a streamlined encoding pipeline for complex multimodal inputs. Beyond Soft Prompts, X-VLA handles (1) high-dimensional inputs (multi-view visuals and languages), and (2) low-dimensional states (proprioception and action tokens). Due to substantial discrepancies in both semantics and dimensionality across these modalities, we employ dedicated encoding strategies to align them effectively, after which vanilla transformer stacks suffice for scalable policy learning. Below, we detail the encoding pipeline with the complete architecture and additional design explorations are provided in Appendix C and Appendix D.

1. High-dimensional observation stream. High-dimensional inputs include multi-view images $\mathrm { I m } { \bf g } = { \bf \Phi }$ $\{ { \mathrm { i m } } g _ { i } \}$ , together with languages ?? specifying task objectives. Unlike most prior approaches (Physical Intelligence et al., 2025; Octo Model Team et al., 2024; Bjorck et al., 2025) that directly feed all views and instructions into VLMs, we disentangle the streams by assigning distinct encoders. A pretrained VLM encoder (Florence-Large (Xiao et al., 2024) in X-VLA) is used for the main vision-language stream (fixed-view and instruction), while auxiliary views such as wrist-views are processed with a shared vision backbone. This design alleviates the semantic gap between generic vision-language reasoning and embodied reasoning: fixed-camera views provide stable, informative context for high-level task reasoning; whereas wrist-camera inputs, though noisy and fast-changing, offer critical cues for finegrained manipulation and are thus encoded separately from the language stream.
2. Low-dimensional proprioceptive–action stream. The proprioceptive states $R _ { t }$ , such as joint positions and end-effector poses, provide embodiment-specific grounding for reasoning and control. The actionrelated tokens $A _ { t }$ consist of noisy action samples used for flow-matching generation. Since both $R _ { t }$ and $A _ { t }$ are compact vectors with closely related physical semantics, we concatenate them along with

their corresponding time embeddings ?? within the flow-matching pipeline. The fused embeddings are projected into a high-dimensional feature space through a lightweight linear layer, enabling early fusion with other modalities and ensuring robust proprioceptive–temporal grounding.

## 4.2. Customized Training Recipe

To fully incentivize the potential of X-VLA, we introduce a carefully designed learning engineering to enhance both stability and effectiveness for X-VLA training. We provide an overview of our training recipe and outline several key techniques that are crucial for the stable and efficient training.

## 4.2.1. Pretraining and Finetuning Pipeline

For pretraining, the backbone $\pi _ { \theta }$ and the soft prompts $P ^ { H }$ are jointly optimized under the flow-matching objective $\mathcal { L } _ { \mathrm { B C } } ^ { \mathrm { F M } }$ . Please refer to Appendix G for detailed pretraining hyperparameters. After pretraining, the backbone becomes an embodiment-agnostic foundation capable of rapid adaptation across heterogeneous robots. To deploy this model on novel domains with new hardware configurations $h _ { \mathrm { { n e w } } }$ , we propose a lightweight two-step adaptation procedure:

(1) Prompt warm-up. We introduce new sets of learnable prompt $p _ { \mathrm { n e w } } \in \mathbb { R } ^ { k }$ for $h _ { \mathrm { { n e w } } }$ . The prompt is first warmed up while keeping the pretrained weights frozen. By doing so, prompts are projected to exploit pretrained embodiment-agnostic features, offering good starts for next-round joint training.
(2) Joint policy adaptation. Then, we jointly optimize both the backbone and the warmed-up prompt, jointly adapt to new domains. This two-stage process first lets $p _ { \mathrm { { n e w } } }$ encode the hardware-specific setups of $h _ { \mathrm { { n e w } } }$ , and then finetunes the full policy for effective specialization, sharing the same philosophy used to adapt LLMs to VLMs (Liu et al., 2023a; Li et al.).

Custom learning rate (LR). A key stabilization technique in both pretraining and adaptation is the use of a reduced learning rate for the soft prompts as well as for the vision–language modules that respond for encoding visual and linguistic inputs. This adjustment reduces the risk of catastrophic drift from pretrained representations, an issue also noted by (Reuss et al., 2025; Driess et al., 2025), leading to smoother optimization during pretraining and more reliable specialization when adapting to novel embodiments. It effectively bridges the general knowledge encoded in vision–language models with the fine-grained spatial localization and action grounding required by VLA models.

## 4.2.2. Enhanced Data Processing

Aligned action representation. Actions are the core supervision signals for VLA models, with their quality directly shaping training outcomes. Therefore, we standardize the action space into end-effector (EEF) pose representation comprising: (1) the Cartesian EEF xyz position, (2) the absolute EEF rotation encoded using the Rotate6D representation (Zhou et al., 2019) to avoid the discontinuities inherent in Euler angles and quaternion representations, and (3) the discretized binary state of the gripper. The position and rotation are optimized using mean-squared-error (MSE) loss, while the gripper state is optimized with binary-cross-entropy (BCE) loss. This ensures consistency across embodiments, providing robust supervision for generalizable policy learning.

Intention abstraction through temporal downsampling. While low-level action trajectories provide the precise manipulation signals required for deployment, they are often too fine-grained and may contain lots

![[5907dc60fe7401fbdd2c604f3da13521ebc695ff9902e715b193b5a42321f39a.jpg]]

![[5a62eb47cf6fc1af0e4e457d8c1d3e94e7f1777b8492844664f0abf7787bf36a.jpg]]

![[fd8bb3a6c360b3ca64225a7b8cbec3bf1aae8c271dcbe6bfd2faa169a2b8a80c.jpg]]
Figure 5 | With increased compute, data diversity, and data volume, X-VLA can output reduced validation prediction error, which can lead to enhanced adaptation performance as discussed by Tab. 1.

of noisy movements due to human randomness, thus are not suitable for achieving high-level grounding and intention modeling for pretraining. To mitigate this issue, we temporarily downsample demonstrations to construct abstract representations of action intentions. Concretely, rather than predicting the full end-effector pose at every time step, the pipeline is designed to generate a sequence of 30 anchor points that summarize the intended trajectory over the next 4 seconds.

Balanced data sampling strategy. In contrast to the common round-robin data sampling strategy (Wang et al., 2024c), we observe that stable training requires a carefully designed data shuffling pipeline. We shuffle samples not only across different domains but also across trajectories within each domain, ensuring exposure to a diverse and balanced data mixture at every iteration. This effectively mitigates distributional bias and reduces overfitting to dominant domains, facilitating smoother convergence during large-scale pretraining.

## 5. Experiments

In this section, we conduct extensive experiments to investigate 1. Scaling behavior: Does X-VLA exhibit scaling properties along model size, data diversity, and data scale? 2. Adaptation performance: Can X-VLA specialize to novel domains with varied characteristics? 3. Interpretability: Do the soft prompts capture meaningful representations that reflect the heterogeneity of mixed data sources?

## 5.1. Scaling Experiments

First, we study the scaling behavior of X-VLA along three axes: (1) model capacity, (2) data diversity, and (3) data volume. As shown in Tab. 1, prediction errors observed during pretraining are strongly correlated with downstream adaptation performance. Therefore, we adopt the $\ell _ { 1 }$ error between predicted actions (after flow-matching denoising) and ground-truth actions on held-out validation sets as our primary evaluation metric. The corresponding results are summarized in Fig. 5, with additional training details presented in Appendix G. Notably, even at the largest tested configuration, X-VLA-0.9B (hidden size 1024, 24 Transformer blocks), trained on 290K episodes from 7 data sources, the scaling trend shows no sign of saturation. This indicates that further increases along these three axes could yield additional performance gains. Due to resource constraints, we adopt the largest configuration as the default model for subsequent experiments.

![[e259249e27707ad7415a1dbe838234b27e33ccacd225d32119a166f7009e8baf.jpg]]
Cross-embodiment Adaptation

![[28afdbe7fa7e4866d0bc0ee450406b3cc465947ff0b1e488a0b5cbcd017f1dee.jpg]]
Simpler-WidowX Simpler-Fractal Simpler-Fractal

![[1f1560da75a21b565776fb6410ed17d6708bd73b88f246e6d3e2cb36bf600886.jpg]]

![[0956b57b6b7079aad867c51da676b97dcbe4f291cc25c02028713b68891e835e.jpg]]
NAVSIM

![[3a98bd16474810200f74ca2705863d9a9d778c29847bb750857ae7eb05128ad2.jpg]]
Cross-environment and -task Adaptation
Libero

![[04f31addb7b21fb20f29ec456839edaa1caa75cd79c4a9f9c89023116130e0d3.jpg]]
RoboTwin-2.0

![[f3461d4c8a196d90cd9abb3511dfc226526612841f6794fb8964e7d3ddcf8179.jpg]]
Calvin

![[f6715c52ff45857a5c0795fbae0f0cb28add143d12a4c7052bc9c1006000a39e.jpg]]
VLABench

![[b7e103f4f61b2272a6b3e510015f2fb0c8f066dfb6ffa5d7c5f752702556532b.jpg]]
Dexterous Task PEFT
Cloth-Folding

![[9dc096a06cdb195b18e814dd2ed8322a5b70a561141d7b140cba11455ae5f14f.jpg]]
AIRBOT
Figure 6 | The evaluated setups in adaptation experiments.

Table 2 | Comparison of specialize and generalize models on simulation benchmarks

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Size</td><td colspan="3">Simpler</td><td colspan="5">LIBERO</td><td rowspan="2">Calvin ABC → D</td><td rowspan="2" colspan="2">RoboTwin-2.0 Easy Hard</td><td rowspan="2">VLABench Avg. PS</td><td rowspan="2">NAVSIM PDMS</td></tr><tr><td>VM</td><td>VA</td><td>WidowX</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg</td></tr><tr><td>LBP (Liu et al., 2025a)</td><td>0.2B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>88.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MoDE (Reuss et al., 2024)</td><td>0.4B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>94.0</td><td>-</td><td>4.01</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SuSIE (Black et al., 2024b)</td><td>1B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>76.3</td><td>-</td><td>2.69</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GHIL-Glue (Hatch et al., 2025)</td><td>1B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.69</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SpatialVLA (Qu et al., 2025)</td><td>4B</td><td>75.1</td><td>70.7</td><td>42.7</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TraceVLA (Zheng et al., 2024b)</td><td>7B</td><td>46.2</td><td>49.1</td><td>-</td><td>84.6</td><td>85.2</td><td>75.1</td><td>54.1</td><td>74.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ThinkAct (Huang et al., 2025)</td><td>7B</td><td>71.5</td><td>65.1</td><td>43.8</td><td>88.3</td><td>91.4</td><td>87.1</td><td>70.9</td><td>84.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FPC-VLA (Yang et al., 2025)</td><td>7B</td><td>78.0</td><td>65.8</td><td>64.6</td><td>86.2</td><td>87.0</td><td>92.0</td><td>82.2</td><td>86.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MemoryVLA (Shi et al., 2025a)</td><td>7B</td><td>77.7</td><td>72.7</td><td>71.9</td><td>98.4</td><td>98.4</td><td>96.4</td><td>93.4</td><td>96.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Octo (Octo Model Team et al., 2024)</td><td>0.1B</td><td>16.8</td><td>1.10</td><td>23.4</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GR-1 (Wu et al., 2023)</td><td>0.2B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.06</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Seer (Tian et al., 2025)</td><td>0.3B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>87.7</td><td>-</td><td>4.28</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UniAct (Zheng et al., 2025)</td><td>0.5B</td><td>-</td><td>-</td><td>-</td><td>77.0</td><td>87.0</td><td>77.0</td><td>70.0</td><td>77.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RDT (Liu et al., 2025b)</td><td>1B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>34.5</td><td>13.7</td><td>-</td><td>-</td></tr><tr><td>FLOWER (Reuss et al., 2025)</td><td>1B</td><td>-</td><td>-</td><td>40.0</td><td>97.1</td><td>96.7</td><td>95.6</td><td>93.5</td><td>95.7</td><td>4.53</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SmolVLA (Shukor et al., 2025)</td><td>2B</td><td>-</td><td>-</td><td>-</td><td>93.0</td><td>94.0</td><td>91.0</td><td>77.0</td><td>88.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GR00T-N1 (Bjorck et al., 2025)</td><td>3B</td><td>45.0</td><td>48.4</td><td>-</td><td>94.4</td><td>97.6</td><td>93.0</td><td>90.6</td><td>93.9</td><td>-</td><td>-</td><td>-</td><td>39.7</td><td>-</td></tr><tr><td>π0 (Black et al., 2024a)</td><td>3B</td><td>58.8</td><td>56.8</td><td>27.8</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.1</td><td>-</td><td>46.4</td><td>16.4</td><td>37.8</td><td>-</td></tr><tr><td>π0+FAST (Pertsch et al., 2025)</td><td>3B</td><td>61.9</td><td>60.5</td><td>39.5</td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>85.5</td><td>-</td><td>-</td><td>-</td><td>34.1</td><td>-</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>7B</td><td>-</td><td>-</td><td>8.30</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td>7B</td><td>63.0</td><td>54.3</td><td>31.3</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DD-VLA (Liang et al., 2025)</td><td>7B</td><td>71.2</td><td>64.1</td><td>49.3</td><td>97.2</td><td>98.6</td><td>97.4</td><td>92.0</td><td>96.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UniVLA (Wang et al., 2025a)</td><td>9B</td><td>-</td><td>-</td><td>69.8</td><td>95.4</td><td>98.8</td><td>93.6</td><td>94.0</td><td>95.4</td><td>4.41</td><td>-</td><td>-</td><td>-</td><td>81.7</td></tr><tr><td>Maximum of Existing SOTA</td><td>-</td><td>78.0</td><td>72.7</td><td>71.9</td><td>98.4</td><td>98.8</td><td>97.9</td><td>94.5</td><td>97.1</td><td>4.53</td><td>46.4</td><td>16.4</td><td>39.7</td><td>81.7</td></tr><tr><td>X-VLA (Ours)</td><td>0.9B</td><td>80.4</td><td>75.7</td><td>95.8</td><td>98.2</td><td>98.6</td><td>97.8</td><td>97.6</td><td>98.1</td><td>4.43</td><td>70.0</td><td>39.0</td><td>51.1</td><td>87.3</td></tr></table>

## 5.2. Adaptation Experiments

We present one of the most comprehensive validation studies to date, evaluating X-VLA-0.9B across 5 simulation environments and 3 real-world robotic platforms. See Appendix D for more experimental results.

Simulation benchmarks. We evaluate on Libero (Liu et al., 2024), Simpler (Li et al., 2025), VLABench (Zhang et al., 2024a), RoboTwin-2.0 (Chen et al., 2025), Calvin (Mees et al., 2022) and NAVSIM (Dauner et al., 2024). These 6 benchmarks encompass hundreds of evaluation setups, spanning single-arm, bi-manual robotic systems, autonomous driving, and assessing diverse axes of generalization, including crossembodiment, cross-environment, and cross-task adaptation. Across FIVE benchmarks, we establish a new SOTA, achieving substantial improvements over aggregated prior models. Remarkably, it attains over $9 0 \%$ success rates on several benchmarks, e.g., Simpler-WidowX $( 9 6 \% )$ , Libero $( 9 8 \% )$ , and Calvin-1st stage. To the best of our knowledge, no prior model has reported such comprehensive evaluation paired with consistently significant gains, underscoring the superior performance of X-VLA-0.9B, which can become a strong baseline for future research to develop advanced models (please refer to Appendix H for details).

![[9fd32c628ddb981b5797f172617a74cfc4c3d510789b1e58fd26c82a60b473e9.jpg]]
Figure 7 | Real-World Evaluation Results. We evaluate our X-VLA model on three distinct real-world embodiments, each under specific task setups, including simple manipulation, dexterous manipulation, and fast adaptation experiments using PEFT techniques. See Appendix J for for details

Real-world experiments. We also evaluate X-VLA-0.9B on physical robotic platforms follow the BridgeDatav2 benchmark (Walke et al., 2023), the evaluation details can be found in Appendix J and the results are reported in Fig. 7. Our X-VLA surpasses other baselines across all five tasks, each for testing distinct axis of capability, demonstrating the superior adaptability of our X-VLA.

Dexterous cloth-folding task. We introduce a challenging dexterous cloth-folding task that requires smoothing highly disordered cloth and folding it neatly. To support this effort, we build a high-quality cloth-folding dataset on the bi-manual Agilex platform, termed Soft-Fold, which consists of 1,200 trajectories collected through a carefully designed pipeline. A detailed description of both the task and the dataset is provided in Appendix F. Importantly, we will release the dataset to facilitate future research in dexterous manipulation. Leveraging this dataset for adaptation, our X-VLA-0.9B model achieves a throughput of nearly $1 0 0 \%$ success rate and 33 completed folds per hour—comparable to the closed-source $\pi _ { 0 }$ -folding model (Physical Intelligence et al., 2025), which is presumably trained on substantially larger and higher-quality datasets. For a fair comparison, we finetuned $\pi _ { 0 }$ -base and trained a ACT (Zhao et al., 2023) model from scratch on Soft-Fold, but they failed to match the throughput of X-VLA-0.9B, underscoring the strong dexterous manipulation capabilities of our model. Additional qualitative results are provided in Appendix F and showcased in our web demos: website.

Parameter efficient finetuning (PEFT) experiments. To evaluate whether the pretrained X-VLA-0.9B backbone encodes embodiment-agnostic features and can be efficiently adapted to new settings, we adopt PEFT techniques such as Low-Rank Adaptation (LoRA) (Hu et al., 2022). We test adaptation on three benchmarks: Libero, Simpler-WidowX, and a cloth-pick task on AIRBOT, a real-world embodiment unseen during pretraining. Tab. 3 and Tab. 7 show that with only 9M tunable parameters (about $1 \%$ of the full model), the backbone can be steered into a strong domain-specialized model, achieving $9 3 \%$ and $5 4 \%$ success rates on Libero and Simpler-WidowX benchmarks, respectively. These scores are comparable to fully finetuned models, e.g., $\pi _ { 0 }$ (Black et al., 2025) achieve $9 4 . 2 \%$ and $5 5 . 7 \%$ on Libero and Simpler-WidowX, respectively, demonstrating the strong adaptation capability of X-VLA.

## 5.3. In-Depth Analysis

We further demystify the effects of soft prompts through both qualitative and quantitative results. We firstly visualize the soft prompts learned after pretraining on the mixed data recipe (Fig. 3) using T-SNE (Maaten & Hinton, 2008). Fig. 8 reveals that the prompts form well-structured clusters that align

![[35f92527becea583924b806db127cf35e8ef15074a465c46daf7b8cae33c8bdc.jpg]]
Figure 8 | T-SNE visualization of soft prompts on 7 data sources.

![[62b8d41ed6da16585f14d87ce5fe8fe4d190a54ec9d5e867f78d6c814e372dfc.jpg]]
Figure 9 | Comparison of different prompts on PEFT.

Table 3 | PEFT performance comparison across benchmarks.

<table><tr><td>Methods</td><td>π0</td><td>Ours-Lora</td></tr><tr><td>#Param</td><td>3B</td><td>9M</td></tr><tr><td>Libero-Spatial</td><td>96.8</td><td>95.4</td></tr><tr><td>Libero-Object</td><td>98.8</td><td>96.6</td></tr><tr><td>Libero-Goal</td><td>95.8</td><td>96.0</td></tr><tr><td>Libero-Long</td><td>85.2</td><td>84.2</td></tr><tr><td>Simpler-WidowX</td><td>55.7</td><td>54.2</td></tr></table>

closely with different hardware configurations, indicating that they successfully capture embodimentspecific information. More excitingly, the two Franka setups (with left and right views) derived from Droid data are intermingled rather than separated, as they only differ in their designated main view. This observation suggests that soft prompts do not merely partition data sources in a brute-force manner but instead leverage cross-embodiment similarities. Further, we evaluate how pretrained soft prompts facilitate efficient adaptation to WidowX, a single-arm robot unseen in pretraining. We conduct PEFT experiments on Simpler, comparing three settings: (1) randomly initialized soft prompts kept frozen, (2) pretrained prompts from another single-arm platform (e.g., UR5) kept frozen, and (3) soft prompts adapted with our two-step adaptation mechanism. In Fig. 9, it’s no surprise that learned prompts converge faster and finally reach higher success rates, whereas random prompts lead to slower adaptation and degraded performance. However, it’s good to see that the frozen pretrained prompts offer strong transfer benefits in the early stage due to the partial similarity between UR5 and WidowX, although the inevitable domain gap limits the final performance. This highlights a promising avenue for cross-embodiment transfer: with pretraining on more diverse robotic platforms, soft prompts may enable zero-shot/few-shot generalization by retrieving prompts aligned with the closest hardware setups.

## 6. Conclusion

In this paper, we introduce X-VLA, a generalist Vision-Language-Action framework capable of operating across heterogeneous robotic platforms. Through a carefully designed training pipeline, adaptation methods, and enhanced data processing, our largest model X-VLA-0.9B achieves SOTA performance across a broad spectrum of benchmarks, setting new records with substantial gains over hundreds of evaluation setups. Remarkably, even with minimal tunable parameters, X-VLA-0.9B delivers results competitive with fully finetuned SOTA models. Importantly, empowered by Soft Prompt mechanism, X-VLA exhibits scalable training trends along all three axes, including model size, data diversity, and data volume without signs of saturation even at our largest test configuration (0.9B parameters, 290K episodes, 7 data sources). This highlights its potential for further scaling to larger models and datasets, paving the way toward more powerful VLA models. Limitations and future works are discussed in Appendix N.

## A. LLM Usage and Ethics Statement

In this paper, we employed Large Language Models (LLMs) solely for polishing the writing. No parts of the technical content, experimental results, or conclusions were generated by LLMs.

A potential ethical concern arises from the use of large-scale pretraining data, which may contain privacysensitive information or embedded biases. To mitigate this, our work is based exclusively on open-sourced robotics datasets (Bu et al., 2025; Wu et al., 2025; Khazatsky et al., 2024; O’Neill et al., 2024), all of which have undergone peer review and are widely adopted in the research community. We believe this substantially reduces the risk of privacy violations or biased data influencing our results.

Nevertheless, we encourage future researchers to exercise caution when curating data for training largescale robotics models, particularly by filtering privacy-sensitive content and addressing potential biases to ensure responsible and fair deployments of embodied AI systems.

## B. Related Work

Vision-Language-Action Models. Developing agents that can interact with the physical world requires integrating three essential modalities: visual perception to understand the environment, language comprehension to interpret task instructions, and action generation to produce executable control signals. Research on Vision-Language-Action (VLA) models (Physical Intelligence et al., 2025; Bjorck et al., 2025; Zheng et al., 2025; Kim et al., 2024; Li et al., 2024c,b) has focused on unifying these modalities to enable embodied agents to perform complex tasks conditioned on natural language commands and visual observations. These models are typically built upon Vision-Language Models (VLMs) (Team et al., 2024; Achiam et al., 2023; Li et al., 2024a; Xiao et al., 2024), which are pretrained on large-scale vision–language corpora. By inheriting strong visual grounding and generalist reasoning capabilities from VLMs, VLA models achieve impressive results on diverse manipulation tasks. More recently, researchers have recognized the inherent gap between general-purpose vision–language reasoning and embodied task requirements (Qu et al., 2025; Black et al., 2025; Mu et al., 2023; Wang et al., 2025b). To address this, various approaches have been explored, such as incorporating embodiment-specific priors (e.g., 3D spatial grounding (Qu et al., 2025; Shi et al., 2025a), instruction-following (Zheng et al., 2024a), historical reasoning (Shi et al., 2025a)) via modality injection (Qu et al., 2025), scaling up domainrelevant datasets (Black et al., 2025), adding extra supervision, or designing specialized models (Shi et al., 2025a). Nevertheless, due to the inherent limitations of current VLMs, these strategies often fall short of achieving the level of generalized reasoning required for embodied tasks with complex visual inputs. In this paper, we demonstrate that a simple yet effective modification of the input streams can better harness the generalization potential of VLMs, leading to significant performance gains.

Heterogeneous Pretraining. Training on large-scale datasets has been a key factor of recent progress in embodied foundation models (Black et al., 2024a; Kim et al., 2024). However, robotics data available for large-scale pretraining often exhibit strong heterogeneity, not only in action spaces but also in hardware setups (O’Neill et al., 2024; Niu et al., 2024). To address this, Wang et al. (2024c) proposed pretraining a standard Transformer on heterogeneous data mixtures with carefully designed architectural modifications, demonstrating promising scaling behavior and transferability. More recently, researchers have observed that pretrained VLMs already possess strong generalization ability in handling diverse vision–language inputs across domains. Consequently, the focus has shifted toward resolving heterogeneity in action spaces (Zheng et al., 2025; NVIDIA et al., 2025; Zawalski et al., 2024). Beyond manually reshaping

action spaces (Liu et al., 2025b) or modifying model architectures to accommodate heterogeneous action labels, several approaches have been proposed to align actions at the semantic level through latent action modeling or representation learning (Ye et al., 2024; Zheng et al., 2025). Nevertheless, learning policies from heterogeneous data sources requires more than aligning action labels, but embodiment-specific and proprioceptive-aware reasoning, since variations in hardware setups directly affect how observations map to actions. Simply feeding heterogeneous data into a shared backbone without explicitly modeling these embodiment-specific factors often leads to unstable training and poor cross-domain generalization. In this paper, we introduce a soft-prompt mechanism that explicitly absorbs embodiment-specific variability while preserving a shared backbone for general reasoning. By associating each hardware configuration with learnable prompt embeddings, the model can flexibly capture domain-specific knowledge, thus enabling stable large-scale pretraining.

Soft Prompt Learning. The concept of soft prompts was originally introduced in the NLP community as a parameter-efficient alternative to full model finetuning. Instead of updating all parameters of a pretrained model, a small set of learnable embeddings are prepended to the input sequence and optimized for downstream tasks (Lester et al., 2021). This approach has proven highly effective in adapting large language models to diverse tasks with minimal additional parameters, inspiring extensive research on prompt-based transfer learning across modalities (Li & Liang, 2021; Wang et al., 2022). Building on this foundation, soft prompts have been extended to multi-modal and multi-task learning settings (Liu et al., 2023c). When combined with the philosophy of meta-learning (Park et al., 2024; Gordon et al., 2019), soft prompts can serve as lightweight carriers of domainor task-specific information. By injecting learnable embeddings that guide the backbone without overwriting its pretrained representations, they provide a flexible and scalable mechanism to handle domain heterogeneity (Wang et al., 2023). In this work, we adopt the soft prompt paradigm for robotics, where heterogeneity arises from embodiment differences such as hardware configurations and action spaces. We demonstrate that soft prompts can effectively absorb embodiment-specific variability, thereby enabling the backbone to focus on learning an embodiment-agnostic generalist policy.

## C. Architecture Design

We provide further details on our proposed X-VLA architecture (Fig. 10). Specifically, we adopt Florence-Large (Xiao et al., 2024) as the vision–language encoder and employ a standard Transformer backbone with 24 layers and a hidden size of 1024 for action generation. Our design introduces a streamlined encoding pipeline that integrates soft prompts and explicitly disentangles highand low-dimensional input streams. This architecture yields improved training stability and consistently stronger validation performance.

## D. More Results

In this section, we present additional results that highlight the strengths of our approach. Specifically, we report: (1) comparisons between our model and alternative architectural designs, (2) evaluations under cross-embodiment joint training, and (3) evaluations in data-constrained settings.

![[2fae79f2caa4b395d4c57d2fae45c88308e3e8391a91ce82213dd1e9bb50ac4d.jpg]]
Figure 10 | Illustration of the detailed architecture of our model. Most parameters are shared across different embodiments, with the exception of the soft prompt and input/output linear projections for action-related tokens. These unshared parameters account for only a small fraction of the total parameters $( 0 . 0 4 \% )$ . For each dataset, the corresponding domain-specific parameters are queried. The image inputs and language instructions are processed by pretrained Vision-Language Models (VLMs). Notably, only the main view is passed through the entire VLM, while additional views, such as the wrist view, are directed only to the vision encoder. This approach helps preserve the pretrained VLM’s capability, as current VLMs have limited multi-view perception. The proprioception and flow-matching time variables are repeated and concatenated with the noise action chunk, which is then projected using its specific projections. These features, along with the soft prompt and multi-modal tokens, are processed by stacking standard self-attention transformer blocks, enabling bi-directional information flow and effective fusion of all modalities. Finally, the control tokens are projected back to action chunks using domain-specific output projections.

Table 4 | Comparison of backbone architectures on validation error. X-VLA achieves the lowest error while maintaining stable training on heterogeneous datasets.

<table><tr><td></td><td>DiT</td><td>MM-DiT</td><td>π0-Style</td><td>Ours</td></tr><tr><td>Validation Error</td><td>0.077</td><td>0.140</td><td>0.056</td><td>0.041</td></tr></table>

Table 5 | Joint adaptation to multiple embodiments. Multi-domain finetuning achieves performance comparable to, and in some cases exceeding, single-domain finetuning, demonstrating the scalability of X-VLA to heterogeneous deployment settings.

<table><tr><td></td><td>Libero-Long</td><td>Simpler-WidowX</td><td>Calvin</td></tr><tr><td>Single-domain FT</td><td>97.6</td><td>96.0</td><td>4.42</td></tr><tr><td>Multi-domain FT</td><td>98.1</td><td>93.8</td><td>4.32</td></tr></table>

## D.1. Alternative Architectural Designs

In this section, we present additional results from alternative architectural designs explored during development. While our final model adopts the X-VLA pipeline, we also implemented several commonly used backbone architectures for comparison. These baselines were evaluated under identical experimental settings, consistent with the preliminary experiments described in Appendix I.

Standard DiT Decoder. A direct application of the Diffusion Transformer (DiT) decoder (Peebles & Xie, 2023) that generates actions conditioned on multimodal features extracted by pretrained vision–language encoders. This serves as the most straightforward extension of DiT to embodied settings.

Standard MM-DiT Decoder. A multimodal variant of DiT that allocates separate parameters for different input modalities and integrates them through attention (Esser et al., 2024). We isolate the action modality from visual–language inputs. Although this design attempts to reduce the semantic gap across modalities, it often destabilizes training and leads to inferior results on heterogeneous datasets and downstream adaptation.

$\pi _ { 0 }$ -Style Decoder. Following (Black et al., 2024a), this design employs a parallel MLP-Mixer (Tolstikhin et al., 2021)–based action module alongside a pretrained VLM for vision–language processing. This leverages the compact nature of action inputs, which can be effectively represented with dense feedforward networks, but comes at the cost of added architectural complexity.

The comparative results across these backbones are summarized in Tab. 4, where our proposed X-VLA consistently achieves the best validation performance while maintaining stable optimization dynamics.

## D.2. Potential to Build Cross-embodiment Generalized Policy

Empowered by soft prompts, X-VLA enables efficient and stable training on heterogeneous datasets, effectively absorbing domain variations and fostering embodiment-agnostic policy learning. Building on this capability, we show that X-VLA can be adapted not only to a single novel embodiment but also to multiple embodiments simultaneously through joint finetuning on demonstrations from diverse data sources. Concretely, we conduct joint finetuning experiments on a mixture of downstream datasets, including Libero, BridgeData, and Calvin-ABC, which include two distinct embodiments and three hardware setups for both data collection and deployment. After joint finetuning using the same training recipe as other finetuning experiments detailed in Appendix H, we report the results in Tab. 5.

Tab. 5 shows the multi-domain adaptation results. X-VLA maintains consistently strong performance across all evaluated embodiments when adapted jointly, demonstrating its ability to scale beyond single-domain

specialization. Interestingly, joint adaptation not only preserves performance within each domain but in some cases even improves success rates compared to single-domain finetuning, suggesting positive crossdomain transfer. This indicates that the soft-prompt mechanism not only absorbs embodiment-specific variations but also enables complementary knowledge sharing across heterogeneous embodiments.

## D.3. Data-efficient Adaptation

Table 6 | Data-efficient adaptation performance of PEFT finetuned X-VLA-0.9B on Libero under limited demonstrations. Even with only 10 demonstrations, the model maintains strong performance.

<table><tr><td># demos</td><td>Libero-Spatial</td><td>Libero-Object</td><td>Libero-Goal</td><td>Libero-Long</td><td>Libero-Avg</td></tr><tr><td>50 (Full &amp; Default)</td><td>96.6</td><td>95.4</td><td>95</td><td>84.2</td><td>92.8</td></tr><tr><td>10</td><td>95.2</td><td>94.2</td><td>93.6</td><td>81.5</td><td>91.1</td></tr></table>

In this section, we investigate whether the learned embodiment-agnostic backbone can be efficiently adapted to novel embodiments under limited supervision. To this end, we finetune X-VLA-0.9B in a PEFT setup on Libero-Goal using only a small number of demonstrations. As shown in Tab. 6, the model achieves a $9 2 . 8 \%$ success rate with 50 demonstrations, and remarkably still retains a strong $9 1 . 1 \%$ success rate with only 10 demonstrations. These results highlight the data efficiency of our two-step adaptation procedure, showing that the pretrained backbone, together with soft prompts, serves as a strong prior that enables effective specialization even under extreme data scarcity.

## E. Failure Attempts for Absorbing Heterogeneity

The core motivation of this paper is to explore strategies for mitigating heterogeneity across mixed data sources and to develop a generalist, embodiment-agnostic policy. Inspired by the philosophy of meta-learning (Gordon et al., 2019), we initially approached this problem from the perspective of heterogeneous parameter learning. Concretely, we assigned distinct parameter sets for each domain, with the expectation that these domain-specific parameters could absorb domain variations while the shared backbone distilled generalizable knowledge across domains. Ultimately, we found that our proposed soft-prompt mechanism provides an effective solution to this challenge. In this section, we present two of our unsuccessful attempts, with the aim of highlighting practical pitfalls and inspiring future work in this direction.

Heterogeneous Low-rank Adapter. Beyond soft prompts, we explored the integration of other parameterefficient learning methods into heterogeneous pretraining. Specifically, we experimented with Low-Rank Adaptation (LoRA)-style modules (Hu et al., 2022), where domain-specific adapters were introduced in parallel with the shared backbone (Liu et al.). Our intuition was that these adapters could capture embodiment-specific variations with efficient parameterization, and meanwhile, the main backbone encodes embodiment-agnostic features. However, we observed that the additional adapters often conflicted with the optimization dynamics of the backbone, leading to instability and degraded generalization across domains.

Heterogeneity-guided MoE framework. We also experimented with a mixture-of-experts (MoE) approach, which has been widely used for scaling model capacity while controlling inference cost. MoE’s sparse activation mechanism (Shazeer et al., 2017) has proven effective in multi-task learning (Pham et al.,

## Soft-FOLD

The first open-source high-quality cloth-folding dataset

![[c8e75589ffd4e69418d409e1edfd06be9fe6bce1d86955f9d7622db6f0feb330.jpg]]
Stage I: Smooth-out

![[673e5dde7cb0bd691ac65d0a5de3a70a51fc438a823772b288b926dbfee22f74.jpg]]

![[f615aeb5755a7ee61e0388f987cc8b756bc2f35e2606a44a6760a49d2a275960.jpg]]

![[f5edfe5923d1d97663f5bdf2cd61ac0de5ea692ff4781c340d791f1a3d3bf9c8.jpg]]
Stage II: Fold

Carefully Designed Folding Pipeline: Daggle-Style Data Collection

![[5184dfad4f5670afc703ecd57715d3b5dccaf8e3ee8ae611aad523eb71801486.jpg]]
Agilex

![[5610859f5ec283f8b17ede80fda8df37695a683e1c93f52d7464119042181dd2.jpg]]
M

![[c702dbe89984f1e86206d322d419165bb522c4d8ed18b5b5983f6dcf99b1d005.jpg]]

![[35aa70d05dc24d7a8868b2cedd6ee9057087f386497daa9d6fbb12d1be182702.jpg]]



Varied Cloth Color and Size

1.2K episode

![[5d53b8d7d498467f9ee22d526ebeec74d492e8ca0973b6309b108cb6e4c8fed4.jpg]]

2M samples

24H Time Cost

![[011222fb84b374d46f9e242179109d65c9eee54a5ad1d3b18dab53671ce949e5.jpg]]
Figure 11 | The illustration of our proposed Soft-Fold datasets.

2023), cross-domain learning (Zhang et al., 2024b), and multi-modal robotics behavior cloning (Reuss et al., 2024). Motivated by these successes, we designed a heterogeneity-guided routing strategy that aimed to activate experts based on embodiment-specific cues. Despite its theoretical appeal, we found that the router tended to collapse, consistently routing most inputs to only a few experts while leaving others underutilized, leading to wasted capacity and only marginal performance gains. To mitigate this, we give another try to introduce load-balancing regularization (Wang et al., 2024a), but the resulting rapid switching across experts often destabilized optimization and degraded overall training dynamics.

## F. Soft-FOLD: Superior Dexterous Manipulation Model with a High-quality Cloth Folding Dataset

We provide qualitative results about our finetuned dexterous manipulation model from the pretrained X-VLA-0.9B and introduce a high-quality cloth folding dataset: Soft-FOLD, as illustrated in Fig 11.

Demonstration collecting strategy. Humans can fold clothes casually and quickly, often using a wide variety of methods in a seemingly random manner. However, this variability poses significant challenges for robotic policy learning, since different folding strategies often correspond to distinct behavioral modes, and not all strategies are equally suitable for training. To reduce the inconsistency in human demonstrations, we decompose the folding task into two stages: (1) smoothing out the cloth from a highly disordered state, and (2) folding the smoothed cloth neatly. We find that the first stage is particularly challenging, as the disordered cloth exhibits highly random dynamics, requiring the policy to capture a universal strategy for unfolding. To address this, we collect demonstrations for stage I in a repetitive

![[08d896a26440d5045714b7c3c5eb23f6f3f9cea9a27a7a22ec286fc479058556.jpg]]
Figure 12 | The folding progress of X-VLA-0.9B.

manner until meaningful keypoints, such as the two corners or two ends of the cloth emerge clearly. At that point, we employ swinging motions to complete the smoothing stage and then transition to stage II. This is critical for cloth folding, as unstructured or randomly collected demonstrations in stage I can entangle policies in inconsistent behaviors, leading to unstable learning dynamics and hindering progression to stage II. For stage II, the data collection becomes far easier, as the cloth behaves less randomly after smooth-out. On average, one full folding episode takes about 1.5 minutes, with one hour of collection yielding 20–25 episodes, including time for resetting and discarding failed attempts. The final dataset includes 1,200 episodes, as shown in Fig. 11.

DAgger-style data collection. To train long-horizon dexterous tasks such as cloth folding with limited episodes, we find it essential to adopt a DAgger-style data collection strategy (Ross et al., 2011), a practice also noted by Hu et al. (2025). Concretely, we train ACT (Zhao et al., 2023) after every 100 collected episodes, identify its failure modes, and then collect targeted demonstrations to address these failures. This iterative refinement enables us to achieve cloth-folding performance comparable to that of closed-source models that are likely trained on substantially larger datasets, using only 1,200 episodes.

Qualitative results of X-VLA-0.9B. Here, we visualize a complete folding progress of our X-VLA-0.9B in Fig. 12. One complete folding covers diverse skills, such as the simple Localization, Pick, Place, and highly dynamic Swing motion, demonstrating the challenges of the cloth-folding tasks.

## G. Pretraining Details

The pretraining of X-VLA-0.9B was carried out on 64 NVIDIA A100 GPUs with a global batch size of 1024, spanning approximately 4 days. The training followed a carefully tuned recipe to ensure stability and efficient convergence across heterogeneous datasets.

Tab. 7 summarizes the core hyperparameters used during pretraining. We adopt the AdamW optimizer with momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 5$ , a learning rate of $1 \times 1 0 ^ { - 4 }$ , and a weight decay of 0.01. Training was performed for 200K iterations with mixed-precision (bfloat16). For visual inputs, images are resized to $2 2 4 \times 2 2 4$ and augmented with mild perturbations using ColorJitter to improve generalization.

In addition to the optimizer configuration, a critical aspect of pretraining is balancing the heterogeneous datasets. Since different sources vary greatly in both scale and quality, we adopt a weighted sampling strategy combined with a carefully designed shuffling pipeline. As highlighted in Sec 4.2.1, this includes cross-domain and cross-trajectory shuffling, ensuring that the model is consistently exposed to a diverse

and balanced mixture of samples at every iteration. We find this design crucial for stabilizing optimization and preventing domain overfitting during large-scale pretraining. Tab. 8 summarizes the data sources, their available trajectories, and the sampling weights applied.

Validation set construction. We conduct open-loop validation experiments to monitor pretraining convergence and ensure fair comparisons across different architectures and methods. To guarantee that the validation loss serves as a clear and reliable X-VLA for downstream task performance, we carefully construct a dedicated validation set. Specifically, we sample trajectories from AGIBOT-beta (Bu et al., 2025) that are excluded from the training split, allowing us to better evaluate cross-embodiment knowledge sharing and generalization. The validation set spans 189 tasks, with three trajectories sampled per task. For evaluation, we report the average $\ell _ { 1 }$ error between the predicted and ground-truth trajectories.

Table 7 | Hyperparameters for pretraining.

<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Learning rate</td><td>1 × 10-4</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Optimizer momentum</td><td>β1,β2=0.9,0.95</td></tr><tr><td>Training iterations</td><td>200K</td></tr><tr><td>Model precision</td><td>bfloat16</td></tr><tr><td>Image Resize</td><td>224x224</td></tr><tr><td>Image Augmentation</td><td>ColorJitter(0.2,0.2,0.2,0)</td></tr></table>

Table 8 | Sampling weights for heterogeneous data sources during pretraining.

<table><tr><td>Data source</td><td>Num. traj</td><td>Sampling weight</td></tr><tr><td>AGIBOT</td><td>141K</td><td>0.4</td></tr><tr><td>Droid-Left</td><td>45K</td><td>0.15</td></tr><tr><td>Droid-Left</td><td>45K</td><td>0.15</td></tr><tr><td>RoboMind-Franka</td><td>19K</td><td>0.1</td></tr><tr><td>RoboMind-Dual-Franka</td><td>2K</td><td>0.03</td></tr><tr><td>RoboMind-UR</td><td>25K</td><td>0.1</td></tr><tr><td>RoboMind-Agilex</td><td>11K</td><td>0.07</td></tr></table>

## H. Finetuning Details

In this section, we provide additional training details for the adaptation experiments. Unless otherwise specified, the optimizer settings (AdamW with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 )$ ), weight decay (0.01), model precision (bfloat16), and learning rate $( 1 \times 1 0 ^ { - 4 } )$ are kept consistent with the pretraining stage. All models are adapted using our proposed two-step procedure: during the first 1,000 iterations, only the soft prompts and action heads are updated while all other parameters remain frozen; this is followed by a 1,000-iteration warm-up phase that gradually restores the learning rate to its default value for joint training.

Tab. 9 summarizes the benchmark-specific hyperparameters. For clarity, Abs EEF denotes the absolute end-effector position control interface, while Rel XYZ $^ +$ Abs Rotation refers to relative Cartesian translation combined with absolute rotation. All rotations are parameterized using the 6D representation, and the gripper state is binarized and predicted via a sigmoid activation. To maximize knowledge transfer from the pretrained backbone, we adopt aligned action representations (Abs EEF) across most downstream benchmarks. However, in the Simpler-Google benchmark, where the camera setup is deliberately altered to test robustness against visual variation, we adopt the Rel XYZ $^ +$ Abs Rotation control interface due to the sensitivity of absolute parameterizations to domain shifts in perception.

Table 9 | Finetuning hyperparameters for each downstream benchmark. Settings follow pretraining defaults unless otherwise specified.

<table><tr><td>Benchmark</td><td>Control Interface</td><td>Batch Size</td><td>Training Steps</td><td>Data Augmentation</td></tr><tr><td>CALVIN-ABC</td><td>Abs EEG</td><td>128</td><td>60K</td><td>ColorJitter</td></tr><tr><td>LIBERO</td><td>Abs EEG</td><td>128</td><td>60K</td><td>-</td></tr><tr><td>RobotWin-2.0</td><td>Abs EEG</td><td>128</td><td>60K</td><td>ColorJitter</td></tr><tr><td>VLA-Bench</td><td>Abs EEG</td><td>128</td><td>60K</td><td>ColorJitter</td></tr><tr><td>BridgeData</td><td>Abs EEG</td><td>128</td><td>60K</td><td>ColorJitter</td></tr><tr><td>FactalData</td><td>Rel XYZ + Abs Rotation</td><td>256</td><td>50K</td><td>RandomResizeCrop + ColorJitter</td></tr><tr><td>SoftFold</td><td>Abs EEG</td><td>256</td><td>400K</td><td>ColorJitter</td></tr><tr><td>PEFT experiments</td><td>Abs EEG</td><td>128</td><td>40K</td><td>ColorJitter</td></tr></table>

## I. Training Details For Preliminary Experiments

In this section, we provide additional details on the preliminary experiments. We adopt Florence-Base (Xiao et al., 2024) as the vision–language encoder and configure the backbone as a standard DiT-Base (12 Transformer layers, hidden size 768, with AdaLN conditioning) to ensure comparability. Training is conducted on the curated heterogeneous data mixture using 8 NVIDIA A100 GPUs with a global batch size of 256 for 200K iterations. Unless otherwise specified, all remaining settings (optimizer, weight decay, augmentation, and shuffling strategy) are kept consistent with the pretraining setup described in Section G. In the following, we provide more implementation details about baseline methods.

HPT-style Methods. Following (Wang et al., 2024b), we implement a cross-attention–based resampler that maps domain-specific observations into a shared representation space before feeding them into the DiT decoder. Each domain is assigned its own resampler and a dedicated action head, while the core Transformer backbone remains shared across all domains. This design aims to mitigate observation heterogeneity while keeping the reasoning backbone general.

Language Prompts. In this setting, we provide embodiment-aware textual descriptions that encode hardware configurations and camera setups for each domain. These descriptions are concatenated with the task instruction and processed by the Florence-Base encoder, enabling the model to explicitly attend to embodiment-specific variations. Tab. 10 lists the language prompt templates used across domains.

## J. Evaluation Details in Real-World Experiments

We provide detailed descriptions of our real-world evaluation setups. We adapt X-VLA-0.9B to three distinct robotic embodiments, each selected to validate different aspects of the model’s adaptability:

WidowX for pick-and-place experiments. Specifically, X-VLA-0.9B finetuned on BridgeData is directly deployed to evaluate its ability to perform robust manipulation on a compact platform. We conduct comprehensive evaluations to assess both manipulation performance and language-instruction following in real-world settings, as illustrated in Fig 13, and each task is evaluated 10 times.

AgileX for dexterous manipulation tasks. As discussed in Appendix F, this setup is designed to test dexterous, fine-grained control on a bi-manual platform equipped with wrist-mounted cameras.

Table 10 | Language prompts designed to provide embodimentand camera-specific descriptions for each domain in the preliminary experiments.

<table><tr><td>Domain</td><td>Language Prompts</td></tr><tr><td>RoboMind-Franka</td><td>Embodiment: Single Franka, Camera Setup: Top View, Freq: 30Hz</td></tr><tr><td>RoboMind-UR</td><td>Embodiment: Single UR, Camera Setup: Top View, Freq: 30Hz</td></tr><tr><td>Droid-Left</td><td>Embodiment: Single Franka, Camera Setup: Left View / Wrist View, Freq: 15Hz</td></tr><tr><td>Droid-Right</td><td>Embodiment: Single Franka, Camera Setup: Right View / Wrist View, Freq: 15Hz</td></tr><tr><td>AGIBOT</td><td>Embodiment: AGIBOT, Camera Setup: Head View / Wrist View, Freq: 30Hz</td></tr><tr><td>RoboMind-Agilex</td><td>Embodiment: AgileX, Camera Setup: Head View / Wrist View, Freq: 30Hz</td></tr><tr><td>RoboMind-Dual-Franka</td><td>Embodiment: Dual Franka, Camera Setup: Front View / Wrist View, Freq: 30Hz</td></tr></table>

![[2be19bbbe453d0d53dee4dd4c210a9f9ddc294d1a152c05e4a47ea9615f35059.jpg]]
Figure 13 | Illustration of tasks used in the WidowX pick-and-place experiments. The selected tasks evaluate different aspects of generalization—Visual, Motion, Physical, and Semantic—following the setup in OpenVLA (Kim et al., 2024).

AIRBOT for parameter-efficient finetuning experiments. AIRBOT is unseen during pretraining. We specifically collect only 200 demonstrations for a cloth-picking task, making it a challenging low-resource setting. This experiment highlights the adaptability of our two-step adaptation procedure under strict data and resource constraints.

Figure 14 shows the hardware setups for these experiments. Each embodiment is equipped with a distinct camera configuration, enabling us to construct a heterogeneous deployment environment for validation.

## K. Training Details of Baselines in Real-World Experiments

In this section, we provide the training details of the real-world baselines.

$\pi _ { 0 }$ in cloth-folding task is finetuned from the official base $\pi _ { 0 }$ model using the Soft-Fold dataset described in Appendix F. The model is trained with a total batch size of 32 across 4 A100 GPUs, requiring approximately 60 hours to complete 150,000 gradient steps.
$\pi _ { 0 }$ in PEFT experiments is finetuned from the official $\pi _ { 0 }$ base model using the official LoRA configuration. We apply LoRA with rank 16 and $\alpha = 1 6$ to both the attention and FFN modules within the PaliGemma-2B VLM. For the action expert, we use rank 32 and $\alpha = 3 2$ . Training is performed with a total batch size of 32 across 4 A800 GPUs, taking approximately 7 hours to complete 30,000 gradient steps.

ACT in cloth-folding task is trained from scratch using the Soft-Fold dataset described in Appendix F.

![[b92d1b244672cff44776372d09e57d3344da6f89a779c8837919f314e5b64494.jpg]]
(a) WidowX

![[78dbc216a07df35e01c9df9835981313429b145f0b658820b4948609a18bb117.jpg]]
(b) AgileX

![[81e60007cb26dab60c2c26b7b71209f85dbd7cabfd1e52aabb6688e73969090d.jpg]]
(c) AIRBOT
Figure 14 | Illustration of the hardware setups used in real-world experiments. We evaluate on three robotic embodiments, including WidowX, AgileX, and AIRBOT, covering diverse camera configurations and task domains to form a heterogeneous validation environment.

Table 11 | Detailed results on NAVSIM benchmark.

<table><tr><td colspan="7">NAVSIM</td></tr><tr><td>Methods</td><td>NC</td><td>DAC</td><td>EP</td><td>TTC</td><td>C</td><td>PDMS</td></tr><tr><td>Transfuser (Chitta et al., 2022)</td><td>97.7</td><td>92.8</td><td>79.2</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><td>UniAD (Hu et al., 2023)</td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><td>UniVLA (Wang et al., 2025a)</td><td>96.9</td><td>91.1</td><td>76.8</td><td>91.7</td><td>96.7</td><td>81.7</td></tr><tr><td>X-VLA (Ours)</td><td>97.5</td><td>96.5</td><td>82.2</td><td>92.9</td><td>100.0</td><td>87.3</td></tr></table>

The model is trained with a total batch size of 256 on 8 A100 GPUs. Since the model capacity of ACT is not as high as large models such as X-VLA-0.9B and $\pi _ { 0 }$ , we train ACT approximately 1M gradient steps for better training.

## L. Evaluation Details on Autonomous Driving Simulation Benchmark

We evaluate our method on the large-scale real-world autonomous driving benchmark NAVSIM (Dauner et al., 2024) using closed-loop assessment. Following the official evaluation protocol, we report the PDM score (higher indicates better performance), which aggregates five key metrics: NC (no-collision rate), DAC (drivable area compliance), TTC (time-to-collision safety), Comfort (acceleration/jerk constraints), and EP (ego progress). All methods are tested under the official closed-loop simulator, and results are averaged over the public test split. As an end-to-end VLA model, our method achieves superior performance over specialized methods designed for autonomous driving, with detailed scores reported in Tab. 11.

## M. Evaluation Details on Robotics Simulation

We report detailed scores for each simulation benchmark in Tab. 12-16.

## N. Limitations and Future Works

In this section, we discuss the limitations of our work and outline potential directions for future research.

Scaling X-VLA with broader data and model sizes. While X-VLA-0.9B achieves strong performance, its scale remains modest compared to large foundation models in the vision–language and language domains. This limitation stems primarily from computational constraints and the limited availability of high-quality robotics data. Despite our efforts to collect and curate open-source datasets (Wu et al., 2025; O’Neill et al., 2024; Bu et al., 2025), the diversity and scale of current robotics corpora still fall short of those in language or vision–language domains. Scaling X-VLA to larger capacities, either by expanding the backbone or leveraging stronger pretrained VLMs, and training on broader, more diverse robotics datasets could further enhance generalization and robustness. Such extensions also raise open questions about the scaling laws of VLA models and how embodiment-specific variability interacts with model capacity.

Enhancing supervision signals for large-scale robotics pretraining. Despite our efforts to mitigate heterogeneity across data sources and to align action spaces for generalized knowledge learning, the supervision provided by low-dimensional action labels remains inherently limited in information content. These labels, while essential for direct control, capture only a narrow view of the underlying task structure and often fail to convey higher-level reasoning, intent, or multi-step dependencies. In this work, we show that a simple temporal downsampling strategy can help abstract action intentions and thereby facilitate more efficient pretraining. However, such heuristics only partially address the problem, as they do not fundamentally enrich the supervision. Future directions include incorporating richer supervisory signals such as 3D spatial reasoning cues, physical dynamics, or intermediate subgoal annotations. Another promising avenue is leveraging self-supervised objectives from raw input streams to complement sparse action labels, thereby enhancing representation learning and improving scalability in heterogeneous, real-world robotics settings.

Towards a generalist model seamlessly deployed to downstream tasks. Our X-VLA demonstrates superior performance across various downstream tasks, showing strong adaptability under fine-tuning and efficient specialization. However, realizing the vision of a truly generalist embodied model that can be seamlessly deployed to arbitrary downstream tasks without additional engineering or retraining remains an open challenge. Currently, deployment still relies on embodiment-specific adaptation, typically involving the collection of a small number of demonstrations for post-training. While these strategies are lightweight compared to full retraining, they nonetheless introduce overhead and prevent the model from serving as a true plug-and-play solution in real-world applications. Moreover, the dependence on embodiment-specific data becomes problematic when scaling to platforms where high-quality demonstrations are scarce, expensive, or risky to collect. Future research should therefore focus on approaches that move closer to seamless deployment. Based on the empirical findings in this paper, a promising direction includes exploring unified embodiment representations: incorporating explicit embodiment-agnostic abstractions (e.g., universal kinematic descriptors, physics-informed priors) to reduce reliance on task-specific adaptation.

Table 12 | Detailed results on Simpler benchmark.

<table><tr><td colspan="14">Simpler</td></tr><tr><td colspan="5">Visual Matching (Google Robot)</td><td colspan="5">Visual Aggregation (Google Robot)</td><td colspan="4">Visual Matching (WidowX Robot)</td></tr><tr><td>Coke</td><td>Near</td><td>Open</td><td>Put</td><td>Average</td><td>Coke</td><td>Near</td><td>Open</td><td>Put</td><td>Average</td><td>Spoon</td><td>Carrot</td><td>Blocks</td><td>Eggplant</td></tr><tr><td>98.3</td><td>97.1</td><td>69.5</td><td>56.5</td><td>80.4</td><td>85.5</td><td>79.8</td><td>61.9</td><td>75.7</td><td>75.7</td><td>100</td><td>91.7</td><td>95.8</td><td>95.8</td></tr></table>

<table><tr><td rowspan="5">Libero</td><td>Libero-Spatial</td><td>98.2</td></tr><tr><td>Libero-Object</td><td>98.6</td></tr><tr><td>Libero-Goal</td><td>97.8</td></tr><tr><td>Libero-Long</td><td>97.6</td></tr><tr><td>Average</td><td>98.1</td></tr></table>

<table><tr><td rowspan="6">Calvin
(ABC→D)</td><td>1</td><td>97.1</td></tr><tr><td>2</td><td>92.6</td></tr><tr><td>3</td><td>88.5</td></tr><tr><td>4</td><td>84.4</td></tr><tr><td>5</td><td>78.8</td></tr><tr><td>Average</td><td>4.43</td></tr></table>

Table 14 | Details on Calvin. Table 15 | Details on VLABench.

<table><tr><td rowspan="5">VLABench</td><td>In Distribution</td><td>67.8</td></tr><tr><td>Cross Category</td><td>25.1</td></tr><tr><td>Common Sense</td><td>48.2</td></tr><tr><td>Semantic Instruction</td><td>63.1</td></tr><tr><td>Average</td><td>51.1</td></tr></table>

Table 13 | Details on Libero.

Table 16 | Detailed results on RoboTwin-2.0 benchmark.

<table><tr><td colspan="9">RoboTwin-2.0</td></tr><tr><td>Task</td><td>Easy</td><td>Hard</td><td>Task</td><td>Easy</td><td>Hard</td><td>Task</td><td>Easy</td><td>Hard</td></tr><tr><td>Adjust Bottle</td><td>97.0</td><td>56.0</td><td>Open Microwave</td><td>85.0</td><td>57.0</td><td>Place Object Stand</td><td>78.0</td><td>33.0</td></tr><tr><td>Beat Block Hammer</td><td>78.0</td><td>18.0</td><td>Pick Diverse Bottles</td><td>27.0</td><td>25.0</td><td>Place Phone Stand</td><td>80.0</td><td>9.00</td></tr><tr><td>Blocks Ranking RGB</td><td>79.0</td><td>26.0</td><td>Pick Dual Bottles</td><td>30.0</td><td>27.0</td><td>Place Shoe</td><td>70.0</td><td>51.0</td></tr><tr><td>Blocks Ranking Size</td><td>42.0</td><td>9.00</td><td>Place A2B Left</td><td>62.0</td><td>21.0</td><td>Press Stapler</td><td>70.0</td><td>13.0</td></tr><tr><td>Click Alarmclock</td><td>96.0</td><td>69.0</td><td>Place A2B Right</td><td>54.0</td><td>17.0</td><td>Put Bottles Dustbin</td><td>0.00</td><td>1.00</td></tr><tr><td>Click Bell</td><td>100</td><td>61.0</td><td>Place Bread Basket</td><td>75.0</td><td>39.0</td><td>Put Object Cabinet</td><td>78.0</td><td>82.0</td></tr><tr><td>Dump Bin Bigbin</td><td>94.0</td><td>59.0</td><td>Place Bread Skillet</td><td>82.0</td><td>17.0</td><td>Rotate QRcode</td><td>78.0</td><td>52.0</td></tr><tr><td>Grab Roller</td><td>99.0</td><td>66.0</td><td>Place Burger Fries</td><td>98.0</td><td>47.0</td><td>Scan Object</td><td>60.0</td><td>44.0</td></tr><tr><td>Handover Block</td><td>27.0</td><td>30.0</td><td>Place Can Basket</td><td>58.0</td><td>18.0</td><td>Shake Horizontally</td><td>99.0</td><td>100.0</td></tr><tr><td>Handover Mic</td><td>100</td><td>38.0</td><td>Place Cans Plasticbox</td><td>100</td><td>85.0</td><td>Shake Bottle</td><td>99.0</td><td>99.0</td></tr><tr><td>Hanging Mug</td><td>34.0</td><td>15.0</td><td>Place Container Plate</td><td>98.0</td><td>60.0</td><td>Stack Blocks Three</td><td>22.0</td><td>15.0</td></tr><tr><td>Lift Pot</td><td>99.0</td><td>75.0</td><td>Place Dual Shoes</td><td>98.0</td><td>28.0</td><td>Stack Blocks Two</td><td>87.0</td><td>55.0</td></tr><tr><td>Move Can Pot</td><td>50.0</td><td>44.0</td><td>Place Empty Cup</td><td>98.0</td><td>34.0</td><td>Stack Bowls Three</td><td>80.0</td><td>42.0</td></tr><tr><td>Move Pillbottle Pad</td><td>52.0</td><td>29.0</td><td>Place Fan</td><td>72.0</td><td>27.0</td><td>Stack Bowls Two</td><td>83.0</td><td>10.0</td></tr><tr><td>Move Playingcard Away</td><td>94.0</td><td>57.0</td><td>Place Mouse Pad</td><td>19.0</td><td>3.00</td><td>Stamp Seal</td><td>52.0</td><td>13.0</td></tr><tr><td>Move Stapler Pad</td><td>58.0</td><td>35.0</td><td>Place Object Basket</td><td>50.0</td><td>0.00</td><td>Turn Switch</td><td>40.0</td><td>13.0</td></tr><tr><td>Open Laptop</td><td>85.0</td><td>73.0</td><td>Place Object Scale</td><td>39.0</td><td>13.0</td><td>Average</td><td>70.0</td><td>39.0</td></tr></table>

## O. Contributions and Acknowledgments

-Model Architecture: Jinliang Zheng, Jianxiong Li
-Training: Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Zhihao Wang
-Data: Jianxiong Li, Jinliang Zheng, Xirui Kang, Zhihao Wang, Dongxiu Liu
-Simulation Evaluation: Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Zhihao Wang, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou
-Real-world Evaluation: Jianxiong Li, Zhihao Wang, Jinliang Zheng, Xirui Kang, Dongxiu Liu
-Writing: Jianxiong $\operatorname { L i } ^ { * }$ , Jinliang Zheng*, Xianyuan Zhan, Jingjing Liu, Dongxiu Liu, Zhihao Wang, Jia Zeng, Yilun Chen, Tai Wang
-Advising: Xianyuan Zhan, Tai Wang, Jia Zeng, Yilun Chen, Jingjing Liu, Jiangmiao Pang, Ya-Qin Zhang
-Team Lead: Jinliang Zheng, Jianxiong Li

This work was supported by funding from the National Key R&D Program of China (2022ZD0160201), Shanghai Artificial Intelligence Laboratory, Wuxi Research Institute of Applied Technologies, Tsinghua University (Grant No. 20242001120), Beijing Academy of Artificial Intelligence (BAAI), Horizon Robotics, and AsiaInfo. We thank Wencong Zhang for the help on robot maintenance, Yiming Meng for the help on surveying simulation benchmarks, and Yiming Chen for the help on real-world data collection.

---

### 关联

### 衍生问题
