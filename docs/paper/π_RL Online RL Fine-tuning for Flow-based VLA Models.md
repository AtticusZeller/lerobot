---
tags:
  - paper-note
  - inbox
title: "$\pi_\texttt{RL}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models"
authors: "Chen, Kang; Liu, Zhihao; Zhang, Tonghe; Guo, Zhen; Xu, Si; Lin, Hao; Zang, Hongzhi; Li, Xiang; Zhang, Quanlu; Yu, Zhaofei; Fan, Guoliang; Huang, Tiejun; Wang, Yu; Yu, Chao"
year: 2025
arxiv: "2510.25889"
status: unread
compiled_from: "http://arxiv.org/abs/2510.25889"
url: "http://arxiv.org/abs/2510.25889"
---

# $\pi _ { \tt R L }$ : Online RL Fine-tuning for Flow-based Vision-Language-Action Models

Kang Chen2,6,∗ Zhihao Liu3,6,∗ Tonghe Zhang4,∗,♯ Zhen Guo5 Si Xu5 Hao Lin5 Hongzhi Zang1 Xiang Li5 Bingwen Wei1 Jiakai Zhou1 Quanlu Zhang5 Zhaofei Yu2 Guoliang Fan3 Tiejun Huang2 Yu Wang 1 † Chao Yu 1 †

![image](images/4f0815bca2ebb5d8a0b8ebd4f28484f66c0d23c7ebaf167233de92867e921979.jpg)

https://github.com/RLinf/RLinf https://huggingface.co/RLinf

![image](images/b667224c46eedc41b731ee83e6820b0d1532e4a7b865befb82d9bb58d9e13fb1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph "Flow-based VLAs"
        A["π₀"] --> B["pre-trained VLA"]
        B --> C["Sig,IP (480M) + Comma (2.68)"]
        D["ViT"] --> E["ViT"]
        F["ViT"] --> G["ViT"]
        H["Fold shirt"] --> I["Folding shirt"]
    end

    subgraph "A Vision-Language-Action Flow Model for General Robot Control"
        J["π₀.₅"] --> K["subtask prediction"]
        K --> L["Pick up the pillow"]
        L --> M["clean the bedroom"]
        M --> N["&quot;high-level prompt"]
        N --> O["low-level command"]
        P["continuous actions"] --> Q["action expert (300M)"]
        Q --> R["noise"]
    end

    subgraph "A Vision-Language-Action Model with Open-World Generalization"
        S["π₀.₅"] --> T["&quot;subtask prediction"]
        T --> U["Pick up the pillow"]
        U --> V["clean the bedroom"]
        V --> W["&quot;high-level prompt"]
        W --> X["low-level command"]
        Y["continuous actions"] --> Z["action expert (300M)"]
        Z --> AA["noise"]
    end
```
</details>

![image](images/52d770d8f4b13e986f4e5ab16f6dcf19bf4341579f98bfdffae8d16ce3cee6ca.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Pretraining"] --> B["Large-scale Datasets"]
    B --> C["Specific Task Datasets"]
    C --> D["Performance Gain"]
    D --> E["Model Weight"]
    E --> F["Buffer Data"]
    F --> G["I. Policy Rollout"]
    F --> H["II. Actor Update"]
    G <--> I["Online Interaction with Env"]
    H <--> J["PP0 GRPO RL with the Buffer"]
    I --> K["π₀/π₀.₅"]
    J --> K
    K --> L["III. RL"]
    L --> M["Performance Gain"]
    M --> N["πRL"]
    N --> O["Data"]
```
</details>

![image](images/14a8fb2fa58a157630e89b97b66d510b3a319d272213633e438fc1821afa86e3.jpg)

<details>
<summary>radar</summary>

| Modality | π₀.₅-SFT | π₀.₅-Flow-SDE | π₀.₅-Flow-Noise |
| -------- | -------- | ------------- | --------------- |
| Instruct | 0.8      | 0.9           | 0.9             |
| Vis-Img  | 0.7      | 0.8           | 0.8             |
| Tex-0.3  | 0.6      | 0.7           | 0.7             |
| Tex-0.5  | 0.5      | 0.6           | 0.6             |
| Whole-0.3| 0.4      | 0.5           | 0.5             |
| Whole-0.5| 0.3      | 0.4           | 0.4             |
| M-Carrot (Test) | 0.2   | 0.3           | 0.3             |
| M-Plate (Train) | 0.1    | 0.2           | 0.2             |
| M-Plate (Test) | 0.2    | 0.3           | 0.3             |
| Pos-Chg  | 0.3      | 0.4           | 0.4             |
</details>

![image](images/063c0e5373ee1bac52c40bb2e9ef83ea8296f9f3fe11db9d6d6bbe2366bf4006.jpg)

<details>
<summary>bar</summary>

In-Distribution RL Training
| Model | SFT (%) | Flow-SDE (%) | Flow-Noise (%) |
| :--- | :--- | :--- | :--- |
| LIBERO | 77.1 | 97.9 | 98.3 |
| ManiSkill | 40.1 | 90.9 | 89.7 |
| MetaWorld | 43.8 | 70.7 | 66.1 |
| CALVIN | 61.3 | 87.0 | 84.5 |
</details>

![image](images/3597b8d712dd6a7cc5a16b1fa7cc691aa83670747625454caf161cbb8d19373e.jpg)

<details>
<summary>line</summary>

| Data Efficiency | Value |
| --------------- | ----- |
| SFT + RL        | 94.0% |
| One-shot SFT    | 43.9% |
</details>

Figure 1. $\pi _ { \mathrm { R L } } \dot { . }$ An online RL framework for flow-based VLAs. Incorporating two solutions, Flow-Noise and Flow-SDE, $\pi _ { \mathrm { R L } }$ enhances the performance and generalization of SFT-aligned models across extensive ID benchmarks and OOD settings. Refined with RL, few-shot SFT policies achieve performance comparable to full dataset baselines. Additionally, we facilitate seamless zero-shot sim-to-real transfer by constructing a simulator with 3D Gaussian Splatting as the rendering engine to narrow the visual domain gap.

# Abstract

Vision-Language-Action (VLA) models enable robots to understand and perform complex tasks from multimodal input. Although recent work explores using reinforcement learning (RL) to automate the laborious data collection process in scaling supervised fine-tuning (SFT), applying RL to large-scale flow-based VLAs (e.g., π0,

π0.5) remains challenging due to intractable action log-likelihoods raised from flow matching. We address this challenge with $\pi _ { \mathrm { R L } }$ , featuring two technical approaches: (1) Flow-Noise models the denoising process as a discrete-time MDP with a learnable noise network for exact log-likelihood computation. (2) Flow-SDE integrates denoising with agent-environment interaction, formulating a two-layer MDP that employs ODE-to-SDE conversion for efficient RL exploration. We evaluate $\pi _ { \mathrm { R L } }$ across various benchmarks, with experiments demonstrating that RL yields significant performance improvements in both in-distribution and out-of-distribution settings.

# 1. Introduction

Vision-Language-Action (VLA) models (Din et al., 2025) have emerged as a leading solution for general-purpose robots, effectively bridging the gap between high-level multimodal reasoning and low-level physical control (Firoozi et al., 2025). Conditioned on sensor inputs and language commands, VLAs (Team et al., 2024; Kim et al., 2024; Black et al., 2024; Intelligence et al., 2025) can translate abstract instructions into executable robotic actions, thereby enabling intuitive and flexible human-robot interaction.

The training methodology for VLAs follows the standard pre-training and supervised fine-tuning (SFT) paradigm as shown in Fig. 1. Building on the pretrained Vision-Language Model (VLM) (Touvron et al., 2023; Beyer et al., 2024), VLAs are fine-tuned on large-scale, heterogeneous human demonstration datasets (O’Neill et al., 2024; Khazatsky et al., 2024), followed by SFT on the target task to align their capabilities with the specific embodiment and environment. However, reliance on SFT introduces a critical challenge: curating large-scale, high-quality expert trajectories is both laborious and costly (Din et al., 2025). Besides, models obtained via SFT tend to overfit to expert demonstrations (Fei et al., 2025), with their performance fundamentally constrained by the quality of expert demonstrations.

Recent efforts (Zang et al., 2025; Li et al., 2025a; Tan et al., 2025; Liu et al., 2025a) have explored expanding the VLA training process with reinforcement learning (RL), establishing a pre-training, SFT, and RL paradigm as shown in Fig. 1, allowing VLAs to improve their performance beyond expert demonstrations through environmental interaction and develop more generalizable policies.

However, these RL advances have been largely confined to autoregressive VLAs, featuring OpenVLA (Kim et al., 2024) and OpenVLA-OFT (Kim et al., 2025), which employ discrete action decoders that generate output in an autoregressive or parallel fashion. This stands in stark contrast to flow-based VLAs, exemplified by the π series models, which generate actions through iterative refinement in flow matching (Lipman et al., 2022), offering the advantages of generating action chunks in high-frequency and performing highly dexterous tasks (Black et al., 2024). Consequently, previous VLA-RL algorithms are incompatible with flowbased VLAs, and the fundamental challenge lies in how to characterize a logarithmic likelihood (Hutchinson, 1989; Chen et al., 2018) for the executed actions.

In this paper, we introduce $\pi _ { \mathrm { R L } }$ , a framework designed for fine-tuning flow-based VLAs with online RL algorithms. To address the intractable log-likelihood estimation problem in flow matching, we propose two solutions. Flow-Noise integrates a learnable noise network into the denoising process and models this stage as a discrete-time Markov decision process (MDP) for exact log-likelihood estimation. Flow-SDE converts the ordinary differential equation (ODE) denoising process into a stochastic differential equation (SDE) while maintaining equivalent marginal distributions for exploration, and builds a two-layer MDP that couples the denoising process with policy-environment interaction. Given the formulated MDP and the exact log-likelihood computation, $\pi _ { \mathrm { R L } }$ undergoes further optimization via the proximal policy optimization (PPO) (Schulman et al., 2017).

We conduct extensive experiments on various benchmarks to evaluate the effectiveness of $\pi _ { \mathrm { R I } }$ on $\pi _ { 0 }$ (Black et al., 2024) and $\pi _ { 0 . 5 }$ (Intelligence et al., 2025) models. Across all benchmarks, the proposed framework consistently yields substantial performance gains over SFT baselines. Furthermore, out of distribution evaluations confirm that our model yields genuine policy enhancement rather than narrow overfitting on the target environment.

To sum up, our contributions are:

• RL for flow-based VLAs. We introduce $\pi _ { \mathrm { R L } }$ , an online RL fine-tuning framework with Flow-Noise and Flow-SDE formulations for flow-based VLAs.   
• Superior Performance. We demonstrate significant performance improvements and enhanced generalization of $\pi _ { \mathrm { R L } }$ across various benchmarks.   
• Comprehensive Ablation. We conduct thorough ablation studies, offering empirical insights to guide future RL research on flow-based VLAs.   
• Open-source Code and Models. We release all codes to ensure reproducibility, hoping that our study helps to advance further research in this field.

# 2. Related Work

# 2.1. Vision-Language-Action Models

VLA models have recently achieved remarkable progress in robotics by integrating multimodal inputs to enable unified perception, reasoning, and control. This development has led to a series of architectures, including Octo (Team et al., 2024), RT (Brohan et al., 2022), OpenVLA, OpenVLA-OFT, π0, π0.5, and GR00T (Bjorck et al., 2025).

# 2.2. Online RL Fine-tuning for VLA Models

Recent research has increasingly focused on enhancing the performance and generalization of VLAs with online RL. For example, SimpleVLA-RL (Li et al., 2025a), building on the OpenVLA-OFT and GRPO, demonstrated that RL can improve long-horizon planning of VLA models under data scarcity. RL4VLA (Liu et al., 2025a) empirically evaluated PPO, GRPO, and direct preference optimization (DPO) (Rafailov et al., 2023) with stage-based sparse rewards. RLinf-VLA (Yu et al., 2025; Zang et al., 2025) provides a unified and efficient framework for scalable RL training of VLA models. These works demonstrate the effectiveness of RL fine-tuning VLA models.

# 2.3. RL Fine-tuning for Flow Models

Integrating RL with flow models is a promising way to transcend the limitations of imitation learning. To this end, Flow-GRPO (Liu et al., 2025b) converts the deterministic ODE into an equivalent SDE to enable stochasticity exploration, a foundation upon which subsequent works like Mix-GRPO (Li et al., 2025b) and TempFlow-GRPO (He et al., 2025) further accelerate training through hybrid ODE-SDE rollouts. ReinFlow (Zhang et al., 2025) injects learnable noise into the flow path and transforms it into a discretetime Markov process with a tractable likelihood for stable policy gradient updates. Flow policy optimization (FPO) (McAllister et al., 2025) reframes policy optimization as maximizing the advantage-weighted ratio of the conditional flow matching loss.

# 3. Preliminary

# 3.1. Problem Formulation

We formulate the task as an MDP, defined by a tuple $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P _ { 0 } , P _ { \mathrm { E N V } } , R _ { \mathrm { E N V } } , \gamma )$ . The state $s _ { t } \in S$ is defined as the robot observation $\mathbf { o } _ { t }$ and $P _ { 0 }$ denotes the initial state distribution. Given the state, the flow policy predicts an action $a _ { t } \sim \pi ( \cdot | s _ { t } ) \in \mathcal { A }$ , resulting in the state transition $s _ { t + 1 } \sim P _ { \mathrm { E N V } } ( \cdot | s _ { t } , a _ { t } )$ and a reward $R _ { \mathrm { E N V } } ( s _ { t } , a _ { t } )$ . The objective is to learn a policy $\pi _ { \theta }$ that maximizes the expected γ-discounted return over a horizon of $T + 1$ :

$$
\mathcal {J} (\pi_ {\theta}) = \mathbb {E} _ {\pi_ {\theta}, P _ {0}} \left[ \sum_ {t = 0} ^ {T} \gamma^ {t} R _ {\mathrm{ENV}} (s _ {t}, a _ {t}) \right]. \tag {1}
$$

With the policy gradient surrogate (Williams, 1992), the gradient of the return expectation can be approximated from sampled trajectories:

$$
\nabla_ {\theta} \mathcal {J} (\pi_ {\theta}) = \mathbb {E} _ {\pi_ {\theta}, P _ {0}} \left[ \sum_ {t = 0} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} (a _ {t} | s _ {t}) A (s _ {t}, a _ {t}) \right]. \tag {2}
$$

The advantage function, $A ( s _ { t } , a _ { t } ) = Q ( s _ { t } , a _ { t } ) - V ( s _ { t } )$ , measures the relative merit of the action value $Q ( s _ { t } , a _ { t } )$ over the state value $V ( s _ { t } )$ , providing a low-variance signal for the policy update.

# 3.2. Flow-based Vision-Language-Action Model

A flow-based VLA model $\pi _ { \theta }$ is designed to map the observation $\mathbf { o } _ { t }$ comprising RGB images, language tokens, and robot proprioception to a sequence of H future actions $\mathbf { A } _ { t } = [ a _ { t , 0 } , . . . , a _ { t , H - 1 } ]$ , formulated as $p ( \mathbf { A } _ { t } | \mathbf { o } _ { t } )$ . Within the model, the VLM extracts features from the visual and language inputs, while the flow matching expert is tasked with generating the actions. Specifically, the model learns a conditional vector field $\mathbf { v } _ { \theta }$ that transforms a standard Gaussian noise distribution into the target action $\mathbf { A } _ { t } .$ This is achieved by minimizing the Conditional Flow Matching (CFM) loss, which aligns the predicted vector field $\mathbf { v } _ { \theta }$ with the ground-truth vector field u:

$$
\mathcal {L} _ {\mathrm{CFM}} = \mathbb {E} _ {\tau , p (\mathbf {A} _ {t}, \mathbf {o} _ {t}), q (\mathbf {A} _ {t} ^ {\tau} | \mathbf {A} _ {t})} \left[ \| \mathbf {v} _ {\theta} (\mathbf {A} _ {t} ^ {\tau}, \mathbf {o} _ {t}) - \mathbf {u} (\mathbf {A} _ {t} ^ {\tau} | \mathbf {A} _ {t}) \| _ {2} ^ {2} \right]. \tag {3}
$$

Here, the conditional probability path $q ( \mathbf { A } _ { t } ^ { \tau } | \mathbf { A } _ { t } )$ generates a noisy action1 ${ \bf A } _ { t } ^ { \tau } = \tau { \bf A } _ { t } + ( 1 - \tau ) \epsilon$ from an action ${ \bf A } _ { t }$ , random noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , and a continuous time $\tau \in [ 0 , 1 ]$ in flow matching. For this specific path, the corresponding ground-truth vector field is defined as ${ \bf u } ( { \bf A } _ { t } ^ { \tau } | { \bf A } _ { t } ) = { \bf A } _ { t } - \epsilon$ .

During the inference, the action sequence is generated by first sampling a noise vector $\mathbf { A } _ { t } ^ { 0 } \sim \mathcal { N } ( 0 , I )$ , which is further iteratively refined by integrating the learned vector field $\mathbf { v } _ { \theta }$ over a fixed number of steps based on the forward Euler method: $\mathbf { A } _ { t } ^ { \tau + \delta } = \mathbf { A } _ { t } ^ { \tau } + \mathbf { v } _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } ) \cdot \delta$ .

# 4. Methodology

Existing VLA-RL approaches leverage base models such as OpenVLA for discrete actions and OpenVLA-OFT for continuous actions. To compute the action log-likelihood log $\pi _ { \boldsymbol { \theta } } \big ( a _ { t } | \boldsymbol { s } _ { t } \big )$ , discrete models (Liu et al., 2025a) apply softmax to the output logits, while continuous models (Li et al., 2025a) treat the action as a Gaussian distribution, employing a prediction head to estimate the variance. As for the flow-based VLAs, directly computing the exact likelihood (Hutchinson, 1989) is inaccurate with few denoising steps. Moreover, the deterministic nature of its ODE sampling process precludes exploration, making its implementation within RL non-trivial. To this end, we propose Flow-Noise and Flow-SDE, two technical approaches that make flowbased VLAs amenable to RL.

# 4.1. Flow-Noise

Inspired by Reinflow (Zhang et al., 2025), we incorporate a learnable noise network into the flow matching denoising process and solve the problem within the standard one-layer MDP framework detailed in Sec. 3.1. By modeling the denoising stage as a discrete MDP, we can directly compute the log-likelihood of the denoised sequence, enabling equivalent policy optimization via RL.

![image](images/02530e36eb49b4b0ba2edef07f7e4aff884342d4973898be7bfd40db302049a7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Vision Language Model"] -->|KV-cache| B["Action Expert"]
    B --> C["Noise Injection"]
    C --> D["Flow-Noise: Learnable Noise from the noise network."]
    D --> E["Flow-SDE: Fixed noise from the ODE-SDE conversion."]
    subgraph Observation
        F["Images: Multi-view RGB camera inputs"]
        G["Language: Goal-oriented command"]
        H["“pick up the black bowl on the stove and place it on the plate”"]
        I["State: Proprioceptive feedback"]
        J["Q1, Q2, Q3, Q4, Q5, Q6, Q7"]
    end
    subgraph Noise Injection
        K["σθ'"] --> L["Vτ"]
        M["στ'"] --> N["στ²"]
        O["μτ = Aτ + vτ* δ"] --> P["Στ = diag(σθ²)"]
        Q["μτ = στ²δ * I"] --> R["Στ = στ²δ * I"]
    end
    subgraph Gaussian Distribution
        S["A⁰"] --> T["Aτ"]
        T --> U["P_FLOW"]
        U --> V["Aτ+δ"]
        V --> W["A¹"]
    end
    style Observation fill:#f9f,stroke:#333
    style Noise Injection fill:#ccf,stroke:#333
```
</details>

Figure 2. Illustration for the noise injection in $\pi _ { \mathrm { R L } }$ .

# 4.1.1. STOCHASTICITY INJECTION

In Flow-Noise, we parameterize the noise schedule with a neural network, allowing the magnitude of the injected noise to be learned dynamically during training for greater flexibility, as shown in Fig. 2. We focus on the generation process within a single environment timestep t. For notational simplicity, we omit the time subscript $t , e . g .$ ., writing $\mathbf { A } ^ { \tau }$ , and denote the predicted velocity $\mathbf { v } _ { \theta } ( \mathbf { A } ^ { \tau } , \mathbf { o } )$ as $\mathbf { v } ^ { \tau }$ .

The step transition during the denoising process is modeled as a Gaussian distribution $p ( \mathbf { A } ^ { \tau + \delta } | \mathbf { A } ^ { \tau } ) \sim \mathcal { N } ( \mu _ { \tau } , \Sigma _ { \tau } )$ , where the mean is determined by the forward Euler update of the original ODE and the variance is controlled by the learnable noise network $\theta ^ { \prime } \colon$ :

$$
\left\{ \begin{array}{l} \mu_ {\tau} = \mathbf {A} ^ {\tau} + \mathbf {v} ^ {\tau} \cdot \delta \\ \Sigma_ {\tau} = \operatorname{diag} (\sigma_ {\theta^ {\prime}} ^ {2}) \end{array} \right. \tag {4}
$$

Here, $\sigma _ { \theta ^ { \prime } } ( \cdot )$ is the standard deviation learned from the noise injection network, conditioned on the action $\mathbf { A } ^ { \tau }$ , and the observation o. The noise network is trained jointly with the velocity but discarded after fine-tuning, leaving a deterministic policy for inference.

# 4.1.2. LOG-LIKELIHOOD ESTIMATION

The primary challenge in applying policy gradient methods to flow-based VLAs stems from the intractable loglikelihood of the final executed action. In Flow-Noise, we address it by substituting the gradient of the joint loglikelihood of the entire denoising process into the policy optimization objective in Eq. (2), which is theoretically grounded in Reinflow (Zhang et al., 2025).

The inference process for action generation is discretized into K uniform steps, which defines a sequence of time points $\{ \tau _ { 0 } , \tau _ { 1 } , \dots , \tau _ { K } \}$ . With the step interval defined as $\delta = 1 / K$ , the discrete timestep at the k-th point is $\tau _ { k } = k \cdot \delta$ , starting from $\tau _ { 0 } = 0$ and culminating at $\tau _ { K } = 1$ . Given the observation $\mathbf { o } ,$ the exact and tractable log probability for the entire denoising sequence $\mathcal { A } = ( \mathbf { A } ^ { 0 } , \ldots , \mathbf { A } ^ { 1 } )$ is depicted in Fig. 3 and formulated as:

![image](images/3c32437367c30e5a43b0b85a6cf02fb91cfc2cc7ad7a15c0f3bf08b1bacb2598.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph One_Layer_MDP
        A["O_t"] -->|P_ENV(·)| B["O_{t+1}"]
        B -->|s_t| C["O_{t+1}"]
        C --> D["O_{t+1}"]
        D --> E["O_{t+1}"]
        E --> F["O_{t+1}"]
        F --> G["O_{t+1}"]
        G --> H["O_{t+1}"]
        H --> I["O_{t+1}"]
        I --> J["O_{t+1}"]
        J --> K["O_{t+1}"]
        K --> L["O_{t+1}"]
        L --> M["O_{t+1}"]
        M --> N["O_{t+1}"]
        N --> O["O_{t+1}"]
        O --> P["O_{t+1}"]
        P --> Q["O_{t+1}"]
        Q --> R["O_{t+1}"]
        R --> S["O_{t+1}"]
        S --> T["O_{t+1}"]
        T --> U["O_{t+1}"]
        U --> V["O_{t+1}"]
        V --> W["O_{t+1}"]
        W --> X["O_{t+1}"]
        X --> Y["O_{t+1}"]
        Y --> Z["O_{t+1}"]
        Z --> AA["O_{t+1}"]
        AA --> AB["O_{t+1}"]
        AB --> AC["O_{t+1}"]
        AC --> AD["O_{t+1}"]
        AD --> AE["O_{t+1}"]
        AE --> AF["O_{t+1}"]
        AF --> AG["O_{t+1}"]
        AG --> AH["O_{t+1}"]
        AH --> AI["O_{t+1}"]
        AI --> AJ["O_{t+1}"]
        AJ --> AK["O_{t+1}"]
        AK --> AL["O_{t+1}"]
        AL --> AM["O_{t+1}"]
        AM --> AN["O_{t+1}"]
        AN --> AO["O_{t+1}"]
        AO --> AP["O_{t+1}"]
        AP --> AQ["O_{t+1}"]
        AQ --> AR["O_{t+1}"]
        AR --> AS["O_{t+1}"]
        AS --> AT["O_{t+1}"]
        AT --> AU["O_{t+1}"]
        AU --> AV["O_{t+1}"]
        AV --> AW["O_{t+1}"]
        AW --> AX["O_{t+1}"]
        AX --> AY["O_{t+1}"]
        AY --> AZ["O_{t+1}"]
        AZ --> BA["O_{t+1}"]
        BA --> BB["O_{t+1}"]
        BB --> BC["O_{t+1}"]
        BC --> BD["O_{t+1}"]
        BD --> BE["O_{t+1}"]
        BE --> BF["O_{t+1}"]
        BF --> BG["O_{t+1}"]
        BG --> BH["O_{t+1}"]
        BH --> BI["O_{t+1}"]
        BI --> BJ["O_{t+1}"]
        BJ --> BK["O_{t+1}"]
        BK --> BL["O_{t+1}"]
        BL --> BM["O_{t+1}"]
        BM --> BN["O_{t+1}"]
        BN --> BO["O_{t+1}"]
        BO --> BP["O_{t+1}"]
        BP --> BQ["O_{t+1}"]
        BQ --> BR["O_{t+1}"]
        BR --> BS["O_{t+1}"]
        BS --> BT["O_{t+1}"]
        BT --> BU["O_{t+1}"]
        BU --> BV["O_{t+1}"]
        BV --> BW["O_{t+1}"]
        BW --> BX["O_{t+1}"]
        BX --> BY["O_{t+1}"]
        BY --> BZ["O_{t+1}"]
        BZ --> CA["O_{t+1}"]
        CA --> CB["O_{t+1}"]
        CB --> CC["O_{t+1}"]
        CC --> CD["O_{t+1}"]
        CD --> CE["O_{t+1}"]
        CE --> CF["O_{t+1}"]
        CF --> CG["O_{t+1}"]
        CG --> CH["O_{t+1}"]
        CH --> CI["O_{t+1}"]
        CI --> CJ["O_{t+1}"]
        CJ --> CK["O_{t+1}"]
    end

    subgraph Flow-Noise
        D["N(0,t)"]
    end

    subgraph Inner_MDP
        D0["O_t"]
        D0a["A_t^0"]
    end

    subgraph Outer_MDP
        D0b["A_t^0"]
    end

    subgraph Two_Layer_MDP
        D0c["S̄_t^0"]
    end

    note right of D
        log_π(A|o) = log(π(A^0|o) ∏_k=0^K-1 π (A^τ_k+i | A^τ_k,o))
    end
    note right of D0c
    note right of D0c
    note right of D0c
    note right of D0c
    note right of D0c
    note right of D0c
    note right of D0c
    note right of D0c
    note right of D0c
    note right of D0c
```
</details>

Figure 3. Illustration of the MDP formulations in $\pi _ { \mathrm { R I } }$ .

$$
\log \pi (\mathcal {A} | \mathbf {o}) = \log \left(\pi (\mathbf {A} ^ {0} | \mathbf {o}) \prod_ {k = 0} ^ {K - 1} \pi (\mathbf {A} ^ {\tau_ {k + 1}} | \mathbf {A} ^ {\tau_ {k}}, \mathbf {o})\right). \tag {5}
$$

Building on this, we can treat flow-based policy optimization within a standard MDP framework.

# 4.2. Flow-SDE

Inspired by Flow-GRPO (Liu et al., 2025b), we enhance stochastic exploration by converting the denoising process from ODE into an SDE formulation. We further construct a two-layer MDP to couple the denoising process with the policy-environment interaction following DPPO (Ren et al., 2024), while leveraging the hybrid ODE-SDE sampling technique to accelerate the training process.

# 4.2.1. STOCHASTICITY INJECTION

In Flow-SDE, we convert the deterministic ODE into an equivalent SDE that preserves the marginal probability density of the generated actions, as shown in Fig. 2.

The deterministic ODE sampling trajectory of the flow matching, especially the Rectified Flow (Liu et al., 2022), is described by the forward Euler method:

$$
d \mathbf {A} ^ {\tau} = \mathbf {v} ^ {\tau} d \tau . \tag {6}
$$

Building on the connection between the probability flow ODE and SDE (Song et al., 2020), we can transform the deterministic ODE in Eq. (6) into an equivalent SDE, with a drift term that corrects the original velocity and a diffusion

term that introduces noise:

$$
d \mathbf {A} ^ {\tau} = \underbrace {\left(\mathbf {v} ^ {\tau} - \frac {1}{2} g ^ {2} (\tau) \nabla \log q _ {\tau} \left(\mathbf {A} ^ {\tau}\right)\right) d \tau} _ {\text { Drift   Term }} + \underbrace {g (\tau) d \mathbf {w}} _ {\text { Diffusion   Term }}, \tag {7}
$$

where $g ( \tau )$ is a scalar function controlling the noise schedule, $\nabla \log q _ { \tau } ( \mathbf { A } ^ { \tau } )$ is the score function of the marginal distribution $q _ { \tau }$ and dw denotes a Wiener process.

As established in Flow-GRPO, the score function and the velocity field are critically linked by ∇ log $q _ { \tau } ( \mathbf { A } ^ { \tau } ) =$ $\begin{array} { r } { - \frac { \mathbf { A } ^ { \tau } } { \tau } - \frac { 1 - \tau } { \tau } \mathbf { v } ^ { \tau } } \end{array}$ . By substituting the score function with the velocity field in Eq. (7) and setting the noise schedule $g ( \tau )$ to $\begin{array} { r } { \sigma _ { \tau } = a \sqrt { \frac { \tau } { 1 - \tau } } } \end{array}$ with a controlling the noise level, we derive the final SDE formulation for the flow-matching sampler:

$$
d \mathbf {A} ^ {\tau} = \left[ \mathbf {v} ^ {\tau} + \frac {\sigma_ {\tau} ^ {2}}{2 \tau} \left(\mathbf {A} ^ {\tau} + (1 - \tau) \mathbf {v} ^ {\tau}\right) \right] d \tau + \sigma_ {\tau} d \mathbf {w} _ {\tau}. (8)
$$

Discretizing this SDE reveals that the transition probability $p ( \mathbf { A } ^ { \tau + \delta } \vert \mathbf { A } ^ { \tau } ) \sim \mathcal { N } ( \mu _ { \tau } , \Sigma _ { \tau } )$ is an isotropic Gaussian distribution, with the mean and variance formulated as:

$$
\left\{ \begin{array}{l} \mu_ {\tau} = \mathbf {A} ^ {\tau} + \left[ \mathbf {v} ^ {\tau} + \frac {\sigma_ {\tau} ^ {2}}{2 \tau} \left(\mathbf {A} ^ {\tau} + (1 - \tau) \mathbf {v} ^ {\tau}\right) \right] \cdot \delta \\ \Sigma_ {\tau} = \sigma_ {\tau} ^ {2} \delta \cdot \mathbf {I} \end{array} . \right. \tag {9}
$$

# 4.2.2. MDP FORMULATION

We couple the denoising process of the flow matching with environmental interaction in Flow-SDE. Specifically, we embed the inner MDP defined during the denoising process into the high-level, outer-loop MDP with the environment $\mathcal { M } _ { \mathrm { E N V } }$ in Sec. 3.1, formulating a two-layer MDP as shown in Fig. 3, with components defined with respect to the environment time t and denoising time $\tau .$

• State $\bar { s } _ { t } ^ { \tau } = ( \mathbf { o } _ { t } , \mathbf { A } _ { t } ^ { \tau } )$ is the tuple of the observation $\mathbf { o } _ { t }$ and the action state $\mathbf { A } _ { t } ^ { \tau }$ .   
• Action $\bar { a } _ { t } ^ { \tau }$ is defined as the next sampled denoised action in the inner-loop and the executed action for the outer loop:

$$
\bar {a} _ {t} ^ {\tau} = \left\{ \begin{array}{l l} \mathbf {A} _ {t} ^ {\tau + \delta} & \text { if   } \tau <   1 \\ \mathbf {A} _ {t} ^ {1} & \text { if   } \tau = 1 \end{array} \right., \tag {10}
$$

where Aτ+δt $\mathbf { A } _ { t } ^ { \tau + \delta } = \mu _ { \tau } + \sigma _ { \tau } \sqrt { \delta } \cdot \epsilon , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ is the randomly sampled noise.

• Transition $\bar { P } ( \bar { s } _ { t ^ { \prime } } ^ { \tau ^ { \prime } } | \bar { s } _ { t } ^ { \tau } , \bar { a } _ { t } ^ { \tau } )$ defines how the state evolves, formulated as:

$$
\bar {s} _ {t ^ {\prime}} ^ {\tau^ {\prime}} = \left\{ \begin{array}{l l} (\mathbf {o} _ {t}, \bar {a} _ {t} ^ {\tau}) & \text { if   } \tau <   1 \\ (\mathbf {o} _ {t + 1}, \mathbf {A} _ {t + 1} ^ {0}) & \text { if   } \tau = 1 \end{array} \right.. \tag {11}
$$

For $\tau < 1$ , the inner loop transition $P _ { \mathrm { F L O W } } ( \cdot )$ occurs between different denoised action states, where the observation $\mathbf { o } _ { t }$ remains fixed and the next action state is set by a¯τt = Aτ +δt . $\bar { a } _ { t } ^ { \tau } = { \bf A } _ { t } ^ { \tau + \delta }$

For $\tau = 1$ , the final action $\bar { a } _ { t } ^ { \tau } = \mathbf { A } _ { t } ^ { 1 }$ interacts with the outer-loop environment, resulting in a new observation $\mathbf { o } _ { t + 1 }$ according to the environment dynamics $P _ { \mathrm { E N V } } ( \cdot )$ Concurrently, the action state is reset from a standard normal distribution $\mathbf { A } _ { t + 1 } ^ { 0 } \sim \mathcal { N } ( 0 , I )$ .

• Reward $\bar { R } ( \bar { s } _ { t } ^ { \tau } , \bar { a } _ { t } ^ { \tau } )$ is granted only upon completion of the denoising process and interaction with the environment:

$$
\bar {R} (\bar {s} _ {t} ^ {\tau}, \bar {a} _ {t} ^ {\tau}) = \left\{ \begin{array}{l l} 0 & \text { if   } \tau <   1 \\ R _ {\mathrm{ENV}} (\mathbf {o} _ {t}, \mathbf {A} _ {t} ^ {1}) & \text { if   } \tau = 1 \end{array} \right.. \tag {12}
$$

Within the two-layer MDP framework, the problem of estimating the action log-likelihood log $\pi ( a _ { t } | s _ { t } )$ is transformed into estimating log $\pi ( \bar { a } _ { t } ^ { \tau } | \bar { s } _ { t } ^ { \tau } )$ , which is straightforward to compute due to the Gaussian nature of the transitions.

# 4.2.3. HYBRID ODE-SDE SAMPLING

The two-layer MDP formulation significantly extends the horizon, increasing training difficulty and computational cost. To mitigate this, we adopt a mixed ODE-SDE rollout strategy (Li et al., 2025b; He et al., 2025). At each step t, we randomly sample a denoising time $\tau _ { t }$ for the stochastic SDE exploration, while treating all remaining denoising steps as deterministic ODE updates. Specifically, the policy acts on the state $\bar { s } _ { t } ^ { \tau _ { t } } = ( \mathbf { o } _ { t } , \mathbf { A } _ { t } ^ { \tau _ { t } } ) ;$ ; subsequently, an environment wrapper executes the remaining ODE steps and the environment transition, ultimately yielding the next state $\bar { s } _ { t + 1 } ^ { \tau _ { t + 1 } } = ( \mathbf { o } _ { t + 1 } , \mathbf { A } _ { t + 1 } ^ { \tau _ { t + 1 } } )$ s¯ t+1 at a newly sampled time $\tau _ { t + 1 }$ . This formulation effectively shortens the MDP horizon while maintaining theoretical consistency with the original twolayer framework.

# 4.3. Policy Optimization

# 4.3.1. ALGORITHM

Given the formulated flow policy MDP, our objective is to learn the optimal parameters $\theta ^ { * }$ for the policy $\pi _ { \theta }$ that maximizes the expected discounted return $\mathcal { I } ( \pi _ { \boldsymbol { \theta } } )$ . To this end, we apply the widely adopted policy gradient algorithm PPO to optimize the policy.

π-series models (Black et al., 2024; Intelligence et al., 2025) adopt a chunk-based approach for action generation. Specifically, the policy outputs an entire sequence of H future actions $\mathbf { A } _ { t } = \left[ a _ { t , 0 } , . . . , a _ { t , H - 1 } \right]$ in response to each observation. In this approach, we treat the entire sequence as a single macro-step and define its corresponding reward $\begin{array} { r } { R _ { t } = \sum _ { j = 0 } ^ { H - 1 } r _ { t , j } } \end{array}$ as the sum of the per-step rewards $\boldsymbol { r } _ { t , j }$ referred to as the chunk-level formulation in RLinf-VLA (Zang et al., 2025).

![image](images/d398ae79b7898279a8b28e28ba4dd0118f7bec6489afb4271e7d9816468948cd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["V_expert (o_t) ≈ E_{τ~U[0,1"]} [V_expert (o_t, A_t^τ)]] --> B["Critic"]
    B --> C["Action"]
    C --> D["Action Expert"]
    D --> E["State"]
    D --> F["Noise A_t^τ"]
    G["fold shirt"] --> H["Image 1"]
    G --> I["Image 2"]
    G --> J["Image 3"]
    K["A_t^τ+δ"] --> L["Action"]
    L --> M["Action Expert"]
    M --> N["State"]
    M --> O["Noise A_t^τ"]
```
</details>

(a) Critic with the action expert, exemplified by $\pi _ { 0 } .$

![image](images/6dc6472ccd0545b60650883d387bfc5835ec1168f926de093ba494a5bff88586.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Critic"] --> B["Vision Language Model"]
    C["Vvlm(σt)"] --> B
    D["&quot;clean room&quot; state"] --> B
    E["Action"] --> F["Action Expert"]
    F --> G["noise Aτ"]
    G --> H["Aτ+δ"]
    H --> I["Action"]
    I --> J["Action Expert"]
```
</details>

(b) Critic with the VLM, exemplified by $\pi _ { 0 . 5 }$ .   
Figure 4. Illustration of the two critic placement configurations.

To effectively guide policy updates, PPO employs Generalized Advantage Estimation (GAE) (Schulman et al., 2015) to compute a low-variance estimate of the advantage, estimated as:

$$
\hat {A} _ {t} = \sum_ {k = 0} ^ {T - t} (\gamma \lambda) ^ {k} \mathcal {T} _ {t + k}, \tag {13}
$$

where the TD-error is $\mathcal T _ { t } = R _ { t } + \gamma V ( s _ { t + 1 } ) - V ( s _ { t } )$ . Here, $V ( \cdot )$ is the state-value function derived from the critic network, γ is the discount factor, and λ is the parameter that balances the trade-off between bias and variance in the advantage estimate.

PPO constrains policy updates to a small trust region to prevent large, destabilizing updates, with the objective function:

$$
\mathcal {J} (\pi_ {\theta}) = \mathbb {E} _ {t} \left[ \min \left(\rho_ {t} (\theta) \hat {A} _ {t}, \operatorname{clip} (\rho_ {t} (\theta), 1 - \epsilon , 1 + \epsilon) \hat {A} _ {t}\right) \right], \tag {14}
$$

where the clip function, governed by a hyperparameter $\epsilon ,$ restricts the ratio $\rho _ { t } ( \theta )$ to the interval $[ 1 - \epsilon , 1 + \epsilon ]$ to ensure training stability.

Here, the probability ratio $\rho _ { t } ( \theta )$ between the updated and old policies takes the form of either:

$$
\rho_ {t} (\theta) = \frac {\pi_ {\theta_ {\text { new }}} (a _ {t} | s _ {t})}{\pi_ {\theta_ {\text { old }}} (a _ {t} | s _ {t})} \quad \text { or } \quad \rho_ {t} (\theta) = \frac {\pi_ {\theta_ {\text { new }}} (\bar {a} _ {t} ^ {\tau} | \bar {s} _ {t} ^ {\tau})}{\pi_ {\theta_ {\text { old }}} (\bar {a} _ {t} ^ {\tau} | \bar {s} _ {t} ^ {\tau})}. \tag {15}
$$

# 4.3.2. CRITIC DESIGN

Following VLA-PPO works (Zang et al., 2025; Liu et al., 2025a), we employ a shared actor-critic architecture for memory-efficient value prediction as shown in Fig. 4. However, the two flow-based VLAs process the proprioceptive state differently: in $\pi _ { 0 } ,$ the state is fed into the action expert model, whereas in $\pi _ { 0 . 5 } ,$ , it is merged with prompt embeddings within the VLM.

To this end, for the $\pi _ { 0 . 5 }$ variant, we attach the critic network directly to the VLM output, providing the value estimate $V _ { \mathrm { v l m } } ( \mathbf { o } _ { t } )$ conditioned on the integrated image, language, and state inputs. Conversely, for the $\pi _ { 0 }$ variant, achieving the value prediction is non-trivial due to the coupled input structure, where the action expert requires both the noisy action $\mathbf { A } _ { t } ^ { \tau }$ and the state. To this end, we approximate $V _ { \mathrm { e x p e r t } } ( \mathbf { o } _ { t } )$ by averaging the value estimates across the entire denoising trajectory, formulated as:

$$
V _ {\text { expert }} (\mathbf {o} _ {t}) \approx \mathbb {E} _ {\tau \sim U [ 0, 1 ]} [ V _ {\text { expert }} (\mathbf {o} _ {t}, \mathbf {A} _ {t} ^ {\tau}) ]. \tag {16}
$$

# 5. Experimental Results

# 5.1. Setup

Benchmarks. We perform experiments on four widelyadopted robot manipulation benchmarks: LIBERO (Liu et al., 2023), ManiSkill (Tao et al., 2024), MetaWorld (McLean et al., 2025) and CALVIN (Mees et al., 2022).

Flow-based VLAs. We conduct our primary experiments based on $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ models. Additionally, we conduct experiments on GR00T in Appendix Sec. H, which validates that our algorithm can be applied to other flow-based VLAs.

# 5.2. Main Results

In this section, we assess the in-distribution (ID) performance of $\pi _ { \mathrm { R I } }$ across various benchmarks, followed by an analysis of its out-of-distribution (OOD) generalization.

# 5.2.1. IN-DISTRIBUTION RL TRAINING

As detailed in Tab. 1, $\pi _ { \mathrm { R L } }$ yields substantial performance gains over SFT baselines across all evaluated benchmarks. Specifically, the $\pi _ { 0 }$ model achieves a maximum average improvement of +29.2%, while the $\pi _ { 0 . 5 }$ variant demonstrates a +31.0% increase in average success rate.

Specifically for LIBERO, we perform few-shot SFT on the $\pi _ { 0 . 5 }$ model followed by RL optimization to achieve a 98.3% success rate, outperforming the 96.9% success rate of the full-dataset SFT baseline. These performance gains extend to other challenging environments, including ManiSkill with its 4,352 pick-and-place task combinations, MetaWorld featuring 50 distinct manipulation primitives, and CALVIN for long-horizon sequential tasks. See Appendix Sec. C for comprehensive experimental details.

Table 1. Comprehensive ID performance comparison across four benchmarks. 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="6">Benchmarks</td></tr><tr><td>LIBERO</td><td>ManiSkill</td><td>MetaWorld</td><td>CALVIN</td><td>Avg.</td><td> $\Delta$  Avg.</td></tr><tr><td rowspan="3"> $\pi_0$ </td><td>SFT</td><td>57.6</td><td>38.4</td><td>50.8</td><td>57.5</td><td>51.1</td><td>—</td></tr><tr><td>Flow-SDE</td><td>96.1</td><td>78.8</td><td>78.1</td><td>61.7</td><td>78.7</td><td>+27.6</td></tr><tr><td>Flow-Noise</td><td>97.6</td><td>77.8</td><td>85.8</td><td>59.9</td><td>80.3</td><td>+29.2</td></tr><tr><td rowspan="3"> $\pi_{0.5}$ </td><td>SFT</td><td>77.1</td><td>40.1</td><td>43.8</td><td>61.3</td><td>55.6</td><td>—</td></tr><tr><td>Flow-SDE</td><td>97.9</td><td>90.9</td><td>70.7</td><td>87.0</td><td>86.6</td><td>+31.0</td></tr><tr><td>Flow-Noise</td><td>98.3</td><td>89.7</td><td>66.1</td><td>84.5</td><td>84.7</td><td>+29.1</td></tr></table>

![image](images/e5822666e28ea8da6d03ed278d254821b2247427fb07cc2b7edec8b1b2bd0b55.jpg)

<details>
<summary>bar</summary>

| Category | Success Rate (%) |
| :--- | :--- |
| SFT | 61.3 |
| RL | 79.1 |
+17.8%
</details>

(a) CALVIN

![image](images/a0814938d52791530458a0e2f4050868049eefc06f6a533991e23635f21e48c7.jpg)

<details>
<summary>radar</summary>

| Category         | π₀.₅-SFT | π₀.₅-RL |
| ---------------- | -------- | ------- |
| Instruct         | 0.4      | 0.8     |
| Vis-Img          | 0.6      | 0.7     |
| Tex-0.3          | 0.5      | 0.6     |
| Tex-0.5          | 0.4      | 0.5     |
| Whole-0.3        | 0.3      | 0.4     |
| Whole-0.5        | 0.2      | 0.3     |
| M-Carrot (Test)  | 0.1      | 0.2     |
| M-Plate (Test)   | 0.2      | 0.3     |
| M-Plate (Train)  | 0.3      | 0.4     |
| Pos-Chg          | 0.4      | 0.5     |
| Pos               | 0.5      | 0.6     |
</details>

(b) ManiSkill

![image](images/258de3599b5511a161ddc0d8ec526eeb4e3a360a5305311d0b03f833132cdcd2.jpg)

<details>
<summary>line</summary>

| Step | IND Train | OOD Eval |
| ---- | --------- | -------- |
| 0    | 0.6       | 0.5      |
| 50   | 0.7       | 0.6      |
| 100  | 0.75      | 0.65     |
| 150  | 0.73      | 0.58     |
| 200  | 0.74      | 0.6      |
| 250  | 0.75      | 0.55     |
</details>

(c) MetaWorld   
Figure 5. Comprehensive OOD evaluation results on CALVIN ABC-D, ManiSkill OOD, and MetaWorld ML45 benchmarks.

# 5.2.2. OUT-OF-DISTRIBUTION RL EVALUATION

While previous experiments demonstrate that RL yields performance improvements, a critical question remains: does RL yield an enhanced policy, or simply overfit to the ID environment driven by provided rewards? In this section, we evaluate RL-finetuned policies in OOD scenarios, where the environment distribution or the task objective deviates from the ID training setup, to assess their generalization capabilities.

As illustrated in Fig. 5, the performance gains achieved in the ID setting effectively transfer to OOD scenarios in ManiSkill and CALVIN, where the domain shift primarily stems from environmental variations. Conversely, for the OOD setting in MetaWorld, which involves distinct manipulation tasks, performance fluctuates without showing significant improvement. This finding suggests that the benefits of RL are primarily localized to action-level refinement rather than broader augmentation of cross-task generalization capabilities. See Appendix Sec. D for more details.

# 5.3. Ablation Study

Given that Flow-SDE achieves performance comparable to Flow-Noise while offering higher computational efficiency, we conduct our ablation studies with the Flow-SDE method. Specifically, we investigate the impact of critic designs, noise injection strategies, and MDP formulations, with additional results on RL algorithms and hyper-parameters provided in Appendix Sec. F.

# 5.3.1. CRITIC DESIGN

Placement. We compare two critic placement strategies, one positioned after the action expert $( V _ { \mathrm { e x p e r t } } )$ and the other after the VLM $( V _ { \mathrm { v l m } } ) .$ , with $\pi _ { 0 }$ model on the LIBERO-Long task suite. As illustrated in Fig. 6, we observe that $V _ { \mathrm { v l m } }$ exhibits slightly superior performance, lower value loss, and higher explained variance, despite not receiving the proprioceptive state as input. This advantage can be attributed to a key difference in their input: $V _ { \mathrm { v l m } }$ learns a direct mapping from observation to value, while $V _ { \mathrm { e x p e r t } }$ must contend with optimization challenges arising from coupled state and noisy action inputs.

Nevertheless, to align with the concept of the value function, we maintain the $V _ { \mathrm { e x p e r t } }$ architecture for the $\pi _ { 0 } ,$ ensuring that state information is incorporated to estimate the value.

![image](images/0ccbb57ac9d3d1f898930ad16a0c44e6ace7fcc280849ffb9be9a54c77e21a5f.jpg)

<details>
<summary>line</summary>

| Step | One-Layer MLP V_expert | Four-Layer MLP V_expert | Four-Layer MLP V_vlm |
| ---- | ---------------------- | ----------------------- | -------------------- |
| 0    | 0.5                    | 0.5                     | 0.5                  |
| 100  | 0.7                    | 0.8                     | 0.85                 |
| 200  | 0.75                   | 0.85                    | 0.88                 |
| 300  | 0.8                    | 0.88                    | 0.9                  |
| 400  | 0.85                   | 0.9                     | 0.9                  |
</details>

(a) Eval

![image](images/8f376341532fc0f777356533eb0bb2299035354497008ca32e5e469f83086157.jpg)

<details>
<summary>line</summary>

| Step | One-Layer MLP V_expert | Four-Layer MLP V_expert | Four-Layer MLP V_vlm |
| ---- | ---------------------- | ----------------------- | -------------------- |
| 0    | 0.065                  | 0.08                    | 0.065                |
| 10   | 0.045                  | 0.04                    | 0.035                |
| 20   | 0.04                   | 0.035                   | 0.03                 |
| 30   | 0.04                   | 0.035                   | 0.03                 |
| 40   | 0.04                   | 0.035                   | 0.03                 |
| 50   | 0.04                   | 0.035                   | 0.03                 |
</details>

(b) Value Loss

![image](images/37076af7e9b4f820c5cbf7d3e4ede071c776452f378302367383ca3da9f36790.jpg)

<details>
<summary>line</summary>

| Step | One-Layer MLP V_expert | Four-Layer MLP V_expert | Four-Layer MLP V_νlm |
| ---- | ---------------------- | ----------------------- | -------------------- |
| 0    | 0.0                    | 0.0                     | 0.0                  |
| 10   | 0.5                    | 0.6                     | 0.7                  |
| 20   | 0.55                   | 0.7                     | 0.75                 |
| 30   | 0.58                   | 0.72                    | 0.76                 |
| 40   | 0.59                   | 0.73                    | 0.76                 |
| 50   | 0.6                    | 0.74                    | 0.76                 |
</details>

(c) Explained Variance

Figure 6. Ablation on the critic design within Flow-SDE π0 on the LIBERO-Long, indicating that the critic $V _ { \mathrm { v l m } }$ attached after the VLM exhibits superior performance. Furthermore, a four-layer MLP demonstrates stronger regression capability in $V _ { \mathrm { e x p e r t } }$ .   
![image](images/70d66677c1369a301730891816c56f29cc5dc885127f9f51141be01d4f1767a3.jpg)

<details>
<summary>line</summary>

| Step | Two-Layer | Hybrid Two-Layer | One-Layer |
| ---- | --------- | ----------------- | --------- |
| 0    | 0.5       | 0.5               | 0.5       |
| 50   | 0.75      | 0.7               | 0.8       |
| 100  | 0.85      | 0.8               | 0.9       |
| 150  | 0.9       | 0.85              | 0.92      |
| 200  | 0.92      | 0.9               | 0.93      |
</details>

(a) Eval

![image](images/992693225940fa4d019a02977025b03b2c961a43e5c836cb379900231322aa2b.jpg)

<details>
<summary>line</summary>

| Step | Two-Layer | Hybrid Two-Layer | One-Layer |
| ---- | --------- | ---------------- | --------- |
| 0    | 0.0       | 0.0              | 0.0       |
| 50   | 0.8       | 0.8              | 0.8       |
| 100  | 0.85      | 0.85             | 0.8       |
| 150  | 0.85      | 0.85             | 0.8       |
| 200  | 0.85      | 0.85             | 0.8       |
</details>

(b) Explained Variance

![image](images/1e38d385d1e6e327468e89422d290c187dc6b96e38ea212c3349d39d288ef301.jpg)

<details>
<summary>bar</summary>

| Model | Update Time (s) |
| :--- | :--- |
| Two-Layer | 814.2 |
| Hybrid Two-Layer | 428.6 |
| One-Layer | 821.4 |
</details>

(c) Update Time

Figure 7. Ablation on the MDP formulation within Flow-SDE of π0 on the LIBERO-Goal.   
![image](images/1db580b3c2fdf1bda2dfa11d81ad19c623411b3ff316bebe3e481d18982f5e4e.jpg)

<details>
<summary>line</summary>

| Step | Fixed Noise | Learnable Noise |
| ---- | ----------- | --------------- |
| 0    | 0.5         | 0.5             |
| 100  | 0.75        | 0.7             |
| 200  | 0.85        | 0.8             |
| 300  | 0.87        | 0.83            |
| 400  | 0.88        | 0.85            |
</details>

(a) Train

![image](images/f8eadd8f5c0d633018ba64585986972f749753fd81066d5fdc2e398bf8944d31.jpg)

<details>
<summary>line</summary>

| Step | Fixed Noise | Learnable Noise |
| ---- | ----------- | --------------- |
| 0    | 0.5         | 0.5             |
| 100  | 0.75        | 0.7             |
| 200  | 0.85        | 0.8             |
| 300  | 0.88        | 0.82            |
| 400  | 0.89        | 0.85            |
</details>

(b) Eval   
Figure 8. Ablation on the injection strategy within Flow-SDE of $\pi _ { 0 }$ on the LIBERO-Long.

Structure. We investigate a four-layer MLP versus a onelayer MLP, which mirrors the action-projection structure in the action expert. Results in Fig. 6 indicate that the fourlayer MLP leads to a more accurate value approximation, resulting in enhanced performance and training stability.

# 5.3.2. FLOW POLICY MDP

With the same fixed noise injection strategy, we evaluate the one-layer MDP of Flow-Noise with the two-layer MDP of Flow-SDE on the LIBERO-Goal, as shown in Fig. 7.

We observe that the one-layer formulation converges most rapidly, but the final success rates remain consistent across all three formulations. In terms of computational efficiency, the hybrid two-layer paradigm achieves a 2× speedup over the standard approach due to its shorter effective MDP chain. Notably, the one-layer MDP yields no substantial wall-clock time advantage over the standard two-layer model, stemming from the requirement to recalculate full denoising trajectories for log likelihood estimation.

# 5.3.3. STOCHASTICITY INJECTION

We compare fixed and learnable noise injection strategies using the Flow-SDE MDP formulation on the LIBERO-Long suite. To ensure a controlled comparison, we set the entropy bonus for the learnable noise to zero, aligning it with the fixed noise approach.

Specifically, we set the fixed noise to 0.5, and lower and upper bounds for the learnable noise log-variance to 0.08 and 0.16. As depicted in Fig. 8, two noise strategies exhibit similar train performance at step 0, which indicates comparable noise magnitudes. Furthermore, the converged performance affirms the efficiency of both injection methods.

Table 2. Ablation study of hyperparameters for Flow-SDE on the LIBERO-Spatial. Train refers to policy performance during the stochastic rollout phase, whereas Eval refers to performance during the deterministic evaluation phase. 

<table><tr><td rowspan="3">Models</td><td rowspan="3">Stage</td><td colspan="10">Hyperparameters</td></tr><tr><td colspan="3">Noise Level</td><td colspan="4">Denoise Step</td><td colspan="3">Action Chunk</td></tr><tr><td>0.2</td><td>0.5</td><td>0.8</td><td>1</td><td>2</td><td>4</td><td>8</td><td>5</td><td>10</td><td>20</td></tr><tr><td rowspan="2">SFT</td><td>Train</td><td>62.3</td><td>56.0</td><td>46.6</td><td>9.4</td><td>28.3</td><td>56.1</td><td>62.6</td><td>56.0</td><td>60.7</td><td>70.3</td></tr><tr><td>Eval</td><td>65.2</td><td>65.2</td><td>65.2</td><td>63.8</td><td>64.9</td><td>65.2</td><td>63.2</td><td>65.2</td><td>70.5</td><td>72.6</td></tr><tr><td rowspan="2">RL</td><td>Train</td><td>59.5</td><td>93.5</td><td>95.3</td><td>73.8</td><td>90.8</td><td>93.5</td><td>84.3</td><td>93.5</td><td>93.3</td><td>87.5</td></tr><tr><td>Eval</td><td>73.1</td><td>94.5</td><td>98.1</td><td>88.5</td><td>97.0</td><td>94.5</td><td>86.7</td><td>94.5</td><td>95.5</td><td>89.2</td></tr></table>

![image](images/e35cacf280aa38952280f0e365f100dbc561668017617c85f1ceaa77395de77d.jpg)

<details>
<summary>line</summary>

| Step | noise=0.2 | noise=0.5 | noise=0.8 |
| ---- | --------- | --------- | --------- |
| 0    | 0.5       | 0.6       | 0.5       |
| 10   | 0.1       | 0.7       | 0.7       |
| 20   | 0.1       | 0.8       | 0.8       |
| 30   | 0.2       | 0.85      | 0.85      |
| 40   | 0.3       | 0.9       | 0.9       |
| 50   | 0.4       | 0.9       | 0.9       |
| 60   | 0.5       | 0.9       | 0.9       |
| 70   | 0.6       | 0.9       | 0.9       |
| 80   | 0.65      | 0.9       | 0.9       |
| 90   | 0.7       | 0.9       | 0.9       |
| 100  | 0.7       | 0.9       | 0.9       |
</details>

(a) Train

![image](images/b569c7d59aa0740a6148ced3a8577d1381cb150f5846ccd20881edae35c5610e.jpg)

<details>
<summary>line</summary>

| Step | noise=0.2 | noise=0.5 | noise=0.8 |
| ---- | --------- | --------- | --------- |
| 0    | 0.6       | 0.6       | 0.6       |
| 100  | 0.6       | 0.9       | 0.9       |
</details>

(b) Eval

![image](images/50bbe0fc8897a886bd7dbf4331050215f3e02f6431a88437f5f7f8fbaf006f43.jpg)

<details>
<summary>line</summary>

| Step | noise=0.2 | noise=0.5 | noise=0.8 |
| ---- | --------- | --------- | --------- |
| 0    | 0.25      | 0.18      | 0.15      |
| 100  | 0.16      | 0.10      | 0.09      |
</details>

(c) Clipped Fraction

Figure 9. Ablation on the noise level a, conducted with the Flow-SDE π0 on the LIBERO-Spatial.   
![image](images/a3a3df5ba0b0ab7f8652ebe716d18403bdd20cd2f69010c6f61baf17e12feaa9.jpg)

<details>
<summary>line</summary>

| Step | Step 1 | Step 2 | Step 4 | Step 8 |
| ---- | ------ | ------ | ------ | ------ |
| 0    | 0.1    | 0.3    | 0.6    | 0.6    |
| 10   | 0.2    | 0.4    | 0.7    | 0.6    |
| 20   | 0.3    | 0.5    | 0.8    | 0.7    |
| 30   | 0.4    | 0.6    | 0.8    | 0.7    |
| 40   | 0.5    | 0.7    | 0.9    | 0.7    |
| 50   | 0.6    | 0.8    | 0.9    | 0.7    |
| 60   | 0.6    | 0.8    | 0.9    | 0.7    |
| 70   | 0.6    | 0.8    | 0.9    | 0.7    |
| 80   | 0.6    | 0.8    | 0.9    | 0.7    |
| 90   | 0.6    | 0.8    | 0.9    | 0.7    |
| 100  | 0.6    | 0.8    | 0.9    | 0.7    |
</details>

(a) Train

![image](images/b828572a37fc9be56ad793f80b5b816568b601350d864bb954aba5a59f58df8d.jpg)

<details>
<summary>line</summary>

| Step | Step 1 | Step 2 | Step 4 | Step 8 |
| ---- | ------ | ------ | ------ | ------ |
| 0    | 0.65   | 0.65   | 0.65   | 0.65   |
| 100  | 0.90   | 0.95   | 0.95   | 0.75   |
</details>

(b) Eval

Figure 10. Ablation on the denoise step, conducted with the Flow-SDE π0 on the LIBERO-Spatial.   
![image](images/82932a50aa19f13a181ad43a9a3aa350e3734f29ea75a6308b6b2eec3ca826f5.jpg)

<details>
<summary>line</summary>

| Step | chunk=5 | chunk=20 | chunk=10 |
| ---- | ------- | -------- | -------- |
| 0    | 0.65    | 0.70     | 0.70     |
| 10   | 0.75    | 0.72     | 0.78     |
| 20   | 0.80    | 0.75     | 0.85     |
| 30   | 0.85    | 0.78     | 0.90     |
| 40   | 0.88    | 0.80     | 0.92     |
| 50   | 0.90    | 0.82     | 0.93     |
| 60   | 0.91    | 0.84     | 0.94     |
| 70   | 0.92    | 0.86     | 0.95     |
| 80   | 0.93    | 0.87     | 0.96     |
| 90   | 0.94    | 0.88     | 0.97     |
| 100  | 0.95    | 0.89     | 0.98     |
</details>

(a) Eval

![image](images/e75ab865cf293ca341a0d5f1788a92fde4fcea6d2b0d0aeb7603a9c076ae691b.jpg)

<details>
<summary>line</summary>

| Step | chunk=5 | chunk=20 | chunk=10 |
| ---- | ------- | -------- | -------- |
| 0    | 0.0     | 0.0      | 0.0      |
| 10   | 0.7     | 0.4      | 0.5      |
| 20   | 0.75    | 0.5      | 0.6      |
| 30   | 0.78    | 0.6      | 0.65     |
| 40   | 0.8     | 0.65     | 0.68     |
| 50   | 0.8     | 0.65     | 0.68     |
| 60   | 0.8     | 0.65     | 0.68     |
| 70   | 0.8     | 0.65     | 0.68     |
| 80   | 0.8     | 0.65     | 0.68     |
| 90   | 0.8     | 0.65     | 0.68     |
| 100  | 0.8     | 0.65     | 0.68     |
</details>

(b) Explained Variance   
Figure 11. Ablation on the chunk size, conducted with the Flow-SDE π0 on the LIBERO-Spatial.

# 5.4. Hyper-Parameters

Building on the Flow-SDE with $\pi _ { 0 }$ model, we investigate the influence of the noise level, denoise step, and action chunk on the LIBERO-Spatial benchmark. We denote the train stage as the phase where the policy generates stochastic actions for exploration, whereas the evaluation stage involves generating deterministic actions. The train and eval success rates for the SFT baseline and the RL fine-tuned model after 100 training steps are presented in Tab. 2.

Noise Level. The noise level a in the Flow-SDE is defined in

Eq. (8), which governs the noise injection magnitude during the denoising process. As shown in Tab. 2, the evaluation performance of the SFT baseline remains identical across all noise levels due to its reliance on deterministic ODE sampling. Conversely, its training performance exhibits a clear degradation as noise increases. This is intuitive, as higher noise levels can distort the flow trajectory, leading to an inaccurate estimation of the marginal action distribution.

Extending this analysis to the RL fine-tuning stage highlights a critical trade-off: while lower noise levels mitigate exploration-induced performance degradation, they simultaneously constrain the capacity for RL refinement. This trade-off is empirically supported by Fig. 9, which shows that training with minimal noise (a = 0.2) leads to instability, characterized by a significantly higher clip fraction. We attribute this instability to the substantially larger gradient magnitudes associated with low-noise regimes.

Denoise Step. The denoise step K defines the number of discretization steps for action generation and is critical for controlling the fidelity of the ODE-to-SDE transition in Eq. (8). In Tab. 2, we observe that while all configurations start with similar eval performance, the train success rate plummets at K = 1, indicating a significant ODE-to-SDE discretization error.

However, a larger K is not unequivocally optimal. As illustrated in Fig. 10, increasing K presents a distinct trade-off: while it enhances rollout performance, it simultaneously increases training complexity and computational overhead due to the extended sequence of denoising steps.

Action chunk. The action chunk refers to the number of consecutive actions the policy executes within a single observation. We ablate the action chunk size across 5, 10, and 20, with results visualized in Fig. 11.

Although a larger chunk size yields marginal performance gains, it inherently reduces the frequency of policyenvironment interactions and obscures precise reward credit assignment. These constraints lead to less reliable advantage estimation, as evidenced by the diminished explained variance. Consequently, while an increased chunk size may offer a superior SFT baseline, it paradoxically limits the ceiling for subsequent RL-driven refinement.

# 6. Conclusion

We introduce $\pi _ { \mathrm { R L } } .$ , a framework that enables flow-based VLAs, $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ , to be fine-tuned with online RL algorithms. We tackle the fundamental challenge of intractable log-likelihoods in flow matching with Flow-Noise and Flow-SDE solutions. Our extensive experiments on the challenging benchmarks demonstrated that $\pi _ { \mathrm { R L } }$ achieves significant performance improvements over SFT baselines. Limitation. Due to the low sample efficiency of online RL, our framework currently relies on sim-to-real deployment. We aim to develop more efficient algorithms to enable realworld RL training in the future.

# References

Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.   
Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.   
Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.   
Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.   
Chen, G., Li, Z., Wang, S., Jiang, J., Liu, Y., Lu, L., Huang, D.-A., Byeon, W., Le, M., Rintamaki, T., et al. Eagle 2.5: Boosting long-context post-training for frontier visionlanguage models. arXiv preprint arXiv:2504.15271, 2025.   
Chen, R. T., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.   
Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11): 1684–1704, 2025.   
Din, M. U., Akram, W., Saoud, L. S., Rosell, J., and Hussain, I. Vision language action models in robotic manipulation: A systematic review. arXiv preprint arXiv:2507.10672, 2025.   
Fan, H., Dai, H., Zhang, J., Li, J., Yan, Q., Zhao, Y., Gao, M., Wu, J., Tang, H., and Dong, H. Twinaligner: Visual-dynamic alignment empowers physics-aware real2sim2real for robotic manipulation. arXiv preprint arXiv:2512.19390, 2025.   
Fei, S., Wang, S., Shi, J., Dai, Z., Cai, J., Qian, P., Ji, L., He, X., Zhang, S., Fei, Z., et al. Libero-plus: In-depth robust-

ness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.   
Firoozi, R., Tucker, J., Tian, S., Majumdar, A., Sun, J., Liu, W., Zhu, Y., Song, S., Kapoor, A., Hausman, K., et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, 44(5):701–739, 2025.   
Guo, R., Lin, X., Liu, M., Gu, J., and Su, H. Mplib: a lightweight motion planning library. https://github.com/haosulab/MPlib, 2025. URL https://motion-planning-lib. readthedocs.io/latest/.   
He, X., Fu, S., Zhao, Y., Li, W., Yang, J., Yin, D., Rao, F., and Zhang, B. Tempflow-grpo: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025.   
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.   
Hutchinson, M. F. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics-Simulation and Computation, 18(3):1059–1076, 1989.   
Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al. π0.5: a vision-language-action model with openworld generalization. arXiv preprint arXiv:2504.16054, 2025.   
Jiang, G., Chang, H., Qiu, R.-Z., Liang, Y., Ji, M., Zhu, J., Dong, Z., Zou, X., and Wang, X. Gsworld: Closed-loop photo-realistic simulation suite for robotic manipulation. arXiv preprint arXiv:2510.20813, 2025.   
Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.   
Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., Karamcheti, S., Nasiriany, S., Srirama, M. K., Chen, L. Y., Ellis, K., et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.   
Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.   
Kim, M. J., Finn, C., and Liang, P. Fine-tuning visionlanguage-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

Li, H., Zuo, Y., Yu, J., Zhang, Y., Yang, Z., Zhang, K., Zhu, X., Zhang, Y., Chen, T., Cui, G., et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025a.   
Li, J., Cui, Y., Huang, T., Ma, Y., Fan, C., Yang, M., and Zhong, Z. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025b.   
Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.   
Li, Y., Wang, Y., Zhu, Y., Zhao, Z., Lu, M., She, Q., and Zhang, S. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025c.   
Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.   
Liu, J., Gao, F., Wei, B., Chen, X., Liao, Q., Wu, Y., Yu, C., and Wang, Y. What can rl bring to vla generalization? an empirical study. arXiv preprint arXiv:2505.19789, 2025a.   
Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025b.   
Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.   
McAllister, D., Ge, S., Yi, B., Kim, C. M., Weber, E., Choi, H., Feng, H., and Kanazawa, A. Flow matching policy gradients. arXiv preprint arXiv:2507.21053, 2025.   
McLean, R., Chatzaroulas, E., McCutcheon, L., Röder, F., Yu, T., He, Z., Zentner, K., Julian, R., Terry, J., Woungang, I., et al. Meta-world+: An improved, standardized, rl benchmark. arXiv preprint arXiv:2505.11289, 2025.   
Mees, O., Hermann, L., Rosete-Beas, E., and Burgard, W. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

O’Neill, A., Rehman, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., Jain, A., et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024.   
Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.   
Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.   
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.   
Ren, A. Z., Lidard, J., Ankile, L. L., Simeonov, A., Agrawal, P., Majumdar, A., Burchfiel, B., Dai, H., and Simchowitz, M. Diffusion policy policy optimization. arXiv preprint arXiv:2409.00588, 2024.   
Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel, P. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.   
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., et al. Smolvla: A vision-languageaction model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.   
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.   
Tan, S., Dou, K., Zhao, Y., and Krähenbühl, P. Interactive post-training for vision-language-action models. arXiv preprint arXiv:2505.17016, 2025.

Tao, S., Xiang, F., Shukla, A., Qin, Y., Hinrichsen, X., Yuan, X., Bao, C., Lin, X., Liu, Y., Chan, T.-k., et al. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. arXiv preprint arXiv:2410.00425, 2024.   
Team, O. M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.   
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Wang, F. and Yu, Z. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.   
Wen, J., Zhu, Y., Li, J., Zhu, M., Tang, Z., Wu, K., Xu, Z., Liu, N., Cheng, R., Shen, C., et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.   
Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.   
Yu, C., Wang, Y., Guo, Z., Lin, H., Xu, S., Zang, H., Zhang, Q., Wu, Y., Zhu, C., Hu, J., et al. Rlinf: Flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation. arXiv preprint arXiv:2509.15965, 2025.   
Zang, H., Wei, M., Xu, S., Wu, Y., Guo, Z., Wang, Y., Lin, H., Shi, L., Xie, Y., Xu, Z., et al. Rlinf-vla: A unified and efficient framework for vla+ rl training. arXiv preprint arXiv:2510.06710, 2025.   
Zhang, T., Yu, C., Su, S., and Wang, Y. Reinflow: Finetuning flow matching policy with online reinforcement learning. arXiv preprint arXiv:2505.22094, 2025.

# A. Appendix

This appendix provides additional technical details and experimental results for $\pi _ { \mathrm { R L } }$ . The content is organized as:

• Appendix B: Experimental Setup.   
• Appendix C: In-Distribution RL Training.   
• Appendix D: Out-of-Distribution RL Evaluation.   
• Appendix E: Case Studies: Single-Task RL Training.   
• Appendix F: Ablation Details.   
• Appendix G: Insights from Large-Scale RL Training.   
• Appendix H: RL for GR00T N1.5.   
• Appendix I: Limitations and Future Work.   
• Appendix J: Training Hyperparameters.

# B. Experimental Setup

# B.1. Benchmarks.

To rigorously assess the performance and generalization of our framework, we conduct evaluations across four diverse benchmarks, with diverse emphases on robotic capability:

• LIBERO focuses on compositional variations and knowledge transfer in tasks. By evaluating across four task suites: Spatial, Object, Goal, and Long, it probes the model’s ability to transfer base skills to variations in object arrangements and long-horizon tasks.   
• ManiSkill emphasizes perceptual and execution robustness under massive environmental diversity. We follow the setup of RL4VLA (Liu et al., 2025a) and built a benchmark with 4,352 unique pick-and-place combinations, which challenges the policy to maintain physical interactions across a vast distribution of objects and receptacles.   
• CALVIN evaluates long-horizon sequential reasoning and vision-language grounding. It evaluates the model’s capacity to execute chains of five random subtasks in a persistent environment, requiring accurate alignment with linguistic instructions.   
• MetaWorld measures skill breadth and multi-task versatility. It requires a single policy to master 50 semantically distinct manipulation primitives, ranging from simple reaching to complex tool usage.

# B.2. Implementation Details.

Given that pre-trained models often struggle to generalize to task-specific benchmarks, we initiate our process with SFT on expert demonstrations. For the SFT stage, we fine-tune the entire 3.3B model following the official setting. In the subsequent RL stage, we freeze the VLM parameters and exclusively fine-tune the 300M action expert model, driven by GPU memory efficiency and the findings from RL4VLA that RL contributes more significantly to action generalization. We build the whole framework upon the RLinf (Yu et al., 2025) codebase, where we adopt a shared, co-located GPU allocation strategy that places the environment, rollout model, and actor model on the same GPU and executes them serially.

For the model configurations, we adhere to the official setting provided by openpi (Black et al., 2024; Intelligence et al., 2025). In these settings, π0 utilizes image, language, and proprioceptive states as input, whereas $\pi _ { 0 . 5 }$ notably omits state information2. Our experiments are conducted on 8 NVIDIA H100 80GB GPUs, and detailed training hyperparameters are available in the Appendix Sec. J.

# C. In-Distribution RL Training

In this section, we evaluate the effectiveness of our framework across various benchmarks to demonstrate its robustness and superior performance in ID settings.

# C.1. LIBERO Benchmark

SFT Procedure. The LIBERO benchmark comprises four task suites, each consisting of 10 distinct subtasks. To facilitate few-shot SFT on LIBERO, a minimum of 40 expert demonstration trajectories is necessary to ensure a positive success rate for each subtask across four task suites, thereby guaranteeing a positive optimization signal for the subsequent RL phase.

We perform few-shot SFT following the official training configurations. For the $\pi _ { 0 }$ model, we fine-tune on a subset of 58 trajectories sampled from the 1,692 total demonstrations3, which serves as the initial checkpoint for subsequent RL training on the Spatial, Object, and Goal task suites. For the Long task suite, a larger pool of 208 trajectories is employed to address its more challenging, long-horizon nature. In contrast, for the $\pi _ { 0 . 5 }$ model, benefiting from a superior pre-trained checkpoint and training configurations, we leverage only 40 trajectories to provide a unified few-shot SFT checkpoint across all task suites.

RL Procedure. In RL, the VLA model receives a multimodal input state comprising: an agent-view and a wristview, natural language guidance, the robot end effector pose, and the gripper state. The model outputs an action to interact with the LIBERO environment, which provides a binary reward of 1 for successful task completion and 0 otherwise.

Table 3. Evaluation results on the LIBERO benchmark, evaluated based on the success rate (%). 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="6">LIBERO</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td><td>Δ Avg.</td></tr><tr><td colspan="8"># Full Dataset SFT</td></tr><tr><td colspan="2">Octo</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td><td>—</td></tr><tr><td colspan="2">OpenVLA</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td><td>—</td></tr><tr><td colspan="2"> $\pi_{\text{fast}}$ </td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>85.5</td><td>—</td></tr><tr><td colspan="2">OpenVLA-OFT</td><td>91.6</td><td>95.3</td><td>90.6</td><td>86.5</td><td>91.0</td><td>—</td></tr><tr><td colspan="2"> $\pi_0$ </td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td><td>—</td></tr><tr><td colspan="2"> $\pi_{0.5}$ </td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td><td>—</td></tr><tr><td colspan="8"># Few-shot SFT + RL</td></tr><tr><td rowspan="3"> $\pi_0$ </td><td>SFT</td><td>65.3</td><td>64.4</td><td>49.8</td><td>51.2</td><td>57.6</td><td>—</td></tr><tr><td>Flow-SDE</td><td>98.4</td><td>99.4</td><td>96.2</td><td>90.2</td><td>96.1</td><td>+38.5</td></tr><tr><td>Flow-Noise</td><td>99.0</td><td>99.2</td><td>98.2</td><td>93.8</td><td>97.6</td><td>+40.0</td></tr><tr><td colspan="8"># Few-shot SFT + RL</td></tr><tr><td rowspan="3"> $\pi_{0.5}$ </td><td>SFT</td><td>84.6</td><td>95.4</td><td>84.6</td><td>43.9</td><td>77.1</td><td>—</td></tr><tr><td>Flow-SDE</td><td>99.6</td><td>100</td><td>98.8</td><td>93.0</td><td>97.9</td><td>+20.8</td></tr><tr><td>Flow-Noise</td><td>99.6</td><td>100</td><td>99.6</td><td>94.0</td><td>98.3</td><td>+21.2</td></tr></table>

Table 4. Evaluation results on the ManiSkill benchmark, with more specific OOD results depicted in Tab. 7. 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="2">IND</td><td colspan="5">OOD</td></tr><tr><td>Avg.</td><td>Δ</td><td>Vision</td><td>Semantic</td><td>Execution</td><td>Avg.</td><td>Δ</td></tr><tr><td rowspan="3"> $\pi_0$ </td><td>SFT</td><td>38.4</td><td>—</td><td>32.6</td><td>8.4</td><td>13.2</td><td>18.1</td><td>—</td></tr><tr><td>Flow-SDE</td><td>78.8</td><td>+40.4</td><td>61.1</td><td>25.4</td><td>31.5</td><td>39.3</td><td>+21.2</td></tr><tr><td>Flow-Noise</td><td>77.8</td><td>+39.4</td><td>63.4</td><td>23.1</td><td>24.2</td><td>36.9</td><td>+18.8</td></tr><tr><td rowspan="3"> $\pi_{0.5}$ </td><td>SFT</td><td>40.1</td><td>—</td><td>40.2</td><td>16.6</td><td>22.4</td><td>26.4</td><td>—</td></tr><tr><td>Flow-SDE</td><td>90.9</td><td>+50.8</td><td>68.0</td><td>34.5</td><td>45.4</td><td>49.3</td><td>+22.9</td></tr><tr><td>Flow-Noise</td><td>89.7</td><td>+49.6</td><td>69.9</td><td>35.5</td><td>54.9</td><td>53.4</td><td>+27.0</td></tr></table>

Experiments. We benchmark the performance of $\pi _ { \mathrm { R L } } ,$ which fine-tunes the few-shot SFT $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ models with Flow-Noise and Flow-SDE, against several state-of-the-art VLAs trained on the entire LIBERO dataset, including Octo, OpenVLA, OpenVLA-OFT, $\pi _ { \mathrm { f a s t } }$ (Pertsch et al., 2025), $\pi _ { 0 } ,$ and $\pi _ { 0 . 5 }$ . We conduct experiments on four LIBERO task suites and report performance as the success rate across all 500 initial states (10 sub-tasks × 50 states each).

Analysis. For the few-shot $\pi _ { 0 }$ model, the SFT baseline performs poorly, with an average success rate of only 57.6%, indicating that the model struggles with limited demonstration data. $\pi _ { \mathrm { R L } }$ substantially boosts performance, with Flow-SDE and Flow-Noise reaching 96.1% and 97.6%, and surpassing the full-dataset $\pi _ { 0 }$ SFT baseline of 94.2%.

While the $\pi _ { 0 . 5 }$ few-shot SFT baseline achieves a decent average performance of 77.1%, it struggles with the challenging Long task suite, scoring only 43.9%. Our proposed $\pi _ { \mathrm { R L } }$ rectifies this deficiency, boosting the Long task success rate from 43.9% to 94.0%, constituting a 50.1% improvement. Notably, despite using only a single trajectory for $\mathrm { S F T } , \pi _ { \mathrm { R L } }$ reaches 98.3% final performance, surpassing the 96.9% full-dataset SFT model.

# C.2. ManiSkill Benchmark

SFT Procedure. In the ManiSkill benchmark, the policy is required to pick from 16 object types and place them onto 17 receptacles across 16 unique table scenes, yielding 4,352 unique task combinations. Given the high complexity of this setting, the SFT dataset consists of 16,384 episodes synthesized using the MPLib motion planning suite (Guo et al., 2025). To reinforce the concept of motion completion, 15 additional frames are appended to the end of each trajectory.

Table 5. Evaluation results on the CALVIN benchmark (Scene D), reporting the average completed subtasks and success rates for task sequences of length 1 to 5. 

<table><tr><td rowspan="2" colspan="2">Methods</td><td colspan="7">CALVIN-D</td></tr><tr><td>Len-1</td><td>Len-2</td><td>Len-3</td><td>Len-4</td><td>Len-5</td><td>Avg.</td><td>Δ Avg.</td></tr><tr><td rowspan="3"> $\pi_0$ </td><td>SFT</td><td>94.7</td><td>84.9</td><td>74.3</td><td>65.2</td><td>57.5</td><td>3.766</td><td>—</td></tr><tr><td>Flow-SDE</td><td>96.4</td><td>88.0</td><td>77.5</td><td>70.8</td><td>61.7</td><td>3.944</td><td>+0.178</td></tr><tr><td>Flow-Noise</td><td>96.9</td><td>88.8</td><td>78.0</td><td>68.3</td><td>59.9</td><td>3.919</td><td>+0.153</td></tr><tr><td rowspan="3"> $\pi_{0.5}$ </td><td>SFT</td><td>92.7</td><td>84.3</td><td>76.7</td><td>68.8</td><td>61.3</td><td>3.838</td><td>—</td></tr><tr><td>Flow-SDE</td><td>99.7</td><td>98.2</td><td>95.8</td><td>91.0</td><td>87.0</td><td>4.717</td><td>+0.879</td></tr><tr><td>Flow-Noise</td><td>99.6</td><td>97.6</td><td>93.9</td><td>89.6</td><td>84.5</td><td>4.652</td><td>+0.814</td></tr></table>

Table 6. Evaluation results on the MetaWorld MT50 benchmark. 

<table><tr><td rowspan="2" colspan="2">Methods</td><td colspan="6">MetaWorld-MT50</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>Very Hard</td><td>Avg.</td><td>Δ Avg.</td></tr><tr><td colspan="2">Diffusion Policy</td><td>23.1</td><td>10.7</td><td>1.9</td><td>6.1</td><td>10.5</td><td>—</td></tr><tr><td colspan="2">TinyVLA</td><td>77.6</td><td>21.5</td><td>11.4</td><td>15.8</td><td>31.6</td><td>—</td></tr><tr><td colspan="2">SmolVLA</td><td>87.1</td><td>51.8</td><td>70.0</td><td>64.0</td><td>68.2</td><td>—</td></tr><tr><td rowspan="3">π0</td><td>SFT</td><td>77.9</td><td>51.8</td><td>53.3</td><td>20.0</td><td>50.8</td><td>—</td></tr><tr><td>Flow-SDE</td><td>92.1</td><td>74.6</td><td>61.7</td><td>84.0</td><td>78.1</td><td>+27.3</td></tr><tr><td>Flow-Noise</td><td>91.1</td><td>81.8</td><td>78.3</td><td>92.0</td><td>85.8</td><td>+35.0</td></tr><tr><td rowspan="3">π0.5</td><td>SFT</td><td>68.2</td><td>37.3</td><td>41.7</td><td>28.0</td><td>43.8</td><td>—</td></tr><tr><td>Flow-SDE</td><td>86.4</td><td>55.5</td><td>75.0</td><td>66.0</td><td>70.7</td><td>+26.9</td></tr><tr><td>Flow-Noise</td><td>86.8</td><td>58.1</td><td>63.3</td><td>56.0</td><td>66.1</td><td>+22.3</td></tr></table>

RL Procedure. In RL, the VLA model receives a thirdperson RGB image, a concise language instruction, and the current joint proprioception. The environment provides a structured reward signal: 1.0 for correct object placement and an auxiliary 0.1 reward for successful gripper-object attachment, intended to encourage stable manipulation and mitigate undesired behaviors such as impulsive throwing.

Experiments. Following the RL4VLA experimental protocol, we conduct RL training on a comprehensive set of 4,352 task combinations and record the performance as the aggregate success rate across these tasks.

Analysis. As detailed in Tab. 4, $\pi _ { \mathrm { R L } }$ significantly boosts performance in the training environment. Specifically, the success rate of $\pi _ { 0 }$ increases from 38.4% to 77.8%, while $\pi _ { 0 . 5 }$ improves from 40.1% to 90.9%. These gains underscore the efficacy of RL in complex settings.

# C.3. CALVIN Benchmark

SFT Procedure. We conduct SFT on the CALVIN ABC dataset4, which comprises approximately 24 hours of unstructured "play data" across three distinct environments (A, B, and C). This dataset includes over 20,000 languagelabeled trajectories covering 34 unique manipulation tasks.

RL Procedure. Each episode consists of a sequence of five randomly sampled subtasks to be completed in succession without environment resets. The reward signal is defined at the subtask level, where the model receives a sparse binary reward of 1.0 for each successfully executed subtask and 0.0 otherwise.

Experiments. We evaluate the performance of $\pi _ { \mathrm { R L } }$ in Scene D over 1,000 episodes. Following the standard CALVIN evaluation protocol, we report two key metrics: (1) the success rate for task sequences of increasing lengths, namely Len-1 to Len-5. (2) the average number of completed subtasks, denoted as Avg. per episode.

Analysis. As detailed in Tab. $5 , \pi _ { \mathrm { R I } }$ yields substantial performance gains in long-horizon sequential execution, particularly with the $\pi _ { 0 . 5 }$ variant. The SFT models inherently struggle with compounding errors across sequential tasks, with $\pi _ { 0 . 5 }$ only achieving a 61.3% success rate on Len-5 sequences. RL effectively mitigates this issue with the average completed sub-tasks of Flow-SDE increasing from 3.838 to 4.717, and its Len-5 success rate surges to 87.0%.

Table 7. Specific generalization evaluation results in the ManiSkill OOD setting. 

<table><tr><td>Environment</td><td>Variation-Version-Type</td><td> $\pi_0$ -SFT</td><td> $\pi_0$ -RL Flow-SDE</td><td> $\pi_0$ -RL Flow-Noise</td><td> $\pi_{0.5}$ -SFT</td><td> $\pi_{0.5}$ -RL Flow-SDE</td><td> $\pi_{0.5}$ -RL Flow-Noise</td></tr><tr><td>In distribution</td><td>Main-v3-train</td><td>38.4</td><td>78.8</td><td>77.8</td><td>40.1</td><td>90.9</td><td>89.7</td></tr><tr><td rowspan="6">Visual-Language Variations</td><td>Instruct-v1-test</td><td>30.1</td><td>64.6</td><td>66.5</td><td>46.6</td><td>77.0</td><td>85.7</td></tr><tr><td>VisionImage-v1-test</td><td>38.3</td><td>68.8</td><td>71.7</td><td>46.2</td><td>78.8</td><td>83.1</td></tr><tr><td>VisionTexture03-v1-test</td><td>35.1</td><td>66.0</td><td>66.8</td><td>36.7</td><td>69.6</td><td>75.0</td></tr><tr><td>VisionTexture05-v1-test</td><td>31.0</td><td>55.8</td><td>60.5</td><td>32.7</td><td>58.0</td><td>62.2</td></tr><tr><td>VisionWhole03-v1-test</td><td>35.4</td><td>62.4</td><td>69.0</td><td>40.1</td><td>69.6</td><td>71.6</td></tr><tr><td>VisionWhole05-v1-test</td><td>28.5</td><td>49.0</td><td>53.9</td><td>30.7</td><td>55.0</td><td>57.0</td></tr><tr><td rowspan="4">Semantic Reasoning (object/receptacle confounders)</td><td>MultiCarrot-v1-test</td><td>7.8</td><td>28.2</td><td>23.0</td><td>16.7</td><td>36.8</td><td>38.2</td></tr><tr><td>MultiCarrot-v1-train</td><td>12.5</td><td>36.5</td><td>31.8</td><td>28.2</td><td>49.5</td><td>50.1</td></tr><tr><td>MultiPlate-v1-test</td><td>5.0</td><td>16.4</td><td>18.3</td><td>11.8</td><td>29.4</td><td>28.3</td></tr><tr><td>MultiPlate-v1-train</td><td>7.3</td><td>20.5</td><td>19.6</td><td>9.7</td><td>22.3</td><td>25.4</td></tr><tr><td rowspan="2">Action Execution</td><td>PositionChangeTo-v1-test</td><td>9.6</td><td>17.4</td><td>10.9</td><td>13.5</td><td>36.2</td><td>54.7</td></tr><tr><td>Position-v1-test</td><td>16.9</td><td>45.6</td><td>37.5</td><td>31.2</td><td>54.5</td><td>55.0</td></tr></table>

Notably, the performance gap between SFT and RL widens significantly as the sequence length increases. For $\pi _ { 0 . 5 } ,$ while Flow-SDE shows a modest 7.0% improvement over SFT in Len-1 tasks, the gap expands to an impressive 25.7% in the most challenging Len-5 sequences.

# C.4. MetaWorld Benchmark

SFT Procedure. We perform SFT on the $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ models using the official dataset5, which consists of 2500 trajectories across 50 different manipulation tasks.

RL Procedure. During the RL procedure, the VLA model processes a multi-modal input comprising a RGB agentview image, language guidance, the robot’s end-effector position, and its gripper state. Based on this input, the model outputs an action to interact with the environment, which in turn provides a sparse reward: 1 for successful task completion and 0 otherwise.

Experiments. We benchmark the performance of $\pi _ { \mathrm { R I } }$ against Diffusion Policy (Chi et al., 2025), TinyVLA (Wen et al., 2025), and SmolVLA (Shukor et al., 2025). For the performance evaluation, we follow the setup from SmolVLA, i.e., classifying 50 tasks into easy, medium, hard, and very hard four categories according to their difficulties.

Analysis. As detailed in Tab. 6, RL fine-tuning substantially boosts performance. The $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ models achieve average success rates of 85.8% and 70.7%, respectively. This marks a significant improvement over their SFT-only counterparts and surpasses the SmolVLA baseline of 68.2%, confirming that RL can effectively enhance model capabilities across a diverse range of manipulation task types.

# D. Out-of-Distribution RL Evaluation

While previous experiments demonstrate significant RLdriven improvements in the ID domain, this section evaluates OOD generalization. As LIBERO lacks a dedicated interface for OOD testing, we utilize the ManiSkill, CALVIN, and MetaWorld benchmarks to investigate whether the RLdriven improvements represent genuine skill acquisition that scales to novel settings, or merely reflect the exploitation of environment-specific biases.

# D.1. ManiSkill

Setup. Following RL4VLA, we evaluate the model’s generalization across three challenging OOD scenarios: (1) Vision, challenging the model with novel backgrounds and textures; (2) Semantics, probing comprehension with unseen objects, varied instructions, and confounding elements like extra objects or receptacles; (3) Execution, assessing robustness against varied initial states, unseen robot poses, and dynamic disturbances.

Results. In the OOD scenarios detailed in Tabs. 4 and 7, we observe that the $\pi _ { 0 } { - } S \mathrm { F T }$ model demonstrates strong generalization for visual information. This can be attributed to the robust foundation of its VLM, which allows it to handle visual disturbances better.

However, the semantic performance of $\pi _ { 0 }$ drops dramatically. This degradation is less pronounced when switching to the $\pi _ { 0 . 5 }$ baseline, a benefit likely stemming from the knowledge generalization of the pre-trained $\pi _ { 0 . 5 }$ model. Regarding action execution, $\pi _ { 0 }$ exhibits a larger performance drop than $\pi _ { 0 . 5 }$ . We hypothesize that this discrepancy arises from the inclusion of joint angle states as input in π0, leading to severe overfitting in the control task. In contrast, π0.5 omits these inputs, thereby avoiding the same degree of performance degradation.

Table 8. Evaluation results on the SIMPLER benchmark for π0 and π0.5 with Flow-Noise method. 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="5">SIMPLER</td></tr><tr><td>Carrot</td><td>Eggplant</td><td>Spoon</td><td>Cube</td><td>Avg.</td></tr><tr><td rowspan="3"> $\pi_0$ </td><td>SFT</td><td>82.7</td><td>87.5</td><td>61.7</td><td>37.1</td><td>67.2</td></tr><tr><td>+RL</td><td>95.7</td><td>96.7</td><td>91.6</td><td>63.0</td><td>86.7</td></tr><tr><td>Δ</td><td>+13.0</td><td>+9.2</td><td>+29.9</td><td>+25.9</td><td>+19.5</td></tr><tr><td rowspan="3"> $\pi_{0.5}$ </td><td>SFT</td><td>70.6</td><td>91.9</td><td>43.5</td><td>31.0</td><td>59.2</td></tr><tr><td>+RL</td><td>82.0</td><td>98.2</td><td>82.8</td><td>53.3</td><td>79.1</td></tr><tr><td>Δ</td><td>+11.4</td><td>+6.3</td><td>+39.3</td><td>+22.3</td><td>+19.9</td></tr></table>

![image](images/e4e00dd2696302ec6b8a3d3278e4e82a59c754ee177f6d4aaecfabd720dfff25.jpg)

<details>
<summary>natural_image</summary>

Sequence of five photos showing a robotic arm performing a 3D printing or stamping process on a patterned floor (no text or symbols visible)
</details>

Figure 12. Real-world deployment of an RL refined policy performing a pick and place task.

As for the RL training, although the performance improvements in OOD scenarios are lower than those in IND settings, the proportional improvements achieved are notably comparable. As indicated in Tab. 4, for the $\pi _ { 0 . 5 }$ model, Flow-SDE enhances the IND success rate by 126.7%, while the OOD similarly increases by 102.3%. This consistency in relative gains indicates that RL-driven optimization promotes the acquisition of generalized action representations rather than merely overfitting the training environment, thus preserving efficacy under distribution shifts.

Nevertheless, we observe a performance gap in relative improvement between the Vision OOD tasks and the IND domain. Specifically, for $\pi _ { 0 . 5 } .$ , the 73.9% gain in visual generalization trails the 126.7% increase observed in the training environment. This discrepancy likely stems from freezing the VLM backbone during the RL stage for computational efficiency, which restricts the model’s ability to adapt its visual grounding features to novel textures and backgrounds.

# D.2. CALVIN

Setup. We evaluate environmental and visual OOD robustness based on the ABC → D protocol in CALVIN. Under this setting, the model is trained on ABC environments and evaluated in a zero-shot manner on Scene D. Scene D introduces significant distribution shifts in terms of visual textures, lighting conditions, and spatial layouts, effectively assessing the agent’s ability to transfer skills to an unfamiliar physical environment.

Results. Under identical D→ D training settings, the RL finetuned policy in the ABC environment reaches a 79.1% success rate in the OOD scene D, improving over the 61.3% SFT baseline as shown in Fig. 5. Aligned with the findings in ManiSkill, this suggests that ID gains can be transferred to OOD settings characterized by visual variations.

# D.3. MetaWorld

Setup. We utilize the ML45 benchmark from MetaWorld to evaluate the task-level generalization. This setup consists of 50 distinct robotic manipulation tasks: the agent is trained on 45 base tasks and subsequently evaluated on 5 heldout, unseen tasks, which require the model to generalize its learned manipulation primitives to entirely novel task objectives and workspace configurations.

Results. As evidenced in the Fig. 5, while success rates show consistent gains within the ID domain, OOD performance is characterized by persistent oscillation throughout the training process. This instability indicates that RL, in its current form, struggles to foster stable cross-category generalization.

Nevertheless, the model retains the OOD skills learned during the SFT phase throughout the RL training process. This highlights a significant advantage over standard SFT, which often causes the model to overfit on expert demonstrations and lose its broader capabilities (Li et al., 2025a). Unlike standard SFT, RL enables ID performance gains while preserving the general knowledge, which indicates that RL provides a more balanced optimization that maintains OOD robustness without catastrophic forgetting.

Table 9. Comparison of the PPO and GRPO with Flow-SDE on the LIBERO. 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="6">LIBERO</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td><td> $\Delta$  Avg.</td></tr><tr><td rowspan="3"> $\pi_0$ </td><td>SFT</td><td>65.3</td><td>64.4</td><td>49.8</td><td>51.2</td><td>57.6</td><td>—</td></tr><tr><td>+GRPO</td><td>97.8</td><td>97.8</td><td>83.2</td><td>81.4</td><td>90.0</td><td>+32.4</td></tr><tr><td>+PPO</td><td>98.4</td><td>99.4</td><td>96.2</td><td>90.2</td><td>96.0</td><td>+38.4</td></tr><tr><td rowspan="3"> $\pi_{0.5}$ </td><td>SFT</td><td>84.6</td><td>95.4</td><td>84.6</td><td>43.9</td><td>77.1</td><td>—</td></tr><tr><td>+GRPO</td><td>97.4</td><td>99.8</td><td>91.2</td><td>77.6</td><td>91.5</td><td>+14.4</td></tr><tr><td>+PPO</td><td>99.6</td><td>100</td><td>98.8</td><td>93.0</td><td>97.9</td><td>+20.8</td></tr></table>

# D.4. Summary

In conclusion, our OOD evaluation demonstrates that RL enhances performance for similar tasks but fails to generalize effectively to novel task objectives.

Specifically, RL training effectively enhances robustness against low-level variations such as the visual and execution shifts observed in ManiSkill and CALVIN. This indicates that the model acquires generalized action representations rather than merely overfitting to the training environment.

Regarding high-level generalization on MetaWorld, the model successfully retains the OOD skills inherited from the SFT phase, demonstrating that RL avoids the catastrophic forgetting and overfitting typical of standard imitation learning. However, transferring its performance gains to entirely novel task objectives remains a significant challenge.

# E. Case Studies: Single-Task RL Training

While the preceding experiments focused on performance across multi-task benchmarks, this section investigates single-task scenarios where the VLA is trained to master a specific task within a relatively static environment. Specifically, we evaluate our approach on the SIMPLER (Li et al., 2024) and a Real2Sim2Real environment.

# E.1. SIMPLER

Setup. In SIMPLER, the experimental setup comprises an 8-DoF WidowX-250S arm evaluated on four standard tasks: (1) Spoon: placing a spoon on a cloth. (2) Carrot: placing a carrot on a plate. (3) Eggplant: placing an eggplant in a basket. (4) Cube: stacking a cube. For the SFT stage, we employ a curated dataset in which each task is trained with 144 demonstration episodes.

Analysis. As detailed in Tab. $8 , \pi _ { \mathrm { R I } }$ increases the average success rate of the $\pi _ { 0 }$ model from 67.2% to 86.7%, with three tasks (carrot, eggplant, and spoon) exceeding 90% success.

# E.2. Real2Sim2Real

While the SIMPLER benchmark demonstrates predictive correlation between simulation and real-world performance, a pronounced visual domain shift remains to be solved. To this end, we leverage recent Real2Sim2Real methodologies (Fan et al., 2025; Jiang et al., 2025) to construct a highfidelity simulation environment with ManiSkill for rigid body dynamics and Gaussian Splatting (Kerbl et al., 2023) for photorealistic rendering.

Setup. Our hardware platform comprises a Franka Panda robotic arm and an Intel RealSense D435 camera serving as the primary visual sensor. We perform manual calibration by aligning simulated viewpoints with real-world camera perspectives to synchronize their extrinsic matrices. As illustrated in Fig. 1, the photorealistic textures and color profiles in our simulator closely mirror the physical environment, effectively minimizing the visual domain shift from simulation to reality.

Results. Following the experimental protocol established in the ManiSkill benchmarks, we initially collect 20 expert trajectories via a motion planner for few-shot SFT, which is subsequently optimized through RL over 100 training iterations. We deploy the RL fine-tuned policies in the real world in a zero-shot manner. Empirical results indicate that while the SFT baseline fails to complete the task, the RLtuned policy achieves a 40% success rate. A representative successful episode is visualized in Fig. 12.

# F. Ablation Details

# F.1. RL algorithms

Given the significant performance gains from PPO on the LIBERO benchmark, we also investigated the effectiveness of GRPO (Shao et al., 2024), another widely used policy gradient method applied in VLA+RL training (Li et al., 2025a). We compare the performance of PPO and GRPO on both the $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ models, with results denoted in Tab. 9.

Conclusion. To sum up, our findings highlight a critical trade-off: parameters tailored for rollout success may adversely impact training stability, ultimately constraining the performance ceiling of RL. Therefore, careful parameter tuning is required to achieve a synergy between high-quality rollouts and stable policy convergence.

![image](images/a66266fb4c191f9af857cf9541e610b14ef78f98c337d8be009d44ab2decce23.jpg)

<details>
<summary>line</summary>

| Step | VLM Frozen | VLM LoRA-I | VLM LoRA-II |
| ---- | ---------- | ---------- | ----------- |
| 0    | 0.5        | 0.5        | 0.5         |
| 50   | 0.7        | 0.6        | 0.7         |
| 100  | 0.8        | 0.7        | 0.8         |
| 150  | 0.85       | 0.75       | 0.85        |
</details>

(a) Eval

![image](images/d2cfccf9e9cf15e7c35bcfea38d49e29a1b1ec6312cb5a05b8190145f6f52d8d.jpg)

<details>
<summary>line</summary>

| Step | VLM Frozen | VLM LoRA-I | VLM LoRA-II |
| ---- | ---------- | ---------- | ----------- |
| 0    | 0.02       | 0.08       | 0.02        |
| 50   | 0.01       | 0.09       | 0.01        |
| 100  | 0.01       | 0.09       | 0.01        |
| 150  | 0.01       | 0.08       | 0.01        |
</details>

(b) KL Divergence

Figure 13. Ablation study on the effectiveness of VLM during RL.   
![image](images/2646da8c6be3d79cfdba007938255d61446983357763419329cb02aef11cb1e2.jpg)

<details>
<summary>line</summary>

| Step | No Scheduler | Cosine Scheduler |
| ---- | ------------ | ---------------- |
| 0    | 0.4          | 0.4              |
| 50   | 0.6          | 0.65             |
| 100  | 0.7          | 0.75             |
| 150  | 0.75         | 0.8              |
| 200  | 0.8          | 0.82             |
| 250  | 0.82         | 0.85             |
| 300  | 0.85         | 0.87             |
| 350  | 0.87         | 0.88             |
</details>

(a) Eval

![image](images/968b10af509f01ca70cf2513f04a80003249ed40d143d08007b333e5b589f69c.jpg)

<details>
<summary>line</summary>

| Step | No Scheduler | Cosine Scheduler |
| ---- | ------------ | ---------------- |
| 0    | 0.005        | 0.005            |
| 100  | 0.008        | 0.007            |
| 200  | 0.012        | 0.010            |
| 300  | 0.025        | 0.012            |
| 350  | 0.035        | 0.013            |
</details>

(b) KL Divergence   
Figure 14. Ablation study on the learning rate scheduler.

# F.2. VLM Fine-tuning Analysis

In our previous experiments, the VLM is frozen, and the optimization is confined exclusively to the action expert during RL. In this subsection, we aim to investigate the role of the VLM during RL. Specifically, we employ Low-Rank Adaptation (LoRA) (Hu et al., 2022) for the VLM, facilitating its joint optimization with the action expert. We set the LoRA rank to r = 32 and the scaling parameter to α = 32, while the action expert remains fully trainable.

We conduct experiments with the π0 model with Flow-SDE on the LIBERO-Long benchmark, comparing three distinct configurations:

• VLM Frozen: $5 e ^ { - 6 }$ learning rate, 4 updates/epoch.   
• VLM LoRA-I: $5 e ^ { - 6 }$ learning rate, 4 updates/epoch.   
• VLM LoRA-II: $1 e ^ { - 6 }$ learning rate, 2 updates/epoch.

As presented in Fig. 13, the VLM LoRA-II configuration achieves a learning trajectory comparable to the VLM frozen baseline. This empirical observation yields two critical inferences: (1) The benefit of fine-tuning the VLM on the LIBERO benchmark is not evident. We conjecture the limited performance gain owing to the limited scene variability within LIBERO, for which the pretrained VLM representations are already sufficiently robust. (2) Fine-tuning VLM together with the action expert requires a more conservative optimization configuration for training stability.

# G. Insights from Large-Scale RL Training

In this section, we elaborate on some empirical insights we gained during RL training.

Hyperparameters. According to the hyperparameters ablation detailed in Sec. 5.4, the performance disparity between the train and eval performance of the initial SFT checkpoint warrants close attention. If this disparity is significant, we recommend either reducing the noise magnitude or increasing the number of denoising steps to mitigate the performance loss when shifting from deterministic to stochastic execution. Furthermore, as previously established, lower noise levels yield larger gradients, requiring a smaller learning rate to maintain training stability.

We also observed that when train performance improves steadily while eval performance oscillates, increasing the number of denoising steps can help alleviate this, benefiting from reduced divergence in the action distributions between the deterministic and stochastic action generation processes. Regarding the action chunk, we empirically found that longhorizon tasks benefit from larger chunk sizes. For instance, we set the chunk size to 10 for LIBERO-Long and 5 for the

![image](images/b69e963e11538fad9d21c1b55a4acf889e526a793cf8762c8d4a4c2d9834c21e.jpg)

<details>
<summary>line</summary>

| Step | π₀    | π₀.₅  |
| ---- | ----- | ----- |
| 0    | 0.3   | 0.4   |
| 10   | 0.65  | 0.65  |
| 20   | 0.75  | 0.75  |
| 30   | 0.8   | 0.8   |
| 40   | 0.85  | 0.85  |
| 50   | 0.85  | 0.85  |
</details>

(a) Eval

![image](images/9b2950459a1beba0ff6708e43a19acdb456425df25d09b8d2baec9960bc832a4.jpg)

<details>
<summary>line</summary>

| Step | π₀    | π₀.₅  |
| ---- | ----- | ----- |
| 0    | 0.2   | 0.0   |
| 10   | 0.6   | 0.45  |
| 20   | 0.7   | 0.5   |
| 30   | 0.75  | 0.55  |
| 40   | 0.75  | 0.55  |
| 50   | 0.75  | 0.55  |
</details>

(b) Explained Variance

Figure 15. Training curves in ManiSkill.   
![image](images/7757c34a6728cb404b5102d106c7ef1ee383b60bca91c27233805182932e7787.jpg)

<details>
<summary>line</summary>

| Step | Flow-SDE | Flow-Noise | Expert (34.5±2.6) |
| ---- | -------- | ---------- | ----------------- |
| 0    | 64.0     | 63.0       | 34.5 ± 2.6        |
| 20   | 50.0     | 48.0       | 34.5 ± 2.6        |
| 40   | 45.0     | 42.0       | 34.5 ± 2.6        |
| 60   | 42.0     | 39.0       | 34.5 ± 2.6        |
| 80   | 40.0     | 37.0       | 34.5 ± 2.6        |
| 100  | 39.0     | 36.0       | 34.5 ± 2.6        |
| 120  | 38.0     | 35.5       | 34.5 ± 2.6        |
| 140  | 37.5     | 35.0       | 34.5 ± 2.6        |
| 150  | 37.0     | 35.0       | 34.5 ± 2.6        |
</details>

Figure 16. Episode length: $\pi _ { 0 . 5 }$ RL training in ManiSkill.

other sub-tasks.

Training. In our $\pi _ { 0 . 5 }$ experiments on the LIBERO-Long benchmark, we observed that the Kullback–Leibler (KL) divergence metric increased steadily throughout training, potentially leading to instability. We mitigated this issue by implementing a learning rate scheduler with cosine annealing. As demonstrated in Fig. 14, this scheduler effectively prevents the KL divergence from escalating, thereby stabilizing the training process.

Critic. In our ManiSkill experiments, we observe that policy evaluation performance exhibits an initial dip before improving for both $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ models, as shown in Fig. 15. We attribute this transient degradation to the critic providing inaccurate signals during its warm-up phase. The subsequent eval improvement correlates directly with the critic’s value estimations stabilizing, as evidenced by the rising explained variance.

Temporal Efficiency. We also study how the rollout of RL in a physical simulator helps shape the policy to achieve expert-level temporal efficiency. We analyze the expert motion planning data used for SFT and then tracked the average episode lengths during the RL training of the $\pi _ { 0 . 5 }$ model in ManiSkill. As shown in Fig. 16, the SFT-initialized policy exhibits significantly longer episodes due to execution errors. In contrast, $\pi _ { 0 . 5 }$ achieves episode lengths that converge to the expert range after RL training, demonstrating a substantial improvement in temporal efficiency.

We attribute this convergence to two factors: (1) RL enhances the policy’s error-correction capabilities, allowing it to recover from execution failures. (2) Our partial reset mechanism incentivizes temporal efficiency through discounted rewards, as faster task completion enables the agent to trigger more resets and accumulate higher total rewards within each update cycle.

# H. RL for GR00T N1.5

# H.1. Setup

GR00T N1.5. We conduct additional experiments based on the GR00T N1.5 model (Bjorck et al., 2025), which is a foundation model tailored for generalist humanoid robot reasoning and manipulation. The architecture integrates an Eagle 2.5 VLM (Chen et al., 2025), optimized for spatial grounding and physical reasoning, with a Diffusion Transformer head (Peebles & Xie, 2023) for action denoising. It facilitates multi-embodiment compatibility through specialized heads, supporting configurations such as humanoids with dexterous hands or grippers, as well as single-arm manipulators.

Regarding the critic implementation, we estimate value functions across the entire denoising trajectory by integrating the critic network directly with the action head. The complete framework is illustrated in Fig. 17.

Benchmark. We evaluate the model performance of GR00T across four manipulation task suites in LIBERO: Spatial, Object, Goal, and Long.

![image](images/914a19ef6a7487a3365855387741fd50d0f1d917956fd39e90766fe01324afd1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Image Observation"] --> B["Language Observation"]
    C["Robot State"] --> D["Engine"]
    B --> E["Engine"]
    D --> F["Engine"]
    E --> G["Input Tokenize"]
    F --> H["Input Tokenize"]
    G --> I["Input Tokenize"]
    H --> J["Input Tokenize"]
    I --> K["Input Tokenize"]
    J --> L["Input Tokenize"]
    K --> M["Input Tokenize"]
    L --> N["Output Token A⁰"]
    M --> O["Output Token A⁰"]
    N --> P["Output Token A⁰"]
    O --> Q["Output Token A⁰"]
    P --> R["Output Token A⁰"]
    Q --> S["Output Token A⁰"]
    R --> T["Output Token A⁰"]
    S --> U["Output Token A⁰"]
    T --> V["Output Token A⁰"]
    U --> W["Output Token A⁰"]
    V --> X["Output Token A⁰"]
    W --> Y["Output Token A⁰"]
    X --> Z["Output Token A⁰"]
    Y --> AA["Output Token A⁰"]
    Z --> AB["Output Token A⁰"]
    AA --> AC["Output Token A⁰"]
    AB --> AD["Output Token A⁰"]
    AC --> AE["Output Token A⁰"]
    AD --> AF["Output Token A⁰"]
    AE --> AG["Output Token A⁰"]
    AF --> AH["Output Token A⁰"]
    AG --> AI["Output Token A⁰"]
    AH --> AJ["Output Token A⁰"]
    AI --> AK["Output Token A⁰"]
    AJ --> AL["Output Token A⁰"]
    AK --> AM["Output Token A⁰"]
    AL --> AN["Output Token A⁰"]
    AM --> AO["Output Token A⁰"]
    AN --> AP["Output Token A⁰"]
    AO --> AQ["Output Token A⁰"]
    AP --> AR["Output Token A⁰"]
    AQ --> AS["Output Token A⁰"]
    AR --> AT["Output Token A⁰"]
    AS --> AU["Output Token A⁰"]
    AT --> AV["Output Token A⁰"]
    AU --> AW["Output Token A⁰"]
    AV --> AX["Output Token A⁰"]
    AW --> AY["Output Token A⁰"]
```
</details>

Figure 17. Illustration for the architecture of GR00T-N1.5.

Table 10. Results of Finetuning GR00T using PPO with Flow-SDE on the LIBERO. 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="6">LIBERO</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td><td>Δ Avg.</td></tr><tr><td rowspan="2">GR00T</td><td>SFT</td><td>41.4</td><td>58.6</td><td>48.2</td><td>61.9</td><td>52.5</td><td>—</td></tr><tr><td>+PPO</td><td>92.5</td><td>96.2</td><td>84.3</td><td>86.6</td><td>89.9</td><td>+37.4</td></tr></table>

Implementation Details. Similar to the $\pi _ { 0 }$ implementation, we initiate our process with SFT on expert demonstrations. For the SFT stage, we fine-tune the entire model following the official setting. In the subsequent RL stage, we exclusively fine-tune the action expert model while keeping the vision-language model parameters fixed.

A crucial methodological refinement in our RL pipeline is the replacement of dropout layers in the expert model with identity layers. Dropout is widely recognized to induce instability during online RL training. Specifically, it introduces non-deterministic perturbations to the effective policy, shifting the standard probability ratio update from:

$$
\rho_ {t} (\theta) = \frac {\pi_ {\theta_ {\text { new }}} (a _ {t} | s _ {t})}{\pi_ {\theta_ {\text { old }}} (a _ {t} | s _ {t})} \tag {17}
$$

to a highly unstable form:

$$
\rho_ {t} (\theta) = \frac {\pi_ {\alpha_ {\text { new }}} (a _ {t} | s _ {t})}{\pi_ {\theta_ {\text { old }}} (a _ {t} | s _ {t})}, \tag {18}
$$

where $\rho _ { t } ( \theta )$ denotes the probability ratio, and $\alpha _ { \mathrm { n e w } }$ represents the policy state post-update as modified by the stochastic dropout mask. This structural stochasticity, compounded with per-step policy updates, severely undermines training convergence. The training hyperparameters are identical to those used for $\pi _ { 0 }$ , with Flow-SDE employed as the primary RL algorithm.

# H.2. Results

Results are summarized in Tab. 10. For the few-shot model, the SFT baseline achieves only a 52.5% success rate, reflecting limited generalization from sparse demonstrations. Conversely, our RL-based Flow-SDE significantly improves performance to 89.9%. These results, obtained using default $\pi _ { 0 }$ configurations, underscore the broad applicability of our method across architectures. While task-specific tuning could further enhance performance, we leave such optimization for future work.

# I. Limitations and Future Work

Noise Injection. Our current noise injection strategy exhibits a performance drop during the ODE-to-SDE conversion. Flow-CPS (Wang & Yu, 2025) attributes this loss to numerical error and proposes an improved coefficientspreserving sampling method. In our experiments, we attempted this configuration. Consistent with our hyperparameter ablation, our experiments showed that while this configuration mitigated the ODE-SDE precision error, it yielded limited RL improvement. Nevertheless, we argue that improving the noise injection strategy holds significant potential, specifically converting the ODE formulation to an SDE formulation while preserving the action distribution undisturbed.

Training Acceleration. Our current implementation of the mixed ODE-SDE rollout is simplistic in Flow-SDE, $i . e . ,$ , it randomly selects one denoising step as an SDE step, while all other steps remain ODE steps. We posit that future investigations into mixed ODE-SDE rollouts, leveraging advances in accelerating flow-based image generation (Li et al., 2025b; He et al., 2025; Liu et al., 2025b; Li et al., 2025c), could further enhance Flow-SDE, leading to faster training and improved performance.

Generalization. Maniskill OOD tests indicate that the semantic generalization of SFT and RL models remains limited. To address this, future work will leverage RL to enhance robustness by training on more diverse task distributions and varied linguistic instructions, thereby fostering better cross-task adaptability.

# J. Training Hyperparameters.

We record the training hyperparameters used to train both π0 and $\pi _ { 0 . 5 }$ on each benchmark, and present them in Tabs. 11 to 13.

Table 11. Hyperparameters across LIBERO. 

<table><tr><td rowspan="3">Parameters</td><td colspan="8">Algorithms and Tasks</td></tr><tr><td colspan="4"> $\pi_0$ </td><td colspan="4"> $\pi_{0.5}$ </td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td></tr><tr><td>Train epochs</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Global batch size</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td></tr><tr><td>Update epochs</td><td>4</td><td>4</td><td>4</td><td>4</td><td>1</td><td>1</td><td>3</td><td>4</td></tr><tr><td>Actor lr</td><td>5e-6</td><td>5e-6</td><td>5e-6</td><td>5e-6</td><td>5e-6</td><td>5e-6</td><td>5e-6</td><td>5e-6</td></tr><tr><td>Critic lr</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>Scheduler</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>True</td></tr><tr><td>Reward discount rate  $\gamma$ </td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>GAE  $\lambda$ </td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>Clip ratio  $\epsilon$ </td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Interaction steps</td><td>240</td><td>240</td><td>320</td><td>480</td><td>240</td><td>320</td><td>320</td><td>480</td></tr><tr><td>Parallel environments</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Rollout epochs</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Action prediction horizon  $H$ </td><td>50</td><td>50</td><td>50</td><td>50</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Action replan horizon  $H'$ </td><td>5</td><td>5</td><td>5</td><td>10</td><td>5</td><td>5</td><td>5</td><td>10</td></tr><tr><td>Denoise steps</td><td>4</td><td>4</td><td>4</td><td>4</td><td>3</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Noise level  $\sigma$  (Flow-SDE)</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.3</td><td>0.3</td><td>0.5</td></tr><tr><td>Max log-var (Flow-Noise)</td><td>0.16</td><td>0.16</td><td>0.16</td><td>0.16</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td></tr><tr><td>Min log-var (Flow-Noise)</td><td>0.08</td><td>0.08</td><td>0.08</td><td>0.08</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.04</td></tr><tr><td>Entropy bonus (Flow-Noise)</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td></tr></table>

Table 12. Hyperparameters across SIMPLER and ManiSkill. 

<table><tr><td rowspan="3">Parameters</td><td colspan="10">Algorithms and Tasks</td></tr><tr><td colspan="5"> $\pi_0$ </td><td colspan="5"> $\pi_{0.5}$ </td></tr><tr><td>Eggplant</td><td>Carrot</td><td>Spoon</td><td>Cube</td><td>ManiSkill</td><td>Eggplant</td><td>Carrot</td><td>Spoon</td><td>Cube</td><td>ManiSkill</td></tr><tr><td>SFT train steps</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>RL train steps</td><td>40</td><td>40</td><td>40</td><td>130</td><td>150</td><td>40</td><td>40</td><td>40</td><td>70</td><td>150</td></tr><tr><td>Global batch size</td><td>2560</td><td>2560</td><td>2560</td><td>2560</td><td>5120</td><td>2560</td><td>2560</td><td>2560</td><td>2560</td><td>5120</td></tr><tr><td>Update epochs</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>5</td></tr><tr><td>Actor lr</td><td>5.6e-6</td><td>5.6e-6</td><td>5.6e-6</td><td>5.6e-6</td><td>7.91e-6</td><td>5.6e-6</td><td>5.6e-6</td><td>5.6e-6</td><td>5.6e-6</td><td>7.91e-6</td></tr><tr><td>Critic lr</td><td>1.1e-4</td><td>1.1e-4</td><td>1.1e-4</td><td>1.1e-4</td><td>1.55e-4</td><td>1.1e-4</td><td>1.1e-4</td><td>1.1e-4</td><td>1.1e-4</td><td>1.55e-4</td></tr><tr><td>Scheduler</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td><td>False</td></tr><tr><td>Reward discount rate  $\gamma$ </td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>GAE  $\lambda$ </td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>Clip ratio  $\epsilon$ </td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Interaction steps</td><td>48</td><td>48</td><td>48</td><td>48</td><td>48</td><td>48</td><td>48</td><td>48</td><td>48</td><td>48</td></tr><tr><td>Parallel environments</td><td>256</td><td>256</td><td>256</td><td>256</td><td>320</td><td>256</td><td>256</td><td>256</td><td>256</td><td>320</td></tr><tr><td>Rollout epochs</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Action prediction horizon  $H$ </td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Action replan horizon  $H'$ </td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Denoise steps</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Noise level  $\sigma$ (Flow-SDE)</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Max log-var (Flow-Noise)</td><td>0.16</td><td>0.16</td><td>0.16</td><td>0.16</td><td>0.16</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td></tr><tr><td>Min log-var (Flow-Noise)</td><td>0.08</td><td>0.08</td><td>0.08</td><td>0.08</td><td>0.08</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.04</td></tr><tr><td>Entropy bonus (Flow-Noise)</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td></tr></table>

Table 13. Hyperparameters across MetaWorld and CALVIN benchmarks. 

<table><tr><td rowspan="3">Parameters</td><td colspan="4">Benchmarks and models</td></tr><tr><td colspan="2">MetaWorld</td><td colspan="2">CALVIN</td></tr><tr><td> $\pi_0$ </td><td> $\pi_{0.5}$ </td><td> $\pi_0$ </td><td> $\pi_{0.5}$ </td></tr><tr><td>Train epochs</td><td>450</td><td>450</td><td>100</td><td>100</td></tr><tr><td>Global batch size</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td></tr><tr><td>Update epochs</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Actor lr</td><td>1e-5</td><td>5e-6</td><td>5e-6</td><td>5e-6</td></tr><tr><td>Critic lr</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>Scheduler</td><td>False</td><td>True</td><td>False</td><td>False</td></tr><tr><td>Reward discount rate  $\gamma$ </td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>GAE  $\lambda$ </td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>Clip ratio  $\epsilon$ </td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Interaction steps</td><td>100</td><td>100</td><td>480</td><td>480</td></tr><tr><td>Parallel environments</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Rollout epochs</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Action prediction horizon  $H$ </td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Action replan horizon  $H'$ </td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Denoise steps</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Noise level  $\sigma$  (Flow-SDE)</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Max log-var (Flow-Noise)</td><td>0.10</td><td>0.10</td><td>0.16</td><td>0.16</td></tr><tr><td>Min log-var (Flow-Noise)</td><td>0.04</td><td>0.04</td><td>0.08</td><td>0.08</td></tr><tr><td>Entropy bonus (Flow-Noise)</td><td>0.005</td><td>0.005</td><td>0.005</td><td>0.005</td></tr></table>

---

## 关联

## 衍生问题
