---
title: "RL Token: Online RL with VLAs | Mochan.org"
source: "https://mochan.org/posts/rlt/"
author:
published:
created: 2026-05-13
description:
tags:
  - "vla"
---
An **RL token** is a compact, learned readout representation that compresses a VLA’s deep understanding of its inputs into a single small vector. This compression acts as a bridge, facilitating incredibly fast and sample-efficient online reinforcement learning (RL). Because directly fine-tuning large-scale VLA models online is computationally expensive and requires significant data, the RL Token acts as a compressed interface that summarizes the high-dimensional internal embeddings of the VLA into a small vector. The concept of an **RL Token** comes from a [blog post](https://www.pi.website/research/rlt) and [published paper](https://www.pi.website/download/rlt.pdf) by Physical Intelligence.

### Overview

#### The Concept: Freezing the VLA

At a high level, this small vector allows the system to freeze the massive, computationally heavy VLA model during the reinforcement learning phase. Instead of updating billions of parameters, the system feeds this summarized “RL Token” into separate, much smaller RL networks. The frozen VLA acts as a powerful foundation, providing a broad understanding of the scene and suggesting an initial action, while the small RL networks use the token to learn precise, task-specific refinements (like the final millimeter adjustments needed for a delicate insertion) through fast, trial-and-error practice in the real world.

#### The Workflow: A Two-Phase Approach

In practice, the system achieves this through a targeted, two-phase approach. First, the VLA model is adapted on a small amount of demonstration data to expose the RL token and provide a strong base policy. Then, during the online learning phase, both the VLA and its token-extractor are completely frozen. The robot executes tasks in the real world, relying on the frozen VLA for the easier, broader movements (like reaching for an object). When the robot enters a high-precision “critical phase” (like inserting a screw), control shifts to the small RL actor network. This actor receives the RL Token (to understand the current state) and a “reference action” chunk from the VLA (as a strong initial suggestion). Guided by a critic network that evaluates its choices based on sparse success or failure rewards, the actor safely explores adjustments to refine the VLA’s suggestions. All of these real-world attempts, along with optional human corrections, are saved into a replay buffer, enabling the lightweight networks to rapidly learn the most difficult parts of a task without starting from scratch.

#### The Reinforcement Learning Mechanism

The core reinforcement learning mechanism relies on an off-policy actor-critic algorithm that learns from this replay buffer. As the robot practices, a human supervisor provides a sparse binary reward at the end of each episode, a simple “success” or “failure” label. The **critic network** uses these recorded experiences to learn a value function, essentially predicting how likely a given action is to lead to task success from the current state (represented by the RL Token). Meanwhile, the **actor network** is continuously updated to output actions that maximize this predicted success score. Crucially, to prevent the robot from making wild, unsafe movements during exploration, the actor’s learning objective includes a regularization term that mathematically anchors its predictions close to the VLA’s original reference action. This turns the reinforcement learning process into a highly efficient, targeted refinement of an already competent strategy, rather than forcing the system to learn the task from scratch.

### Extracting the RL Token

The RL token vector is obtained by appending a special learned token (referred to as the `<rl>` token, mathematically denoted as $e_{r l}$) to the final-layer token embeddings produced by the pretrained VLA. (In a transformer-based VLA, these **final-layer token embeddings** are the dense vector representations that emerge after the input images and text have passed through all the layers of the model’s visual-language backbone, representing the model’s highest-level, most contextualized understanding of the scene before action generation).

This augmented sequence is then processed through a small, additional encoder transformer, and the output at this special token’s position becomes the RL token. This token then serves as the **state observation** (the crucial input summarizing the current state of the environment) for smaller **off-policy actor and critic networks**.

To break this down: in this reinforcement learning setup, the **“actor”** network decides what precise actions the robot should take next, while the **“critic”** network evaluates how good those proposed actions are. Because these networks are **“smaller”** (meaning they have far fewer parameters than the VLA and are lightweight to compute) and **“off-policy”** (meaning they can learn highly efficiently from past experiences and human interventions stored in a replay buffer, rather than only from live trial-and-error), they can be trained incredibly fast.

By utilizing the RL token, the system can bootstrap from the VLA’s broad perceptual understanding and action priors, while relying on lightweight online RL to refine precision-critical behaviors. This division of labor enables robots to master challenging manipulation tasks with just a few hours of real-world practice without needing to update the full foundational model.

#### Architecture Overview

Here is a visual representation of how the RL Token is extracted and used:

![RL Token Architecture](https://mochan.org/images/rlt_architecture.svg)

### Training the RL Token Extractor

Before the online reinforcement learning phase can begin, the system must learn how to compress the massive VLA state down into the single RL Token. This is accomplished using a small dataset of task-specific human demonstrations.

The extractor is trained using an **encoder-decoder** architecture. While the encoder’s job is to compress the VLA’s final-layer embeddings into the single `<rl>` token, a temporary decoder transformer is tasked with the opposite: taking that single RL Token and attempting to autoregressively reconstruct the original high-dimensional VLA embeddings from it.

By forcing the decoder to rebuild the full context from just this one vector, the system creates an **information bottleneck**. This mathematically guarantees that the RL Token captures and retains all the crucial, task-relevant visual and contextual information from the scene. Once this initial training phase is complete, the decoder is discarded, the token-extracting encoder is frozen, and the system is ready to begin online trial-and-error learning.

![RL Token Training](https://mochan.org/images/rlt_training.svg)

### The Online Actor and Critic System

Once the VLA is frozen and the RL Token is available, the system relies on a lightweight off-policy actor-critic architecture to refine the robot’s actions. This system requires three distinct inputs to function:

1. **The RL Token:** The compressed visual and language understanding of the scene (the “eyes”).
2. **Proprioceptive State:** The physical sensory data from the robot itself, such as the current position and velocity of its joints (the “body awareness”).
3. **The Reference Action Chunk:** The frozen VLA’s initial suggestion of what to do next (used specifically by the actor).

Here is how the two networks work together:

- **The Critic ($Q_{\psi}$):** The critic network takes the current state and a sequence of proposed actions (an “action chunk”) and estimates its value, essentially predicting whether this sequence of actions will lead to task success. It learns by evaluating past experiences from the replay buffer.
- **The Actor ($\pi_{\theta}$):** The actor network is responsible for generating the action chunk. Crucially, it doesn’t generate actions from scratch. It takes the state *and a reference action chunk* generated by the frozen VLA. The actor is trained to output a refinement of this reference chunk that maximizes the critic’s value.

It is important to understand *why* the actor needs both the action chunk and the RL Token. The VLA’s **action chunk** provides a “blind” suggestion (e.g., “move the gripper down by 2cm”). However, to know *how* to refine that suggestion, the actor needs to see the current state of the world (e.g., “the screw is slightly tilted”). Because these small actor and critic networks do not have complex image-processing layers of their own, they rely entirely on the **RL Token** to act as their “eyes,” providing a highly compressed, rich contextual summary of the scene.

Structurally, these networks are incredibly lightweight, highlighting the efficiency of the RL Token approach. Both the actor and critic are implemented as simple **Multi-Layer Perceptrons (MLPs)** initialized from scratch. For most tasks (like zip tie fastening, Ethernet insertion, and charger insertion), they are just **2-layer MLPs with a hidden dimension of 256**. For more complex, high-precision tasks (like screw installation), the networks are scaled up slightly to a **3-layer MLP with a hidden dimension of 512**. This is a stark contrast to the billion-parameter foundational models they are attached to.

To ensure the system learns efficiently without completely forgetting the VLA’s strong priors, the researchers introduced two important techniques:

1. **Policy Regularization:** The actor is penalized if its output deviates too far from the VLA’s reference action chunk. This forces the online RL to act as a local refinement editor rather than wildly exploring new, potentially unsafe behaviors.
2. **Reference Action Dropout:** If the actor always relies on the VLA’s reference action, it might simply learn to copy it blindly. To prevent this, the reference chunk is randomly masked out (replaced with zeros) during training. This forces the actor to maintain an independent ability to generate actions, while still taking advantage of the VLA’s suggestions when they are present.

### The RLT Algorithm Workflow

Here is a step-by-step breakdown of the entire RLT (RL Token) algorithm, integrating the formal mathematical equations from the original paper:

**Require:** Frozen VLA backbone $f_{\theta_{\text{vla}}}$ and VLA action distribution $\pi_{\text{vla}}$; demo data $\mathcal{D}$, chunk length $C$, replay buffer $\mathcal{B}$, warmup steps $N_{\text{warm}}$, ratio $G$, VLA fine-tuning weight $\alpha$, policy constraint $\beta$.

**Phase 1: Train RL token and (optionally) fine-tune the VLA**

- **Train $\phi$** using $\mathbf{z}_{i} = f_{i} \left(\right. \mathbf{s} , ℓ , \theta_{\text{vla}} \left.\right)$, $\mathbf{z}_{\text{rl}} = g_{\phi} \left(\right. \left[\right. \mathbf{z}_{1 : M} , \mathbf{e}_{\text{rl}} \left]\right. \left.\right)_{M + 1}$, and $\theta_{\text{vla}}$. During this phase, the VLA is optionally fine-tuned on the human demonstration data to create a specialized “strong base policy” before the robot starts practicing. If the weight $\alpha$ is set to zero, the VLA remains frozen even during this step. 
	$$
	\mathcal{L}_{\text{ro}} \left(\right. \phi \left.\right) = \mathbb{E}_{\mathcal{D}} \left[\right. \sum_{i = 1}^{M} \parallel h_{\phi} \left(\right. d_{\phi} \left(\right. \left[\right. \mathbf{z}_{\text{rl}} , \bar{\mathbf{z}}_{1 : i - 1} \left]\right. \left.\right) \left.\right)_{i} - \bar{\mathbf{z}}_{i} \parallel^{2} \left]\right.
	$$
	 
	$$
	\phi , \theta_{\text{vla}} = arg ⁡ \underset{\phi , \theta_{\text{vla}}}{min} \mathcal{L}_{\text{ro}} \left(\right. \phi \left.\right) + \alpha \mathcal{L}_{\text{vla}} \left(\right. \theta_{\text{vla}} \left.\right)
	$$

**Phase 2: Train RL actor and critic**

- **Initialize** critic $Q_{\psi}$ and RL Policy $\pi_{\theta}$.
- **for** environment steps $t = 0 , C , 2 C \ldots$ **do**
	- Sample VLA reference chunk $\overset{\sim}{\mathbf{a}}_{t : t + C - 1} sim \pi_{\text{vla}} \left(\right. \mathbf{s}_{t} \left.\right)$.
		- Form RL state $\mathbf{x}_{t} = \left(\right. \mathbf{z}_{\text{rl}} \left(\right. \mathbf{s}_{t} \left.\right) , \mathbf{s}_{t}^{p} \left.\right)$.
		- **Choose action** based on three priority levels: human teleoperation during interventions, the base VLA model’s suggestions during the initial warmup phase (to pre-fill the replay buffer), or the RL actor’s refined actions once the training is underway. 
		$$
		\mathbf{a}_{t : t + C - 1} \leftarrow \left{\right. \mathbf{a}^{\text{human}} & \text{if intervention} \\ \overset{\sim}{\mathbf{a}}_{t : t + C - 1} & \text{if } t < N_{\text{warm}} \\ sim \pi_{\theta} \left(\right. \cdot \mid \mathbf{x}_{t} , \overset{\sim}{\mathbf{a}} \left.\right) & \text{otherwise}
		$$
		- Execute $\mathbf{a}_{t : t + C - 1}$ and observe $r_{t}$, $\mathbf{s}_{t + 1}$, $\mathbf{s}_{t + 1}^{p}$
		- $\overset{\sim}{\mathbf{a}}_{t : t + C - 1} \leftarrow \mathbf{a}^{\text{human}}$ if intervention
		- Store transition in $\mathcal{B}$: $\langle \mathbf{x}_{t} , \mathbf{a}_{t : t + C - 1} , \overset{\sim}{\mathbf{a}} , r_{t} , \mathbf{x}_{t + 1} \rangle$
		- **for** $g = 1 , \ldots , G$ **do**
		- Sample batch of data $\mathbf{b} sim \mathcal{B}$.
				- Compute target Q values: 
			$$
			\hat{Q} = \sum_{t^{'} = 1}^{C} \gamma^{t^{'} - 1} r_{t^{'}} + \gamma^{C} \mathbb{E}_{\mathbf{a}^{'} sim \pi_{\theta}} \left[\right. Q_{\psi^{'}} \left(\right. \mathbf{x}^{'} , \mathbf{a}^{'} \left.\right) \left]\right.
			$$
				- Train Critic with TD backup: 
			$$
			\mathcal{L}_{Q} \left(\right. \psi \left.\right) = \mathbb{E}_{\mathbf{b}} \left[\right. \left(\right. \hat{Q} - Q_{\psi} \left(\right. \mathbf{x} , \mathbf{a} \left.\right) \left.\right)^{2} \left]\right.
			$$
				- Train Policy $\mathbf{a} sim \pi_{\theta} \left(\right. \cdot \mid \mathbf{s} , \overset{\sim}{\mathbf{a}} \left.\right)$: 
			$$
			\mathcal{L}_{\pi} \left(\right. \theta \left.\right) = \mathbb{E}_{\mathbf{b}} \left[\right. - Q_{\psi} \left(\right. \mathbf{x} , \mathbf{a} \left.\right) + \beta \parallel \mathbf{a} - \overset{\sim}{\mathbf{a}} \parallel_{2}^{2} \left]\right.
			$$
		- **end for**
- **end for**

### Algorithm Symbols Dictionary

Here is a reference table of the mathematical symbols used in the algorithm:

| Symbol | Meaning |
| --- | --- |
| **$f_{\theta_{v l a}}$** | The frozen Vision-Language-Action (VLA) backbone model |
| **$\pi_{v l a}$** | The pretrained VLA action policy/distribution |
| **$\mathbf{s}$** | The raw input state (primarily camera images) |
| **$ℓ$** | The natural language instruction given to the robot |
| **$\mathcal{D}$** | The dataset of human demonstrations |
| **$C$** | The length of the “action chunk” (how many steps the RL policy outputs at once) |
| **$\mathcal{B}$** | The replay buffer used to store past experiences |
| **$N_{w a r m}$** | Number of initial warmup steps where the VLA strictly controls the robot to pre-fill the buffer |
| **$G$** | Gradient updates ratio (how many training steps to take per environment step) |
| **$\alpha$** | Weight determining how much to fine-tune the VLA (0 means completely frozen) |
| **$\beta$** | Policy constraint coefficient (forces the Actor to stay close to the VLA’s suggestion) |
| **$z_{i}$, $z_{r l}$** | The VLA’s final-layer token embeddings ($z_{i}$) and the extracted RL Token ($z_{r l}$) |
| **$\phi$** | The learnable parameters/weights of the RL Token’s encoder-decoder transformer |
| **$d_{\phi}$** | The decoder transformer used to reconstruct VLA embeddings from the RL token |
| **$h_{\phi}$** | The linear output projection layer attached to the decoder |
| **$\theta_{v l a}$** | The learnable parameters/weights of the VLA model |
| **$\mathcal{L}_{r o}$, $\mathcal{L}_{v l a}$** | The loss functions for reconstructing the RL token and fine-tuning the VLA |
| **$Q_{\psi}$** | The Critic network (parameterized by weights $\psi$) |
| **$\pi_{\theta}$** | The Actor policy network (parameterized by weights $\theta$) |
| **$t$** | The current timestep in the environment |
| **$\overset{\sim}{a}_{t : t + C - 1}$** | The reference action chunk proposed by the VLA |
| **$x_{t}$** | The total RL state observation (RL token + proprioceptive state) |
| **$s_{t}^{p}$** | The proprioceptive state (joint positions, velocity, etc.) |
| **$a_{h u m a n}$** | Human intervention action (if a human takes over) |
| **$r_{t}$** | The reward observed after taking an action |
| **$\gamma$** | Discount factor (used by the Critic to value future rewards) |
| **$\hat{Q}$** | The target Q-value used to train the Critic |

### Training the Actor and Critic (TD3)

The online RL phase uses a specialized algorithm called **Twin Delayed Deep Deterministic Policy Gradient (TD3)**. TD3 is an “off-policy” algorithm, meaning it can learn from any experience in the replay buffer, regardless of whether it was generated by the robot’s current strategy, its earlier (worse) strategies, or even human demonstrations.

To make the learning stable and prevent the robot from “over-optimizing” on accidental successes, the system uses three key tricks:

1. **Target Networks:** It keeps a separate “delayed” copy of the networks to calculate target values, preventing the model from chasing its own tail.
2. **Two Critics:** It actually trains two separate critic networks and uses the most conservative (lowest) estimate between them to avoid over-optimizing on lucky guesses.
3. **Delayed Updates:** The actor network is updated less frequently than the critic, ensuring the critic’s evaluations are accurate before the actor changes its behavior.

### Simplified Algorithm Equations

If you find the formal math in Algorithm 1 a bit dense, here is a mapping of the core logic to the actual equations:

**1\. The Goal of the Critic (Value Prediction)**

$$
\underset{\text{Target Value}}{\underbrace{\hat{Q}}} = \underset{\text{Immediate Reward}}{\underbrace{\sum_{t^{'} = 1}^{C} \gamma^{t^{'} - 1} r_{t^{'}}}} + \underset{\text{Discounted Future Value}}{\underbrace{\gamma^{C} \mathbb{E}_{\mathbf{a}^{'} sim \pi_{\theta}} \left[\right. Q_{\psi^{'}} \left(\right. \mathbf{x}^{'} , \mathbf{a}^{'} \left.\right) \left]\right.}}
$$
 
$$
\underset{\text{Critic Error }=\text{ }(\text{Target Value }-\text{ Predicted Value})^{2}}{\underbrace{\mathcal{L}_{Q} \left(\right. \psi \left.\right) = \mathbb{E}_{\mathbf{b}} \left[\right. \left(\right. \hat{Q} - Q_{\psi} \left(\right. \mathbf{x} , \mathbf{a} \left.\right) \left.\right)^{2} \left]\right.}}
$$

*The critic’s only job is to reduce this error so it becomes an expert at predicting success from the current state and action.*

**2\. The Goal of the Actor (Action Refinement)** To train the actor policy (mathematically written as $\mathbf{a} sim \pi_{\theta} \left(\right. \cdot \mid \mathbf{s} , \overset{\sim}{\mathbf{a}} \left.\right)$), we minimize a “Loss Function” ($\mathcal{L}_{\pi} \left(\right. \theta \left.\right)$). Sampling from $\pi_{\theta}$ means the robot picks an action from a probability distribution created by the actor network, conditioned on the current state $\mathbf{s}$ and the VLA suggestion $\overset{\sim}{\mathbf{a}}$.

$$
\underset{\text{Actor Score }=}{\underbrace{\mathcal{L}_{\pi} \left(\right. \theta \left.\right) = \mathbb{E}_{\mathbf{b}} \left[\right.}} \underset{\text{Maximize Predicted Value}}{\underbrace{- Q_{\psi} \left(\right. \mathbf{x} , \mathbf{a} \left.\right)}} + \underset{\text{Penalty for straying from VLA}}{\underbrace{\beta \parallel \mathbf{a} - \overset{\sim}{\mathbf{a}} \parallel_{2}^{2}}} \left]\right.
$$

*The actor tries to minimize this total loss—which is mathematically equivalent to maximizing the predicted value while staying close to the VLA’s safe advice (the $\beta$ penalty).*