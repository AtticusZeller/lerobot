---
tags:
  - paper-note
  - inbox
title: "$\pi _ { 0 . 5 } \colon$ a Vision-Language-Action Model with Open-World Generalization"
authors: "Intelligence, Physical; Black, Kevin; Brown, Noah; Darpinian, James; Dhabalia, Karan; Driess, Danny; Esmail, Adnan; Equi, Michael; Finn, Chelsea; Fusai, Niccolo; Galliker, Manuel Y.; Ghosh, Dibya; Groom, Lachy; Hausman, Karol; Ichter, Brian; Jakubczak, Szymon; Jones, Tim; Ke, Liyiming; LeBlanc, Devin; Levine, Sergey; Li-Bell, Adrian; Mothukuri, Mohith; Nair, Suraj; Pertsch, Karl; Ren, Allen Z.; Shi, Lucy Xiaoyang; Smith, Laura; Springenberg, Jost Tobias; Stachowicz, Kyle; Tanner, James; Vuong, Quan; Walke, Homer; Walling, Anna; Wang, Haohuan; Yu, Lili; Zhilinsky, Ury"
year: 2025
arxiv: "2504.16054"
status: unread
compiled_from: "http://arxiv.org/abs/2504.16054"
url: "http://arxiv.org/abs/2504.16054"
---

# $\pi _ { 0 . 5 } \colon$ a Vision-Language-Action Model with Open-World Generalization

# Physical Intelligence

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, Ury Zhilinsky https://pi.website/blog/pi05

![image](images/683c72c7e9cb3a3e111fc61c20dff1b303a3d7de1cdb2786f08375a258c762eb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Subtask Commands"] --> B["Close the microwave"]
    A --> C["Pick up the mitten"]
    D["Object Detection"] --> E["Dress"]
    D --> F["Coffee"]
    G["Multimodal Web Data"] --> H["Q: How many desks are in the image?"]
    G --> I["A: 12"]
    J["Multimodal Data"] --> K["Q: Detect and label all objects in the scene."]
    J --> L["A: <loc0112>"]
    J --> M["<loc0234>..."]
    J --> N["Q: What kind of pie is on this plate?"]
    J --> O["A: Chocolate"]
    
    B --> P["π₀.₅ Vision-Language-Action Policy"]
    P --> Q["High-Level"]
    P --> R["Low-Level"]
    P --> S["Action Expert"]
    P --> T["Language Instruction"]
    T --> U["Deploy out-of-the-box in new homes"]
    
    V["Robot Action Data"] --> W["In-the-wild Mobile Robot"]
    V --> X["Shirt in basket"]
    V --> Y["Make bed"]
    
    W --> Z["In-the-wild Static Robot"]
    W --> AA["Fold linen"]
    W --> AB["Item in drawer"]
    
    V --> AC["In-Lab Static Robot"]
    V --> AD["Fold laundry"]
    V --> AE["Sweep table"]
    
    V --> AF["General Robot Data"]
```
</details>

Fig. 1: The $\pi _ { 0 . 5 }$ model transfers knowledge from a heterogeneous range of data sources, including other robots, high-level subtask prediction, verbal instructions, and data from the web, in order to enable broad generalization across environments and objects. $\pi _ { 0 . 5 }$ can control a mobile manipulator to clean kitchens and bedrooms in new homes that were not present in the training data, performing complex multi-stage behaviors with durations of 10 to 15 minutes.

Abstract—In order for robots to be useful, they must perform practically relevant tasks in the real world, outside of the lab. While vision-language-action (VLA) models have demonstrated impressive results for end-to-end robot control, it remains an open question how far such models can generalize in the wild. We describe $\pi _ { 0 . 5 } ,$ a new model based on $\pi _ { 0 }$ that uses co-training on heterogeneous tasks to enable broad generalization. π0.5 uses data from multiple robots, high-level semantic prediction, web data, and other sources to enable broadly generalizable realworld robotic manipulation. Our system uses a combination of co-training and hybrid multi-modal examples that combine image observations, language commands, object detections, semantic subtask prediction, and low-level actions. Our experiments show that this kind of knowledge transfer is essential for effective generalization, and we demonstrate for the first time that an end-to-end learning-enabled robotic system can perform longhorizon and dexterous manipulation skills, such as cleaning a kitchen or bedroom, in entirely new homes.

# I. INTRODUCTION

Stuff your eyes with wonder... See the world. It’s more fantastic than any dream made or paid for in factories.

Ray Bradbury, Fahrenheit 451

Open-world generalization represents one of the biggest open problems in physical intelligence: embodied systems such as robotic arms, humanoids, and autonomous vehicles only truly become useful when they can leave the lab and handle the diverse situations and unexpected events that occur in the real world. Learning-based systems offer a path to enabling broad generalization, particularly with recent advances that have enabled scalable learning systems in domains ranging from natural language processing [79, 21, 10, 78] to computer vision [34, 66, 35, 43]. However, the diversity of situations that a robot might encounter in the real world requires more than just scale: we need to design training recipes that can provide the breadth of knowledge that will allow robots to generalize at many levels of abstraction. For example, if a mobile robot is asked to clean up a kitchen that it has never seen before, some behaviors generalize readily if they are well represented in the data with a sufficient range of scenes and objects (e.g., picking up a knife or plate), others might require adapting or modifying existing skills to use them in a new way or in a new sequence, and yet others might require understanding the semantics of the scene based on prior knowledge (e.g., which drawer to open, or which object on the counter is most likely to be a drying rack). How can we structure a training recipe for a robotic learning system that can enable this kind of flexible generalization?

![image](images/a55cf421e3d3b86cff775eea5f907d4d4071425666d5ea7149cbbb8249e63fad.jpg)

<details>
<summary>text_image</summary>

"close the cabinets"
"put the items in the drawer"
"wipe the spill"
"place the dishes in the sink"
</details>

Fig. 2: π0.5 cleaning a new kitchen. The robot is tasked with cleaning a kitchen in a home that was not in the training data. The model is given general tasks (close the cabinets, put the items in the drawer, wipe the spill, and put the dishes in the sink), which it performs by both predicting subtasks to accomplish (e.g., pick up the plate) and emitting low-level actions.

A person can draw on a lifetime of experience to synthesize appropriate solutions to each of these challenges. Not all of this experience is firsthand, and not all of it comes from rote practice – for example, we might use facts that we were told by others or read in a book, together with bits of insight from other tasks we have performed in different contexts, combined with direct experience in the target domain. Analogously, we might hypothesize that generalizable robotic learning systems must be able to transfer experience and knowledge from a variety of information sources. Some of these sources are firsthand experience with direct relevance to the task at hand, some require transfer from other robot embodiments, environments, or domains, and some represent entirely different data types, such as verbal instructions, perceptual tasks based on web data, or prediction of high-level semantic commands. The heterogeneity of these different sources of data present a major obstacle, but fortunately recent advances in visionlanguage-action (VLA) models provide us with a toolkit that can make this possible: by casting different modalities into the same sequence modeling framework, VLAs can be adapted to train on robot data, language data, computer vision tasks, and combinations of the above.

In this paper, we leverage this observation to design a cotraining framework for VLAs that can utilize heterogeneous and diverse knowledge sources to enable broad generalization. Building on the $\pi _ { 0 }$ VLA, we propose to include a range of different data sources to create the $\pi _ { 0 . 5 }$ model (“pi oh five”), which can control mobile manipulators to perform a variety of household tasks even in homes that were never seen during training. $\pi _ { 0 . 5 }$ draws on experience from many sources: in addition to a medium-sized dataset collected directly with mobile manipulators in a variety of real homes (about 400 hours), $\pi _ { 0 . 5 }$ uses data from other non-mobile robots, data of related tasks collected under laboratory conditions, training examples that require predicting “high-level” semantic tasks based on robot observation, verbal language instructions provided to the robot by human supervisors, and a variety of multi-modal examples created from web data, such as image captioning, question answering, and object localization (see Figure 1). The overwhelming majority of training examples provided to $\pi _ { 0 . 5 }$ (97.6% during the first training phase) do not come from mobile manipulators performing household tasks, but from these other sources, such as other robots or data from the web. Nonetheless, $\pi _ { 0 . 5 }$ is able to control mobile manipulators in entirely new homes not seen during training, perform intricate tasks such as hanging up towels or making beds, and can carry out long-horizon manipulation skills 10 to 15 minutes in length, cleaning an entire kitchen or bedroom based on only a high-level prompt.

The design of $\pi _ { 0 . 5 }$ follows a simple hierarchical architecture: we first pre-train the model on the heterogeneous mixture of training tasks, and then fine-tune it specifically for mobile manipulation with both low-level action examples and high-level “semantic” actions, which correspond to predicting subtask labels such as “pick up the cutting board” or “rearrange the pillow.” At runtime, during each step of inference, the model first predicts the semantic subtask, inferring the behavior that is appropriate to perform next based on the task structure and the semantics of the scene, and then predicts the low-level robot action chunk based on this subtask. This simple architecture provides both the ability to reason about long-horizon multi-stage tasks and the ability to leverage different sources of knowledge for the two levels: the low-level action inference procedure readily benefits from action data collected by other robots, including simpler static robots in other environments, while the high-level inference procedure benefits from semantic examples from the web, high-level annotation prediction, and even verbal commands that can be provided to the robot by human “supervisors” that walk the robot through complex tasks step by step, instructing it (much like how they might instruct a person) on the appropriate subtasks to perform to complete a complex task such as cleaning a room. We illustrate this design in Figure 1.

Our central contribution is a system for training a highly generalizable VLA, $\pi _ { 0 . 5 } ,$ together with a proof of concept that generalization can emerge from this model when it is trained on appropriately diverse data. We provide a detailed empirical evaluation of both $\pi _ { 0 . 5 } \mathrm { ^ { 5 } s }$ generalization capabilities and the relevance of different co-training ingredients. To our knowledge, our work is the first to demonstrate an end-to-end learning-enabled robotic system that can perform long-horizon and dexterous manipulation skills, such as cleaning a kitchen or bedroom, in entirely new homes. Our experiments and comparisons further show that this is enabled by transferring knowledge from other robots, high-level semantic prediction, verbal language instruction from human supervisors, web data, and other sources.

# II. RELATED WORK

Generalist robot manipulation policies. Recent works have demonstrated that broadening the training data distribution for robot manipulation policies from narrow, single-task datasets to diverse datasets that span many scenes and tasks [17, 25, 80, 63, 41, 6, 30, 67, 1] allows the resulting policies to not only solve a wider range of tasks out of the box, but also improves their ability to generalize to new scenes and tasks [9, 63, 62, 22]. Training such generalist policies requires new modeling approaches that can handle the scale and diversity of datasets that often span hundreds of different tasks and scenes. Vision-language-action models (VLAs) [23, 92, 42, 8, 83, 90, 55, 45, 3, 75, 64, 76, 84, 7, 37] offer an appealing solution: by fine-tuning pre-trained visionlanguage models for robot control, VLAs can leverage the semantic knowledge acquired from web-scale pretraining and bring it to bear on the robotics problem. When combined with highly expressive action decoding mechanisms like flow matching [8], diffusion [55, 84, 52], or advanced action tokenization schemes [64], VLAs can perform a wide range of complex manipulation tasks in the real world. However, despite impressive language following abilities, VLAs are still typically evaluated in environments that closely match their training data. While some studies suggest that simple skills like picking up objects or opening drawers can be made to generalize simply by collecting robot data in a broader set of environments [14, 67, 28, 49, 64], it is challenging to apply the same approach to more complex, long-horizon tasks like cleaning up a kitchen, where achieving broad coverage of plausible scenarios via brute-force scaling of robot data collection is infeasible. In our experiments, we evaluate π0.5 in entirely new scenes, such as new kitchens and bedrooms that were not seen in training, showing that our VLA can generalize to entirely new scenes by leveraging not only direct first-hand experience on the target mobile manipulator platform, but also information from other data sources. These sources include data from other (non-mobile) robots, highlevel semantic subtask prediction, and data from the web.

Non-robot data co-training. A number of prior works have sought to use diverse non-robot data to improve the generalization of robot policies. Prior methods have explored initializing vision encoders from computer vision datasets [85, 58, 57, 18], or leveraging off-the-shelf task planners [38, 48, 73, 81]. VLA policies are typically initialized from a pre-trained visionlanguage model, which has been exposed to large amounts of internet vision and language data [23, 92, 42]. Notably, the VLA architecture is flexible and allows to map between input and output sequences of multi-modal vision, language, and action tokens. As such, VLAs broaden the design space of possible transfer approaches beyond simple weight initialization, by supporting the co-training of a single, unified architecture on not just robot action imitation data, but any dataset that interleaves one or multiple of the aforementioned modalities. Prior works have demonstrated that co-training VLAs with data mixtures used for VLM training [23, 92, 86] can improve their generalization ability, e.g., when interacting with new objects or unseen scene backgrounds. In this work, we go beyond VLM data co-training and design a system for cotraining VLAs with a broader set of robotics-relevant supervision sources, including data from other robots, high-level semantic subtask predictions, and verbal language instructions. While multitask training and co-training are not new ideas, we show that the specific combination of data sources in our system enables mobile robots to perform complex and longhorizon behaviors in entirely new environments. We believe that this level of generalization, particularly when accounting for the complexity of the tasks, goes significantly beyond the results demonstrated in prior works.

Robot reasoning and planning with language. A number of prior works have shown that augmenting end-to-end policies with high-level reasoning can significantly improve performance for long-horizon tasks [2, 36, 44, 74, 71, 4, 16, 11, 53, 88, 51, 59, 13, 70, 91, 65, 72, 47, 76, 89], particularly when high-level subtask inference can benefit from large pretrained LLMs and VLMs. Our method also uses a two-stage inference procedure, where we first infer a high-level semantic subtask (e.g., “pick up the plate”), and then predict the action based on this subtask. Many prior methods have employed two separate models for this purpose, with a VLM predicting semantic steps and a separate low-level policy executing those steps [2, 71, 13, 24, 70, 72, 47]. Our method uses the same exact model for both high-level and low-level inference, in a recipe that more closely resembles chain-of-thought [82] or test-time compute [39] methods, though unlike embodied chain-of-thought methods [88, 46, 61], the high-level inference process still runs at a lower frequency than low-level action inference.

Robotic learning systems with open-world generalization. While most robotic learning systems are evaluated in environments that closely match the training data, a number of prior works have explored broader open-world generalization. When the robot’s tasks are restricted to a more narrow set of basic primitives, such as picking up objects, methods that allow for task-specific assumptions (e.g., grasp prediction, or incorporating model-based planning and control) have been shown to generalize broadly, even to entirely new homes [40, 20, 60, 56, 29]. However, such methods do not readily generalize to the full range of possible tasks that a generalist robot might need to perform. More recently, large-scale datasets collected across many domains [41, 68, 63, 67, 14, 49] have been shown to enable generalization of simple but end-to-end learned tasks to new environments [33, 31, 67, 69, 26, 49, 28, 64]. However, the tasks in these demonstrations are still relatively simple, typically less than a minute in length and often with relatively low success rates. We show that $\pi _ { 0 . 5 }$ can perform long, multistage tasks, such as putting all of the dishes in the sink or picking all of the clothing off the floor of a new bedroom, while generalizing to entirely new homes.

![image](images/889c09e7f014b762d1ef657835f50cc2f3f5ae643bd9cd6f6f744769251a5a00.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["pre-training"] --> B["language subtasks"]
    B --> C["put the plate in the sink"]
    B --> D["discretized actions"]
    D --> E["-17 12 34 142 -72 -135"]
    D --> F["open vocabulary captions"]
    F --> G["a dog catches a frisbee"]
    F --> H["bounding boxes"]
    H --> I["3 35 145 223"]
    I --> J["pre-trained VLM\nSigLIP (400M) + Gemma (2.6B)"]
    J --> K["clean the kitchen"]
    J --> L["pick up the pillow"]
    J --> M["caption the image"]
    J --> N["localize the gripper"]
    K --> O["multimodal web & robot data"]
    L --> O
    M --> O
    N --> O
    O --> P["task-specific prompts"]
```
</details>

![image](images/c413fd1b3f276c7a2b9a4ed6a7aeff1a378877255af62945130752ef2fae3b33.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["post-training & inference"] --> B["subtask prediction"]
    B --> C["pick up the pillow"]
    C --> D["pre-trained VLA"]
    D --> E["clean the bedroom"]
    E --> F["high-level prompt"]
    E --> G["low-level command"]
    G --> H["pick up the pillow"]
    H --> I["continuous actions"]
    I --> J["-1.7 1.25 3.14 1.42"]
    J --> K["action expert (300M)"]
    K --> L["noise"]
```
</details>

Fig. 3: Model overview. $\pi _ { 0 . 5 }$ is trained in two stages. First, a pre-training stage combines all of the different data sources to produce an initial VLA with discrete tokens. This stage uses data from diverse robotic platforms, high-level semantic action prediction, and data from the web. Robotic data uses the FAST action tokenizer to represent actions as discrete tokens [64]. Second, a post-training stage specializes the model for low-level and high-level inferences for mobile manipulation, leveraging the most task-relevant data, including verbal instructions from human supervisors. This stage uses flow matching to represent the action distribution, enabling efficient real-time inference and the ability to represent fine-grained continuous action sequences. At inference time, the model first infers a high-level subtask, and then predicts the actions based on this subtask.

# III. PRELIMINARIES

Vision-language-action models (VLAs) are typically trained via imitation learning on diverse robot demonstration datasets ${ \mathcal { D } } ,$ by maximizing the log-likelihood of an action $\mathbf { a } _ { t }$ (or, more generally, an action chunk $\mathbf { a } _ { t : t + H } )$ given an observation $\mathbf { o } _ { t }$ and a natural language task instruction ℓ: maxθ $\mathbb { E } _ { ( \mathbf { a } _ { t : t + H } , \mathbf { o } _ { t } , \ell ) \sim \mathcal { D } } \log \big ( \pi _ { \theta } ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \ell ) \big )$ . The observation typically contains one or more images $\mathbf { I } _ { t } ^ { 1 } , . . . , \mathbf { I } _ { t } ^ { n }$ and proprioceptive state $\mathbf { q } _ { t } ,$ which captures the position of the robot’s joints. VLA architectures follow the design of modern language and vision-language models, with modality-specific tokenizers that map inputs and outputs to discrete (“hard”) or continuous (“soft”) token representations, and a large, autoregressive transformer backbone that is trained to map from input to output tokens. The weights of these models are initialized from pre-trained vision-language models. By encoding policy inputs and outputs into tokenized representations, the imitation learning problem described above can be cast as a simple next-token-prediction problem over a sequence of observation, instruction and action tokens, and we can leverage the scalable tools of modern machine learning to optimize it. In practice, the choice of tokenizers for image and text inputs follows those of modern vision-language models. For actions, prior work has developed effective, compressionbased tokenization approaches [64], which we use in this work during pretraining. A number of recent VLA models have also proposed to represent the action distribution via diffusion [55, 84, 52] or flow matching [8], providing a more expressive representation over continuous-valued action chunks. During the post-training phase of our model, we will build on the design of the $\pi _ { 0 }$ model [8], which represents the action distribution via flow matching. In this design, the tokens corresponding to actions receive the partially denoised actions from the previous step of flow matching as input, and output the flow matching vector field. These tokens also use a different set of model weights, which we refer to as an “action expert,” analogously to a mixture of experts architecture. This action expert can specialize to flow matching-based action generation, and can be significantly smaller than the rest of the LLM backbone.

# IV. THE $\pi _ { 0 . 5 }$ MODEL AND TRAINING RECIPE

We provide an overview of the $\pi _ { 0 . 5 }$ model and training recipe in Figure 3. The model weights are initialized from a standard VLM trained on data from the web, and training then proceeds in two stages: a pre-training stage intended to adapt the model to diverse robotic tasks, and a post-training stage intended to specialize it to mobile manipulation and equip it with the mechanisms for efficient test-time inference. During pre-training, all tasks, including tasks with robot actions, are represented with discrete tokens, which leads to simple, scalable, and efficient training [64]. During post-training, we adapt the model to also have an action expert, as with $\pi _ { 0 } ,$ , in order to both represent actions with finer granularity and enable more compute-efficient inference for real-time control. At inferencetime, the model first produces a high-level subtask for the robot to perform and then, conditioned on this subtask, predicts the low-level actions via the action expert. We describe the model architecture below, followed by a description of each of the phases and their corresponding training tasks.

# A. The $\pi _ { 0 . 5 }$ architecture

The $\pi _ { 0 . 5 }$ architecture can flexibly represent both action chunk distributions and tokenized text outputs, with the latter used both for co-training tasks (e.g., question-answering) and for outputting high-level subtask predictions during hierarchical inference. The distribution captured by the model can be written as $\pi _ { \theta } ( \mathbf { a } _ { t : t + H } , \hat { \ell } | \mathbf { o } _ { t } , \ell )$ , where $\mathbf o _ { t } = [ \mathbf I _ { t } ^ { 1 } , . . . , \mathbf I _ { t } ^ { n } , \mathbf q _ { t } ]$ consists of the images from all of the cameras and the robot’s configuration (joint angles, gripper pose, torso lift pose, and base velocity), ℓ is the overall task prompt (e.g., “put away the dishes”), ˆℓ represents the model’s (tokenized) textual output, which could be either a predicted high-level subtask (e.g., “pick up the plate”) or the answer to a vision-language prompt in web data, and $\mathbf { a } _ { t : t + H }$ is a predicted action chunk. We decompose the distribution as

$$
\pi_ {\theta} (\mathbf {a} _ {t: t + H}, \hat {\ell} | \mathbf {o} _ {t}, \ell) = \pi_ {\theta} (\mathbf {a} _ {t: t + H} | \mathbf {o} _ {t}, \hat {\ell}) \pi_ {\theta} (\hat {\ell} | \mathbf {o} _ {t}, \ell),
$$

where the action distribution does not depend on $\ell ,$ only on $\hat { \ell } .$ Thus, high-level inference captures $\pi _ { \boldsymbol { \theta } } ( \widehat { \ell } | _ { \mathbf { o } _ { t } , \ell } )$ , and low-level inference captures $\pi _ { \boldsymbol { \theta } } \big ( \mathbf { a } _ { t : t + H } | \mathbf { o } _ { t } , \boldsymbol { \hat { \ell } } \big )$ , with both distributions represented by the same model.

The model corresponds to a transformer that takes in N multimodal input tokens $x _ { 1 : N }$ (we use the term token loosely here, referring to both discretized and continuous inputs) and produces a sequence of multimodal outputs $y _ { 1 : N } .$ , which we can write as $y _ { 1 : N } = f { \big ( } x _ { 1 : N } , A ( x _ { 1 : N } ) , \rho ( x _ { 1 : N } ) { \big ) }$ . Each $x _ { i }$ can be a text token $( x _ { i } ^ { w } \in \mathbb { N } )$ , an image patch $( \boldsymbol { x } _ { i } ^ { I } \in \mathbb { R } ^ { p \times p \times 3 } )$ , or an intermediate denoising value of a robot action in flow matching $( x _ { i } ^ { a } \in \mathbb { R } ^ { d } )$ . The observations $\mathbf { o } _ { t }$ and ℓ form the prefix part of $x _ { 1 : N }$ . Depending on the token type, as indicated by $\rho ( x _ { i } )$ , each token can be processed not only by a different encoder, but also by different expert weights within the transformer. For example, image patches are fed through a vision encoder, and text tokens are embedded with an embedding matrix. Following π0 [8], we linearly project action tokens $\boldsymbol { x } _ { i } ^ { a }$ into the transformer embedding space and use separate expert weights in the transformer to process the action tokens. The attention matrix $A ( x _ { 1 : N } ) \in [ 0 , \dot { 1 } ] ^ { N \times N }$ indicates if a token can attend to another token. Compared to standard causal attention in LLMs, image patch, textual prompt, and continuous action tokens use bidirectional attention.

As we want our model to output both text (to answer questions about the scene or to output next tasks to accomplish) and actions (to act in the world), the output of f is split into text token logits and action output tokens, respectively $\left( y _ { 1 : M } ^ { \ell } , y _ { 1 : H } ^ { a } \right)$ . The first M correspond to text token logits that can be used to sample $\hat { \ell }$ and the later H tokens are produced by a separate action expert, as in $\pi _ { 0 }$ , and projected via a linear mapping to continuous outputs used to obtain $\mathbf { a } _ { t : t + H }$ (see next section). Note that $M + H \leq N$ , i.e., not all outputs are associated with a loss. The robot proprioceptive state is discretized and input to the model as text tokens. More details about the architecture are in Appendix E.

# B. Combining discrete & continuous action representations

Similarly to $\pi _ { 0 } ,$ , we use flow-matching [50] to predict continuous actions in the final model. Given $\mathbf { a } _ { t : t + H } ^ { \tau , \omega } = \tau \mathbf { a } _ { t : t + H } +$ $( 1 - \tau ) \omega , \omega \sim \mathcal { N } ( 0 , \mathbf { I } )$ , where $\tau \in [ 0 , 1 ]$ is the flow matching time index, the model is trained to predict the flow vector field $\omega - \mathbf { a } _ { t }$ . However, as shown in [64], VLA training can be much faster when actions are represented by discrete tokens, particularly when using a tokenization scheme that is efficient for compressing the action chunks (e.g., FAST). Unfortunately, such discrete representations are less well-suited for realtime inference, because they require expensive autoregressive decoding for inference [64]. Therefore, an ideal model design would train on discretized actions but still allow for use of flow matching to produce continuous actions at inference time.

Our model is therefore trained to predict actions both through autoregressive sampling of tokens (using the FAST tokenizer) and iterative integration of the flow field, combining the best of both worlds. We use the attention matrix to ensure that the different action representations do not attend to each other. Our model is optimized to minimize the combined loss

$$
\begin{array}{l} \mathbb {E} _ {\mathcal {D}, \tau , \omega} \left[ H \left(x _ {1: M}, f _ {\theta} ^ {\ell} (\mathbf {o} _ {t}, \ell)\right) \right. \\ \left. + \alpha \left\| \omega - \mathbf {a} _ {t: t + H} - f _ {\theta} ^ {a} \left(\mathbf {a} _ {t: t + H} ^ {\tau , \omega}, \mathbf {o} _ {t}, \ell\right) \right\| ^ {2} \right], \tag {1} \\ \end{array}
$$

where $H ( x _ { 1 : M } , y _ { 1 : M } ^ { \ell } )$ is the cross entropy loss between the text tokens and predicted logits (including the FAST encoded action tokens), $y _ { 1 : H } ^ { a } = f _ { \theta } ^ { a } ( \mathbf { a } _ { t : t + H } ^ { \tau , \omega } , \mathbf { o } _ { t } , \ell )$ is the output from the (smaller) action expert, and $\alpha \in \mathbb { R }$ is a trade-off parameter. This scheme enables us to first pre-train our model as a standard VLM transformer model by mapping actions to text tokens $( \alpha = 0 )$ , and then add additional action expert weights predicting continuous action tokens in a non-autoregressive fashion for fast inference in a post-training stage. We find that following this procedure, which is further explained below, leads to stable pre-training and excellent language following abilities of the VLA model. At inference time we then use standard autoregressive decoding for text tokens ˆℓ followed by 10 denoising steps, conditioned on text tokens, to produce actions $\mathbf { a } _ { t : t + H }$ .

![image](images/0cb084987ee7af9e7a640ef647109f68e873c843c229fde1a8b260d7d349ff29.jpg)  
Bounding boxes:   
<loc0405><loc0011><loc0911><loc0197>closet Subtask: move to closet   
Bounding boxes:   
<loc0571><loc0376><loc0815><loc0484>mitten   
<loc0787><loc0346><loc1003><loc0490>drawer   
Subtask: move left arm forward and pick up mitten   
Describe this region: <loc0470><loc0390><loc0605><loc0484> Front legs of elephant   
What kind of pie is this? This is a delicious-looking pecan pie. The image shows a classic pecan pie with its characteristic dark brown filling studded with pecans.

# Post-training

Fig. 4: Examples from pre-training and post-training tasks. π0.5 is pre-trained on data from mobile manipulators (MM), non-mobile robots in diverse environments (ME), and cross-embodiment data collected under laboratory conditions (CE), as well as high-level subtask prediction (HL), and multi-modal web data (WD). In a post-training phase, we additionally use verbal instructions (VI), and omit the laboratory cross-embodiment data (CE) to focus the model on mobile manipulation and diverse environments. The figure displays an exemplary subset of the tasks in each category.

# C. Pre-training

In the first training stage, π0.5 is trained with a broad range of robot and non-robot data, which we summarize below and illustrate in Figure 4. It is trained as a standard auto-regressive transformer, performing next-token prediction of text, object locations, and FAST encoded action tokens.

Diverse Mobile Manipulator data (MM). We use about 400 hours of data of mobile manipulators performing household tasks in about 100 different home environments, some of which are shown in Figure 7, using the robots in Section IV-E. This slice of the training set is the most directly relevant to our evaluation tasks, which consist of similar cleaning and tidying tasks in new, unseen, home environments.

Diverse Multi-Environment non-mobile robot data (ME). We also collected non-mobile robot data, either with a single arm or two arms, in a variety of home environments. These arms were fixed to surfaces or mounting platforms, and because they are significantly lighter and easier to transport, we were able to gather a more diverse dataset in a wider range of homes with them. However, this ME data comes from a different embodiment than the mobile robots.

Cross-Embodiment laboratory data (CE). We collected data for a wide range of tasks (e.g., bussing a table, folding shirts) in the laboratory, with simpler tabletop environments and a variety of robot types. Some of these tasks are highly relevant to our evaluation (e.g., putting dishes in a bin), while others are not (e.g., grinding coffee beans). This data includes singlearm and dual-arm manipulators, and both static and mobile bases. We also include the open-source OXE dataset [15]. This dataset is an extended version of the dataset used by π0[8].

High-Level subtask prediction (HL). Breaking down highlevel task commands such as “clean the bedroom” into shorter subtasks like “adjust the blanket” and “pick up pillow”, similar to chain-of-thought prompting for language models, can help a trained policy reason about the current scene and better determine the next action. For robot data in MM, ME, and CE where the task involves multiple subtasks, we manually annotate all data with semantic descriptions of the subtasks and train $\pi _ { 0 . 5 }$ to jointly predict the subtask labels (as text) as well as the actions (conditioned on the subtask label) based on the current observation and high-level command. This naturally leads to a model that can act both as a high-level policy (outputting subtasks) and low-level policy that executes actions for these subtasks. We also label relevant bounding boxes shown in the current observation and train $\pi _ { 0 . 5 }$ to predict them before predicting the subtask.

Multi-modal Web Data (WD). Finally we include a diverse set of web data involving image captioning (CapsFusion [87], COCO [12]), question answering (Cambrian-7M [77], PixMo [19], VQAv2 [32]), and object localization in pre-training. For object localization, we further extend the standard datasets with additional web data of indoor scenes and household objects with bounding box annotations.

For all action data, we train the model to predict target joint and end-effector poses. To differentiate the two, we add ‘<control mode> joint/end effector <control mode>’ to the text prompt. All action data is normalized to [−1, 1] using the 1% and 99% quantile of each action dimension of the individual dataset. We set the dimensionality of the action a to a fixed number to accommodate the largest action space among all the datasets. For robots with lower-dimensional configuration and action spaces, we zero-pad the action vectors.

# D. Post-training

After pre-training the model with discrete tokens for 280k gradient steps, we perform a second stage of training that we refer to as post-training. The purpose of this stage is to both specialize the model to our use-case (mobile manipulation in homes), and to add an action expert that can produce continuous action chunks via flow matching. This stage jointly trains with next-token prediction, to preserve text prediction capabilities, and flow matching for the action expert (which is initialized with random weights at the beginning of posttraining). We optimize the objective in Equation (1), with $\alpha = 1 0 . 0$ for 80k additional steps. The post-training action dataset consists of the MM and ME robot data, filtered down to successful episodes that are below a fixed length threshold. We include web data (WD) to preserve the model’s semantic and visual capabilities, and the slice of HL data corresponding to the multi-environment datasets. Additionally, to improve the model’s ability to predict appropriate high-level subtasks, we collect verbal instruction demonstrations (VI), which are constructed by expert users providing “language demonstrations,” selecting appropriate sub-task commands to command the robot to perform mobile manipulation tasks step by step. These examples are collected by “teleoperating” the robot in real time with language to perform tasks with the learned low level policy, essentially providing demonstrations of good high-level subtask outputs for a trained policy.

# E. Robot system details

The robot systems used in our mobile manipulation experiments are illustrated in Figure 5. We conducted all of our experiments using two types of mobile manipulators. Both platforms are equipped with two 6 DoF arms with parallel jaw grippers and wrist-mounted monocular RGB cameras, a wheeled holonomic base, and a torso lift mechanism. The state and action spaces for the base correspond to linear (2D) and angular (1D) velocity, and the torso lift mechanism is either 1D (up/down) or 2D (up/down and forward/backward). In addition to the two wrist cameras, the robots have a forward and backward facing camera mounted between the arms. We use all four cameras for high-level inference, and the wrist and forward cameras for the low-level inference process. The total dimensionality of the state and action spaces is 18 or 19, depending on the platform.

![image](images/28885acd59d6e5d24dd5e74691f287b882a977647ee16a2d97718b3cb2833a03.jpg)

<details>
<summary>text_image</summary>

4x images
front & rear camera
2x 6 DoF arm + 1 DoF gripper
2x wrist camera
1-2 DoF lift mechanism
3 DoF holonomic base
</details>

Fig. 5: Robot system overview. We use two mobile manipulator platforms – each has four cameras (forward, backward, and both wrists), two 6 DoF arms with parallel jaw grippers, a mobile base, and a torso lift mechanism. The $\pi _ { 0 . 5 }$ model controls the joints and grippers of each arm, base velocity, and the lift position, resulting in 18-19 DoF state and action spaces.

The control system is very simple: the $\pi _ { 0 . 5 }$ model directly commands target poses for the arms, gripper, and torso lift, and the target base velocities at 50 Hz (with action chunking). These targets are tracked with simple PD controllers, without any additional trajectory planning or collision detection. All manipulation and navigation control is fully end-to-end.

# V. EXPERIMENTAL EVALUATION

The $\pi _ { 0 . 5 }$ model is designed to generalize broadly to new environments. While it is common to evaluate VLAs in environments that match the training data, we conduct all of our experiments in novel environments that were not seen in training. For quantitative comparisons, we use a set of mock home environments to provide a controlled and reproducible setup, while the most realistic final evaluation is conducted in three real homes that were not part of the training set (see Figure 6). Our experiments focus on the following questions:

1) Can $\pi _ { 0 . 5 }$ effectively generalize to complex multi-stage tasks in entirely new homes?   
2) How does the generalization of $\pi _ { 0 . 5 }$ scale with the number of distinct environments in the training data?   
3) How do the individual co-training ingredients in the $\pi _ { 0 . 5 }$ training mixture contribute to its final performance?   
4) How does $\pi _ { 0 . 5 }$ compare to the $\pi _ { 0 }$ VLA?   
5) How important is the high-level inference component of $\pi _ { 0 . 5 } ,$ and how does it compare to flat, low-level inference as well as oracle high-level baselines?

# A. Can $\pi _ { 0 . 5 }$ generalize to real homes?

To answer Question (1), we evaluated $\pi _ { 0 . 5 }$ in three real homes that were not present in the training set, using both types of robots. In each of the homes, the robots were instructed to perform a bedroom and kitchen cleaning task. The evaluation rubrics for each task are provided in Appendix B and roughly correspond to the percentage of steps in each task that were completed successfully (e.g., placing half the dishes in the sink corresponds to around 50%). The results in Figure 7

![image](images/d0b28c7722da8f43f98eaf825f0fb37b42de2f9c81b34432bc34d5d481ae528b.jpg)

<details>
<summary>natural_image</summary>

Interior view of a kitchen with a bench, bowl, and camera rig (no visible text or symbols)
</details>

![image](images/fd26d57449e603447c9e631c8546c7328698fbbc3271419c9951998bcb4df486.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with green walls, stainless steel cabinets, and a dog on the floor (no visible text or symbols)
</details>

![image](images/3efc4110ff43eb7338d7c39f35a571f0fbd77d6fe270e23ad69a38fcd8a37170.jpg)  
Mock Kitchens

![image](images/61b13e22fc4a237c01d493da34a46c03ce386545de926d08f671c86c748332b8.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with electric stove, built-in robotic arm, and wooden furniture (no visible text or symbols)
</details>

![image](images/f5123630550ed54d650001958b885d5b40568f07a27adad84d0ac91384a95a54.jpg)

![image](images/9bcdddd94c9c238ab45d82a28a81df9dba29face6b732c282a390c03ae5a73dd.jpg)  
Real Kitchens

![image](images/f4a174007d8cfdaf7e55fb0c584f23c6dad90e13b81e36031c1436b232fc6dd8.jpg)

<details>
<summary>natural_image</summary>

Interior view of a bedroom with a bed, patterned bedding, and wall-mounted equipment (no visible text or symbols)
</details>

![image](images/768cef789eee942dab467fdab6421f3ce3fd6878d17d8e1724df1079409926e8.jpg)

![image](images/56eeb55831f5c73d5fa14fca7e51b360de6f84fbc1499fc0dd969d1fd20dc5ff.jpg)  
Mock Bedrooms

![image](images/7cab422f8fa5b0712279ae95374bed2add2d39a0c17bd0796a8e1d5e9e0d9ce4.jpg)

<details>
<summary>natural_image</summary>

Interior view of a bedroom with a robotic arm and camera on a bed, no visible text or symbols
</details>

![image](images/8af73a4b0c6a05c7dc460bad1a2623a386d12aeb863f1e606cdb9f55a4769d9c.jpg)

![image](images/76e6d8797630a1b858268ecdc1d7ede0cac6bb6cad78c7dd4767bf59b84f3ff7.jpg)  
Real Bedrooms   
Fig. 6: Evaluation environments. We evaluate $\pi _ { 0 . 5 }$ in entirely new kitchens and bedrooms that were not seen during training, with novel objects, backgrounds, and layouts. We use a set of mock rooms for controlled, reproducible quantitative comparisons (left) and real homes for a realistic final evaluation (right).   
Home 1 Human: “put the items in the drawer”   
HL prediction:

![image](images/3cff27ed897179fabb89413e648773fb8e96f785cbc59adbfc36b51aa6d7422c.jpg)  
pull out the drawer

![image](images/adba2b819e02a75e340b5b3a0f7d62dee8b854b1af59d44818be069de658d2f1.jpg)  
pull out the top right drawer

![image](images/14ad7617035dc1d7f62b7d9ec533f8676cc437a54c86384b700750e44101bd26.jpg)  
pick up tong

![image](images/92f8babafa5af3acec1d14b41bb000ff40d8f06a5ec54e67d274d0064e0b6f4c.jpg)  
put tong into drawer

![image](images/8c894e4769a6833bf2e40a7bef9e11f835465d8a88dfe51d026f061792a794f8.jpg)  
push the top drawer   
HL prediction:

Home 2 Human: “place the dishes in the sink”

![image](images/d1b02128485febcca4d9cd86006b8f4bc893fc41e54cd7541da6552d21ec0d4c.jpg)  
pick up plate

![image](images/681d1ed8ccdaa812378e0c14e15518f91fb8c5ff1e3690ec1d631ad9312de3f3.jpg)  
put plate in the sink

![image](images/cb90cde201954c37cf66cb749490cb3c2a5e3bf6200ae9780d35db9f47b26802.jpg)  
put cup in the sink

![image](images/33be82c65df42f2eb5656adbce2a640e1e2571e277b681591a995c461da6b552.jpg)  
pick up the spoon

![image](images/dfcb398b9e527f3c05e7136d5f3e0b3da97bb56535b4cf422603e202042d1127.jpg)  
pick up bowl

Home 3 Human: “put the laundry in the laundry basket” HL prediction:   
![image](images/5bdee2444aaa89859ecf412be896ad8225c1123e9af47fa81ea8b56ee1af905d.jpg)  
pick up shirt

![image](images/fa856abf19f744bf3ed3cf1406782a5a77a7934f04eb560388806d36d91456d7.jpg)  
pick up shirt

![image](images/41fe6f635667398584db6efe5262cb50296e0d199f44f2b0c07129f3ac55ac83.jpg)  
put clothes in the laundry basket

![image](images/9ca413224ec3730fa11a4cb39b2a8517adad8af8bda20d8b922799f0e756bea7.jpg)  
put the shirt in the laundry basket

![image](images/70f27f742d9ea406ec33ceee6e7eb2344448e679b287eab84d9f3afdd534785a.jpg)  
put clothes in laundry basket

100% 80%

![image](images/6b80f2b3bf6474638fd7145553252959516d0805ba4376c4a502708d7902f6b7.jpg)  
40% 40%

![image](images/677c7c89a86c1517bce13913015e8df17cf5f155a14111cf746e8ab55f028e6c.jpg)

![image](images/b2c333679db1c5a815aab6c8a6b69d37196996f9d132946f66312bcf136ef9b8.jpg)

![image](images/6ef9018552c778e7489578cf8b8e869abee1def54f6e36849856fc1d6b6e3a0e.jpg)

![image](images/3a8cb59544368f9a15af682a76c414ab99f3bc9914b468ab185ee912cf1f1169.jpg)

![image](images/1b40c5ce44effb7319200e28b022bfbd2268c9e955ac360d4769b0a59a5fd0fc.jpg)  
20%   
Real Home   
1 Real Home 2   
Real Home 3   
Mock Environments

(a) Example rollouts. We visualize an exemplary $\pi _ { 0 . 5 }$ episode for one task from each home. Top to bottom: putting items in a drawer in Home 1, followed by putting dishes in the sink in Home 2, and putting clothes in the laundry basket in Home 3. The human instruction for each is given on the left, and the high-level subtask prediction from $\pi _ { 0 . 5 }$ is shown beneath each frame in blue.

(b) Quantitative evaluation. We show the task progress per task and environment averaged over 10 trials. We find that $\pi _ { 0 . 5 } ^ { } \ '$ s performance in the mock evaluation setups is representative of its performance in real homes.

Fig. 7: Evaluation in real homes. We evaluated $\pi _ { 0 . 5 }$ in three kitchens and three bedrooms in real homes that were not seen during training. We evaluate the tasks ‘items in drawer’, ‘laundry basket’, and ‘dishes in sink,’ and find $\pi _ { 0 . 5 }$ to be successful at these tasks in these completely new, real homes.

show that $\pi _ { 0 . 5 }$ was able to consistently succeed on a variety of tasks in each home (we note that, additionally, the model is capable of performing many more tasks than used in our quantitative evaluation). Many of the tasks involve multiple stages (e.g., moving multiple objects) lasting about 2 to 5 minutes. For these trials, the model is provided with a simple high-level command (e.g., “place the dishes in the sink”), and the high-level inference process autonomously determines appropriate steps (e.g., “pick up the cup”). This level of inthe-wild generalization goes significantly beyond the results demonstrated with prior vision-language-action models, both in terms of the degree of novelty that the model must handle, and the task duration and complexity.

# B. How does generalization scale with the number of scenes?

In the next set of experiments, we aim to measure how generalization scales with the number of environments seen in the training data. We vary the number of environments in the mobile manipulation data and measure its impact on generalization by training with data from 3, 12, 22, 53, 82, and 104 locations. Since applying the entire pre-training and post-training recipe to each of these datasets is prohibitively

compute-intensive, for these experiments we pre-train on the mixture of robot action prediction data without mobile manipulation data, and then compare models post-trained on datasets that comprise mobile manipulation data from varying numbers of environments. While the datasets split by location in principle differ in size, in practice the number of training steps (40k) is chosen such that each model sees the same number of unique data samples, which allows us to control for dataset size when varying the number of locations used within a post-training experiment.

Each model is evaluated in the mock environments shown in Figure 6, which are not seen in training. We conduct two types of evaluations. First, to evaluate overall performance on multi-stage tasks, we use the standard rubric in Appendix B and the mock test homes to evaluate each model’s end-to-end performance on putting dishes in the sink, packing items into a drawer, putting away laundry, and making a bed. Second, we conduct a more fine-grained evaluation of each model’s ability to follow language instructions and interact with novel objects, where the robot must pick up specific objects from a kitchen counter based on language commands. These experiments use both in-distribution objects from similar categories as those in the training data (but new instances), as well as out-of-distribution objects from unseen categories. The latter necessitates broad semantic generalization.

![image](images/ae04f1dc8e043467d6fdcc831544b802f83262d9f241bde9491a548318c8cc7b.jpg)

<details>
<summary>line</summary>

| Number of Locations | ours for different numbers of locations | in-domain data | in-domain data no pre-training | 104 locations no pre-training |
| ------------------- | ---------------------------------------- | -------------- | ------------------------------ | ------------------------------ |
| 0                   | 15%                                      | -              | -                              | -                              |
| 20                  | 60%                                      | -              | -                              | -                              |
| 40                  | 75%                                      | -              | -                              | -                              |
| 60                  | 80%                                      | -              | -                              | -                              |
| 80                  | 65%                                      | -              | -                              | -                              |
| 100                 | 85%                                      | 83%            | 39%                            | 5%                             |
</details>

Fig. 8: Evaluating performance with different numbers of locations. Performance over the four test tasks — “dishes in sink”, “items in drawer”, “laundry basket”, “make bed” — improves with more training environments. The dashed green line and green bar show a baseline model that includes the test homes in the training set. Compared to this model, our best model achieves similar performance, despite not seeing any data from the test homes.

The results of the first experiment are shown in Figure 8. The average performance among the tasks generally improves with more training locations. To quantify how much the final model (with 104 locations) bridges the generalization gap, we include a control (shown in green) that is trained directly on data from the test homes. This control attains similar performance as the final 104-location model, suggesting that our co-training recipe effectively enables broad generalization, reaching similar performance to a model trained on the test environment. To confirm that this generalization performance requires our full co-training recipe, we additionally include two baselines that do not use any of the other co-training tasks in the pre-training phase, but instead train directly on either data from the test environment (light green) or mobile manipulation data from the 104 training locations (light yellow). The performance for both those baselines is significantly worse — this indicates that the other data sources leveraged by our full training recipe are essential for good generalization, even when the policy has seen robot data from test homes. When not using data from test homes, pre-training with our recipe is especially important, as can be seen by the large gap between the green bars and light yellow bar in Figure 8.

The results of the second experiment (language following) are shown in Figure 9. We report the language following rate, which measures how often the robot selects the object indicated in the language command, and success rate, which measures how often the robot successfully places that object in the correct location (either inside the drawer or inside the sink, depending on the test scenario). We separately measure performance on object categories seen in training (but new object instances) and unseen (“out-of-distribution”) object categories. Details of this experiment are shown and discussed in Appendix C. Figure 9 shows that, as the number of locations in the training data increases, both language following performance and success rate improve. As expected, the performance on in-distribution objects improves more quickly than that of out-of-distribution objects. As each new environment introduces new household items, the model becomes generally more robust and starts to generalize to task categories that were not present in the training data.

![image](images/1c7d4904322ee2c5b5fe894db3f75d56af40924db99a6220e900b3a311a1a27f.jpg)

<details>
<summary>line</summary>

| Number of Locations | In-Distribution | Out-of-Distribution |
| ------------------- | --------------- | ------------------- |
| 20                  | 45%             | 25%                 |
| 40                  | 60%             | 35%                 |
| 60                  | 65%             | 45%                 |
| 80                  | 60%             | 55%                 |
| 100                 | 65%             | 70%                 |
</details>

![image](images/51ec6baca50bc6e3e29c63c9d17c6086df259d11d8356224787c582ff326d7b2.jpg)

<details>
<summary>line</summary>

| Number of Locations | In-Distribution | Out-of-Distribution |
| ------------------- | --------------- | ------------------- |
| 20                  | 20%             | 15%                 |
| 40                  | 40%             | 15%                 |
| 60                  | 55%             | 30%                 |
| 80                  | 55%             | 30%                 |
| 100                 | 65%             | 55%                 |
</details>

Fig. 9: Evaluating language following with different numbers of training locations. We evaluate language following rate and success rate for picking up user-indicated items and placing them into drawers or sinks, averaged over seen object categories (“in-distribution”) or unseen categories (“out-ofdistribution”). Performance increases steadily as we increase the number of training locations.

# C. How important is each part of our co-training recipe?

To study Question (3), we compare our full $\pi _ { 0 . 5 }$ model to other training mixtures to study the importance of each mixture component, again using end-to-end task performance in the mock homes and the language following evaluation described in Section V-B. As a reminder, our full recipe uses data from mobile manipulators in many environments (MM), static manipulators in many environments (ME), and diverse cross-embodiment data collected in laboratory settings (CE). It also includes high-level data where the prediction corresponds to a high-level language command (HL), and web data corresponding to captioning, VQA, and object localization tasks (WD). Post-training also uses verbal instruction data (VI), which we analyze in Section V-E. In these experiments, we ablate different parts of the mixture:

1) no WD: this ablation excludes web data.   
2) no ME: this ablation excludes multi-environment nonmobile data.   
3) no CE: this ablation excludes the laboratory crossembodiment data.   
4) no ME or CE: this ablation excludes both data sources from other robots, such that the model is trained on only data from the target mobile manipulator platform as well as web data.

The results on the full mock home tasks are shown in Figure 10 (detailed breakdown of performance on each task in Appendix D). First, we see in the results that excluding either of the two cross-embodiment data sources (ME and CE) significantly degrades performance, indicating that π0.5 benefits considerably from cross-embodiment transfer, from both other environments (ME) and other tasks (CE). Excluding both sources harms performance even more. Interestingly, the difference in performance with the no WD ablation is not statistically significant in this experiment, though we show later that web data has a large impact on language following (below) and high-level subtask inference (Section V-E).

![image](images/9a0b239ba027c18eda8288b81e4161ba39f5ba29d6eba1e84c7fd3358836a42c.jpg)

<details>
<summary>bar</summary>

| Group       | Average Task Progress |
| ----------- | --------------------- |
| π₀.₅        | 80%                   |
| no WD       | 75%                   |
| no CE       | 50%                   |
| no ME       | 55%                   |
| no CE or ME | 40%                   |
</details>

Fig. 10: Training recipe ablations, mock homes. We evaluate variants of our model that exclude different parts of the training mixture on all four test tasks (10 trials per policy and task). Including cross-embodiment data, both in diverse environments (ME) and for diverse tasks in laboratory settings (CE) is important for good performance, with large degradation when either or both of these data sources are removed. Web data (WD) does not make a significant difference in these experiments, but we will see in Figures 11 and 13 that it impacts object generalization and high-level performance.

![image](images/e873349d6938cd14f7032600782aa3da4a37d9ea96772ea360001bfb68e67ea6.jpg)

<details>
<summary>bar</summary>

| Category | π0.5 (%) | no WD (%) | no CE (%) | no ME (%) | no ME or CE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| In-Dist. Follow Rate | 86 | 86 | 74 | 66 | 56 |
| In-Dist. Success Rate | 83 | 82 | 67 | 57 | 50 |
| OOD Follow Rate | 94 | 80 | 67 | 33 | 33 |
| OOD Success Rate | 94 | 73 | 49 | 31 | 33 |
</details>

Fig. 11: Training recipe ablations, language following. Evaluating language following with in-distribution and out-of-distribution objects after training on different numbers of locations. Including web data (WD) is important for outof-distribution (OOD) performance in particular. Cross-embodiment (CE) and diverse environment (ME) data both have a large impact on in-distribution and out-of-distribution performance.

The results of the language following experiment, shown in Figure 11, show a similar trend as Figure 10 — excluding ME or/and CE data leads to a significant degradation in performance. What differs now is that removing web data (no WD) causes significantly worse performance on out-ofdistribution (OOD) objects — we conjecture that training with web data, which contains very broad knowledge of physical objects, allows the model to understand and follow language commands involving unseen object categories.

# D. How does $\pi _ { 0 . 5 }$ compare to other VLAs?

We compare $\pi _ { 0 . 5 }$ to the original $\pi _ { 0 }$ VLA as well as an improved version of $\pi _ { 0 }$ which we denote as π0-FAST+Flow. This version is trained via the joint diffusion and FAST action prediction formulation from Equation (1), but on action data only, without the HL or WD datasets. These models provide a strong point of comparison, since $\pi _ { 0 }$ has been demonstrated to perform strongly on complex and dexterous mobile manipulation tasks, and the enhancement in $\pi _ { 0 } .$ -FAST+Flow brings it as close to $\pi _ { 0 . 5 }$ as possible. $\pi _ { 0 . 5 }$ builds on these models with a combination of co-training tasks. For a fair comparison, all models receive the same cross-embodiment robot training set and are trained for a comparable number of steps. The differences then are: (1) $\pi _ { 0 . 5 }$ additionally uses HL and WD data; (2) $\pi _ { 0 . 5 }$ uses a hybrid training procedure, with discrete tokenized training in the pre-training phase, and training with a flow matching action expert only in the posttraining phase, while $\pi _ { 0 }$ always uses the action expert. $\pi _ { 0 ^ { - } }$ FAST+Flow follows the hybrid training recipe but is trained only with data containing robot actions and thus cannot perform high-level inference. The results in Figure 12 show that $\pi _ { 0 . 5 }$ significantly outperforms both $\pi _ { 0 }$ and our enhanced version. This result holds even when we allow for longer training up to 300k training steps of $\pi _ { 0 } .$ , confirming that as in Pertsch et al. [64] training with FAST tokens is more effective in terms of compute than pure diffusion based training.

![image](images/be511a7adbeee43e721b2f367a0fe3b3cf6adb3d242b9d73a965bc71780aedf2.jpg)

<details>
<summary>bar</summary>

| Category | π₀.5 (%) | π₀-FAST+ (Flow (no HL)) (%) | π₀ 300k (%) | π₀ 80k (%) |
|---|---|---|---|---|
| dishes in sink | 92 | 56 | 42 | 20 |
| items in drawer | 85 | 71 | 33 | 6 |
| laundry basket | 81 | 74 | 31 | 39 |
| make bed | 59 | 48 | 21 | 14 |
</details>

Fig. 12: Comparing π0.5 with other models. Our full model significantly outperforms both π0 and π0-FAST+Flow in the mock home test environments.

# E. How important is high-level inference?

Finally, we evaluate the importance of high-level inference, and compare the performance of several alternative high-level inference methods. The high-level inference mechanism in $\pi _ { 0 . 5 }$ takes in a high-level command (e.g., “clean the bedroom”) and outputs the subtask to complete (e.g., “pick up pillow”), which is then used as context for inferring the lowerlevel actions, analogously to chain of thought inference [82]. While $\pi _ { 0 . 5 }$ uses a unified architecture where the same model performs both high-level and low-level inference, we can also construct baseline methods that either forego the highlevel inference process and feed the task prompt directly into the low-level system, as is common in standard VLA models [92, 8], or use another model for high-level inference to ablate the importance of different dataset components in terms of their impact on the high-level policy. We consider the following methods and ablations, all of which use the full π0.5 low-level inference process with different high-level policies:

1) $\pi _ { 0 . 5 }$ model for high-level and low-level inference.   
2) no WD: an ablation of $\pi _ { 0 . 5 }$ that excludes web data.

![image](images/a14d1a3b2e564f8e91d513c36cac646ddefd131ab6c8c87cbd8cb6b53dfc0e74.jpg)

<details>
<summary>bar</summary>

| Method       | Average Task Progress |
| ------------ | --------------------- |
| π₀.₅         | 78%                   |
| implicit HL  | 70%                   |
| no HL        | 62%                   |
| no VI        | 60%                   |
| no WD        | 60%                   |
| GPT-4 HL     | 58%                   |
| human HL     | 63%                   |
</details>

Fig. 13: Evaluation of the high-level inference process. While the full π0.5 model with high-level and low-level inference attains the best results, using only low-level inference (“implicit HL”) with the full $\pi _ { 0 . 5 }$ model also benefits from the inclusion of high-level subtask examples in training. In contrast, excluding verbal instructions (no VI) or web data (no WD) leads to a significant degradation in performance, and zero-shot prompting a large API-based model (GPT-4) performs worse.

3) no VI: an ablation of $\pi _ { 0 . 5 }$ that excludes the verbal instruction (VI) data.   
4) implicit HL: no high-level inference at runtime but includes high-level data in training, which may teach the model about subtasks implicitly.   
5) no HL: no high-level inference, and no high-level data in training at all.   
6) GPT-4: use GPT-4 as the high-level policy, evaluating the importance of training the high-level policy on robot data. To align the model with our domain, we prompt GPT-4 with a description of the task and a list of the most used labels to choose from.   
7) human HL: use an expert human as an “oracle” highlevel policy, to provide an upper bound on performance.

The results of these experiments are shown in Figure 13. The full $\pi _ { 0 . 5 }$ model performs the best, and outperforms even the human HL “oracle” baseline. Perhaps surprisingly, the second best model is the implicit HL ablation, which does not perform any high-level inference, but includes the full data mixture, i.e. also subtask prediction, in training. This strongly suggests the importance of the co-training recipe used by our model: while there is a benefit to explicitly infer highlevel subtasks, a significant portion of that benefit is already obtained simply by including subtask prediction data in the training mixture. The no HL ablation, excluding HL task even in training, performs significantly worse. The results also show that the relatively small verbal instruction dataset, which only constitutes about 11% of the high-level mobile manipulation examples, is critical to strong performance as the no VI ablation is significantly weaker. The no WD ablation is also significantly worse, indicating that much of the benefit of web data (perhaps unsurprisingly) lies in improving the high-level policy. Finally, the zero-shot GPT-4 ablation attains the worst performance, indicating the importance of adapting VLMs with robot data. We provide a detailed breakdown of performance on each task in Appendix D, Figure 17.

# VI. DISCUSSION AND FUTURE WORK

We described $\pi _ { 0 . 5 }$ , a co-trained model that builds on the $\pi _ { 0 }$ VLA to integrate a variety of data sources and enable generalization to new environments. The $\pi _ { 0 . 5 }$ VLA can control mobile manipulators to perform tasks in homes that were never seen in the training data, cleaning kitchens and bedrooms, making beds, hanging towels, and performing other multistage and dexterous behaviors. $\pi _ { 0 . 5 }$ is trained on about 400 hours of mobile manipulation data, but includes a much larger amount of data from other robots, including non-mobile manipulators in diverse environments and data collected under laboratory conditions. It is also co-trained jointly with data from the web, as well as high-level prediction data for outputting language commands based on robot observations. The generalization capabilities of $\pi _ { 0 . 5 }$ demonstrate that this cotraining recipe facilitates effective transfer, enabling highly generalizable control of a mobile manipulator with only a medium-sized mobile manipulation dataset.

π0.5 is not without its limitations. While our VLA exhibits broad generalization, it still makes mistakes. Some environments present persistent challenges (e.g., unfamiliar handles on drawers, or cabinets that are physically hard for the robot to open), some behaviors present challenges with partial observability (e.g., the robot arm occluding a spill that should be wiped), and in some cases the high-level subtask inference is easily distracted (e.g., closing and opening a drawer multiple times while putting away items). Addressing these challenges with better co-training, transfer, and larger datasets is a promising direction for future work. Other future work directions could address the technical constraints of our method. While $\pi _ { 0 . 5 }$ can perform a variety of behaviors to clean up kitchens and bedrooms, it processes relatively simple prompts. The complexity of the prompts that the model can accommodate is determined by the training data, and more complex preferences and instructions could be incorporated by producing more intricate and diverse annotations, either with human labelers or synthetically. The model also uses a relatively modest context, and incorporating richer context and memory could make the model significantly more capable in settings with more partial observability, such as tasks that require navigating between different rooms or remembering where objects are stored. More broadly, $\pi _ { 0 . 5 }$ explores a particular combination of heterogeneous data sources, but the specific sources of data can be explored even more broadly. For instance, the ability of our system to learn from verbal instructions provides a powerful new supervision modality, and future work could explore this and other ways that people can provide robots with additional contextual knowledge. We hope that our work will serve as a foundation for a new generation of VLAs that exhibit broad generalization to diverse real-world environments.

# ACKNOWLEDGEMENTS

We thank our robot operators for data collection, evaluations, logistics, and video recording. See Appendix A for a full contributions statement.

---

## 衍生问题
