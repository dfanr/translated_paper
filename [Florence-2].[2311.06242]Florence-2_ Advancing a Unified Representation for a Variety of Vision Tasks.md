# Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks

## 0. Abstract

| 【概述】原文 | 【概述】翻译 |
| ---- | ---- |
| ✅ We introduce Florence-2, a novel vision foundation model with a unified, prompt-based representation for a variety of computer vision and vision-language tasks. While existing large vision models excel in transfer learning, they struggle to perform a diversity of tasks with simple instructions, a capability that implies handling the complexity of various spatial hierarchy and semantic granularity. Florence-2 was designed to take text-prompt as task instructions and generate desirable results in text forms, whether it be captioning, object detection, grounding or segmentation. This multi-task learning setup demands large-scale, high-quality annotated data. To this end, we co-developed FLD-5B that consists of 5.4 billion comprehensive visual annotations on 126 million images, using an iterative strategy of automated image annotation and model refinement. We adopted a sequence-to-sequence structure to train Florence-2 to perform versatile and comprehensive vision tasks. Extensive evaluations on numerous tasks demonstrated Florence-2 to be a strong vision foundation model contender with unprecedented zero-shot and fine-tuning capabilities. | ✅ We introduce Florence-2, a novel vision foundation model with a unified, prompt-based representation for a variety of computer vision and vision-language tasks. While existing large vision models excel in transfer learning, they struggle to perform a diversity of tasks with simple instructions, a capability that implies handling the complexity of various spatial hierarchy and semantic granularity. Florence-2 was designed to take text-prompt as task instructions and generate desirable results in text forms, whether it be captioning, object detection, grounding or segmentation. This multi-task learning setup demands large-scale, high-quality annotated data. To this end, we co-developed FLD-5B that consists of 5.4 billion comprehensive visual annotations on 126 million images, using an iterative strategy of automated image annotation and model refinement. We adopted a sequence-to-sequence structure to train Florence-2 to perform versatile and comprehensive vision tasks. Extensive evaluations on numerous tasks demonstrated Florence-2 to be a strong vision foundation model contender with unprecedented zero-shot and fine-tuning capabilities. |

## 1 Introduction

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x1.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 1:  We aim to build a vision foundation model to enable extensive perception capabilities including spatial hierarchy and semantic granularity. To achieve this, a single unified model Florence-2 is pre-trained on our FLD-5B dataset encompassing a total of 5.4B comprehensive annotations across 126M images, which are collected by our Florence data engine.  | ✅ Figure 1:  We aim to build a vision foundation model to enable extensive perception capabilities including spatial hierarchy and semantic granularity. To achieve this, a single unified model Florence-2 is pre-trained on our FLD-5B dataset encompassing a total of 5.4B comprehensive annotations across 126M images, which are collected by our Florence data engine.  |

| 【第1节，第1段】原文 | 【第1节，第1段】翻译 |
| ---- | ---- |
| ✅ In the realm of Artificial General Intelligence (AGI) systems, there has been a notable shift towards utilizing pre-trained, versatile representations, acknowledged for task-agnostic benefits accross diverse applications. | ✅ 在通用人工智能 (AGI) 系统领域，人们已经明显转向使用预先训练的、多功能的表示形式，这种表示形式因其在不同应用中具有与任务无关的优势而受到认可。 |
| ✅ This trend is evident in natural language processing (NLP), where advanced models ( **1. On the opportunities and risks of foundation models.** ｜ **2. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.** ｜ **3. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.** ｜ **4. Exploring the limits of transfer learning with a unified text-to-text transformer.** ｜ **5. Language models are few-shot learners.** ｜ **6. Language models are unsupervised multitask learners.** ) show adaptability with comprehensive knowledge spanning various domains and tasks with simple instructions. | ✅ 这一趋势在自然语言处理 (NLP) 中很明显，其中高级模型 ( **1. On the opportunities and risks of foundation models.** ｜ **2. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.** ｜ **3. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.** ｜ **4. Exploring the limits of transfer learning with a unified text-to-text transformer.** ｜ **5. Language models are few-shot learners.** ｜ **6. Language models are unsupervised multitask learners.** ) 表现出适应性，具有涵盖各个领域和任务的全面知识以及简单的指令。 |
| ✅ The success of NLP motivates a parallel approach in computer vision. | ✅ NLP 的成功激发了计算机视觉领域的并行方法。 |

| 【第1节，第2段】原文 | 【第1节，第2段】翻译 |
| ---- | ---- |
| ✅ Universal representation for diverse vision-related tasks presents unique challenges, notably the need for comprehensive perceptual abilities. | ✅ 与各种视觉相关的任务的通用表示提出了独特的挑战，特别是需要全面的感知能力。 |
| ✅ Unlike NLP, which deals mainly with text, computer vision requires handling intricate visual data like object location, masked contours, and attributes. | ✅ 与主要处理文本的 NLP 不同，计算机视觉需要处理复杂的视觉数据，如对象位置、蒙版轮廓和属性。 |
| ✅ Attaining universal representation in computer vision demands adept management of a spectrum of complex tasks, organized two-dimensionally as illustrated in Figure 1 : | ✅ 要实现计算机视觉的通用表示，需要熟练地管理一系列复杂的任务，这些任务以二维形式组织，如 Figure 1 所示： |

| 【第1节，第3段】原文 | 【第1节，第3段】翻译 |
| ---- | ---- |
| ✅ Spatial Hierarchy : The model must discern spatial details across varying scales, understanding image-level concepts and fine-grained pixel specifics. | ✅ Spatial Hierarchy：模型必须辨别不同尺度的空间细节，理解图像级概念和细粒度像素细节。 |
| ✅ Accommodating the intricate spatial hierarchy within vision demands the model’s proficiency in handling diverse levels of granularity. | ✅ 适应视觉内复杂的空间层次要求模型能够熟练地处理不同粒度级别。 |

| 【第1节，第4段】原文 | 【第1节，第4段】翻译 |
| ---- | ---- |
| ✅ Semantic Granularity : Universal representation in computer vision should span a spectrum of semantic granularity. | ✅ Semantic Granularity：计算机视觉中的通用表示应该涵盖语义粒度的范围。 |
| ✅ The model transitions from high-level captions to nuanced descriptions, enabling versatile understanding for diverse applications. | ✅ 该模型从高级标题过渡到细致入微的描述，为不同的应用提供多种理解。 |

| 【第1节，第5段】原文 | 【第1节，第5段】翻译 |
| ---- | ---- |
| ✅ This pursuit is characterized by distinctiveness and substantial challenges. | ✅ 这一追求具有独特性和重大挑战性。 |
| ✅ A key hurdle is the scarcity of comprehensive visual annotations , hindering the development of a foundational model capable of capturing the intricate nuances of spatial hierarchy and semantic granularity. | ✅ 一个关键的障碍是 comprehensive visual annotations 的稀缺性，这阻碍了能够捕捉空间层次和语义粒度的复杂细微差别的基础模型的开发。 |
| ✅ Existing datasets, such as ImageNet ( **Imagenet: A large-scale hierarchical image database.** ) , COCO ( **Microsoft coco: Common objects in context.** ) , and Flickr30k Entities ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** ) , tailored for specialized applications, are extensively labeled by humans. | ✅ 现有的数据集，例如 ImageNet ( **Imagenet: A large-scale hierarchical image database.** )、COCO ( **Microsoft coco: Common objects in context.** ) 和 Flickr30k Entities ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** )，都是为专门应用而定制的，并由人工进行大量标记。 |
| ✅ To overcome this constraint, it is imperative to generate extensive annotations for each image on a larger scale. | ✅ 为了克服这一限制，必须为更大规模的每幅图像生成大量注释。 |

| 【第1节，第6段】原文 | 【第1节，第6段】翻译 |
| ---- | ---- |
| ✅ Another challenge is the absence of a unified pre-training framework with a singular network architecture that seamlessly integrates spatial hierarchy and semantic granularity in computer vision. | ✅ 另一个挑战是缺少一个能够无缝集成计算机视觉中的空间层次和语义粒度的 unified pre-training framework with a singular network architecture。 |
| ✅ Traditional models excel in tasks like object detection ( **1. Mask r-cnn.** ｜ **2. Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) , semantic segmentation ( **1. Masked-attention mask transformer for universal image segmentation.** ｜ **2. Unified perceptual parsing for scene understanding.** ) , and image captioning ( **1. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ｜ **2. Git: A generative image-to-text transformer for vision and language, 2022.** ) with task-specific design. | ✅ 传统模型通过针对任务的特定设计，在对象检测 ( **1. Mask r-cnn.** ｜ **2. Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** )、语义分割 ( **1. Masked-attention mask transformer for universal image segmentation.** ｜ **2. Unified perceptual parsing for scene understanding.** ) 和图像字幕 ( **1. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ｜ **2. Git: A generative image-to-text transformer for vision and language, 2022.** ) 等任务上表现出色。 |
| ✅ However, it is essential to develop a comprehensive, unified model that is capable of adapting across various vision tasks in a task-agnostic manner, even accommodating new tasks with minimal or no task-specific fine-tuning. | ✅ 然而，必须开发一个全面、统一的模型，该模型能够以与任务无关的方式适应各种视觉任务，甚至可以通过最少或没有特定于任务的微调来适应新任务。 |

| 【第1节，第7段】原文 | 【第1节，第7段】翻译 |
| ---- | ---- |
| ✅ The model Florence   ( **Florence: A new foundation model for computer vision.** ) pioneers the integration of spatial, temporal, and multi-modal aspects in computer vision through unified pre-training and network architecture. | ✅ 模型 Florence   ( **Florence: A new foundation model for computer vision.** ) 通过统一的预训练和网络架构，率先实现了计算机视觉中空间、时间和多模态方面的整合。 |
| ✅ The first evolutionary version ( **Florence: A new foundation model for computer vision.** ) excels in transfer learning via pre-training with noisy text-image pairs and task-specific fine-tuning using specialized adapters. | ✅ 第一个进化版本 ( **Florence: A new foundation model for computer vision.** ) 通过使用嘈杂的文本-图像对进行预训练以及使用专门的适配器进行特定任务的微调，在迁移学习方面表现出色。 |
| ✅ However, it relies on large task-specific datasets and adapters, leaving gaps in addressing the above dual key challenges. | ✅ 然而，它依赖于大型特定任务的数据集和适配器，在解决上述双键挑战方面存在差距。 |

| 【第1节，第8段】原文 | 【第1节，第8段】翻译 |
| ---- | ---- |
| ✅ In this paper, we introduce Florence-2 , a universal backbone achieved through multitask learning with extensive visual annotations. | ✅ 在本文中，我们介绍了 Florence-2，一种通过多任务学习和大量视觉注释实现的通用主干。 |
| ✅ This results in a unified, prompt-based representation for diverse vision tasks, effectively addressing the challenges of limited comprehensive data and the absence of a unified architecture. | ✅ 这使得针对不同视觉任务的表示具有统一性、基于提示性，从而有效地解决了综合数据有限和缺乏统一架构的挑战。 |

| 【第1节，第9段】原文 | 【第1节，第9段】翻译 |
| ---- | ---- |
| ✅ Multitask learning necessitates large-scale, high-quality annotated data. | ✅ 多任务学习需要大规模、高质量的注释数据。 |
| ✅ Our data engine, instead of relying on labor-intensive manual annotation, autonomously generates a comprehensive visual dataset called FLD-5B , encompassing a total of 5.4B annotations for 126M images. | ✅ 我们的数据引擎不再依赖劳动密集型的人工注释，而是自主生成一个名为 FLD-5B 的综合视觉数据集，该数据集包含 126M 张图像的总共 54 亿条注释。 |
| ✅ This engine consists of two efficient processing modules. | ✅ 该引擎由两个高效的处理模块组成。 |
| ✅ The first module uses specialized models to collaboratively and autonomously annotate images, moving away from the traditional single and manual annotation approach. | ✅ 第一个模块使用专门的模型来协作和自主地注释图像，摆脱传统的单一和手动注释方法。 |
| ✅ Multiple models work together to reach a consensus, reminiscent of the wisdom of crowds concept ( **1. The wisdom of the crowd in combinatorial problems.** ｜ **2. Wisdom of the crowd.** ｜ **3. Power of the few vs. wisdom of the crowd: Wikipedia and the rise of the bourgeoisie.** ) , ensuring a more reliable and unbiased image understanding. | ✅ 多个模型共同努力达成共识，让人联想到群体智慧概念( **1. The wisdom of the crowd in combinatorial problems.** ｜ **2. Wisdom of the crowd.** ｜ **3. Power of the few vs. wisdom of the crowd: Wikipedia and the rise of the bourgeoisie.** )，确保更可靠、更公正的图像理解。 |
| ✅ The second module iteratively refines and filters these automated annotations using well-trained foundational models. | ✅ 第二个模块使用训练有素的基础模型迭代地细化和过滤这些自动注释。 |

| 【第1节，第10段】原文 | 【第1节，第10段】翻译 |
| ---- | ---- |
| ✅ By utilizing this extensive dataset, our model employs a sequence-to-sequence (seq2seq) architecture ( **1. Sequence to sequence learning with neural networks.** ｜ **2. Learning phrase representations using rnn encoder-decoder for statistical machine translation.** ｜ **3. Exploring the limits of transfer learning with a unified text-to-text transformer.** ｜ **4. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.** ) , which integrates an image encoder and a multi-modality encoder-decoder. | ✅ 通过利用这个广泛的数据集，我们的模型采用了序列到序列（seq2seq）架构 ( **1. Sequence to sequence learning with neural networks.** ｜ **2. Learning phrase representations using rnn encoder-decoder for statistical machine translation.** ｜ **3. Exploring the limits of transfer learning with a unified text-to-text transformer.** ｜ **4. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.** )，它集成了图像编码器和多模态编码器-解码器。 |
| ✅ This design accommodates a spectrum of vision tasks without the need for task-specific architectural modifications, aligning with the ethos of the NLP community for versatile model development with a consistent underlying structure. | ✅ 该设计适用于一系列视觉任务，而无需针对特定任务进行架构修改，这符合 NLP 社区关于具有一致底层结构的多功能模型开发的精神。 |
| ✅ All annotations in the dataset FLD-5B , are uniformly standardized into textual outputs, facilitating a unified multi-task learning approach with consistent optimization with the same loss function as the objective. | ✅ 数据集 FLD-5B 中的所有注释都被统一标准化为文本输出，从而促进了统一的多任务学习方法，并以相同的损失函数作为目标进行一致的优化。 |
| ✅ The outcome is a versatile vision foundation model, Florence-2 , capable of performing a variety of tasks, such as object detection, captioning, and grounding, all within a single model governed by a uniform set of parameters. | ✅ 最终成果是一个多功能视觉基础模型 Florence-2，它能够执行各种任务，例如对象检测、字幕和基础，所有这些都在由统一的一组参数控制的单个模型中完成。 |
| ✅ Task activation is achieved through textual prompts, reflecting the approach used by Large Language Models (LLMs) ( **Language models are unsupervised multitask learners.** ) . | ✅ 任务激活是通过文本提示实现的，反映了大型语言模型 (LLM) ( **Language models are unsupervised multitask learners.** ) 使用的方法。 |

| 【第1节，第11段】原文 | 【第1节，第11段】翻译 |
| ---- | ---- |
| ✅ Our approach attains a universal representation, demonstrating broad applicability across various visual tasks. | ✅ 我们的方法实现了通用表示，展示了在各种视觉任务中的广泛适用性。 |
| ✅ Key results include: | ✅ 主要成果包括： |

| 【第1节，第12段】原文 | 【第1节，第12段】翻译 |
| ---- | ---- |
| ✅ As a versatile vision foundation model, Florence-2 achieves new state-of-the-art zero-shot performance in tasks such as captioning on COCO ( **Microsoft coco: Common objects in context.** ) , visual grounding on Flick30k ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** ) , and referring expression comprehension on RefCOCO/+/g ( **1. Referitgame: Referring to objects in photographs of natural scenes.** ｜ **2. Modeling context in referring expressions.** ｜ **3. Generation and comprehension of unambiguous object descriptions.** ) . | ✅ 作为一个多功能视觉基础模型，Florence-2 在 COCO ( **Microsoft coco: Common objects in context.** ) 上的字幕、Flick30k ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** ) 上的视觉基础、以及 RefCOCO/+/g ( **1. Referitgame: Referring to objects in photographs of natural scenes.** ｜ **2. Modeling context in referring expressions.** ｜ **3. Generation and comprehension of unambiguous object descriptions.** ) 上的指称表达理解等任务中实现了新的最先进的零样本性能。 |

| 【第1节，第13段】原文 | 【第1节，第13段】翻译 |
| ---- | ---- |
| ✅ After fine-tuning with public human-annotated data, Florence-2 , despite its compact size, competes with larger specialist models. | ✅ 在使用公共的人工注释数据进行微调后，Florence-2 尽管体积小，但仍能与更大的专业模型相媲美。 |
| ✅ Notably, the fine-tuned Florence-2 establishes new state-of-the-art results on the benchmarks on RefCOCO/+/g. | ✅ 值得注意的是，经过微调的 Florence-2 在 RefCOCO/+/g 的基准上建立了新的最先进结果。 |

| 【第1节，第14段】原文 | 【第1节，第14段】翻译 |
| ---- | ---- |
| ✅ The pre-trained Florence-2 backbone enhances performance on downstream tasks, e.g. | ✅ 预先训练的 Florence-2 主干增强了下游任务 e.g 的性能。 |
| ✅  COCO object detection and instance segmentation, and ADE20K semantic segmentation, surpassing both supervised and self-supervised models. | ✅  COCO 对象检测和实例分割，以及 ADE20K 语义分割，超越了监督和自监督模型。 |
| ✅ Compared to pre-trained models on ImageNet, ours improves training efficiency by 4  $\times$  and achieves substantial improvements of 6.9, 5.5, and 5.9 points on COCO ( **Microsoft coco: Common objects in context.** ) and ADE20K ( **Scene parsing through ade20k dataset.** ) datasets, using Mask-RCNN ( **Mask r-cnn.** ) , DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) , and UperNet ( **Unified perceptual parsing for scene understanding.** ) frameworks respectively. | ✅ 与 ImageNet 上的预训练模型相比，我们的模型提高了 4  $\times$  的训练效率，并在 COCO ( **Microsoft coco: Common objects in context.** ) 和 ADE20K ( **Scene parsing through ade20k dataset.** ) 数据集上分别使用 Mask-RCNN ( **Mask r-cnn.** )、DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) 和 UperNet ( **Unified perceptual parsing for scene understanding.** ) 框架取得了 6.9、5.5 和 5.9 点的大幅提升。 |

## 2 Rethinking Vision Model Pre-training

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x2.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 2:  Florence-2  consists of an image encoder and standard multi-modality encoder-decoder. We train Florence-2 on our FLD-5B data in a unified multitask learning paradigm, resulting in a generaslist vision foundation model, which can perform various vision tasks. | ✅ Figure 2:  Florence-2  consists of an image encoder and standard multi-modality encoder-decoder. We train Florence-2 on our FLD-5B data in a unified multitask learning paradigm, resulting in a generaslist vision foundation model, which can perform various vision tasks. |

| 【第2节，第1段】原文 | 【第2节，第1段】翻译 |
| ---- | ---- |
| ✅ In pursuit of a versatile vision foundation model, we revisit three predominant pre-training paradigms: supervised ( e.g. | ✅ 为了追求多功能的视觉基础模型，我们重新审视了三种主要的预训练范式：监督（e.g）。 |
| ✅  , ImageNet classification ( **Imagenet: A large-scale hierarchical image database.** ) ), self-supervised ( e.g. | ✅ 、ImageNet 分类 ( **Imagenet: A large-scale hierarchical image database.** )）、自监督（e.g）。 |
| ✅  , SimCLR ( **A simple framework for contrastive learning of visual representations.** ) , MoCo ( **Momentum contrast for unsupervised visual representation learning.** ) , BEiT ( **BEiT: BERT pre-training of image transformers.** ) , MAE ( **Masked autoencoders are scalable vision learners.** ) ), and weakly supervised ( e.g. | ✅ 、SimCLR ( **A simple framework for contrastive learning of visual representations.** )、MoCo ( **Momentum contrast for unsupervised visual representation learning.** )、BEiT ( **BEiT: BERT pre-training of image transformers.** )、MAE ( **Masked autoencoders are scalable vision learners.** )）和弱监督（e.g）。 |
| ✅  , CLIP ( **Learning transferable visual models from natural language supervision.** ) , Florence ( **Florence: A new foundation model for computer vision.** ) , SAM ( **Segment anything.** ) ). | ✅ 、CLIP ( **Learning transferable visual models from natural language supervision.** )、佛罗伦萨 ( **Florence: A new foundation model for computer vision.** )、SAM ( **Segment anything.** )）。 |
| ✅ Each paradigm captures unique aspects of visual data but is inherently limited by the constraints of single-task learning frameworks. | ✅ 每个范式都捕捉视觉数据的独特方面，但本质上受到单任务学习框架的限制。 |
| ✅ Supervised pre-training excels in object recognition but lacks adaptability ( **Imagenet classification with deep convolutional neural networks.** ) ; self-supervised algorithms reveal intricate features but may overemphasize certain attributes ( **Unsupervised learning of visual features by contrasting cluster assignments.** ) ; weakly supervised methods leverage unstructured textual annotations but yield only image-level understanding ( **Learning transferable visual models from natural language supervision.** ). | ✅ 监督预训练在物体识别方面表现出色，但缺乏适应性 ( **Imagenet classification with deep convolutional neural networks.** )；自监督算法揭示复杂的特征，但可能会过分强调某些属性 ( **Unsupervised learning of visual features by contrasting cluster assignments.** )；弱监督方法利用非结构化文本注释，但只能产生图像级别的理解 ( **Learning transferable visual models from natural language supervision.** )。 |
| ✅ To build a unified vision foundation model suitable for various applications, we must explore innovative pre-training strategies that overcome single-task limitations and integrate both textual and visual semantics. | ✅ 为了构建适用于各种应用的统一视觉基础模型，我们必须探索创新的预训练策略，以克服单任务限制并整合文本和视觉语义。 |

| 【第2节，第2段】原文 | 【第2节，第2段】翻译 |
| ---- | ---- |
| ✅ Image understanding necessitates capturing multiple levels of granularity, from global semantics to local details, and comprehending spatial relationships between objects and entities in their semantic context. | ✅ 图像理解需要捕捉多层次的粒度，从全局语义到局部细节，并理解语义环境中对象和实体之间的空间关系。 |
| ✅ To address these core aspects of image understanding, our approach incorporates a diverse set of annotations, effectively capturing visual understanding nuances and bridging the gap between vision and language understanding. | ✅ 为了解决图像理解的这些核心方面，我们的方法结合了多种注释，有效地捕捉视觉理解的细微差别并弥合视觉和语言理解之间的差距。 |

### 2.1 Comprehensive Multitask Learning

| 【第2.1节，第1段】原文 | 【第2.1节，第1段】翻译 |
| ---- | ---- |
| ✅ To develop a versatile vision foundation model, we formulate a range of multitask learning objectives, each tailored to address specific aspects of visual comprehension. | ✅ 为了开发多功能的视觉基础模型，我们制定了一系列多任务学习目标，每个目标都针对视觉理解的特定方面进行定制。 |
| ✅ These objectives align with our predefined criteria: spatial hierarchy and semantic granularity, inspired by recent research on multitask learning ( **1. Flamingo: a visual language model for few-shot learning.** ｜ **2. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** ｜ **3. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **4. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **5. Pali-x: On scaling up a multilingual vision and language model.** ｜ **6. Pali-3 vision language models: Smaller, faster, stronger, 2023.** ). | ✅ 这些目标与我们预定义的标准一致：空间层次和语义粒度，受到最近对多任务学习 ( **1. Flamingo: a visual language model for few-shot learning.** ｜ **2. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** ｜ **3. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **4. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **5. Pali-x: On scaling up a multilingual vision and language model.** ｜ **6. Pali-3 vision language models: Smaller, faster, stronger, 2023.** ) 的研究的启发。 |
| ✅ Our multitask learning approach incorporates three distinct learning objectives, each addressing a different level of granularity and semantic understanding: | ✅ 我们的多任务学习方法包含三个不同的学习目标，每个目标针对不同级别的粒度和语义理解： |

| 【第2.1节，第2段】原文 | 【第2.1节，第2段】翻译 |
| ---- | ---- |
| ✅ Image-level understanding tasks capture high-level semantics and foster a comprehensive understanding of images through linguistic descriptions ( **1. Microsoft coco captions: Data collection and evaluation server.** ｜ **2. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions.** ｜ **3. Imagenet: A large-scale hierarchical image database.** ｜ **4. A hierarchical approach for generating descriptive image paragraphs.** ). | ✅ Image-level understanding 任务捕获高级语义并通过语言描述 ( **1. Microsoft coco captions: Data collection and evaluation server.** ｜ **2. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions.** ｜ **3. Imagenet: A large-scale hierarchical image database.** ｜ **4. A hierarchical approach for generating descriptive image paragraphs.** ) 促进对图像的全面理解。 |
| ✅ They enable the model to comprehend the overall context of an image and grasp semantic relationships and contextual nuances in the language domain. | ✅ 它们使模型能够理解图像的整体背景并掌握语言领域中的语义关系和上下文细微差别。 |
| ✅ Exemplar tasks include image classification, captioning, and visual question answering. | ✅ 示例任务包括图像分类、字幕和视觉问答。 |

| 【第2.1节，第3段】原文 | 【第2.1节，第3段】翻译 |
| ---- | ---- |
| ✅ Region/pixel-level recognition tasks facilitate detailed object and entity localization within images, capturing relationships between objects and their spatial context. | ✅ Region/pixel-level recognition 任务有助于在图像中实现详细的对象和实体定位，捕捉对象与其空间环境之间的关系。 |
| ✅ Tasks include object detection, segmentation, and referring expression comprehension. | ✅ 任务包括对象检测、分割和指称表达理解。 |

| 【第2.1节，第4段】原文 | 【第2.1节，第4段】翻译 |
| ---- | ---- |
| ✅ Fine-grained visual-semantic alignment tasks require fine-grained understanding of both text and image. | ✅ Fine-grained visual-semantic alignment 任务需要对文本和图像进行细粒度的理解。 |
| ✅ It involves locating the image regions that correspond to the text phrases, such as objects, attributes, or relations. | ✅ 它涉及定位与文本短语（例如对象、属性或关系）相对应的图像区域。 |
| ✅ These tasks challenge the ability to capture the local details of visual entities and their semantic contexts, as well as the interactions between textual and visual elements. | ✅ 这些任务挑战了捕捉视觉实体的局部细节及其语义背景以及文本和视觉元素之间交互的能力。 |

| 【第2.1节，第5段】原文 | 【第2.1节，第5段】翻译 |
| ---- | ---- |
| ✅ By combining these three learning objectives in a multitask learning framework, our foundation model learns to handle different levels of detail and semantic understanding. | ✅ 通过在多任务学习框架中结合这三个学习目标，我们的基础模型可以学习处理不同级别的细节和语义理解。 |
| ✅ This strategic alignment enables our model to deal with various spatial details, distinguish levels of detail in understanding, and go beyond surface-level recognition—ultimately learning a universal representation for vision understanding. | ✅ 这种战略协调使我们的模型能够处理各种空间细节，区分理解中的细节层次，并超越表面层次的识别——最终学习视觉理解的通用表示。 |

## 3 Model

| 【第3节，第1段】原文 | 【第3节，第1段】翻译 |
| ---- | ---- |
| ✅ We present the foundation model Florence-2 , designed for universal representation learning, capable of handling various vision tasks with a single set of weights and a unified architecture. | ✅ 我们提出了基础模型 Florence-2，专为通用表示学习而设计，能够使用一组权重和统一的架构处理各种视觉任务。 |
| ✅ As depicted in Figure 2 , Florence-2 employs a sequence-to-sequence learning paradigm ( **Attention is all you need.** ) , integrating all tasks, described in Section 2 , under a common language modeling objective. | ✅ 如 Figure 2 所示，Florence-2 采用序列到序列学习范式 ( **Attention is all you need.** )，将 Section 2 中描述的所有任务整合在一个共同的语言建模目标之下。 |
| ✅ The model takes images coupled with task-prompt as task instructions, and generates the desirable results in text forms. | ✅ 该模型以图像加上任务提示作为任务指令，以文本形式生成所需的结果。 |
| ✅ It uses a vision encoder to convert images into visual token embeddings, which are then concatenated with text embeddings and processed by a transformer-based multi-modal encoder-decoder to generate the response. | ✅ 它使用视觉编码器将图像转换为视觉标记嵌入，然后将其与文本嵌入连接，并由基于变压器的多模态编码器解码器处理以生成响应。 |
| ✅ In the following sections, we will provide a detailed explanation of each model component. | ✅ 在接下来的章节中，我们将对每个模型组件进行详细的解释。 |

#### 3.1 Task formulation.

| 【第3.1节，第1段】原文 | 【第3.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We adopt a sequence-to-sequence framework ( **1. Attention is all you need.** ｜ **2. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **3. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **4. Pix2seq: A language modeling framework for object detection, 2022.** ) to address various vision tasks in a unified manner. | ✅ 我们采用序列到序列框架( **1. Attention is all you need.** ｜ **2. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **3. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **4. Pix2seq: A language modeling framework for object detection, 2022.** )来统一处理各种视觉任务。 |
| ✅ As shown in Table 13 , we formulate each task as a translation problem: Given an input image and a task-specific prompt, we generate the corresponding output response. | ✅ 如 Table 13 所示，我们将每个任务制定为一个翻译问题：给定一个输入图像和一个特定于任务的提示，我们生成相应的输出响应。 |
| ✅ Depending on the task, the prompt and response can be either text or region: | ✅ 根据任务，提示和响应可以是文本或区域： |

| 【第3.1节，第2段】原文 | 【第3.1节，第2段】翻译 |
| ---- | ---- |
| ✅ Text : When the prompt or answer is plain text without special formatting, we maintain it in our final sequence-to-sequence format. | ✅ Text：当提示或答案是没有特殊格式的纯文本时，我们会将其保留在最终的序列到序列格式中。 |

| 【第3.1节，第3段】原文 | 【第3.1节，第3段】翻译 |
| ---- | ---- |
| ✅ Region : For region-specific tasks, we add location tokens to the tokenizer’s vocabulary list, representing quantized coordinates. | ✅ Region：对于特定区域的任务，我们将位置标记添加到标记器的词汇列表中，表示量化坐标。 |
| ✅ We create  $1,000$  bins, similar to ( **1. Pix2seq: A language modeling framework for object detection, 2022.** ｜ **2. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **3. A unified sequence interface for vision tasks.** ｜ **4. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** ) , and represent regions using formats tailored to task requirements: | ✅ 我们创建类似于 ( **1. Pix2seq: A language modeling framework for object detection, 2022.** ｜ **2. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **3. A unified sequence interface for vision tasks.** ｜ **4. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** ) 的  $1,000$  箱，并使用根据任务要求定制的格式来表示区域： | | 【第3.1节，第4段】原文 | 【第3.1节，第4段】翻译 |
| ---- | ---- |
| ✅ Box representation (x0,y0,x1,y1)subscript𝑥0subscript𝑦0subscript𝑥1subscript𝑦1(x_{0},y_{0},x_{1},y_{1}) : Utilized in tasks such as object detection and dense region captioning, with location tokens corresponding to the box coordinates. | ✅ Box representation (x0,y0,x1,y1)subscript𝑥0subscript𝑦0subscript𝑥1subscript𝑦1(x_{0},y_{0},x_{1},y_{1})：用于对象检测和密集区域字幕等任务，位置标记与框坐标相对应。 |
| ✅ The location tokens are the coordinates of the top-left and bottom-right corners of the box. | ✅ 位置标记是框左上角和右下角的坐标。 | | 【第3.1节，第5段】原文 | 【第3.1节，第5段】翻译 |
| ---- | ---- |
| ✅ Quad box representation (x0,y0,…,x3,y3)subscript𝑥0subscript𝑦0…subscript𝑥3subscript𝑦3(x_{0},y_{0},...,x_{3},y_{3}) : For text detection and recognition tasks, using location tokens for each coordinate of the quadrilateral enclosing the text. | ✅ Quad box representation (x0,y0,…,x3,y3)subscript𝑥0subscript𝑦0…subscript𝑥3subscript𝑦3(x_{0},y_{0},...,x_{3},y_{3})：对于文本检测和识别任务，使用位置标记来表示包围文本的四边形的每个坐标。 |
| ✅ The location tokens are the coordinates of each corner of the quad box, starting from the top-left and going clockwise. | ✅ 位置标记是四边形框每个角的坐标，从左上角开始顺时针旋转。 | | 【第3.1节，第6段】原文 | 【第3.1节，第6段】翻译 |
| ---- | ---- |
| ✅ Polygon Representation (x0,y0,…,xn,yn)subscript𝑥0subscript𝑦0…subscript𝑥𝑛subscript𝑦𝑛(x_{0},y_{0},...,x_{n},y_{n}) : For referring segmentation tasks, with location tokens representing the vertices of the polygon. | ✅ Polygon Representation (x0,y0,…,xn,yn)subscript𝑥0subscript𝑦0…subscript𝑥𝑛subscript𝑦𝑛(x_{0},y_{0},...,x_{n},y_{n})：用于引用分割任务，其中位置标记代表多边形的顶点。 |
| ✅ The location tokens are the coordinates of the vertices of the polygon, in clockwise order. | ✅ 位置标记是多边形顶点的坐标，按顺时针顺序排列。 |

| 【第3.1节，第7段】原文 | 【第3.1节，第7段】翻译 |
| ---- | ---- |
| ✅ By extending the tokenizer’s vocabulary to include location tokens, we enable the model to process region-specific information in a unified learning format. | ✅ 通过扩展标记器的词汇表以包含位置标记，我们使模型能够以统一的学习格式处理特定于区域的信息。 |
| ✅ This eliminates the need to design task-specific heads for different tasks and allows for a more data-centric approach. | ✅ 这样就无需为不同任务设计特定任务的头部，而可以采用更加以数据为中心的方法。 |

#### 3.2 Vision encoder.

| 【第3.2节，第1段】原文 | 【第3.2节，第1段】翻译 |
| ---- | ---- |
| ✅ We employ DaViT ( **Davit: Dual attention vision transformers.** ) as the vision encoder. | ✅ 我们采用 DaViT ( **Davit: Dual attention vision transformers.** ) 作为视觉编码器。 |
| ✅ It processes an input image  $\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$  (with  $H$  and  $W$  denoting height and width, respectively) into flattened visual token embeddings  $\mathbf{V}\in\mathbb{R}^{N_{v}\times D_{v}}$  , where  $N_{v}$  and  $D_{v}$  represent the number and dimensionality of vision tokens, respectively. | ✅ 它将输入图像  $\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$ （ $H$  和  $W$  分别表示高度和宽度）处理成扁平的视觉标记嵌入  $\mathbf{V}\in\mathbb{R}^{N_{v}\times D_{v}}$ ，其中  $N_{v}$  和  $D_{v}$  分别表示视觉标记的数量和维数。 |

#### 3.3 Multi-modality encoder decoder.

| 【第3.3节，第1段】原文 | 【第3.3节，第1段】翻译 |
| ---- | ---- |
| ✅ We use a standard encoder-decoder transformer architecture to process visual and language token embeddings. | ✅ 我们使用标准的编码器-解码器转换器架构来处理视觉和语言标记嵌入。 |
| ✅ We first obtain prompt text embeddings  $\mathbf{T}_{prompt}\in\mathbf{R}^{N_{t}\times D}$  using our extended language tokenizer and word embedding layer ( **Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.** ). | ✅ 我们首先使用扩展的语言标记器和词嵌入层 ( **Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.** ) 获得提示文本嵌入  $\mathbf{T}_{prompt}\in\mathbf{R}^{N_{t}\times D}$ 。 |
| ✅ Then, we concatenate vision token embeddings with prompt embeddings to form the multi-modality encoder module input,  $\mathbf{X}=[\mathbf{V}^{\prime},\mathbf{T}_{prompt}]$  , where  $\mathbf{V}^{\prime}\in\mathbb{R}^{N_{v}\times D}$  is obtained by applying a linear projection and LayerNorm layer ( **Layer normalization, 2016.** ) to  $\mathbf{V}$  for dimensionality alignment. | ✅ 然后，我们将视觉标记嵌入与提示嵌入连接起来以形成多模态编码器模块输入  $\mathbf{X}=[\mathbf{V}^{\prime},\mathbf{T}_{prompt}]$ ，其中  $\mathbf{V}^{\prime}\in\mathbb{R}^{N_{v}\times D}$  是通过应用线性投影和 LayerNorm 层 ( **Layer normalization, 2016.** ) 到  $\mathbf{V}$  进行维度对齐获得的。 |

#### 3.4 Optimization objective.

| 【第3.4节，第1段】原文 | 【第3.4节，第1段】翻译 |
| ---- | ---- |
| ✅ Given the input  $x$  combined from the image and the prompt, and the target  $y$  , we use the standard language modeling with cross-entropy loss for all the tasks. | ✅ 给定由图像和提示组合而成的输入  $x$  以及目标  $y$ ，我们对所有任务使用具有交叉熵损失的标准语言建模。 |

**公式(1):** 
$$ \mathcal{L}=-\sum_{i=1}^{|y|}logP_{\theta}(y_{i}|y_{<i},x) $$

| 【第3.4节，第2段】原文 | 【第3.4节，第2段】翻译 |
| ---- | ---- |
| ✅ where  $\theta$  are the network parameters,  $ \vert y \vert $  is the number of target tokens. | ✅ 其中 $\theta$ 是网络参数， $ \vert y \vert $ 是目标令牌的数量。 |

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x3.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 3:  Florence-2  data engine consists of three essential phrases: (1) initial annotation employing specialist models, (2) data filtering to correct errors and remove irrelevant annotations, and (3) an iterative process for data refinement. Our final dataset (FLD-5B) of over 5B annotations contains 126M images, 500M text annotations, 1.3B region-text annotations, and 3.6B text-phrase-region annotations. | ✅ Figure 3:  Florence-2  data engine consists of three essential phrases: (1) initial annotation employing specialist models, (2) data filtering to correct errors and remove irrelevant annotations, and (3) an iterative process for data refinement. Our final dataset (FLD-5B) of over 5B annotations contains 126M images, 500M text annotations, 1.3B region-text annotations, and 3.6B text-phrase-region annotations. |

## 4 Data Engine

| 【第4节，第1段】原文 | 【第4节，第1段】翻译 |
| ---- | ---- |
| ✅ To train our Florence-2 model, we require a comprehensive, large-scale, high-quality multitask dataset encompassing various image data aspects. | ✅ 为了训练我们的 Florence-2 模型，我们需要一个涵盖各种图像数据方面的全面、大规模、高质量的多任务数据集。 |
| ✅ Given the scarcity of such data, we have developed a new multitask image dataset. | ✅ 鉴于此类数据的稀缺性，我们开发了一个新的多任务图像数据集。 |
| ✅ This dataset FLD-5B includes 126M images, 500M text annotations, and 1.3B text-region annotations, and 3.6B text-phrase-region annotations across different tasks. | ✅ 该数据集FLD-5B包括跨不同任务的126M图像、500M文本注释、1.3B文本区域注释和3.6B文本短语区域注释。 |
| ✅ We extensively explain our data collection and annotation procedures, encompassing adaptations for various annotation types. | ✅ 我们广泛解释了我们的数据收集和注释程序，涵盖了对各种注释类型的适应性。 |
| ✅ The data engine pipeline, shown in Figure 3 , will be discussed in subsequent sections. | ✅ Figure 3 中所示的数据引擎管道将在后续章节中讨论。 |

### 4.1 Image Collection

| 【第4.1节，第1段】原文 | 【第4.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We construct our data by gathering a diverse collection of images from various sources. | ✅ 我们通过收集来自各种来源的多样化图像来构建数据。 |
| ✅ We begin with the identification of three key tasks that act as primary sources for our image corpus: image classification, object detection, and image captioning. | ✅ 我们首先确定作为图像语料库主要来源的三个关键任务：图像分类、对象检测和图像字幕。 |
| ✅ Consequently, we curate and combine five distinct datasets originating from the aforementioned tasks: ImageNet-22k ( **Imagenet: A large-scale hierarchical image database.** ) , Object 365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ) , Open Images ( **The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.** ) , Conceptual Captions ( **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.** ) , and LAION ( **Laion-400m: Open dataset of clip-filtered 400 million image-text pairs.** ) filtered by ( **Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ). | ✅ 因此，我们从上述任务中整理并组合了五个不同的数据集：ImageNet-22k ( **Imagenet: A large-scale hierarchical image database.** )、Object 365 ( **Objects365: A large-scale, high-quality dataset for object detection.** )、Open Images ( **The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.** )、Conceptual Captions ( **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.** ) 和通过 ( **Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ) 过滤的 LAION ( **Laion-400m: Open dataset of clip-filtered 400 million image-text pairs.** )。 |
| ✅ This combination results in a dataset of 126 million images in total. | ✅ 这种组合产生了总计 1.26 亿张图像的数据集。 |

### 4.2 Data Annotation

| 【第4.2节，第1段】原文 | 【第4.2节，第1段】翻译 |
| ---- | ---- |
| ✅ Our primary objective is to generate comprehensive annotations that can support multitask learning effectively. | ✅ 我们的主要目标是生成能够有效支持多任务学习的综合注释。 |
| ✅ Accordingly, our annotation endeavors span a comprehensive range of tasks, encapsulated within three discrete annotation categories: text , region-text pairs, and text-phrase-region triplets, which is illustrated in Figure 4. | ✅ 因此，我们的注释工作涵盖了广泛的任务，封装在三个离散的注释类别中：text、region-text 对和 text-phrase-region 三元组，如 Figure 4 中所示。 |
| ✅ The data annotation workflow consists of three essential phases, each of which ensures the accuracy and quality of the annotations: (1) initial annotation employing specialist models, (2) data filtering to correct errors and remove irrelevant annotations, and (3) an iterative process for data refinement. | ✅ 数据注释工作流程包括三个基本阶段，每个阶段都确保注释的准确性和质量：（1）采用专家模型进行初始注释，（2）数据过滤以纠正错误并删除不相关的注释，以及（3）数据细化的迭代过程。 |

#### 4.2.1 Initial annotation with specialist models.

| 【第4.2.1节，第1段】原文 | 【第4.2.1节，第1段】翻译 |
| ---- | ---- |
| ✅ To initiate the annotation process for each annotation type, we employ synthetic labels obtained from specialist models. | ✅ 为了启动每种注释类型的注释过程，我们采用从专家模型获得的合成标签。 |
| ✅ These specialist models are a combination of offline models trained on a diverse range of publicly available datasets and online services hosted on cloud platforms. | ✅ 这些专业模型是在各种公开数据集上训练的离线模型和托管在云平台上的在线服务的组合。 |
| ✅ They are specifically tailored to excel in annotating their respective annotation types. | ✅ 它们经过专门定制，能够出色地注释各自的注释类型。 |

| 【第4.2.1节，第2段】原文 | 【第4.2.1节，第2段】翻译 |
| ---- | ---- |
| ✅ It is worth noting that certain image datasets may already contain partial annotations for some annotation types. | ✅ 值得注意的是，某些图像数据集可能已经包含某些注释类型的部分注释。 |
| ✅ For instance, the Object 365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ) dataset already includes human-annotated bounding boxes and corresponding categories as region-text annotations. | ✅ 例如，Object 365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ) 数据集已经包含人工注释的边界框和相应的类别作为区域文本注释。 |
| ✅ In such cases, we merge the pre-existing annotations with the synthetic labels generated by the specialist models. | ✅ 在这种情况下，我们将预先存在的注释与专家模型生成的合成标签合并。 |
| ✅ This approach enhances the coverage and diversity of the annotations. | ✅ 这种方法增强了注释的覆盖率和多样性。 |

| 【第4.2.1节，第3段】原文 | 【第4.2.1节，第3段】翻译 |
| ---- | ---- |
| ✅ Moreover, specific annotations, such as detailed descriptions in the text annotation type, are represented by datasets of a considerably small size. | ✅ 此外，特定注释（例如文本注释类型中的详细描述）由相当小的数据集表示。 |
| ✅ This inherently poses challenges in obtaining high-performance specialist models. | ✅ 这本质上对获得高性能专家模型带来了挑战。 |
| ✅ Consequently, we opt to omit these tasks during the initial annotation phase. | ✅ 因此，我们选择在初始注释阶段省略这些任务。 |
| ✅ Annotations for these tasks are generated later during the iterative data refinement process. | ✅ 这些任务的注释稍后在迭代数据细化过程中生成。 |

| 【第4.2.1节，第4段】原文 | 【第4.2.1节，第4段】翻译 |
| ---- | ---- |
| ✅ In summation, through the rigorous initial annotation procedures, we ensure that the aggregated dataset of 126 million images is comprehensively labeled across the majority of annotation types. | ✅ 总而言之，通过严格的初始注释程序，我们确保 1.26 亿张图像的聚合数据集在大多数注释类型中得到全面标记。 |

#### 4.2.2 Data filtering and enhancement.

| 【第4.2.2节，第1段】原文 | 【第4.2.2节，第1段】翻译 |
| ---- | ---- |
| ✅ The initial annotations obtained from the specialist models, while comprehensive, are susceptible to noise and imprecision. | ✅ 从专家模型获得的初始注释虽然全面，但容易受到噪音和不精确的影响。 |
| ✅ In response to this challenge, we have implemented a multifaceted filtering process to refine and eliminate undesired annotations. | ✅ 为了应对这一挑战，我们实施了多方面的过滤过程来改进和消除不需要的注释。 |
| ✅ Our general filtering protocol mainly focuses on two data types in the annotations: text and region data. | ✅ 我们的通用过滤协议主要关注注释中的两种数据类型：文本和区域数据。 |

| 【第4.2.2节，第2段】原文 | 【第4.2.2节，第2段】翻译 |
| ---- | ---- |
| ✅ First, pertaining to textual annotations, we are inspired by DiHT ( **Filtering, distillation, and hard negatives for vision-language pre-training.** ) and develop a parsing tool based on SpaCy ( **spacy: Industrial-strength natural language processing in python.** ) to extract objects, attributes, and actions. | ✅ 首先，关于文本注释，我们受到 DiHT ( **Filtering, distillation, and hard negatives for vision-language pre-training.** ) 的启发，并开发了一个基于 SpaCy ( **spacy: Industrial-strength natural language processing in python.** ) 的解析工具来提取对象、属性和动作。 |
| ✅ We filter out texts containing excessive objects, as they tend to introduce noise and may not accurately reflect the actual content in the corresponding images. | ✅ 我们过滤掉包含过多对象的文本，因为它们往往会引入噪音，并且可能无法准确反映相应图像中的实际内容。 |
| ✅ Additionally, we assess the complexity of the actions and objects by measuring their degree of node in the dependency parsing tree. | ✅ 此外，我们通过测量依赖解析树中的节点度来评估动作和对象的复杂性。 |
| ✅ We retain texts with a certain minimum action and object complexity to ensure the richness of visual concepts in the images. | ✅ 我们保留具有一定最小动作和对象复杂度的文本，以确保图像中视觉概念的丰富性。 |

| 【第4.2.2节，第3段】原文 | 【第4.2.2节，第3段】翻译 |
| ---- | ---- |
| ✅ Second, in relation to the region annotations, specifically bounding boxes, we remove the noisy boxes under a confidence score threshold. | ✅ 其次，针对区域注释，特别是边界框，我们删除了置信度分数阈值以下的噪声框。 |
| ✅ Complementing this, we also employ non-maximum suppression to reduce redundant or overlapping bounding boxes. | ✅ 除此之外，我们还采用非最大抑制来减少冗余或重叠的边界框。 |

#### 4.2.3 Iterative data refinement.

| 【第4.2.3节，第1段】原文 | 【第4.2.3节，第1段】翻译 |
| ---- | ---- |
| ✅ Using our filtered initial annotations, we trained a multitask model that processes sequences of data. | ✅ 使用我们过滤的初始注释，我们训练了一个处理数据序列的多任务模型。 |
| ✅ Upon evaluating this model against our training images, we discerned a marked enhancement in its predictions, particularly in instances where original labels were marred by inaccuracies or extraneous noise, such as in alt-texts. | ✅ 在根据我们的训练图像评估该模型后，我们发现其预测效果有了明显增强，特别是在原始标签因不准确或外部噪音（例如替代文本）而受损的情况下。 |
| ✅ Motivated by these findings, we integrated these updated annotations with our original ones and subjected the model to another training iteration. | ✅ 在这些发现的启发下，我们将这些更新的注释与我们原来的注释相结合，并对模型进行了另一次训练迭代。 |
| ✅ This cyclical refinement process incrementally improves the quality of our training dataset. | ✅ 这个循环的改进过程逐步提高了我们的训练数据集的质量。 |

| 【第4.2.3节，第2段】原文 | 【第4.2.3节，第2段】翻译 |
| ---- | ---- |
| ✅ In the case of tasks we initially bypassed due to insufficient data for the training of a robust specialist model, we leveraged the iteratively trained model for pre-training purposes. | ✅ 对于我们最初由于数据不足以训练强大的专家模型而绕过的任务，我们利用迭代训练的模型进行预训练。 |
| ✅ Subsequent fine-tuning of this pre-trained model with the sparse dataset showcased superior performance compared to a model trained from scratch on the same data. | ✅ 使用稀疏数据集对该预训练模型进行后续微调，与使用相同数据从头开始训练的模型相比，其性能更为出色。 |
| ✅ Thus, we harness the fine-tuned model as a specialist for annotating our expansive dataset comprising 126 million images, ensuring comprehensive annotation coverage. | ✅ 因此，我们利用微调模型作为专家来注释我们包含 1.26 亿张图像的广泛数据集，确保全面的注释覆盖。 |

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x4.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 4:  An illustrative example of an image and its corresponding annotations in FLD-5B dataset. Each image in FLD-5B is annotated with text, region-text pairs, and text-phrase-region triplets by Florence data engine, which covers multiple spatial hierarchies, brief-to-detailed progressive granularity, and a wide semantics spectrum, enabling more comprehensive visual understanding from diverse perspectives. | ✅ Figure 4:  An illustrative example of an image and its corresponding annotations in FLD-5B dataset. Each image in FLD-5B is annotated with text, region-text pairs, and text-phrase-region triplets by Florence data engine, which covers multiple spatial hierarchies, brief-to-detailed progressive granularity, and a wide semantics spectrum, enabling more comprehensive visual understanding from diverse perspectives. |

### 4.3 Annotation-specific Variations

| 【第4.3节，第1段】原文 | 【第4.3节，第1段】翻译 |
| ---- | ---- |
| ✅ In Section 4.2 , we introduce our general annotation workflow. | ✅ 在Section 4.2中，我们介绍了一般的注释工作流程。 |
| ✅ This section delves into each annotation type and the corresponding variations of the annotation procedure. | ✅ 本节深入探讨每种注释类型以及注释过程的相应变化。 |

#### 4.3.1 Text.

| 【第4.3.1节，第1段】原文 | 【第4.3.1节，第1段】翻译 |
| ---- | ---- |
| ✅ Text annotations categorize images using three types of granularities: brief, detailed, and more detailed. | ✅ 文本注释使用三种粒度对图像进行分类：简要、详细和更详细。 |
| ✅ The brief text includes only one sentence that demonstrates the most salient objects and activities, which is similar to COCO caption ( **Microsoft coco captions: Data collection and evaluation server.** ). | ✅ 简短的文本仅包含一句话，展示了最突出的物体和活动，与 COCO 标题 ( **Microsoft coco captions: Data collection and evaluation server.** ) 类似。 |
| ✅ In contrast, the detailed text and more detailed text contain multiple sentences that describe the image with richer objects, attributes, and actions. | ✅ 相比之下，详细文本和更详细的文本包含多个句子，用更丰富的对象、属性和动作来描述图像。 |

| 【第4.3.1节，第2段】原文 | 【第4.3.1节，第2段】翻译 |
| ---- | ---- |
| ✅ For the brief text, a Florence-2 model is trained as the specialist on publicly available image caption and image-text datasets, creating an image-to-text model for initial annotations. | ✅ 对于简短的文本，Florence-2 模型作为公开可用的图像标题和图像文本数据集的专家进行训练，从而创建用于初始注释的图像到文本模型。 |
| ✅ Iterative refinement is used to minimize noise in these texts. | ✅ 使用迭代细化来最小化这些文本中的噪音。 |
| ✅ For the detailed text, prompts including existing image annotations like the brief text and region-text annotations, are fed to large language models (LLMs) or large multimodal models (LMMs) to generate comprehensive descriptions. | ✅ 对于详细文本，包括现有图像注释（如简短文本和区域文本注释）的提示被输入到大型语言模型（LLM）或大型多模态模型（LMM）以生成全面的描述。 |
| ✅ Due to the high cost of the large models, only a small set of detailed text and more detailed text are generated. | ✅ 由于大型模型成本高，因此只能生成一小部分详细文本和更详细的文本。 |
| ✅ These are used to fine-tune the caption specialist, developing a detailed description specialist for further annotations. | ✅ 这些用于微调字幕专家，开发详细的描述专家以供进一步注释。 |

#### 4.3.2 Region-text pairs.

| 【第4.3.2节，第1段】原文 | 【第4.3.2节，第1段】翻译 |
| ---- | ---- |
| ✅ The region-text pairs provide descriptive textual annotation for semantic regions in the image. | ✅ 区域-文本对为图像中的语义区域提供描述性文本注释。 |
| ✅ Semantic regions include regions of visual objects as well as text regions. | ✅ 语义区域包括视觉对象区域以及文本区域。 |
| ✅ The region is represented by a tight bounding box surrounds the region. | ✅ 该区域由围绕该区域的紧密边界框表示。 |
| ✅ Moreover, each region can be annotated with varying degrees of granularity, including phrases and sentences, that contribute to a richer understanding of the region. | ✅ 此外，每个区域都可以用不同程度的粒度进行注释，包括短语和句子，从而有助于更深入地了解该区域。 |

| 【第4.3.2节，第2段】原文 | 【第4.3.2节，第2段】翻译 |
| ---- | ---- |
| ✅ Region-text pairs are annotated differently for text regions and visual object regions. | ✅ 区域-文本对对于文本区域和视觉对象区域的注释不同。 |
| ✅ Text regions are labeled using Azure AI Services’ OCR API ( **https://azure.microsoft.com/en-us/products/ai-services?activetab=pivot:azureopenaiservicetab.** ) , while visual objects are initially annotated with a DINO object detector ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) trained on public datasets. | ✅ 文本区域使用 Azure AI 服务的 OCR API ( **https://azure.microsoft.com/en-us/products/ai-services?activetab=pivot:azureopenaiservicetab.** ) 进行标记，而视觉对象最初使用在公共数据集上训练的 DINO 对象检测器 ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) 进行注释。 |
| ✅ Data filtering, including confidence thresholding and non-maximum suppression, removes noisy boxes. | ✅ 数据过滤（包括置信度阈值和非最大抑制）可以消除噪声框。 |
| ✅ Textual annotations for the visual object regions are further enriched by brief text generated from an image-to-text model with cropped image regions. | ✅ 通过从具有裁剪图像区域的图像到文本模型生成的简短文本，进一步丰富了视觉对象区域的文本注释。 |
| ✅ Each region then receives three textual annotations: phrase from object category, brief text, and noun phrase chunks from the brief text. | ✅ 然后，每个区域会收到三个文本注释：来自对象类别的短语、简短文本和来自简短文本的名词短语块。 |
| ✅ The Florence-1 ( **Florence: A new foundation model for computer vision.** ) model determines the most similar textual annotation to each image region. | ✅ Florence-1 ( **Florence: A new foundation model for computer vision.** ) 模型确定每个图像区域最相似的文本注释。 |

#### 4.3.3 Text-phrase-region triplets.

| 【第4.3.3节，第1段】原文 | 【第4.3.3节，第1段】翻译 |
| ---- | ---- |
| ✅ Text-phrase-region triplets consist of a descriptive text of the image, noun phrases in this text related to image objects, and region annotations for these objects. | ✅ 文本-​​短语-区域三元组由图像的描述性文本、与图像对象相关的文本中的名词短语以及这些对象的区域注释组成。 |
| ✅ The text includes brief, detailed, and more detailed text generated earlier. | ✅ 文本包括先前生成的简短、详细和更详细的文本。 |
| ✅ For each text, the Grounding DINO model ( **Grounding dino: Marrying dino with grounded pre-training for open-set object detection.** ) identifies noun phrases and creates bounding boxes for them. | ✅ 对于每篇文本，Grounding DINO 模型 ( **Grounding dino: Marrying dino with grounded pre-training for open-set object detection.** ) 会识别名词短语并为它们创建边界框。 |
| ✅ Additionally, the SAM model ( **Segment anything.** ) generates segmentation masks for each box, offering more precise object localization. | ✅ 此外，SAM 模型 ( **Segment anything.** ) 为每个框生成分割掩码，提供更精确的对象定位。 |
| ✅ During data filtering, a confidence score threshold is applied to both noun phrases and bounding boxes to ensure relevance. | ✅ 在数据过滤期间，对名词短语和边界框应用置信度分数阈值以确保相关性。 |
| ✅ A blacklist is also used to exclude irrelevant noun phrases like pronouns and abstract concepts. | ✅ 黑名单还用于排除不相关的名词短语，如代词和抽象概念。 |

<table class="ltx_tabular ltx_centering ltx_align_middle" id="S4.T1.2"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S4.T1.2.1.1"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt" id="S4.T1.2.1.1.1" style="padding-left:7.5pt;padding-right:7.5pt;">Dataset</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt" id="S4.T1.2.1.1.2" style="padding-left:7.5pt;padding-right:7.5pt;">Rep. Model</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_tt" id="S4.T1.2.1.1.3" style="padding-left:7.5pt;padding-right:7.5pt;">#Images</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_tt" id="S4.T1.2.1.1.4" style="padding-left:7.5pt;padding-right:7.5pt;">#Annotations</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt" id="S4.T1.2.1.1.5" style="padding-left:7.5pt;padding-right:7.5pt;">Spatial hierarchy</td><td class="ltx_td ltx_align_left ltx_border_tt" id="S4.T1.2.1.1.6" style="padding-left:7.5pt;padding-right:7.5pt;">Semantics granularity</td></tr><tr class="ltx_tr" id="S4.T1.2.2.2"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S4.T1.2.2.2.1" style="padding-left:7.5pt;padding-right:7.5pt;">JFT300M <html><body><p>( <strong>An image is worth 16x16 words: Transformers for image recognition atscale, 2021.</strong> )</p></body></html></td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S4.T1.2.2.2.2" style="padding-left:7.5pt;padding-right:7.5pt;">ViT</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S4.T1.2.2.2.3" style="padding-left:7.5pt;padding-right:7.5pt;">300M</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S4.T1.2.2.2.4" style="padding-left:7.5pt;padding-right:7.5pt;">300M</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S4.T1.2.2.2.5" style="padding-left:7.5pt;padding-right:7.5pt;">Image-level</td><td class="ltx_td ltx_align_left ltx_border_t" id="S4.T1.2.2.2.6" style="padding-left:7.5pt;padding-right:7.5pt;">Coarse</td></tr><tr class="ltx_tr" id="S4.T1.2.3.3"><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.3.3.1" style="padding-left:7.5pt;padding-right:7.5pt;">WIT <html><body><p>( <strong>Learning transferable visual models from natural languagesupervision.</strong> )</p></body></html></td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.3.3.2" style="padding-left:7.5pt;padding-right:7.5pt;">CLIP</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.3.3.3" style="padding-left:7.5pt;padding-right:7.5pt;">400M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.3.3.4" style="padding-left:7.5pt;padding-right:7.5pt;">400M</td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.3.3.5" style="padding-left:7.5pt;padding-right:7.5pt;">Image-level</td><td class="ltx_td ltx_align_left" id="S4.T1.2.3.3.6" style="padding-left:7.5pt;padding-right:7.5pt;">Coarse</td></tr><tr class="ltx_tr" id="S4.T1.2.4.4"><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.4.4.1" style="padding-left:7.5pt;padding-right:7.5pt;">SA-1B <html><body><p>( <strong>Segment anything.</strong> )</p></body></html></td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.4.4.2" style="padding-left:7.5pt;padding-right:7.5pt;">SAM</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.4.4.3" style="padding-left:7.5pt;padding-right:7.5pt;">11M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.4.4.4" style="padding-left:7.5pt;padding-right:7.5pt;">1B</td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.4.4.5" style="padding-left:7.5pt;padding-right:7.5pt;">Region-level</td><td class="ltx_td ltx_align_left" id="S4.T1.2.4.4.6" style="padding-left:7.5pt;padding-right:7.5pt;">Non-semantic</td></tr><tr class="ltx_tr" id="S4.T1.2.5.5"><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.5.5.1" style="padding-left:7.5pt;padding-right:7.5pt;">GrIT <html><body><p>( <strong>Kosmos-2: Grounding multimodal large language models to the world.</strong> )</p></body></html></td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.5.5.2" style="padding-left:7.5pt;padding-right:7.5pt;">Kosmos-2</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.5.5.3" style="padding-left:7.5pt;padding-right:7.5pt;">91M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.5.5.4" style="padding-left:7.5pt;padding-right:7.5pt;">137M</td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.5.5.5" style="padding-left:7.5pt;padding-right:7.5pt;">Image &amp; Region-level</td><td class="ltx_td ltx_align_left" id="S4.T1.2.5.5.6" style="padding-left:7.5pt;padding-right:7.5pt;">Fine-grained</td></tr><tr class="ltx_tr" id="S4.T1.2.6.6"><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.6.6.1" style="padding-left:7.5pt;padding-right:7.5pt;">M3W <html><body><p>( <strong>Flamingo: a visual language model for few-shot learning.</strong> )</p></body></html></td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.6.6.2" style="padding-left:7.5pt;padding-right:7.5pt;">Flamingo</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.6.6.3" style="padding-left:7.5pt;padding-right:7.5pt;">185M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S4.T1.2.6.6.4" style="padding-left:7.5pt;padding-right:7.5pt;">43.3M*</td><td class="ltx_td ltx_align_left ltx_border_r" id="S4.T1.2.6.6.5" style="padding-left:7.5pt;padding-right:7.5pt;">Multi-image-level</td><td class="ltx_td ltx_align_left" id="S4.T1.2.6.6.6" style="padding-left:7.5pt;padding-right:7.5pt;">Fine-grained</td></tr><tr class="ltx_tr" id="S4.T1.2.7.7" style="background-color:#E6E6E6;"><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="S4.T1.2.7.7.1" style="padding-left:7.5pt;padding-right:7.5pt;"><em class="ltx_emph ltx_font_italic" id="S4.T1.2.7.7.1.1" style="background-color:#E6E6E6;">FLD-5B</em> (ours)</td><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="S4.T1.2.7.7.2" style="padding-left:7.5pt;padding-right:7.5pt;"><em class="ltx_emph ltx_font_italic" id="S4.T1.2.7.7.2.1" style="background-color:#E6E6E6;">Florence-2</em> (ours)</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S4.T1.2.7.7.3" style="padding-left:7.5pt;padding-right:7.5pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S4.T1.2.7.7.4" style="padding-left:7.5pt;padding-right:7.5pt;">5B</td><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="S4.T1.2.7.7.5" style="padding-left:7.5pt;padding-right:7.5pt;">Image &amp; Region-level</td><td class="ltx_td ltx_align_left ltx_border_bb" id="S4.T1.2.7.7.6" style="padding-left:7.5pt;padding-right:7.5pt;">Coarse to fine-grained</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 1:  Comparison with datasets in vision foundation model training. *Flamingo’s annotations are counted in the number of documents, where each document may have multiple images. | ✅ Table 1:  Comparison with datasets in vision foundation model training. *Flamingo’s annotations are counted in the number of documents, where each document may have multiple images. |

## 5 Dataset

| 【第5节，第1段】原文 | 【第5节，第1段】翻译 |
| ---- | ---- |
| ✅ This section introduces the statistics and analysis of FLD-5B that we built using the data engine in Section 4. | ✅ 本节介绍我们利用Section 4中的数据引擎构建的FLD-5B的统计和分析。 |
| ✅ We begin with an overview of the dataset and compare it with the recent works. | ✅ 我们首先概述数据集，并将其与最近的研究进行比较。 |
| ✅ We then show further analyses of detailed annotation statistics, semantic coverage and spatial coverage in the established dataset. | ✅ 然后，我们进一步分析已建立的数据集中的详细注释统计、语义覆盖率和空间覆盖率。 |

### 5.1 Overview

| 【第5.1节，第1段】原文 | 【第5.1节，第1段】翻译 |
| ---- | ---- |
| ✅ Following the data engine, we build a large-scale training set ( FLD-5B ) of 126M images, more than 500M text annotations, 1.3B region-text annotations, and 3.6B text-phrase-region annotations. | ✅ 按照数据引擎，我们构建了包含126M张图像的大规模训练集（FLD-5B）、超过500M的文本标注、1.3B的区域文本标注、以及3.6B的文本短语区域标注。 |
| ✅ Each image is annotated with text, region-text pairs, and text-phrase-region triplets and each annotation type has multiple instances varying in diverse granularity. | ✅ 每个图像都带有文本、区域-文本对和文本-短语-区域三元组注释，并且每种注释类型都有多个不同粒度的实例。 |
| ✅ An illustrative example of an image and its corresponding annotations can be found in Figure 4 . | ✅ 在 Figure 4 中可以找到图像及其相应注释的说明性示例。 |

| 【第5.1节，第2段】原文 | 【第5.1节，第2段】翻译 |
| ---- | ---- |
| ✅ We provide a comparison between our data set and the existing data sets that are commonly used for training foundation models in Table 1. | ✅ 我们对我们的数据集和常用于训练 Table 1 基础模型的现有数据集进行了比较。 |
| ✅ Our data set has several advantages over the previous ones, such as having more annotations in total and per image. | ✅ 我们的数据集比以前的数据集有几个优势，例如总体和每个图像有更多的注释。 |
| ✅ Moreover, the annotations in our data set span multiple levels of spatial and semantic granularity, which allows for more diverse and comprehensive visual understanding tasks. | ✅ 此外，我们数据集中的注释涵盖了多个空间和语义粒度级别，从而可以实现更加多样化和全面的视觉理解任务。 |

### 5.2 Data Analysis

#### 5.2.1 Annotation statistics.

| 【第5.2.1节，第1段】原文 | 【第5.2.1节，第1段】翻译 |
| ---- | ---- |
| ✅ The statistics for each annotation type within our dataset are presented in Table 2 . | ✅ 我们的数据集中每种注释类型的统计数据都显示在 Table 2 中。 |

| 【第5.2.1节，第2段】原文 | 【第5.2.1节，第2段】翻译 |
| ---- | ---- |
| ✅ Firstly, we have around 500M text annotations, including brief, detailed, and more detailed texts with different lengths. | ✅ 首先，我们有大约500M个文本注释，包括不同长度的简短、详细和更详细的文本。 |
| ✅ It is noteworthy that our detailed and more detailed text has 4x and 9x number of tokens compared with the brief text that is similar to COCO captions ( **Microsoft coco captions: Data collection and evaluation server.** ). | ✅ 值得注意的是，与类似于 COCO 字幕 ( **Microsoft coco captions: Data collection and evaluation server.** ) 的简短文本相比，我们的详细和更详细的文本具有 4 倍和 9 倍的标记数量。 |
| ✅ These lengthy annotations provide much richer information for comphrensive visual understanding. | ✅ 这些冗长的注释为全面的视觉理解提供了更丰富的信息。 |

| 【第5.2.1节，第3段】原文 | 【第5.2.1节，第3段】翻译 |
| ---- | ---- |
| ✅ In addition, our dataset has around 1.3B region-text annotations, which is more than 30x larger than the academic object detection datasets such as OpenImages ( **The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.** ) and Object 365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ). | ✅ 此外，我们的数据集有大约 1.3B 区域文本注释，比 OpenImages ( **The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.** ) 和 Object 365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ) 等学术对象检测数据集大 30 倍以上。 |
| ✅ On average, each image has around 5 regions, and each region is annotated with either a phrase or a relatively longer brief text. | ✅ 平均而言，每张图像有大约 5 个区域，每个区域都用短语或相对较长的简短文本进行注释。 |
| ✅ Note that the regional brief text (2.55 avg tokens) is shorter than typical brief text annotation (7.95 avg tokens), as the regional brief text annotation actually includes a mixture of phrase, noun chunks, and brief text based on the Florence-1 score. | ✅ 请注意，区域简短文本（2.55 个平均标记）比典型的简短文本注释（7.95 个平均标记）短，因为区域简短文本注释实际上包括基于 Florence-1 分数的短语、名词块和简短文本的混合。 |
| ✅ More details can be found from Section 4.3 - region-text pairs. | ✅ 更多详细信息可参见 Section 4.3 - 区域-文本对。 |

| 【第5.2.1节，第4段】原文 | 【第5.2.1节，第4段】翻译 |
| ---- | ---- |
| ✅ Moreover, we collect text-phrase-region annotations that include more than 3.6B phrase-region pairs for the 500M text annotations. | ✅ 此外，我们为 500M 文本注释收集了包含超过 3.6B 短语区域对的文本短语区域注释。 |
| ✅ Specifically, the brief text annotation has 4.27 average phrase-region pairs, while detailed and more detailed text annotation has more than 10 pairs, indicating that the richer text annotation covers more objects and their corresponding phrases in the text. | ✅ 具体来说，简短的文本注释平均有 4.27 个短语-区域对，而详细和更详细的文本注释平均有 10 对以上，这表明更丰富的文本注释涵盖了文本中更多的对象及其对应的短语。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S5.T2.2"><thead class="ltx_thead"><tr class="ltx_tr" id="S5.T2.2.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S5.T2.2.1.1.1" style="padding:1.6pt 5.5pt;">Annotation Type</th><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S5.T2.2.1.1.2" style="padding:1.6pt 5.5pt;">Text Type</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S5.T2.2.1.1.3" style="padding:1.6pt 5.5pt;">#Image Annotations</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S5.T2.2.1.1.4" style="padding:1.6pt 5.5pt;">#Avg Tokens</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S5.T2.2.1.1.5" style="padding:1.6pt 5.5pt;">#Regions</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S5.T2.2.1.1.6" style="padding:1.6pt 5.5pt;">#Avg Regions</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt" id="S5.T2.2.1.1.7" style="padding:1.6pt 5.5pt;">#Avg Regional Tokens</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S5.T2.2.2.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S5.T2.2.2.1.1" style="padding:1.6pt 5.5pt;">Text</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S5.T2.2.2.1.2" style="padding:1.6pt 5.5pt;">Brief</th><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.2.1.3" style="padding:1.6pt 5.5pt;">235M</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.2.1.4" style="padding:1.6pt 5.5pt;">7.95</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.2.1.5" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.2.1.6" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_t" id="S5.T2.2.2.1.7" style="padding:1.6pt 5.5pt;">-</td></tr><tr class="ltx_tr" id="S5.T2.2.3.2"><th class="ltx_td ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.3.2.1" style="padding:1.6pt 5.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.3.2.2" style="padding:1.6pt 5.5pt;">Detailed</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.3.2.3" style="padding:1.6pt 5.5pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.3.2.4" style="padding:1.6pt 5.5pt;">31.65</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.3.2.5" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.3.2.6" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right" id="S5.T2.2.3.2.7" style="padding:1.6pt 5.5pt;">-</td></tr><tr class="ltx_tr" id="S5.T2.2.4.3"><th class="ltx_td ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.4.3.1" style="padding:1.6pt 5.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.4.3.2" style="padding:1.6pt 5.5pt;">More detailed</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.4.3.3" style="padding:1.6pt 5.5pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.4.3.4" style="padding:1.6pt 5.5pt;">70.53</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.4.3.5" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.4.3.6" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right" id="S5.T2.2.4.3.7" style="padding:1.6pt 5.5pt;">-</td></tr><tr class="ltx_tr" id="S5.T2.2.5.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S5.T2.2.5.4.1" style="padding:1.6pt 5.5pt;">Region-Text</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S5.T2.2.5.4.2" style="padding:1.6pt 5.5pt;">Phrase</th><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.5.4.3" style="padding:1.6pt 5.5pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.5.4.4" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.5.4.5" style="padding:1.6pt 5.5pt;">681M</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.5.4.6" style="padding:1.6pt 5.5pt;">5.42</td><td class="ltx_td ltx_align_right ltx_border_t" id="S5.T2.2.5.4.7" style="padding:1.6pt 5.5pt;">1.19</td></tr><tr class="ltx_tr" id="S5.T2.2.6.5"><th class="ltx_td ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.6.5.1" style="padding:1.6pt 5.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.6.5.2" style="padding:1.6pt 5.5pt;">Brief</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.6.5.3" style="padding:1.6pt 5.5pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.6.5.4" style="padding:1.6pt 5.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.6.5.5" style="padding:1.6pt 5.5pt;">681M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.6.5.6" style="padding:1.6pt 5.5pt;">5.42</td><td class="ltx_td ltx_align_right" id="S5.T2.2.6.5.7" style="padding:1.6pt 5.5pt;">2.55</td></tr><tr class="ltx_tr" id="S5.T2.2.7.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S5.T2.2.7.6.1" style="padding:1.6pt 5.5pt;">Text-Phrase-Region</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S5.T2.2.7.6.2" style="padding:1.6pt 5.5pt;">Brief</th><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.7.6.3" style="padding:1.6pt 5.5pt;">235M</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.7.6.4" style="padding:1.6pt 5.5pt;">7.95</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.7.6.5" style="padding:1.6pt 5.5pt;">1007M</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T2.2.7.6.6" style="padding:1.6pt 5.5pt;">4.27</td><td class="ltx_td ltx_align_right ltx_border_t" id="S5.T2.2.7.6.7" style="padding:1.6pt 5.5pt;">1.93</td></tr><tr class="ltx_tr" id="S5.T2.2.8.7"><th class="ltx_td ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.8.7.1" style="padding:1.6pt 5.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T2.2.8.7.2" style="padding:1.6pt 5.5pt;">Detailed</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.8.7.3" style="padding:1.6pt 5.5pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.8.7.4" style="padding:1.6pt 5.5pt;">31.65</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.8.7.5" style="padding:1.6pt 5.5pt;">1289M</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T2.2.8.7.6" style="padding:1.6pt 5.5pt;">10.25</td><td class="ltx_td ltx_align_right" id="S5.T2.2.8.7.7" style="padding:1.6pt 5.5pt;">1.49</td></tr><tr class="ltx_tr" id="S5.T2.2.9.8"><th class="ltx_td ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S5.T2.2.9.8.1" style="padding:1.6pt 5.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S5.T2.2.9.8.2" style="padding:1.6pt 5.5pt;">More detailed</th><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S5.T2.2.9.8.3" style="padding:1.6pt 5.5pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S5.T2.2.9.8.4" style="padding:1.6pt 5.5pt;">70.53</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S5.T2.2.9.8.5" style="padding:1.6pt 5.5pt;">1278M</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S5.T2.2.9.8.6" style="padding:1.6pt 5.5pt;">10.17</td><td class="ltx_td ltx_align_right ltx_border_bb" id="S5.T2.2.9.8.7" style="padding:1.6pt 5.5pt;">1.35</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 2:  Annotation statistics of FLD-5B dataset. | ✅ Table 2:  FLD-5B数据集的注释统计。 |

#### 5.2.2 Semantic coverage.

| 【第5.2.2节，第1段】原文 | 【第5.2.2节，第1段】翻译 |
| ---- | ---- |
| ✅ Our text annotations comprise various text types, addressing different levels of detail. | ✅ 我们的文本注释包含各种文本类型，涉及不同级别的细节。 |
| ✅ To assess semantic coverage, we employ SpaCy ( **spacy: Industrial-strength natural language processing in python.** ) for tokenization and parsing, inspired by DiHT ( **Filtering, distillation, and hard negatives for vision-language pre-training.** ). | ✅ 为了评估语义覆盖范围，我们采用 SpaCy ( **spacy: Industrial-strength natural language processing in python.** ) 进行标记和解析，受到 DiHT ( **Filtering, distillation, and hard negatives for vision-language pre-training.** ) 的启发。 |
| ✅ This process yields part-of-speech (POS) tags and the dependency parsing tree among tokens. | ✅ 该过程产生了词性 (POS) 标记和标记之间的依赖关系解析树。 |
| ✅ We establish heuristic rules based on POS tags, categorizing tokens into semantic element types, e.g. | ✅ 我们根据 POS 标签建立启发式规则，将标记分类为语义元素类型 e.g。 |
| ✅  , objects, attributes, actions, and proper nouns. | ✅ 、对象、属性、动作和专有名词。 |
| ✅ Additionally, we introduce the concept of token complexity , measured by the total degrees of the token in the dependency parsing tree when treated as an undirected graph. | ✅ 此外，我们引入了 token complexity 的概念，当将依赖关系解析树视为无向图时，以标记在依赖关系解析树中的总度数来衡量。 |
| ✅ This complexity reflects the richness of semantic connections. | ✅ 这种复杂性反映了语义联系的丰富性。 |
| ✅ In our study, we focus on measuring the complexity of objects and actions. | ✅ 在我们的研究中，我们重点测量物体和动作的复杂性。 |

| 【第5.2.2节，第2段】原文 | 【第5.2.2节，第2段】翻译 |
| ---- | ---- |
| ✅ Table 3 presents the statistics on the average number of semantic elements and their corresponding complexity. | ✅ Table 3 显示了语义元素的平均数量及其对应的复杂度的统计数据。 |
| ✅ The results indicate that all measurements increase with the inclusion of more details in text annotations. | ✅ 结果表明，随着文本注释中包含更多细节，所有测量值都会增加。 |
| ✅ Notably, average actions experience the most significant boost, with detailed and more detailed text exhibiting 7  $\times$  and 15  $\times$  increases, respectively, compared to brief text. | ✅ 值得注意的是，平均操作经历了最显著的提升，与简短文本相比，详细文本和更详细的文本分别显示了 7  $\times$  和 15  $\times$  的增加。 |
| ✅ This highlights the limitations of traditional brief text annotations in describing image actions. | ✅ 这凸显了传统简短文本注释在描述图像动作方面的局限性。 |
| ✅ Conversely, the increment in proper nouns is relatively low, potentially because specialists often describe objects more generally than using specific proper nouns. | ✅ 相反，专有名词的增量相对较低，这可能是因为专家通常更笼统地描述对象而不是使用特定的专有名词。 |
| ✅ In terms of complexity measurements, both objects and actions show more semantic connections in detailed text annotations. | ✅ 在复杂性测量方面，对象和动作在详细的文本注释中都表现出更多的语义联系。 |
| ✅ The complexity of actions exhibits a higher improvement, aligning with our observation of the increasing number of actions. | ✅ 动作的复杂性表现出更高的改进，这与我们对动作数量不断增加的观察一致。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S5.T3.2"><thead class="ltx_thead"><tr class="ltx_tr" id="S5.T3.2.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S5.T3.2.1.1.1" style="padding-left:5.1pt;padding-right:5.1pt;">Text Type</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S5.T3.2.1.1.2" style="padding-left:5.1pt;padding-right:5.1pt;">Brief</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S5.T3.2.1.1.3" style="padding-left:5.1pt;padding-right:5.1pt;">Detailed</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt" id="S5.T3.2.1.1.4" style="padding-left:5.1pt;padding-right:5.1pt;">More detailed</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S5.T3.2.2.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S5.T3.2.2.1.1" style="padding-left:5.1pt;padding-right:5.1pt;">#Image Annotations</th><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T3.2.2.1.2" style="padding-left:5.1pt;padding-right:5.1pt;">235M</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S5.T3.2.2.1.3" style="padding-left:5.1pt;padding-right:5.1pt;">126M</td><td class="ltx_td ltx_align_right ltx_border_t" id="S5.T3.2.2.1.4" style="padding-left:5.1pt;padding-right:5.1pt;">126M</td></tr><tr class="ltx_tr" id="S5.T3.2.3.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T3.2.3.2.1" style="padding-left:5.1pt;padding-right:5.1pt;">#Avg Tokens</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.3.2.2" style="padding-left:5.1pt;padding-right:5.1pt;">7.95</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.3.2.3" style="padding-left:5.1pt;padding-right:5.1pt;">31.65</td><td class="ltx_td ltx_align_right" id="S5.T3.2.3.2.4" style="padding-left:5.1pt;padding-right:5.1pt;">70.53</td></tr><tr class="ltx_tr" id="S5.T3.2.4.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T3.2.4.3.1" style="padding-left:5.1pt;padding-right:5.1pt;">#Avg Objects</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.4.3.2" style="padding-left:5.1pt;padding-right:5.1pt;">3.23</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.4.3.3" style="padding-left:5.1pt;padding-right:5.1pt;">13.31</td><td class="ltx_td ltx_align_right" id="S5.T3.2.4.3.4" style="padding-left:5.1pt;padding-right:5.1pt;">28.06</td></tr><tr class="ltx_tr" id="S5.T3.2.5.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T3.2.5.4.1" style="padding-left:5.1pt;padding-right:5.1pt;">#Avg Attributes</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.5.4.2" style="padding-left:5.1pt;padding-right:5.1pt;">2.80</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.5.4.3" style="padding-left:5.1pt;padding-right:5.1pt;">7.27</td><td class="ltx_td ltx_align_right" id="S5.T3.2.5.4.4" style="padding-left:5.1pt;padding-right:5.1pt;">16.25</td></tr><tr class="ltx_tr" id="S5.T3.2.6.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T3.2.6.5.1" style="padding-left:5.1pt;padding-right:5.1pt;">#Avg Actions</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.6.5.2" style="padding-left:5.1pt;padding-right:5.1pt;">0.58</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.6.5.3" style="padding-left:5.1pt;padding-right:5.1pt;">4.21</td><td class="ltx_td ltx_align_right" id="S5.T3.2.6.5.4" style="padding-left:5.1pt;padding-right:5.1pt;">8.76</td></tr><tr class="ltx_tr" id="S5.T3.2.7.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T3.2.7.6.1" style="padding-left:5.1pt;padding-right:5.1pt;">#Proper Nouns</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.7.6.2" style="padding-left:5.1pt;padding-right:5.1pt;">1.10</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.7.6.3" style="padding-left:5.1pt;padding-right:5.1pt;">2.40</td><td class="ltx_td ltx_align_right" id="S5.T3.2.7.6.4" style="padding-left:5.1pt;padding-right:5.1pt;">2.41</td></tr><tr class="ltx_tr" id="S5.T3.2.8.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S5.T3.2.8.7.1" style="padding-left:5.1pt;padding-right:5.1pt;">Avg Object Complexity</th><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.8.7.2" style="padding-left:5.1pt;padding-right:5.1pt;">2.80</td><td class="ltx_td ltx_align_right ltx_border_r" id="S5.T3.2.8.7.3" style="padding-left:5.1pt;padding-right:5.1pt;">4.00</td><td class="ltx_td ltx_align_right" id="S5.T3.2.8.7.4" style="padding-left:5.1pt;padding-right:5.1pt;">4.02</td></tr><tr class="ltx_tr" id="S5.T3.2.9.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S5.T3.2.9.8.1" style="padding-left:5.1pt;padding-right:5.1pt;">Avg Action Complexity</th><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S5.T3.2.9.8.2" style="padding-left:5.1pt;padding-right:5.1pt;">1.14</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S5.T3.2.9.8.3" style="padding-left:5.1pt;padding-right:5.1pt;">3.63</td><td class="ltx_td ltx_align_right ltx_border_bb" id="S5.T3.2.9.8.4" style="padding-left:5.1pt;padding-right:5.1pt;">4.38</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 3:  Statistics of the average number of semantic elements and corresponding complexity in FLD-5B dataset. | ✅ Table 3:  FLD-5B数据集中语义元素的平均数量及相应的复杂度统计。 |

#### 5.2.3 Spatial coverage.

| 【第5.2.3节，第1段】原文 | 【第5.2.3节，第1段】翻译 |
| ---- | ---- |
| ✅ Our region-text and text-phrase-region annotations, represented by bounding boxes and masks, capture the location of visual concepts within images. | ✅ 我们的区域文本和文本短语区域注释以边界框和蒙版表示，捕捉图像内视觉概念的位置。 |
| ✅ The distribution of box areas, as shown in Figure 5(a) , reveals more small boxes in region-text pairs and a uniform box size distribution in text-phrase-region triplets. | ✅ 框区域分布如 Figure 5(a) 所示，表明区域-文本对中存在更多小框，而文本-短语-区域三元组中的框大小分布均匀。 |
| ✅ This difference stems from the the divergent origins of these boxes: object detectors for region-text pairs and a grounding model for text-phrase-region triplets, which aligns boxes to textual phrases representing both localized and overarching image concepts. | ✅ 这种差异源于这些框的不同来源：用于区域-文本对的对象检测器和用于文本-短语-区域三元组的接地模型，该模型将框与代表局部和总体图像概念的文本短语对齐。 |
| ✅ In Figure 5(b) , the log-format distribution of aspect ratios is illustrated. | ✅ 在 Figure 5(b) 中，显示了长宽比的对数格式分布。 |
| ✅ Region-text pairs and text-phrase-region triplets exhibit similar symmetric distributions, covering a wide range of aspect ratios. | ✅ 区域-文本对和文本-短语-区域三元组表现出相似的对称分布，涵盖了广泛的纵横比。 |
| ✅ Heatmaps of the box center for each annotation type, shown in Figures. 5(c) and 5(d) , indicate a center bias, with region-text pairs displaying a more uniform distribution than text-phrase-region triplets. | ✅ Figures. 5(c) 和 5(d) 中显示的每种注释类型的框中心热图表明存在中心偏差，其中区域-文本对比文本-短语-区域三元组显示出更均匀的分布。 |

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x5.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ (a)  | ✅ (a)  |

## 6 Experiments

| 【第6节，第1段】原文 | 【第6节，第1段】翻译 |
| ---- | ---- |
| ✅ Our Florence-2 models are trained on FLD-5B to learn a universal image representation. | ✅ 我们的 Florence-2 模型在 FLD-5B 上进行训练，以学习通用图像表示。 |
| ✅ We conduct our experiments in three main parts: (1) We evaluate the zero-shot performance of our method on various tasks to show its inherent ability to handle multiple tasks without any extra fine-tuning on task-specific data using one single generalist model. | ✅ 我们的实验主要分为三个部分：（1）我们评估我们的方法在各种任务上的 zero-shot 性能，以显示其处理多项任务的固有能力，而无需使用 one single generalist 模型对特定于任务的数据进行任何额外的微调。 |
| ✅ (2) We show the adaptability of our method by further training one single generalist model with additional supervised data on a wide range of tasks, achieving competitive state-of-the-art performance. | ✅ （2）我们通过在广泛任务上使用额外的监督数据进一步训练 one single generalist 模型来展示我们方法的适应性，并实现了具有竞争力的最先进的性能。 |
| ✅ (3) We examine the performance of the learned visual representation on the downstream tasks as the backbone to show the superiority of our pre-training method over previous approaches. | ✅ （3）我们以学习到的视觉表征在下游任务中的表现为骨干，展示我们的预训练方法相对于以前的方法的优越性。 |

### 6.1 Setup

| 【第6.1节，第1段】原文 | 【第6.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We investigate two model variants with different sizes: Florence-2-B model with 232 million parameters and Florence-2-L model with 771 million parameters. | ✅ 我们研究了两种不同大小的模型变体：具有 2.32 亿个参数的 Florence-2-B 模型和具有 7.71 亿个参数的 Florence-2-L 模型。 |
| ✅ The detailed architectures of each model are given in Table 15. | ✅ 每个模型的详细架构在Table 15中给出。 |
| ✅ We initialize the weights of the image encoder and multi-modality encoder-decoder from UniCL ( **Unified contrastive learning in image-text-label space, 2022.** ) and BART ( **Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.** ) , respectively. | ✅ 我们分别从 UniCL ( **Unified contrastive learning in image-text-label space, 2022.** ) 和 BART ( **Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.** ) 初始化图像编码器和多模态编码器-解码器的权重。 |

| 【第6.1节，第2段】原文 | 【第6.1节，第2段】翻译 |
| ---- | ---- |
| ✅ We adopt AdamW ( **Decoupled weight decay regularization, 2019.** ) with cosine learning rate decay ( **Sgdr: Stochastic gradient descent with warm restarts, 2017.** ) for training our models. | ✅ 我们采用 AdamW ( **Decoupled weight decay regularization, 2019.** ) 和余弦学习率衰减 ( **Sgdr: Stochastic gradient descent with warm restarts, 2017.** ) 来训练我们的模型。 |
| ✅ We leverage Deepspeed ( **Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.** ) and mixed precision to improve the training efficiency. | ✅ 我们利用 Deepspeed ( **Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.** ) 和混合精度来提高训练效率。 |
| ✅ The maximum learning rate is set at  $1e-4$  for the base model and  $1e-5$  for the large model. | ✅ 基础模型的最大学习率设置为  $1e-4$ ，大型模型的最大学习率设置为  $1e-5$ 。 |
| ✅ A linear warm-up to the maximum learning rate is applied during the first 5,000 optimization steps. | ✅ 在前 5,000 个优化步骤中，应用线性预热至最大学习率。 |

| 【第6.1节，第3段】原文 | 【第6.1节，第3段】翻译 |
| ---- | ---- |
| ✅ We train our models with a mini-batch size of 2048/3072 (base/large) and an image size of 384  $\times$  384 until reaching 3 billion effective training samples. | ✅ 我们使用 2048/3072（基础/大）的小批量和 384  $\times$  384 的图像大小来训练我们的模型，直到达到 30 亿有效训练样本。 |
| ✅ Similar to ( **1. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **2. Learning transferable visual models from natural language supervision.** ｜ **3. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ｜ **4. Florence: A new foundation model for computer vision.** ｜ **5. Coca: Contrastive captioners are image-text foundation models, 2022.** ) , we further conduct high-resolution tuning with an image size of 768  $\times$  768 for 0.5 billion samples for the base model and 0.1 billion samples for the large model. | ✅ 与 ( **1. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **2. Learning transferable visual models from natural language supervision.** ｜ **3. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ｜ **4. Florence: A new foundation model for computer vision.** ｜ **5. Coca: Contrastive captioners are image-text foundation models, 2022.** ) 类似，我们进一步对图像大小为 768  $\times$  768 的基础模型进行 5 亿个样本的高分辨率调优，对大型模型进行 1 亿个样本的高分辨率调优。 |

### 6.2 Zero-shot Evaluation Across Tasks

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T4.2"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T4.2.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T4.2.1.1.1" rowspan="3" style="padding-left:2.4pt;padding-right:2.4pt;">Method</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T4.2.1.1.2" rowspan="3" style="padding-left:2.4pt;padding-right:2.4pt;">#params</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.3" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.4" style="padding-left:2.4pt;padding-right:2.4pt;">COCO Cap.</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.5" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.6" style="padding-left:2.4pt;padding-right:2.4pt;">NoCaps</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.7" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.8" style="padding-left:2.4pt;padding-right:2.4pt;">TextCaps</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.9" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.10" style="padding-left:2.4pt;padding-right:2.4pt;">COCO Det.</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.11" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.12" style="padding-left:2.4pt;padding-right:2.4pt;">Flickr30k</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.13" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="3" id="S6.T4.2.1.1.14" style="padding-left:2.4pt;padding-right:2.4pt;">Refcoco</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.15" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="3" id="S6.T4.2.1.1.16" style="padding-left:2.4pt;padding-right:2.4pt;">Refcoco+</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.17" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="2" id="S6.T4.2.1.1.18" style="padding-left:2.4pt;padding-right:2.4pt;">Refcocog</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.19" style="padding-left:2.4pt;padding-right:2.4pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T4.2.1.1.20" style="padding-left:2.4pt;padding-right:2.4pt;">Refcoco RES</th></tr><tr class="ltx_tr" id="S6.T4.2.2.2"><td class="ltx_td" id="S6.T4.2.2.2.1" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.2" style="padding-left:2.4pt;padding-right:2.4pt;">test</td><td class="ltx_td" id="S6.T4.2.2.2.3" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.4" style="padding-left:2.4pt;padding-right:2.4pt;">val</td><td class="ltx_td" id="S6.T4.2.2.2.5" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.6" style="padding-left:2.4pt;padding-right:2.4pt;">val</td><td class="ltx_td" id="S6.T4.2.2.2.7" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.8" style="padding-left:2.4pt;padding-right:2.4pt;">val2017</td><td class="ltx_td" id="S6.T4.2.2.2.9" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.10" style="padding-left:2.4pt;padding-right:2.4pt;">test</td><td class="ltx_td" id="S6.T4.2.2.2.11" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.12" style="padding-left:2.4pt;padding-right:2.4pt;">val</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.13" style="padding-left:2.4pt;padding-right:2.4pt;">test-A</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.14" style="padding-left:2.4pt;padding-right:2.4pt;">test-B</td><td class="ltx_td" id="S6.T4.2.2.2.15" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.16" style="padding-left:2.4pt;padding-right:2.4pt;">val</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.17" style="padding-left:2.4pt;padding-right:2.4pt;">test-A</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.18" style="padding-left:2.4pt;padding-right:2.4pt;">test-B</td><td class="ltx_td" id="S6.T4.2.2.2.19" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.20" style="padding-left:2.4pt;padding-right:2.4pt;">val</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.21" style="padding-left:2.4pt;padding-right:2.4pt;">test</td><td class="ltx_td" id="S6.T4.2.2.2.22" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.2.2.23" style="padding-left:2.4pt;padding-right:2.4pt;">val</td></tr><tr class="ltx_tr" id="S6.T4.2.3.3"><td class="ltx_td" id="S6.T4.2.3.3.1" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.3.3.2" style="padding-left:2.4pt;padding-right:2.4pt;">CIDEr</td><td class="ltx_td" id="S6.T4.2.3.3.3" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.3.3.4" style="padding-left:2.4pt;padding-right:2.4pt;">CIDEr</td><td class="ltx_td" id="S6.T4.2.3.3.5" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.3.3.6" style="padding-left:2.4pt;padding-right:2.4pt;">CIDEr</td><td class="ltx_td" id="S6.T4.2.3.3.7" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.3.3.8" style="padding-left:2.4pt;padding-right:2.4pt;">mAP</td><td class="ltx_td" id="S6.T4.2.3.3.9" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.3.3.10" style="padding-left:2.4pt;padding-right:2.4pt;">R@1</td><td class="ltx_td" id="S6.T4.2.3.3.11" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" colspan="3" id="S6.T4.2.3.3.12" style="padding-left:2.4pt;padding-right:2.4pt;">Accuracy</td><td class="ltx_td" id="S6.T4.2.3.3.13" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" colspan="3" id="S6.T4.2.3.3.14" style="padding-left:2.4pt;padding-right:2.4pt;">Accuracy</td><td class="ltx_td" id="S6.T4.2.3.3.15" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" colspan="2" id="S6.T4.2.3.3.16" style="padding-left:2.4pt;padding-right:2.4pt;">Accuracy</td><td class="ltx_td" id="S6.T4.2.3.3.17" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.3.3.18" style="padding-left:2.4pt;padding-right:2.4pt;">mIoU</td></tr><tr class="ltx_tr" id="S6.T4.2.4.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T4.2.4.4.1" style="padding-left:2.4pt;padding-right:2.4pt;">Flamingo <html><body><p>( <strong>Flamingo: a visual language model for few-shot learning.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T4.2.4.4.2" style="padding-left:2.4pt;padding-right:2.4pt;">80B</th><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.3" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.4" style="padding-left:2.4pt;padding-right:2.4pt;">84.3</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.5" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.6" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.7" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.8" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.9" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.10" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.11" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.12" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.13" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.14" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.15" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.16" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.17" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.18" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.19" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.20" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.21" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.22" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.23" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T4.2.4.4.24" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.4.4.25" style="padding-left:2.4pt;padding-right:2.4pt;">-</td></tr><tr class="ltx_tr" id="S6.T4.2.5.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T4.2.5.5.1" style="padding-left:2.4pt;padding-right:2.4pt;">Kosmos-2 <html><body><p>( <strong>Kosmos-2: Grounding multimodal large language models to the world.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T4.2.5.5.2" style="padding-left:2.4pt;padding-right:2.4pt;">1.6B</th><td class="ltx_td" id="S6.T4.2.5.5.3" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.4" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td" id="S6.T4.2.5.5.5" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.6" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td" id="S6.T4.2.5.5.7" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.8" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td" id="S6.T4.2.5.5.9" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.10" style="padding-left:2.4pt;padding-right:2.4pt;">-</td><td class="ltx_td" id="S6.T4.2.5.5.11" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.12" style="padding-left:2.4pt;padding-right:2.4pt;">78.7</td><td class="ltx_td" id="S6.T4.2.5.5.13" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.14" style="padding-left:2.4pt;padding-right:2.4pt;">52.3</td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.15" style="padding-left:2.4pt;padding-right:2.4pt;">57.4</td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.16" style="padding-left:2.4pt;padding-right:2.4pt;">47.3</td><td class="ltx_td" id="S6.T4.2.5.5.17" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.18" style="padding-left:2.4pt;padding-right:2.4pt;">45.5</td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.19" style="padding-left:2.4pt;padding-right:2.4pt;">50.7</td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.20" style="padding-left:2.4pt;padding-right:2.4pt;">42.2</td><td class="ltx_td" id="S6.T4.2.5.5.21" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.22" style="padding-left:2.4pt;padding-right:2.4pt;">60.6</td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.23" style="padding-left:2.4pt;padding-right:2.4pt;">61.7</td><td class="ltx_td" id="S6.T4.2.5.5.24" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center" id="S6.T4.2.5.5.25" style="padding-left:2.4pt;padding-right:2.4pt;">-</td></tr><tr class="ltx_tr" id="S6.T4.2.6.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T4.2.6.6.1" style="padding-left:2.4pt;padding-right:2.4pt;"><em class="ltx_emph ltx_font_italic" id="S6.T4.2.6.6.1.1" style="font-size:90%;">Florence-2-B</em></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T4.2.6.6.2" style="padding-left:2.4pt;padding-right:2.4pt;">0.23B</th><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.3" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.4" style="padding-left:2.4pt;padding-right:2.4pt;">133.0</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.5" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.6" style="padding-left:2.4pt;padding-right:2.4pt;">118.7</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.7" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.8" style="padding-left:2.4pt;padding-right:2.4pt;">70.1</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.9" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.10" style="padding-left:2.4pt;padding-right:2.4pt;">34.7</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.11" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.12" style="padding-left:2.4pt;padding-right:2.4pt;">83.6</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.13" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.14" style="padding-left:2.4pt;padding-right:2.4pt;">53.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.15" style="padding-left:2.4pt;padding-right:2.4pt;">58.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.16" style="padding-left:2.4pt;padding-right:2.4pt;">49.7</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.17" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.18" style="padding-left:2.4pt;padding-right:2.4pt;">51.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.19" style="padding-left:2.4pt;padding-right:2.4pt;">56.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.20" style="padding-left:2.4pt;padding-right:2.4pt;">47.9</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.21" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.22" style="padding-left:2.4pt;padding-right:2.4pt;">66.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.23" style="padding-left:2.4pt;padding-right:2.4pt;">65.1</td><td class="ltx_td ltx_border_t" id="S6.T4.2.6.6.24" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T4.2.6.6.25" style="padding-left:2.4pt;padding-right:2.4pt;">34.6</td></tr><tr class="ltx_tr" id="S6.T4.2.7.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T4.2.7.7.1" style="padding-left:2.4pt;padding-right:2.4pt;"><em class="ltx_emph ltx_font_italic" id="S6.T4.2.7.7.1.1" style="font-size:90%;">Florence-2-L</em></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T4.2.7.7.2" style="padding-left:2.4pt;padding-right:2.4pt;">0.77B</th><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.3" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.4" style="padding-left:2.4pt;padding-right:2.4pt;">135.6</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.5" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.6" style="padding-left:2.4pt;padding-right:2.4pt;">120.8</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.7" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.8" style="padding-left:2.4pt;padding-right:2.4pt;">72.8</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.9" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.10" style="padding-left:2.4pt;padding-right:2.4pt;">37.5</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.11" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.12" style="padding-left:2.4pt;padding-right:2.4pt;">84.4</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.13" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.14" style="padding-left:2.4pt;padding-right:2.4pt;">56.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.15" style="padding-left:2.4pt;padding-right:2.4pt;">61.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.16" style="padding-left:2.4pt;padding-right:2.4pt;">51.4</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.17" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.18" style="padding-left:2.4pt;padding-right:2.4pt;">53.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.19" style="padding-left:2.4pt;padding-right:2.4pt;">57.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.20" style="padding-left:2.4pt;padding-right:2.4pt;">49.9</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.21" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.22" style="padding-left:2.4pt;padding-right:2.4pt;">68.0</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.23" style="padding-left:2.4pt;padding-right:2.4pt;">67.0</td><td class="ltx_td ltx_border_bb" id="S6.T4.2.7.7.24" style="padding-left:2.4pt;padding-right:2.4pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T4.2.7.7.25" style="padding-left:2.4pt;padding-right:2.4pt;">35.8</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 4:  Zero-shot performance of generalist vision foundation models. | ✅ Table 4:  Zero-shot 通用视觉基础模型的性能。 |
| ✅ The models do not see the training data of the evaluation tasks during training. | ✅ 模型在训练期间看不到评估任务的训练数据。 |
| ✅ Florence-2 models are pre-trained on FLD-5B dataset. | ✅ Florence-2 模型在 FLD-5B 数据集上进行了预训练。 |
| ✅ Karpathy test split is used for COCO caption evaluation. | ✅ Karpathy 测试分割用于 COCO 字幕评估。 |

| 【第6.2节，第1段】原文 | 【第6.2节，第1段】翻译 |
| ---- | ---- |
| ✅ We present a powerful vision foundation model that does not require task-specific supervised annotations for fine-tuning. | ✅ 我们提出了一个强大的视觉基础模型，该模型不需要针对特定​​任务的监督注释进行微调。 |
| ✅ The zero-shot performance of our model is shown in Table 4. | ✅ 我们的模型的zero-shot性能显示在Table 4中。 |
| ✅ For image-level tasks, Florence-2-L achieves a 135.6 CIDEr score on the COCO caption benchmark ( **Microsoft coco: Common objects in context.** ) , utilizing less than 1% of the parameters compared to the 80B Flamingo ( **Flamingo: a visual language model for few-shot learning.** ) model (which has an 84.3 CIDEr score). | ✅ 对于图像级任务，Florence-2-L 在 COCO 标题基准 ( **Microsoft coco: Common objects in context.** ) 上获得了 135.6 CIDEr 分数，与 80B Flamingo ( **Flamingo: a visual language model for few-shot learning.** ) 模型（其具有 84.3 CIDEr 分数）相比，使用的参数不到 1％。 |
| ✅ For region-level grounding and referring expression comprehension tasks, Florence-2-L establishes a new record in zero-shot performance achieving a 5.7 improvement in Flickr30k ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** ) Recall@1, and approximately 4%, 8%, and 8% absolute improvements on Refcoco, Refcoco+, and Refcocog ( **Modeling context in referring expressions.** ) , respectively, compared to the Kosmos-2 ( **Kosmos-2: Grounding multimodal large language models to the world.** ) model, which has 1.6B parameters. | ✅ 对于区域级基础定位和指称表达理解任务，与拥有 16 亿参数的 Kosmos-2 ( **Kosmos-2: Grounding multimodal large language models to the world.** ) 模型相比，Florence-2-L 在零样本性能方面创下了新纪录，在 Flickr30k ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** ) Recall@1 上实现了 5.7 的提升，在 Refcoco、Refcoco+ 和 Refcocog ( **Modeling context in referring expressions.** ) 上分别实现了约 4%、8% 和 8% 的绝对提升。 |
| ✅ Additionally, our pre-trained model attains a 35.8% mIOU in the Refcoco referring expression segmentation (RES) ( **Modeling context in referring expressions.** ) task, a capability not supported by prior foundation models. | ✅ 此外，我们预训练的模型在 Refcoco 指称表达分割 (RES) ( **Modeling context in referring expressions.** ) 任务中达到了 35.8% mIOU，这是之前的基础模型所不支持的功能。 |

### 6.3 Generalist Model with Public Supervised Data

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T5.6"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T5.6.7.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T5.6.7.1.1" rowspan="3" style="padding-left:9.4pt;padding-right:9.4pt;">Method</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T5.6.7.1.2" rowspan="3" style="padding-left:9.4pt;padding-right:9.4pt;">#params</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T5.6.7.1.3" style="padding-left:9.4pt;padding-right:9.4pt;">COCO Caption</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T5.6.7.1.4" style="padding-left:9.4pt;padding-right:9.4pt;">NoCaps</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T5.6.7.1.5" style="padding-left:9.4pt;padding-right:9.4pt;">TextCaps</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T5.6.7.1.6" style="padding-left:9.4pt;padding-right:9.4pt;">VQAv2</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T5.6.7.1.7" style="padding-left:9.4pt;padding-right:9.4pt;">TextVQA</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T5.6.7.1.8" style="padding-left:9.4pt;padding-right:9.4pt;">VizWiz VQA</th></tr><tr class="ltx_tr" id="S6.T5.6.8.2"><td class="ltx_td ltx_align_center" id="S6.T5.6.8.2.1" style="padding-left:9.4pt;padding-right:9.4pt;">Karpathy test</td><td class="ltx_td ltx_align_center" id="S6.T5.6.8.2.2" style="padding-left:9.4pt;padding-right:9.4pt;">val</td><td class="ltx_td ltx_align_center" id="S6.T5.6.8.2.3" style="padding-left:9.4pt;padding-right:9.4pt;">val</td><td class="ltx_td ltx_align_center" id="S6.T5.6.8.2.4" style="padding-left:9.4pt;padding-right:9.4pt;">test-dev</td><td class="ltx_td ltx_align_center" id="S6.T5.6.8.2.5" style="padding-left:9.4pt;padding-right:9.4pt;">test-dev</td><td class="ltx_td ltx_align_center" id="S6.T5.6.8.2.6" style="padding-left:9.4pt;padding-right:9.4pt;">test-dev</td></tr><tr class="ltx_tr" id="S6.T5.6.9.3"><td class="ltx_td ltx_align_center" id="S6.T5.6.9.3.1" style="padding-left:9.4pt;padding-right:9.4pt;">CIDEr</td><td class="ltx_td ltx_align_center" id="S6.T5.6.9.3.2" style="padding-left:9.4pt;padding-right:9.4pt;">CIDEr</td><td class="ltx_td ltx_align_center" id="S6.T5.6.9.3.3" style="padding-left:9.4pt;padding-right:9.4pt;">CIDEr</td><td class="ltx_td ltx_align_center" id="S6.T5.6.9.3.4" style="padding-left:9.4pt;padding-right:9.4pt;">Acc</td><td class="ltx_td ltx_align_center" id="S6.T5.6.9.3.5" style="padding-left:9.4pt;padding-right:9.4pt;">Acc</td><td class="ltx_td ltx_align_center" id="S6.T5.6.9.3.6" style="padding-left:9.4pt;padding-right:9.4pt;">Acc</td></tr><tr class="ltx_tr" id="S6.T5.6.10.4"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" colspan="8" id="S6.T5.6.10.4.1" style="padding-left:9.4pt;padding-right:9.4pt;">Specialist Models</th></tr><tr class="ltx_tr" id="S6.T5.6.11.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T5.6.11.5.1" style="padding-left:9.4pt;padding-right:9.4pt;">CoCa <html><body><p>( <strong>Coca: Contrastive captioners are image-text foundation models, 2022.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T5.6.11.5.2" style="padding-left:9.4pt;padding-right:9.4pt;">2.1B</th><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T5.6.11.5.3" style="padding-left:9.4pt;padding-right:9.4pt;">143.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T5.6.11.5.4" style="padding-left:9.4pt;padding-right:9.4pt;">122.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T5.6.11.5.5" style="padding-left:9.4pt;padding-right:9.4pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T5.6.11.5.6" style="padding-left:9.4pt;padding-right:9.4pt;">82.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T5.6.11.5.7" style="padding-left:9.4pt;padding-right:9.4pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T5.6.11.5.8" style="padding-left:9.4pt;padding-right:9.4pt;">-</td></tr><tr class="ltx_tr" id="S6.T5.6.12.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.12.6.1" style="padding-left:9.4pt;padding-right:9.4pt;">BLIP-2 <html><body><p>( <strong>Blip-2: Bootstrapping language-image pre-training with frozen imageencoders and large language models.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.12.6.2" style="padding-left:9.4pt;padding-right:9.4pt;">7.8B</th><td class="ltx_td ltx_align_center" id="S6.T5.6.12.6.3" style="padding-left:9.4pt;padding-right:9.4pt;">144.5</td><td class="ltx_td ltx_align_center" id="S6.T5.6.12.6.4" style="padding-left:9.4pt;padding-right:9.4pt;">121.6</td><td class="ltx_td ltx_align_center" id="S6.T5.6.12.6.5" style="padding-left:9.4pt;padding-right:9.4pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T5.6.12.6.6" style="padding-left:9.4pt;padding-right:9.4pt;">82.2</td><td class="ltx_td ltx_align_center" id="S6.T5.6.12.6.7" style="padding-left:9.4pt;padding-right:9.4pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T5.6.12.6.8" style="padding-left:9.4pt;padding-right:9.4pt;">-</td></tr><tr class="ltx_tr" id="S6.T5.6.13.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.13.7.1" style="padding-left:9.4pt;padding-right:9.4pt;">GIT2 <html><body><p>( <strong>Git: A generative image-to-text transformer for vision and language,2022.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.13.7.2" style="padding-left:9.4pt;padding-right:9.4pt;">5.1B</th><td class="ltx_td ltx_align_center" id="S6.T5.6.13.7.3" style="padding-left:9.4pt;padding-right:9.4pt;">145</td><td class="ltx_td ltx_align_center" id="S6.T5.6.13.7.4" style="padding-left:9.4pt;padding-right:9.4pt;">126.9</td><td class="ltx_td ltx_align_center" id="S6.T5.6.13.7.5" style="padding-left:9.4pt;padding-right:9.4pt;">148.6</td><td class="ltx_td ltx_align_center" id="S6.T5.6.13.7.6" style="padding-left:9.4pt;padding-right:9.4pt;">81.7</td><td class="ltx_td ltx_align_center" id="S6.T5.6.13.7.7" style="padding-left:9.4pt;padding-right:9.4pt;">67.3</td><td class="ltx_td ltx_align_center" id="S6.T5.6.13.7.8" style="padding-left:9.4pt;padding-right:9.4pt;">71.0</td></tr><tr class="ltx_tr" id="S6.T5.6.14.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.14.8.1" style="padding-left:9.4pt;padding-right:9.4pt;">Flamingo <html><body><p>( <strong>Flamingo: a visual language model for few-shot learning.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.14.8.2" style="padding-left:9.4pt;padding-right:9.4pt;">80B</th><td class="ltx_td ltx_align_center" id="S6.T5.6.14.8.3" style="padding-left:9.4pt;padding-right:9.4pt;">138.1</td><td class="ltx_td ltx_align_center" id="S6.T5.6.14.8.4" style="padding-left:9.4pt;padding-right:9.4pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T5.6.14.8.5" style="padding-left:9.4pt;padding-right:9.4pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T5.6.14.8.6" style="padding-left:9.4pt;padding-right:9.4pt;">82.0</td><td class="ltx_td ltx_align_center" id="S6.T5.6.14.8.7" style="padding-left:9.4pt;padding-right:9.4pt;">54.1</td><td class="ltx_td ltx_align_center" id="S6.T5.6.14.8.8" style="padding-left:9.4pt;padding-right:9.4pt;">65.7</td></tr><tr class="ltx_tr" id="S6.T5.3.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T5.3.3.4" style="padding-left:9.4pt;padding-right:9.4pt;">PaLI <html><body><p>( <strong>Pali: A jointly-scaled multilingual language-image model, 2022.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T5.3.3.5" style="padding-left:9.4pt;padding-right:9.4pt;">17B</th><td class="ltx_td ltx_align_center" id="S6.T5.3.3.6" style="padding-left:9.4pt;padding-right:9.4pt;">149.1</td><td class="ltx_td ltx_align_center" id="S6.T5.3.3.7" style="padding-left:9.4pt;padding-right:9.4pt;">127.0</td><td class="ltx_td ltx_align_center" id="S6.T5.1.1.1" style="padding-left:9.4pt;padding-right:9.4pt;">160.0<sup class="ltx_sup" id="S6.T5.1.1.1.2">△</sup></td><td class="ltx_td ltx_align_center" id="S6.T5.3.3.8" style="padding-left:9.4pt;padding-right:9.4pt;">84.3</td><td class="ltx_td ltx_align_center" id="S6.T5.2.2.2" style="padding-left:9.4pt;padding-right:9.4pt;">58.8 / 73.1<sup class="ltx_sup" id="S6.T5.2.2.2.2">△</sup></td><td class="ltx_td ltx_align_center" id="S6.T5.3.3.3" style="padding-left:9.4pt;padding-right:9.4pt;">71.6 / 74.4<sup class="ltx_sup" id="S6.T5.3.3.3.2">△</sup></td></tr><tr class="ltx_tr" id="S6.T5.6.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.6.4" style="padding-left:9.4pt;padding-right:9.4pt;">PaLI-X <html><body><p>( <strong>Pali-x: On scaling up a multilingual vision and language model.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.6.5" style="padding-left:9.4pt;padding-right:9.4pt;">55B</th><td class="ltx_td ltx_align_center" id="S6.T5.6.6.6" style="padding-left:9.4pt;padding-right:9.4pt;">149.2</td><td class="ltx_td ltx_align_center" id="S6.T5.6.6.7" style="padding-left:9.4pt;padding-right:9.4pt;">126.3</td><td class="ltx_td ltx_align_center" id="S6.T5.4.4.1" style="padding-left:9.4pt;padding-right:9.4pt;">147 / 163.7<sup class="ltx_sup" id="S6.T5.4.4.1.2">△</sup></td><td class="ltx_td ltx_align_center" id="S6.T5.6.6.8" style="padding-left:9.4pt;padding-right:9.4pt;">86.0</td><td class="ltx_td ltx_align_center" id="S6.T5.5.5.2" style="padding-left:9.4pt;padding-right:9.4pt;">71.4 / 80.8<sup class="ltx_sup" id="S6.T5.5.5.2.2">△</sup></td><td class="ltx_td ltx_align_center" id="S6.T5.6.6.3" style="padding-left:9.4pt;padding-right:9.4pt;">70.9 / 74.6<sup class="ltx_sup" id="S6.T5.6.6.3.2">△</sup></td></tr><tr class="ltx_tr" id="S6.T5.6.15.9"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" colspan="8" id="S6.T5.6.15.9.1" style="padding-left:9.4pt;padding-right:9.4pt;">Generalist Models</th></tr><tr class="ltx_tr" id="S6.T5.6.16.10"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" id="S6.T5.6.16.10.1" style="padding-left:9.4pt;padding-right:9.4pt;">Unified-IO <html><body><p>( <strong>Unified-io: A unified model for vision, language, and multi-modaltasks, 2022.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" id="S6.T5.6.16.10.2" style="padding-left:9.4pt;padding-right:9.4pt;">2.9B</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T5.6.16.10.3" style="padding-left:9.4pt;padding-right:9.4pt;">-</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T5.6.16.10.4" style="padding-left:9.4pt;padding-right:9.4pt;">100</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T5.6.16.10.5" style="padding-left:9.4pt;padding-right:9.4pt;">-</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T5.6.16.10.6" style="padding-left:9.4pt;padding-right:9.4pt;">77.9</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T5.6.16.10.7" style="padding-left:9.4pt;padding-right:9.4pt;">-</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T5.6.16.10.8" style="padding-left:9.4pt;padding-right:9.4pt;">57.4</th></tr><tr class="ltx_tr" id="S6.T5.6.17.11"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.17.11.1" style="padding-left:9.4pt;padding-right:9.4pt;"><em class="ltx_emph ltx_font_italic" id="S6.T5.6.17.11.1.1" style="font-size:90%;">Florence-2-B</em></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T5.6.17.11.2" style="padding-left:9.4pt;padding-right:9.4pt;">0.23B</th><td class="ltx_td ltx_align_center" id="S6.T5.6.17.11.3" style="padding-left:9.4pt;padding-right:9.4pt;">140.0</td><td class="ltx_td ltx_align_center" id="S6.T5.6.17.11.4" style="padding-left:9.4pt;padding-right:9.4pt;">116.7</td><td class="ltx_td ltx_align_center" id="S6.T5.6.17.11.5" style="padding-left:9.4pt;padding-right:9.4pt;">143.9</td><td class="ltx_td ltx_align_center" id="S6.T5.6.17.11.6" style="padding-left:9.4pt;padding-right:9.4pt;">79.7</td><td class="ltx_td ltx_align_center" id="S6.T5.6.17.11.7" style="padding-left:9.4pt;padding-right:9.4pt;">63.6</td><td class="ltx_td ltx_align_center" id="S6.T5.6.17.11.8" style="padding-left:9.4pt;padding-right:9.4pt;">63.6</td></tr><tr class="ltx_tr" id="S6.T5.6.18.12"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T5.6.18.12.1" style="padding-left:9.4pt;padding-right:9.4pt;"><em class="ltx_emph ltx_font_italic" id="S6.T5.6.18.12.1.1" style="font-size:90%;">Florence-2-L</em></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T5.6.18.12.2" style="padding-left:9.4pt;padding-right:9.4pt;">0.77B</th><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T5.6.18.12.3" style="padding-left:9.4pt;padding-right:9.4pt;">143.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T5.6.18.12.4" style="padding-left:9.4pt;padding-right:9.4pt;">124.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T5.6.18.12.5" style="padding-left:9.4pt;padding-right:9.4pt;">151.1</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T5.6.18.12.6" style="padding-left:9.4pt;padding-right:9.4pt;">81.7</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T5.6.18.12.7" style="padding-left:9.4pt;padding-right:9.4pt;">73.5</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T5.6.18.12.8" style="padding-left:9.4pt;padding-right:9.4pt;">72.6</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 5:  Performance of specialist and generalist models on captioning and VQA tasks. | ✅ Table 5:  专家模型和通才模型在字幕和 VQA 任务中的表现。 |
| ✅ Specialist Models refer to those that are fine-tuned specifically for each task, while Generalist Models denote a single model fine-tuned in a task-agnostic manner, applicable across all tasks. | ✅ Specialist Models 指针对每个任务专门进行微调的模型，而 Generalist Models 表示以与任务无关的方式进行微调的单一模型，适用于所有任务。 |
| ✅ △ indicates usage of external OCR as input. | ✅ △ 表示使用外部 OCR 作为输入。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T6.2"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T6.2.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="S6.T6.2.1.1.1" rowspan="3" style="padding-left:3.3pt;padding-right:3.3pt;">Method</th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="S6.T6.2.1.1.2" rowspan="3" style="padding-left:3.3pt;padding-right:3.3pt;">#params</th><td class="ltx_td ltx_border_tt" id="S6.T6.2.1.1.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T6.2.1.1.4" style="padding-left:3.3pt;padding-right:3.3pt;">COCO Det.</td><td class="ltx_td ltx_border_tt" id="S6.T6.2.1.1.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T6.2.1.1.6" style="padding-left:3.3pt;padding-right:3.3pt;">Flickr30k</td><td class="ltx_td ltx_border_tt" id="S6.T6.2.1.1.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" colspan="3" id="S6.T6.2.1.1.8" style="padding-left:3.3pt;padding-right:3.3pt;">Refcoco</td><td class="ltx_td ltx_border_tt" id="S6.T6.2.1.1.9" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" colspan="3" id="S6.T6.2.1.1.10" style="padding-left:3.3pt;padding-right:3.3pt;">Refcoco+</td><td class="ltx_td ltx_border_tt" id="S6.T6.2.1.1.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" colspan="2" id="S6.T6.2.1.1.12" style="padding-left:3.3pt;padding-right:3.3pt;">Refcocog</td><td class="ltx_td ltx_border_tt" id="S6.T6.2.1.1.13" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T6.2.1.1.14" style="padding-left:3.3pt;padding-right:3.3pt;">Refcoco RES</td></tr><tr class="ltx_tr" id="S6.T6.2.2.2"><td class="ltx_td" id="S6.T6.2.2.2.1" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.2" style="padding-left:3.3pt;padding-right:3.3pt;">val2017</td><td class="ltx_td" id="S6.T6.2.2.2.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.4" style="padding-left:3.3pt;padding-right:3.3pt;">test</td><td class="ltx_td" id="S6.T6.2.2.2.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.6" style="padding-left:3.3pt;padding-right:3.3pt;">val</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.7" style="padding-left:3.3pt;padding-right:3.3pt;">test-A</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.8" style="padding-left:3.3pt;padding-right:3.3pt;">test-B</td><td class="ltx_td" id="S6.T6.2.2.2.9" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.10" style="padding-left:3.3pt;padding-right:3.3pt;">val</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.11" style="padding-left:3.3pt;padding-right:3.3pt;">test-A</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.12" style="padding-left:3.3pt;padding-right:3.3pt;">test-B</td><td class="ltx_td" id="S6.T6.2.2.2.13" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.14" style="padding-left:3.3pt;padding-right:3.3pt;">val</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.15" style="padding-left:3.3pt;padding-right:3.3pt;">test</td><td class="ltx_td" id="S6.T6.2.2.2.16" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.2.2.17" style="padding-left:3.3pt;padding-right:3.3pt;">val</td></tr><tr class="ltx_tr" id="S6.T6.2.3.3"><td class="ltx_td" id="S6.T6.2.3.3.1" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.3.3.2" style="padding-left:3.3pt;padding-right:3.3pt;">mAP</td><td class="ltx_td" id="S6.T6.2.3.3.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.3.3.4" style="padding-left:3.3pt;padding-right:3.3pt;">R@1</td><td class="ltx_td" id="S6.T6.2.3.3.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" colspan="3" id="S6.T6.2.3.3.6" style="padding-left:3.3pt;padding-right:3.3pt;">Accuracy</td><td class="ltx_td" id="S6.T6.2.3.3.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" colspan="3" id="S6.T6.2.3.3.8" style="padding-left:3.3pt;padding-right:3.3pt;">Accuracy</td><td class="ltx_td" id="S6.T6.2.3.3.9" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" colspan="2" id="S6.T6.2.3.3.10" style="padding-left:3.3pt;padding-right:3.3pt;">Accuracy</td><td class="ltx_td" id="S6.T6.2.3.3.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.3.3.12" style="padding-left:3.3pt;padding-right:3.3pt;">mIoU</td></tr><tr class="ltx_tr" id="S6.T6.2.4.4"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" colspan="19" id="S6.T6.2.4.4.1" style="padding-left:3.3pt;padding-right:3.3pt;">Specialist Models</th></tr><tr class="ltx_tr" id="S6.T6.2.5.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T6.2.5.5.1" style="padding-left:3.3pt;padding-right:3.3pt;">SeqTR <html><body><p>( <strong>Seqtr: A simple yet universal network for visual grounding.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T6.2.5.5.2" style="padding-left:3.3pt;padding-right:3.3pt;">-</th><td class="ltx_td ltx_border_t" id="S6.T6.2.5.5.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.4" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T6.2.5.5.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.6" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T6.2.5.5.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.8" style="padding-left:3.3pt;padding-right:3.3pt;">83.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.9" style="padding-left:3.3pt;padding-right:3.3pt;">86.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.10" style="padding-left:3.3pt;padding-right:3.3pt;">81.2</td><td class="ltx_td ltx_border_t" id="S6.T6.2.5.5.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.12" style="padding-left:3.3pt;padding-right:3.3pt;">71.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.13" style="padding-left:3.3pt;padding-right:3.3pt;">76.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.14" style="padding-left:3.3pt;padding-right:3.3pt;">64.9</td><td class="ltx_td ltx_border_t" id="S6.T6.2.5.5.15" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.16" style="padding-left:3.3pt;padding-right:3.3pt;">74.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.17" style="padding-left:3.3pt;padding-right:3.3pt;">74.2</td><td class="ltx_td ltx_border_t" id="S6.T6.2.5.5.18" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.5.5.19" style="padding-left:3.3pt;padding-right:3.3pt;">-</td></tr><tr class="ltx_tr" id="S6.T6.2.6.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.6.6.1" style="padding-left:3.3pt;padding-right:3.3pt;">PolyFormer <html><body><p>( <strong>Polyformer: Referring image segmentation as sequential polygongeneration.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.6.6.2" style="padding-left:3.3pt;padding-right:3.3pt;">-</th><td class="ltx_td" id="S6.T6.2.6.6.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.4" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td" id="S6.T6.2.6.6.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.6" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td" id="S6.T6.2.6.6.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.8" style="padding-left:3.3pt;padding-right:3.3pt;">90.4</td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.9" style="padding-left:3.3pt;padding-right:3.3pt;">92.9</td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.10" style="padding-left:3.3pt;padding-right:3.3pt;">87.2</td><td class="ltx_td" id="S6.T6.2.6.6.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.12" style="padding-left:3.3pt;padding-right:3.3pt;">85.0</td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.13" style="padding-left:3.3pt;padding-right:3.3pt;">89.8</td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.14" style="padding-left:3.3pt;padding-right:3.3pt;">78.0</td><td class="ltx_td" id="S6.T6.2.6.6.15" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.16" style="padding-left:3.3pt;padding-right:3.3pt;">85.8</td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.17" style="padding-left:3.3pt;padding-right:3.3pt;">85.9</td><td class="ltx_td" id="S6.T6.2.6.6.18" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.6.6.19" style="padding-left:3.3pt;padding-right:3.3pt;">76.9</td></tr><tr class="ltx_tr" id="S6.T6.2.7.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.7.7.1" style="padding-left:3.3pt;padding-right:3.3pt;">UNINEXT <html><body><p>( <strong>Universal instance perception as object discovery and retrieval.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.7.7.2" style="padding-left:3.3pt;padding-right:3.3pt;">0.74B</th><td class="ltx_td" id="S6.T6.2.7.7.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.4" style="padding-left:3.3pt;padding-right:3.3pt;">60.6</td><td class="ltx_td" id="S6.T6.2.7.7.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.6" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td" id="S6.T6.2.7.7.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.8" style="padding-left:3.3pt;padding-right:3.3pt;">92.6</td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.9" style="padding-left:3.3pt;padding-right:3.3pt;">94.3</td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.10" style="padding-left:3.3pt;padding-right:3.3pt;">91.5</td><td class="ltx_td" id="S6.T6.2.7.7.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.12" style="padding-left:3.3pt;padding-right:3.3pt;">85.2</td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.13" style="padding-left:3.3pt;padding-right:3.3pt;">89.6</td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.14" style="padding-left:3.3pt;padding-right:3.3pt;">79.8</td><td class="ltx_td" id="S6.T6.2.7.7.15" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.16" style="padding-left:3.3pt;padding-right:3.3pt;">88.7</td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.17" style="padding-left:3.3pt;padding-right:3.3pt;">89.4</td><td class="ltx_td" id="S6.T6.2.7.7.18" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.7.7.19" style="padding-left:3.3pt;padding-right:3.3pt;">-</td></tr><tr class="ltx_tr" id="S6.T6.2.8.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.8.8.1" style="padding-left:3.3pt;padding-right:3.3pt;">Ferret <html><body><p>( <strong>Ferret: Refer and ground anything anywhere at any granularity, 2023.</strong> )</p></body></html></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.8.8.2" style="padding-left:3.3pt;padding-right:3.3pt;">13B</th><td class="ltx_td" id="S6.T6.2.8.8.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.4" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td" id="S6.T6.2.8.8.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.6" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td" id="S6.T6.2.8.8.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.8" style="padding-left:3.3pt;padding-right:3.3pt;">89.5</td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.9" style="padding-left:3.3pt;padding-right:3.3pt;">92.4</td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.10" style="padding-left:3.3pt;padding-right:3.3pt;">84.4</td><td class="ltx_td" id="S6.T6.2.8.8.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.12" style="padding-left:3.3pt;padding-right:3.3pt;">82.8</td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.13" style="padding-left:3.3pt;padding-right:3.3pt;">88.1</td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.14" style="padding-left:3.3pt;padding-right:3.3pt;">75.2</td><td class="ltx_td" id="S6.T6.2.8.8.15" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.16" style="padding-left:3.3pt;padding-right:3.3pt;">85.8</td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.17" style="padding-left:3.3pt;padding-right:3.3pt;">86.3</td><td class="ltx_td" id="S6.T6.2.8.8.18" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.8.8.19" style="padding-left:3.3pt;padding-right:3.3pt;">-</td></tr><tr class="ltx_tr" id="S6.T6.2.9.9"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" colspan="19" id="S6.T6.2.9.9.1" style="padding-left:3.3pt;padding-right:3.3pt;">Generalist Models</th></tr><tr class="ltx_tr" id="S6.T6.2.10.10"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T6.2.10.10.1" style="padding-left:3.3pt;padding-right:3.3pt;">UniTAB <html><body><p>( <strong>Unitab: Unifying text and box outputs for grounded vision-languagemodeling.</strong> )</p></body></html></th><th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T6.2.10.10.2" style="padding-left:3.3pt;padding-right:3.3pt;"></th><td class="ltx_td ltx_border_t" id="S6.T6.2.10.10.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.4" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T6.2.10.10.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.6" style="padding-left:3.3pt;padding-right:3.3pt;">-</td><td class="ltx_td ltx_border_t" id="S6.T6.2.10.10.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.8" style="padding-left:3.3pt;padding-right:3.3pt;">88.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.9" style="padding-left:3.3pt;padding-right:3.3pt;">91.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.10" style="padding-left:3.3pt;padding-right:3.3pt;">83.8</td><td class="ltx_td ltx_border_t" id="S6.T6.2.10.10.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.12" style="padding-left:3.3pt;padding-right:3.3pt;">81.0</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.13" style="padding-left:3.3pt;padding-right:3.3pt;">85.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.14" style="padding-left:3.3pt;padding-right:3.3pt;">71.6</td><td class="ltx_td ltx_border_t" id="S6.T6.2.10.10.15" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.16" style="padding-left:3.3pt;padding-right:3.3pt;">84.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.17" style="padding-left:3.3pt;padding-right:3.3pt;">84.7</td><td class="ltx_td ltx_border_t" id="S6.T6.2.10.10.18" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T6.2.10.10.19" style="padding-left:3.3pt;padding-right:3.3pt;">-</td></tr><tr class="ltx_tr" id="S6.T6.2.11.11"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.11.11.1" style="padding-left:3.3pt;padding-right:3.3pt;"><em class="ltx_emph ltx_font_italic" id="S6.T6.2.11.11.1.1" style="font-size:90%;">Florence-2-B</em></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T6.2.11.11.2" style="padding-left:3.3pt;padding-right:3.3pt;">0.23B</th><td class="ltx_td" id="S6.T6.2.11.11.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.4" style="padding-left:3.3pt;padding-right:3.3pt;">41.4</td><td class="ltx_td" id="S6.T6.2.11.11.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.6" style="padding-left:3.3pt;padding-right:3.3pt;">84.0</td><td class="ltx_td" id="S6.T6.2.11.11.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.8" style="padding-left:3.3pt;padding-right:3.3pt;">92.6</td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.9" style="padding-left:3.3pt;padding-right:3.3pt;">94.8</td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.10" style="padding-left:3.3pt;padding-right:3.3pt;">91.5</td><td class="ltx_td" id="S6.T6.2.11.11.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.12" style="padding-left:3.3pt;padding-right:3.3pt;">86.8</td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.13" style="padding-left:3.3pt;padding-right:3.3pt;">91.7</td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.14" style="padding-left:3.3pt;padding-right:3.3pt;">82.2</td><td class="ltx_td" id="S6.T6.2.11.11.15" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.16" style="padding-left:3.3pt;padding-right:3.3pt;">89.8</td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.17" style="padding-left:3.3pt;padding-right:3.3pt;">82.2</td><td class="ltx_td" id="S6.T6.2.11.11.18" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center" id="S6.T6.2.11.11.19" style="padding-left:3.3pt;padding-right:3.3pt;">78.0</td></tr><tr class="ltx_tr" id="S6.T6.2.12.12"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T6.2.12.12.1" style="padding-left:3.3pt;padding-right:3.3pt;"><em class="ltx_emph ltx_font_italic" id="S6.T6.2.12.12.1.1" style="font-size:90%;">Florence-2-L</em></th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T6.2.12.12.2" style="padding-left:3.3pt;padding-right:3.3pt;">0.77B</th><td class="ltx_td ltx_border_bb" id="S6.T6.2.12.12.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.4" style="padding-left:3.3pt;padding-right:3.3pt;">43.4</td><td class="ltx_td ltx_border_bb" id="S6.T6.2.12.12.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.6" style="padding-left:3.3pt;padding-right:3.3pt;">85.2</td><td class="ltx_td ltx_border_bb" id="S6.T6.2.12.12.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.8" style="padding-left:3.3pt;padding-right:3.3pt;">93.4</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.9" style="padding-left:3.3pt;padding-right:3.3pt;">95.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.10" style="padding-left:3.3pt;padding-right:3.3pt;">92.0</td><td class="ltx_td ltx_border_bb" id="S6.T6.2.12.12.11" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.12" style="padding-left:3.3pt;padding-right:3.3pt;">88.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.13" style="padding-left:3.3pt;padding-right:3.3pt;">92.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.14" style="padding-left:3.3pt;padding-right:3.3pt;">83.6</td><td class="ltx_td ltx_border_bb" id="S6.T6.2.12.12.15" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.16" style="padding-left:3.3pt;padding-right:3.3pt;">91.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.17" style="padding-left:3.3pt;padding-right:3.3pt;">91.7</td><td class="ltx_td ltx_border_bb" id="S6.T6.2.12.12.18" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T6.2.12.12.19" style="padding-left:3.3pt;padding-right:3.3pt;">80.5</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 6:  Performance of specialist and generalist models on region-level tasks. | ✅ Table 6:  专家模型和通才模型在区域级任务上的表现。 |
| ✅ Specialist Models refer to those that are fine-tuned specifically for each task, while Generalist Models denote a single model fine-tuned in a task-agnostic manner, applicable across all tasks. | ✅ Specialist Models 指针对每个任务专门进行微调的模型，而 Generalist Models 表示以与任务无关的方式进行微调的单一模型，适用于所有任务。 |

| 【第6.3节，第1段】原文 | 【第6.3节，第1段】翻译 |
| ---- | ---- |
| ✅ We demonstrate the versatility and effectiveness of our model as a vision foundation that can be transferred to various downstream tasks. | ✅ 我们证明了我们的模型作为可转移到各种下游任务的视觉基础的多功能性和有效性。 |
| ✅ We fine-tune Florence-2 models by adding a collection of public datasets that cover image-level, region-level, pixel-level tasks, yielding one generalist model for various vision tasks. | ✅ 我们通过添加涵盖图像级、区域级、像素级任务的公共数据集集合来微调 Florence-2 模型，从而产生适用于各种视觉任务的 one 通用模型。 |
| ✅ The details of the dataset collection are provided in Table 14. | ✅ 数据集收集的详细信息在 Table 14 中提供。 |
| ✅ Tables 5 and 6 compare our model with other state-of-the-art models. | ✅ Tables 5 和 6 将我们的模型与其他最先进的模型进行比较。 |
| ✅ Our key findings are: | ✅ 我们的主要发现是： |

#### 6.3.1 Simple design for strong performance.

| 【第6.3.1节，第1段】原文 | 【第6.3.1节，第1段】翻译 |
| ---- | ---- |
| ✅ Florence-2 demonstrates strong performance with standard multi-modality Transformer encoder-decoder without special designs, particularly for region-level and pixel-level tasks. | ✅ Florence-2 使用 standard 多模态 Transformer 编码器-解码器展示了 strong 的性能，无需特殊设计，特别是对于区域级和像素级任务。 |
| ✅ For example, Florence-2-L outperforms PolyFormer ( **Polyformer: Referring image segmentation as sequential polygon generation.** ) on both RefCOCO REC task and RES task by 3.0 Accuracy@0.5 and 3.54 mIOU respectively, where PolyFormer ( **Polyformer: Referring image segmentation as sequential polygon generation.** ) adapts specifically designed regression-based prediction head for coordinates. | ✅ 例如，Florence-2-L 在 RefCOCO REC 任务和 RES 任务上的表现分别比 PolyFormer ( **Polyformer: Referring image segmentation as sequential polygon generation.** ) 高出 3.0 Accuracy@0.5 和 3.54 mIOU，其中 PolyFormer ( **Polyformer: Referring image segmentation as sequential polygon generation.** ) 采用专门设计的基于回归的坐标预测头。 |
| ✅ Florence-2-L also outperforms previous SOTA method UNINEXT ( **Universal instance perception as object discovery and retrieval.** ) on RefCOCO by 0.8 Accuracy@0.5, where UNINEXT ( **Universal instance perception as object discovery and retrieval.** ) is based on advanced object detector Deformable DETR ( **Deformable detr: Deformable transformers for end-to-end object detection.** ) and DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) . | ✅ Florence-2-L 在 RefCOCO 上的表现也比之前的 SOTA 方法 UNINEXT ( **Universal instance perception as object discovery and retrieval.** ) 高出 0.8 个准确度@0.5，其中 UNINEXT ( **Universal instance perception as object discovery and retrieval.** ) 基于先进的物体检测器 Deformable DETR ( **Deformable detr: Deformable transformers for end-to-end object detection.** ) 和 DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** )。 |

#### 6.3.2 Competitive performance with fewer parameters.

| 【第6.3.2节，第1段】原文 | 【第6.3.2节，第1段】翻译 |
| ---- | ---- |
| ✅ Florence-2-L achieves competitive performance without the need for LLMs, showcasing efficiency in handling diverse tasks while maintaining a compact size. | ✅ Florence-2-L 无需 LLM 即可实现具有竞争力的性能，展现出在保持紧凑尺寸的同时处理各种任务的效率。 |
| ✅ For instance, Florence-2-L attains a CIDEr score of 140.0 on the COCO Caption karpathy test split ( **Deep visual-semantic alignments for generating image descriptions.** ) , outperforming models with significantly more parameters, such as Flamingo (80B parameters, 138.1 CIDEr score). | ✅ 例如，Florence-2-L 在 COCO Caption karpathy 测试分割 ( **Deep visual-semantic alignments for generating image descriptions.** ) 上获得了 140.0 的 CIDEr 分数，其表现优于具有更多参数的模型，例如 Flamingo（80B 参数，138.1 CIDEr 分数）。 |

#### 6.3.3 Adaptable generalization across task levels.

| 【第6.3.3节，第1段】原文 | 【第6.3.3节，第1段】翻译 |
| ---- | ---- |
| ✅ Florence-2 demonstrates competitive performance across image-level, pixel-level, and region-level tasks, emphasizing its adaptability and effectiveness in addressing various challenges in computer vision and natural language processing. | ✅ Florence-2 在图像级、像素级和区域级任务中展现出极具竞争力的性能，凸显了其在解决计算机视觉和自然语言处理中的各种挑战方面的适应性和有效性。 |
| ✅ For example, in the TextVQA task, Florence-2-L sets a new state-of-the-art performance with an accuracy of 81.5 without any external OCR token input, surpassing previous SOTA methods ( **1. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **2. Pali-x: On scaling up a multilingual vision and language model.** ) . | ✅ 例如，在 TextVQA 任务中，Florence-2-L 在没有任何外部 OCR token 输入的情况下，以 81.5 的准确率创下了新的 SOTA 性能，超越了之前的 SOTA 方法 ( **1. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **2. Pali-x: On scaling up a multilingual vision and language model.** )。 |

| 【第6.3.3节，第2段】原文 | 【第6.3.3节，第2段】翻译 |
| ---- | ---- |
| ✅ These achievements emphasize Florence-2 ’s efficiency in handling diverse tasks while maintaining a compact size, making it a unique and valuable asset in the ever-evolving landscape of AI research and applications. | ✅ 这些成就凸显了Florence-2在保持紧凑尺寸的同时处理多样化任务的效率，使其成为不断发展的人工智能研究和应用领域中独特而宝贵的资产。 |

### 6.4 Downstream Tasks Fine-tuning

| 【第6.4节，第1段】原文 | 【第6.4节，第1段】翻译 |
| ---- | ---- |
| ✅ In this section, we investigate the performance of our single model fine-tuning on downstream tasks. | ✅ 在本节中，我们研究单一模型微调在下游任务上的性能。 |
| ✅ This experiment highlights the superiority of Florence-2 pre-training over previous approaches, as it demonstrates the effectiveness of the learned universal image representation. | ✅ 该实验凸显了 Florence-2 预训练相对于以前方法的优越性，因为它证明了所学习的通用图像表示的有效性。 |
| ✅ We use the base size model with about 80M parameters in our experiments to ensure fair comparison with other methods. | ✅ 我们在实验中使用具有约 80M 个参数的基本尺寸模型，以确保与其他方法进行公平比较。 |

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x9.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ (a)  Mask-RCNN on COCO detection. | ✅ (a)  Mask-RCNN on COCO detection. |

#### 6.4.1 Object detection and segmentation.

| 【第6.4.1节，第1段】原文 | 【第6.4.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We conduct COCO object detection and instance segmentation ( **Microsoft coco: Common objects in context.** ) experiments with Mask R-CNN ( **Mask r-cnn.** ) , and COCO object detection ( **Microsoft coco: Common objects in context.** ) experiments with DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) to further demonstrate the effectiveness of Florence-2 pre-training. | ✅ 我们利用 Mask R-CNN ( **Mask r-cnn.** ) 进行 COCO 对象检测和实例分割 ( **Microsoft coco: Common objects in context.** ) 实验，利用 DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) 进行 COCO 对象检测 ( **Microsoft coco: Common objects in context.** ) 实验，进一步证明 Florence-2 预训练的有效性。 |
| ✅ We train on the train2017 split and evaluate on the val2017 split. | ✅ 我们在 train2017 分割上进行训练，并在 val2017 分割上进行评估。 |

| 【第6.4.1节，第2段】原文 | 【第6.4.1节，第2段】翻译 |
| ---- | ---- |
| ✅ For Mask R-CNN ( **Mask r-cnn.** ) experiments, we follow the common setup used in ( **1. Swin transformer: Hierarchical vision transformer using shifted windows, 2021.** ｜ **2. Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) , we use the standard 1  $\times$  (12 epochs) schedule with multi-scale training for all experiments. | ✅ 对于 Mask R-CNN ( **Mask r-cnn.** ) 实验，我们遵循 ( **1. Swin transformer: Hierarchical vision transformer using shifted windows, 2021.** ｜ **2. Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) 中使用的通用设置，我们对所有实验使用标准 1  $\times$ （12 个时期）计划和多尺度训练。 |
| ✅ The learning rate is stepped down by a factor of 0.1 at the 67% and 89% of training epochs. | ✅ 在训练阶段的 67% 和 89% 时，学习率降低了 0.1 倍。 |
| ✅ We do not use any additional augmentation (such as random crop, mosaic, etc) or optimization techniques (such as EMA, weight normalization) during training to ensure a fair comparison. | ✅ 为了确保公平比较，我们在训练期间不使用任何额外的增强（如随机裁剪、马赛克等）或优化技术（如 EMA、权重标准化）。 |
| ✅ We do not use any test time augmentation (TTA) either. | ✅ 我们也不使用任何测试时间增强（TTA）。 |
| ✅ Thanks to the strong universal representation learned by Florence-2 pre-training, we do not require longer training epochs, such as 36 epochs in ( **1. Convnext v2: Co-designing and scaling convnets with masked autoencoders.** ｜ **2. Swin transformer: Hierarchical vision transformer using shifted windows, 2021.** ｜ **3. Focal self-attention for local-global interactions in vision transformers.** ｜ **4. Focal modulation networks.** ) , or 100 epochs in ( **Exploring plain vision transformer backbones for object detection.** ) , to achieve better results. | ✅ 得益于 Florence-2 预训练学习到的强大的通用表征，我们不需要更长的训练周期（例如 ( **1. Convnext v2: Co-designing and scaling convnets with masked autoencoders.** ｜ **2. Swin transformer: Hierarchical vision transformer using shifted windows, 2021.** ｜ **3. Focal self-attention for local-global interactions in vision transformers.** ｜ **4. Focal modulation networks.** ) 中的 36 个周期或 ( **Exploring plain vision transformer backbones for object detection.** ) 中的 100 个周期）即可获得更好的结果。 |

| 【第6.4.1节，第3段】原文 | 【第6.4.1节，第3段】翻译 |
| ---- | ---- |
| ✅ For DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) experiments, we train DINO-4scale ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) detector for 12 epochs (1  $\times$  ) using the same data augmentation strategy as employed by ( **End-to-end object detection with transformers.** ) . | ✅ 对于 DINO ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) 实验，我们使用与 ( **End-to-end object detection with transformers.** ) 相同的数据增强策略，对 DINO-4scale ( **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.** ) 检测器进行 12 个时期（1  $\times$ ）的训练。 |

| 【第6.4.1节，第4段】原文 | 【第6.4.1节，第4段】翻译 |
| ---- | ---- |
| ✅ First, our base model achieves a strong performance improvement compared to other approaches. | ✅ 首先，与其他方法相比，我们的基础模型实现了显著的性能提升。 |
| ✅ As shown in Table 7 , our DaViT-B model pre-trained by Florence-2 surpasses previous best base model (ConvNext v2-B), which is pre-trained by FCMAE ( **Convnext v2: Co-designing and scaling convnets with masked autoencoders.** ) , by 0.7  $AP_{b}$  using Mask RCNN. | ✅ 如 Table 7 所示，我们通过 Florence-2 预训练的 DaViT-B 模型比之前使用 FCMAE ( **Convnext v2: Co-designing and scaling convnets with masked autoencoders.** ) 预训练的最佳基础模型（ConvNext v2-B）高出 0.7 个  $AP_{b}$ （使用 Mask RCNN）。 |
| ✅ Importantly, while ConvNeXt v2-B leverages a 3  $\times$  schedule (36 epochs), our model efficiently employs a 1  $\times$  schedule (12 epochs) thanks to our powerful pre-trained universal representation. | ✅ 重要的是，虽然 ConvNeXt v2-B 利用了 3  $\times$  计划（36 个时期），但由于我们强大的预训练通用表示，我们的模型有效地采用了 1  $\times$  计划（12 个时期）。 |
| ✅ For DINO framework, our model significantly outperforms the ViT-B, achieving a notable improvement of 4.2 AP. | ✅ 对于DINO框架，我们的模型明显优于ViT-B，实现了4.2 AP的显著提升。 |

| 【第6.4.1节，第5段】原文 | 【第6.4.1节，第5段】翻译 |
| ---- | ---- |
| ✅ Second, our pre-training demonstrates higher training efficiency. | ✅ 其次，我们的预训练表现出更高的训练效率。 |
| ✅ As shown in Table 8 and Figure 6 , compared to the model with supervised ImageNet-1k pre-training, our model with Florence-2 pre-training achieves 4x efficiency and a significant improvement of 6.9 AP and 5.5 AP with Mask-RCNN and DINO framework, respectively. | ✅ 如Table 8和Figure 6所示，与有监督的ImageNet-1k预训练的模型相比，我们采用Florence-2预训练的模型在使用Mask-RCNN和DINO框架时分别实现了4倍的效率和6.9 AP和5.5 AP的显著提升。 |

| 【第6.4.1节，第6段】原文 | 【第6.4.1节，第6段】翻译 |
| ---- | ---- |
| ✅ Third, our pre-training provides a good generic representation without extensive fine-tuning. | ✅ 第三，我们的预训练提供了良好的通用表示，无需进行大量的微调。 |
| ✅ Table 8 indicates that the models with Florence-2 pre-training maintains competitive performances when the first two stages are frozen with only 0.3 and 0.2 drops for Mask-RCNN and DINO, respectively. | ✅ Table 8 表示当前两个阶段冻结时，使用 Florence-2 预训练的模型保持了有竞争力的性能，对于 Mask-RCNN 和 DINO，分别只有 0.3 和 0.2 的下降。 |
| ✅ Moreover, our approach with completely frozen backbone can outperform the model with supervised ImageNet-1k pre-training by 1.6 and 2.4 for Mask-RCNN and DINO. | ✅ 此外，对于 Mask-RCNN 和 DINO，我们采用完全冻结主干的方法可以比使用监督 ImageNet-1k 预训练的模型表现更好 1.6 和 2.4。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T7.2"><thead class="ltx_thead"><tr class="ltx_tr" id="S6.T7.2.3.1"><th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T7.2.3.1.1" style="padding-left:3.0pt;padding-right:3.0pt;"></th><th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T7.2.3.1.2" style="padding-left:3.0pt;padding-right:3.0pt;"></th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T7.2.3.1.3" style="padding-left:3.0pt;padding-right:3.0pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="2" id="S6.T7.2.3.1.4" style="padding-left:3.0pt;padding-right:3.0pt;">Mask R-CNN</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T7.2.3.1.5" style="padding-left:3.0pt;padding-right:3.0pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T7.2.3.1.6" style="padding-left:3.0pt;padding-right:3.0pt;">DINO</th></tr><tr class="ltx_tr" id="S6.T7.2.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r" id="S6.T7.2.2.3" style="padding-left:3.0pt;padding-right:3.0pt;">Backbone</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r" id="S6.T7.2.2.4" style="padding-left:3.0pt;padding-right:3.0pt;">Pretrain</th><th class="ltx_td ltx_th ltx_th_column" id="S6.T7.2.2.5" style="padding-left:3.0pt;padding-right:3.0pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T7.1.1.1" style="padding-left:3.0pt;padding-right:3.0pt;">AP<sub class="ltx_sub" id="S6.T7.1.1.1.2">b</sub></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T7.2.2.2" style="padding-left:3.0pt;padding-right:3.0pt;">AP<sub class="ltx_sub" id="S6.T7.2.2.2.2">m</sub></th><th class="ltx_td ltx_th ltx_th_column" id="S6.T7.2.2.6" style="padding-left:3.0pt;padding-right:3.0pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T7.2.2.7" style="padding-left:3.0pt;padding-right:3.0pt;">AP</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T7.2.4.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T7.2.4.1.1" style="padding-left:3.0pt;padding-right:3.0pt;">ViT-B <html><body><p>( <strong>Exploring plain vision transformer backbones for object detection.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T7.2.4.1.2" style="padding-left:3.0pt;padding-right:3.0pt;">MAE, IN-1k</th><td class="ltx_td ltx_border_t" id="S6.T7.2.4.1.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T7.2.4.1.4" style="padding-left:3.0pt;padding-right:3.0pt;">51.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T7.2.4.1.5" style="padding-left:3.0pt;padding-right:3.0pt;">45.9</td><td class="ltx_td ltx_border_t" id="S6.T7.2.4.1.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T7.2.4.1.7" style="padding-left:3.0pt;padding-right:3.0pt;">55.0</td></tr><tr class="ltx_tr" id="S6.T7.2.5.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.5.2.1" style="padding-left:3.0pt;padding-right:3.0pt;">Swin-B <html><body><p>( <strong>Swin transformer: Hierarchical vision transformer using shiftedwindows, 2021.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.5.2.2" style="padding-left:3.0pt;padding-right:3.0pt;">Sup IN-1k</th><td class="ltx_td" id="S6.T7.2.5.2.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.5.2.4" style="padding-left:3.0pt;padding-right:3.0pt;">50.2</td><td class="ltx_td ltx_align_center" id="S6.T7.2.5.2.5" style="padding-left:3.0pt;padding-right:3.0pt;">-</td><td class="ltx_td" id="S6.T7.2.5.2.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.5.2.7" style="padding-left:3.0pt;padding-right:3.0pt;">53.4</td></tr><tr class="ltx_tr" id="S6.T7.2.6.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.6.3.1" style="padding-left:3.0pt;padding-right:3.0pt;">Swin-B <html><body><p>( <strong>Swin transformer: Hierarchical vision transformer using shiftedwindows, 2021.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.6.3.2" style="padding-left:3.0pt;padding-right:3.0pt;">SimMIM <html><body><p>( <strong>Simmim: A simple framework for masked image modeling.</strong> )</p></body></html></th><td class="ltx_td" id="S6.T7.2.6.3.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.6.3.4" style="padding-left:3.0pt;padding-right:3.0pt;">52.3</td><td class="ltx_td ltx_align_center" id="S6.T7.2.6.3.5" style="padding-left:3.0pt;padding-right:3.0pt;">-</td><td class="ltx_td" id="S6.T7.2.6.3.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.6.3.7" style="padding-left:3.0pt;padding-right:3.0pt;">-</td></tr><tr class="ltx_tr" id="S6.T7.2.7.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.7.4.1" style="padding-left:3.0pt;padding-right:3.0pt;">FocalAtt-B <html><body><p>( <strong>Focal self-attention for local-global interactions in visiontransformers.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.7.4.2" style="padding-left:3.0pt;padding-right:3.0pt;">Sup IN-1k</th><td class="ltx_td" id="S6.T7.2.7.4.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.7.4.4" style="padding-left:3.0pt;padding-right:3.0pt;">49.0</td><td class="ltx_td ltx_align_center" id="S6.T7.2.7.4.5" style="padding-left:3.0pt;padding-right:3.0pt;">43.7</td><td class="ltx_td" id="S6.T7.2.7.4.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.7.4.7" style="padding-left:3.0pt;padding-right:3.0pt;">-</td></tr><tr class="ltx_tr" id="S6.T7.2.8.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.8.5.1" style="padding-left:3.0pt;padding-right:3.0pt;">FocalNet-B <html><body><p>( <strong>Focal modulation networks.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.8.5.2" style="padding-left:3.0pt;padding-right:3.0pt;">Sup IN-1k</th><td class="ltx_td" id="S6.T7.2.8.5.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.8.5.4" style="padding-left:3.0pt;padding-right:3.0pt;">49.8</td><td class="ltx_td ltx_align_center" id="S6.T7.2.8.5.5" style="padding-left:3.0pt;padding-right:3.0pt;">44.1</td><td class="ltx_td" id="S6.T7.2.8.5.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.8.5.7" style="padding-left:3.0pt;padding-right:3.0pt;">54.4</td></tr><tr class="ltx_tr" id="S6.T7.2.9.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.9.6.1" style="padding-left:3.0pt;padding-right:3.0pt;">ConvNeXt v1-B <html><body><p>( <strong>A convnet for the 2020s.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.9.6.2" style="padding-left:3.0pt;padding-right:3.0pt;">Sup IN-1k</th><td class="ltx_td" id="S6.T7.2.9.6.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.9.6.4" style="padding-left:3.0pt;padding-right:3.0pt;">50.3</td><td class="ltx_td ltx_align_center" id="S6.T7.2.9.6.5" style="padding-left:3.0pt;padding-right:3.0pt;">44.9</td><td class="ltx_td" id="S6.T7.2.9.6.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.9.6.7" style="padding-left:3.0pt;padding-right:3.0pt;">52.6</td></tr><tr class="ltx_tr" id="S6.T7.2.10.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.10.7.1" style="padding-left:3.0pt;padding-right:3.0pt;">ConvNeXt v2-B <html><body><p>( <strong>Convnext v2: Co-designing and scaling convnets with maskedautoencoders.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.10.7.2" style="padding-left:3.0pt;padding-right:3.0pt;">Sup IN-1k</th><td class="ltx_td" id="S6.T7.2.10.7.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.10.7.4" style="padding-left:3.0pt;padding-right:3.0pt;">51.0</td><td class="ltx_td ltx_align_center" id="S6.T7.2.10.7.5" style="padding-left:3.0pt;padding-right:3.0pt;">45.6</td><td class="ltx_td" id="S6.T7.2.10.7.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.10.7.7" style="padding-left:3.0pt;padding-right:3.0pt;">-</td></tr><tr class="ltx_tr" id="S6.T7.2.11.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.11.8.1" style="padding-left:3.0pt;padding-right:3.0pt;">ConvNeXt v2-B <html><body><p>( <strong>Convnext v2: Co-designing and scaling convnets with maskedautoencoders.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T7.2.11.8.2" style="padding-left:3.0pt;padding-right:3.0pt;">FCMAE</th><td class="ltx_td" id="S6.T7.2.11.8.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.11.8.4" style="padding-left:3.0pt;padding-right:3.0pt;">52.9</td><td class="ltx_td ltx_align_center" id="S6.T7.2.11.8.5" style="padding-left:3.0pt;padding-right:3.0pt;">46.6</td><td class="ltx_td" id="S6.T7.2.11.8.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center" id="S6.T7.2.11.8.7" style="padding-left:3.0pt;padding-right:3.0pt;">-</td></tr><tr class="ltx_tr" id="S6.T7.2.12.9" style="background-color:#E6E6E6;"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T7.2.12.9.1" style="padding-left:3.0pt;padding-right:3.0pt;">DaViT-B <html><body><p>( <strong>Davit: Dual attention vision transformers.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T7.2.12.9.2" style="padding-left:3.0pt;padding-right:3.0pt;"><em class="ltx_emph ltx_font_italic" id="S6.T7.2.12.9.2.1" style="font-size:90%;background-color:#E6E6E6;">Florence-2</em></th><td class="ltx_td ltx_border_bb" id="S6.T7.2.12.9.3" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T7.2.12.9.4" style="padding-left:3.0pt;padding-right:3.0pt;">53.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T7.2.12.9.5" style="padding-left:3.0pt;padding-right:3.0pt;">46.4</td><td class="ltx_td ltx_border_bb" id="S6.T7.2.12.9.6" style="padding-left:3.0pt;padding-right:3.0pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T7.2.12.9.7" style="padding-left:3.0pt;padding-right:3.0pt;">59.2</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 7:  COCO object detection and instance segmentation results using Mask-RCNN framework, and COCO object detection results using DINO-4scale framework. | ✅ Table 7: 、COCO object detection and instance segmentation results采用Mask-RCNN框架，COCO object detection results采用DINO-4scale框架。 |
| ✅ All the entries use a base size model to ensure a fair comparison. | ✅ 所有参赛作品均采用基本尺寸模型，以确保公平比较。 |
| ✅ For Mask-RCNN experiments, our method utilizes 1  $\times$  schedule (12 epochs), ViT-B use 100 epochs, all others use 3  $\times$  (36 epochs). | ✅ 对于 Mask-RCNN 实验，我们的方法采用 1  $\times$  计划（12 个 epoch），ViT-B 使用 100 个 epoch，其他所有方法均使用 3  $\times$ （36 个 epoch）。 |
| ✅ For DINO experiments, all the entries use 1  $\times$  schedule except for ViT-B which uses 50 epochs. | ✅ 对于 DINO 实验，除 ViT-B 使用 50 个时期外，所有条目都使用 1  $\times$  计划。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T8.2"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T8.2.3.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt" id="S6.T8.2.3.1.1" rowspan="2" style="padding-left:2.5pt;padding-right:2.5pt;">Pretrain</th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="S6.T8.2.3.1.2" rowspan="2" style="padding-left:2.5pt;padding-right:2.5pt;">Frozen stages</th><td class="ltx_td ltx_border_tt" id="S6.T8.2.3.1.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" colspan="2" id="S6.T8.2.3.1.4" style="padding-left:2.5pt;padding-right:2.5pt;">Mask R-CNN</td><td class="ltx_td ltx_border_tt" id="S6.T8.2.3.1.5" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T8.2.3.1.6" style="padding-left:2.5pt;padding-right:2.5pt;">DINO</td><td class="ltx_td ltx_border_tt" id="S6.T8.2.3.1.7" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T8.2.3.1.8" style="padding-left:2.5pt;padding-right:2.5pt;">UperNet</td></tr><tr class="ltx_tr" id="S6.T8.2.2"><td class="ltx_td" id="S6.T8.2.2.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.1.1.1" style="padding-left:2.5pt;padding-right:2.5pt;">AP<sub class="ltx_sub" id="S6.T8.1.1.1.2">b</sub></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.2.2" style="padding-left:2.5pt;padding-right:2.5pt;">AP<sub class="ltx_sub" id="S6.T8.2.2.2.2">m</sub></td><td class="ltx_td" id="S6.T8.2.2.4" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.2.5" style="padding-left:2.5pt;padding-right:2.5pt;">AP</td><td class="ltx_td" id="S6.T8.2.2.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.2.7" style="padding-left:2.5pt;padding-right:2.5pt;">mIoU</td></tr><tr class="ltx_tr" id="S6.T8.2.4.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t" id="S6.T8.2.4.2.1" style="padding-left:2.5pt;padding-right:2.5pt;">Sup IN1k</th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T8.2.4.2.2" style="padding-left:2.5pt;padding-right:2.5pt;">n/a</th><td class="ltx_td ltx_border_t" id="S6.T8.2.4.2.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.4.2.4" style="padding-left:2.5pt;padding-right:2.5pt;">46.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.4.2.5" style="padding-left:2.5pt;padding-right:2.5pt;">42.0</td><td class="ltx_td ltx_border_t" id="S6.T8.2.4.2.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.4.2.7" style="padding-left:2.5pt;padding-right:2.5pt;">53.7</td><td class="ltx_td ltx_border_t" id="S6.T8.2.4.2.8" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.4.2.9" style="padding-left:2.5pt;padding-right:2.5pt;">49</td></tr><tr class="ltx_tr" id="S6.T8.2.5.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row" id="S6.T8.2.5.3.1" style="padding-left:2.5pt;padding-right:2.5pt;">UniCL <html><body><p>( <strong>Unified contrastive learning in image-text-label space, 2022.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T8.2.5.3.2" style="padding-left:2.5pt;padding-right:2.5pt;">n/a</th><td class="ltx_td" id="S6.T8.2.5.3.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.5.3.4" style="padding-left:2.5pt;padding-right:2.5pt;">50.4</td><td class="ltx_td ltx_align_center" id="S6.T8.2.5.3.5" style="padding-left:2.5pt;padding-right:2.5pt;">45.0</td><td class="ltx_td" id="S6.T8.2.5.3.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.5.3.7" style="padding-left:2.5pt;padding-right:2.5pt;">57.3</td><td class="ltx_td" id="S6.T8.2.5.3.8" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.5.3.9" style="padding-left:2.5pt;padding-right:2.5pt;">53.6</td></tr><tr class="ltx_tr" id="S6.T8.2.6.4" style="background-color:#E6E6E6;"><th class="ltx_td ltx_align_left ltx_th ltx_th_row" id="S6.T8.2.6.4.1" style="padding-left:2.5pt;padding-right:2.5pt;"><em class="ltx_emph ltx_font_italic" id="S6.T8.2.6.4.1.1" style="font-size:90%;background-color:#E6E6E6;">Florence-2</em></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T8.2.6.4.2" style="padding-left:2.5pt;padding-right:2.5pt;">n/a</th><td class="ltx_td" id="S6.T8.2.6.4.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.6.4.4" style="padding-left:2.5pt;padding-right:2.5pt;">53.6</td><td class="ltx_td ltx_align_center" id="S6.T8.2.6.4.5" style="padding-left:2.5pt;padding-right:2.5pt;">46.4</td><td class="ltx_td" id="S6.T8.2.6.4.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.6.4.7" style="padding-left:2.5pt;padding-right:2.5pt;">59.2</td><td class="ltx_td" id="S6.T8.2.6.4.8" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.6.4.9" style="padding-left:2.5pt;padding-right:2.5pt;">54.9</td></tr><tr class="ltx_tr" id="S6.T8.2.7.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t" id="S6.T8.2.7.5.1" style="padding-left:2.5pt;padding-right:2.5pt;"><em class="ltx_emph ltx_font_italic" id="S6.T8.2.7.5.1.1" style="font-size:90%;">Florence-2</em></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T8.2.7.5.2" style="padding-left:2.5pt;padding-right:2.5pt;">[1]</th><td class="ltx_td ltx_border_t" id="S6.T8.2.7.5.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.7.5.4" style="padding-left:2.5pt;padding-right:2.5pt;">53.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.7.5.5" style="padding-left:2.5pt;padding-right:2.5pt;">46.3</td><td class="ltx_td ltx_border_t" id="S6.T8.2.7.5.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.7.5.7" style="padding-left:2.5pt;padding-right:2.5pt;">59.2</td><td class="ltx_td ltx_border_t" id="S6.T8.2.7.5.8" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T8.2.7.5.9" style="padding-left:2.5pt;padding-right:2.5pt;">54.1</td></tr><tr class="ltx_tr" id="S6.T8.2.8.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row" id="S6.T8.2.8.6.1" style="padding-left:2.5pt;padding-right:2.5pt;"><em class="ltx_emph ltx_font_italic" id="S6.T8.2.8.6.1.1" style="font-size:90%;">Florence-2</em></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T8.2.8.6.2" style="padding-left:2.5pt;padding-right:2.5pt;">[1, 2]</th><td class="ltx_td" id="S6.T8.2.8.6.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.8.6.4" style="padding-left:2.5pt;padding-right:2.5pt;">53.3</td><td class="ltx_td ltx_align_center" id="S6.T8.2.8.6.5" style="padding-left:2.5pt;padding-right:2.5pt;">46.1</td><td class="ltx_td" id="S6.T8.2.8.6.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.8.6.7" style="padding-left:2.5pt;padding-right:2.5pt;">59.0</td><td class="ltx_td" id="S6.T8.2.8.6.8" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.8.6.9" style="padding-left:2.5pt;padding-right:2.5pt;">54.4</td></tr><tr class="ltx_tr" id="S6.T8.2.9.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row" id="S6.T8.2.9.7.1" style="padding-left:2.5pt;padding-right:2.5pt;"><em class="ltx_emph ltx_font_italic" id="S6.T8.2.9.7.1.1" style="font-size:90%;">Florence-2</em></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T8.2.9.7.2" style="padding-left:2.5pt;padding-right:2.5pt;">[1, 2, 3]</th><td class="ltx_td" id="S6.T8.2.9.7.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.9.7.4" style="padding-left:2.5pt;padding-right:2.5pt;">49.5</td><td class="ltx_td ltx_align_center" id="S6.T8.2.9.7.5" style="padding-left:2.5pt;padding-right:2.5pt;">42.9</td><td class="ltx_td" id="S6.T8.2.9.7.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.9.7.7" style="padding-left:2.5pt;padding-right:2.5pt;">56.7</td><td class="ltx_td" id="S6.T8.2.9.7.8" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center" id="S6.T8.2.9.7.9" style="padding-left:2.5pt;padding-right:2.5pt;">49.6</td></tr><tr class="ltx_tr" id="S6.T8.2.10.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb" id="S6.T8.2.10.8.1" style="padding-left:2.5pt;padding-right:2.5pt;"><em class="ltx_emph ltx_font_italic" id="S6.T8.2.10.8.1.1" style="font-size:90%;">Florence-2</em></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T8.2.10.8.2" style="padding-left:2.5pt;padding-right:2.5pt;">[1, 2, 3, 4]</th><td class="ltx_td ltx_border_bb" id="S6.T8.2.10.8.3" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T8.2.10.8.4" style="padding-left:2.5pt;padding-right:2.5pt;">48.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T8.2.10.8.5" style="padding-left:2.5pt;padding-right:2.5pt;">44.5</td><td class="ltx_td ltx_border_bb" id="S6.T8.2.10.8.6" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T8.2.10.8.7" style="padding-left:2.5pt;padding-right:2.5pt;">56.1</td><td class="ltx_td ltx_border_bb" id="S6.T8.2.10.8.8" style="padding-left:2.5pt;padding-right:2.5pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T8.2.10.8.9" style="padding-left:2.5pt;padding-right:2.5pt;">45.9</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 8:  Downstream task fine-tuning on COCO and ADE20K dataset. | ✅ Table 8:  在 COCO 和 ADE20K 数据集上进行下游任务微调。 |
| ✅ COCO object detection using Mask R-CNN and DINO. | ✅ COCO object detection 使用 Mask R-CNN 和 DINO。 |
| ✅ ADE20K semantic segmentation using UperNet. | ✅ ADE20K semantic segmentation 使用 UpperNet。 |
| ✅ All entries use DaViT-B with 80M parameters as the backbone and standard 1  $\times$  schedule. | ✅ 所有参赛作品均采用具有 80M 参数的 DaViT-B 作为主干和标准 1  $\times$  时间表。 |

#### 6.4.2 Semantic segmentation.

| 【第6.4.2节，第1段】原文 | 【第6.4.2节，第1段】翻译 |
| ---- | ---- |
| ✅ We conduct semantic segmentation experiments with UperNet ( **Unified perceptual parsing for scene understanding.** ) framework on ADE20k ( **Scene parsing through ade20k dataset.** ) dataset. | ✅ 我们使用UperNet ( **Unified perceptual parsing for scene understanding.** )框架在ADE20k ( **Scene parsing through ade20k dataset.** )数据集上进行语义分割实验。 |
| ✅ We mostly follow the training and evaluation protocols from Swin ( **Swin transformer: Hierarchical vision transformer using shifted windows, 2021.** ). | ✅ 我们主要遵循 Swin ( **Swin transformer: Hierarchical vision transformer using shifted windows, 2021.** ) 的培训和评估协议。 |
| ✅ Specifically, we use input size 512  $\times$  512 and train the model for 40k iterations with a batch size of 64. | ✅ 具体来说，我们使用输入大小 512  $\times$  512 并以批量大小 64 对模型进行 40k 次迭代训练。 |
| ✅ We adopt the AdamW ( **Decoupled weight decay regularization, 2019.** ) optimizer with the optimal learning rate searched from {8e-4,4e-4,2e-4,1e-4}. | ✅ 我们采用 AdamW ( **Decoupled weight decay regularization, 2019.** ) 优化器，最佳学习率从 {8e-4,4e-4,2e-4,1e-4} 中搜索。 |

| 【第6.4.2节，第2段】原文 | 【第6.4.2节，第2段】翻译 |
| ---- | ---- |
| ✅ Our results show a similar trend to the object detection experiments. | ✅ 我们的结果显示出与物体检测实验相似的趋势。 |
| ✅ As illustrated in Table 9 , our base model outperforms the previous SoTA model, which is BEiT pre-trained ViT-B ( **BEiT: BERT pre-training of image transformers.** ) , by 1.3 and 1.4 points in single-scale and multi-scale testing protocol, respectively. | ✅ 如 Table 9 所示，我们的基础模型在单尺度和多尺度测试协议中分别比之前的 SoTA 模型（即 BEiT 预训练的 ViT-B ( **BEiT: BERT pre-training of image transformers.** )）高出 1.3 和 1.4 个点。 |
| ✅ With the same backbone architecture of DaViT-B ( **Davit: Dual attention vision transformers.** ) , Florence-2 pre-trained model achieves a remarkable improvement of 4.9 points and 4  $\times$  efficiency compared to the ImageNet-1k pre-trained counterpart as demonstrated in Tables 8 and 6 . | ✅ 使用与 DaViT-B ( **Davit: Dual attention vision transformers.** ) 相同的主干架构，Florence-2 预训练模型与 ImageNet-1k 预训练模型相比，实现了 4.9 分和 4  $\times$  效率的显著提升，如 Tables 8 和 6 所示。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T9.6"><thead class="ltx_thead"><tr class="ltx_tr" id="S6.T9.6.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T9.6.1.1.1" style="padding-left:5.5pt;padding-right:5.5pt;">Backbone</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T9.6.1.1.2" style="padding-left:5.5pt;padding-right:5.5pt;">Pretrain</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T9.6.1.1.3" style="padding-left:5.5pt;padding-right:5.5pt;">mIoU</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T9.6.1.1.4" style="padding-left:5.5pt;padding-right:5.5pt;">ms-mIoU</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T9.6.2.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T9.6.2.1.1" style="padding-left:5.5pt;padding-right:5.5pt;">ViT-B <html><body><p>( <strong>Masked autoencoders are scalable vision learners.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T9.6.2.1.2" style="padding-left:5.5pt;padding-right:5.5pt;">Sup IN-1k</th><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T9.6.2.1.3" style="padding-left:5.5pt;padding-right:5.5pt;">47.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T9.6.2.1.4" style="padding-left:5.5pt;padding-right:5.5pt;">-</td></tr><tr class="ltx_tr" id="S6.T9.6.3.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.3.2.1" style="padding-left:5.5pt;padding-right:5.5pt;">ViT-B <html><body><p>( <strong>Masked autoencoders are scalable vision learners.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.3.2.2" style="padding-left:5.5pt;padding-right:5.5pt;">MAE IN-1k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.3.2.3" style="padding-left:5.5pt;padding-right:5.5pt;">48.1</td><td class="ltx_td ltx_align_center" id="S6.T9.6.3.2.4" style="padding-left:5.5pt;padding-right:5.5pt;">-</td></tr><tr class="ltx_tr" id="S6.T9.6.4.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.4.3.1" style="padding-left:5.5pt;padding-right:5.5pt;">ViT-B <html><body><p>( <strong>BEiT: BERT pre-training of image transformers.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.4.3.2" style="padding-left:5.5pt;padding-right:5.5pt;">BEiT</th><td class="ltx_td ltx_align_center" id="S6.T9.6.4.3.3" style="padding-left:5.5pt;padding-right:5.5pt;">53.6</td><td class="ltx_td ltx_align_center" id="S6.T9.6.4.3.4" style="padding-left:5.5pt;padding-right:5.5pt;">54.1</td></tr><tr class="ltx_tr" id="S6.T9.6.5.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.5.4.1" style="padding-left:5.5pt;padding-right:5.5pt;">ViT-B <html><body><p>( <strong>BEiT v2: Masked image modeling with vector-quantized visualtokenizers.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.5.4.2" style="padding-left:5.5pt;padding-right:5.5pt;">BEiTv2 IN-1k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.5.4.3" style="padding-left:5.5pt;padding-right:5.5pt;">53.1</td><td class="ltx_td ltx_align_center" id="S6.T9.6.5.4.4" style="padding-left:5.5pt;padding-right:5.5pt;">-</td></tr><tr class="ltx_tr" id="S6.T9.6.6.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.6.5.1" style="padding-left:5.5pt;padding-right:5.5pt;">ViT-B <html><body><p>( <strong>BEiT v2: Masked image modeling with vector-quantized visualtokenizers.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.6.5.2" style="padding-left:5.5pt;padding-right:5.5pt;">BEiTv2 IN-22k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.6.5.3" style="padding-left:5.5pt;padding-right:5.5pt;">53.5</td><td class="ltx_td ltx_align_center" id="S6.T9.6.6.5.4" style="padding-left:5.5pt;padding-right:5.5pt;">-</td></tr><tr class="ltx_tr" id="S6.T9.6.7.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.7.6.1" style="padding-left:5.5pt;padding-right:5.5pt;">Swin-B <html><body><p>( <strong>Swin transformer: Hierarchical vision transformer using shiftedwindows, 2021.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.7.6.2" style="padding-left:5.5pt;padding-right:5.5pt;">Sup IN-1k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.7.6.3" style="padding-left:5.5pt;padding-right:5.5pt;">48.1</td><td class="ltx_td ltx_align_center" id="S6.T9.6.7.6.4" style="padding-left:5.5pt;padding-right:5.5pt;">49.7</td></tr><tr class="ltx_tr" id="S6.T9.6.8.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.8.7.1" style="padding-left:5.5pt;padding-right:5.5pt;">Swin-B <html><body><p>( <strong>Swin transformer: Hierarchical vision transformer using shiftedwindows, 2021.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.8.7.2" style="padding-left:5.5pt;padding-right:5.5pt;">Sup IN-22k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.8.7.3" style="padding-left:5.5pt;padding-right:5.5pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T9.6.8.7.4" style="padding-left:5.5pt;padding-right:5.5pt;">51.8</td></tr><tr class="ltx_tr" id="S6.T9.6.9.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.9.8.1" style="padding-left:5.5pt;padding-right:5.5pt;">Swin-B <html><body><p>( <strong>Swin transformer: Hierarchical vision transformer using shiftedwindows, 2021.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.9.8.2" style="padding-left:5.5pt;padding-right:5.5pt;">SimMIM <html><body><p>( <strong>Simmim: A simple framework for masked image modeling.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S6.T9.6.9.8.3" style="padding-left:5.5pt;padding-right:5.5pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T9.6.9.8.4" style="padding-left:5.5pt;padding-right:5.5pt;">52.8</td></tr><tr class="ltx_tr" id="S6.T9.6.10.9"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.10.9.1" style="padding-left:5.5pt;padding-right:5.5pt;">FocalAtt-B <html><body><p>( <strong>Focal self-attention for local-global interactions in visiontransformers.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.10.9.2" style="padding-left:5.5pt;padding-right:5.5pt;">Sup IN-1k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.10.9.3" style="padding-left:5.5pt;padding-right:5.5pt;">49.0</td><td class="ltx_td ltx_align_center" id="S6.T9.6.10.9.4" style="padding-left:5.5pt;padding-right:5.5pt;">50.5</td></tr><tr class="ltx_tr" id="S6.T9.6.11.10"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.11.10.1" style="padding-left:5.5pt;padding-right:5.5pt;">FocalNet-B <html><body><p>( <strong>Focal modulation networks.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.11.10.2" style="padding-left:5.5pt;padding-right:5.5pt;">Sup IN-1k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.11.10.3" style="padding-left:5.5pt;padding-right:5.5pt;">50.5</td><td class="ltx_td ltx_align_center" id="S6.T9.6.11.10.4" style="padding-left:5.5pt;padding-right:5.5pt;">51.4</td></tr><tr class="ltx_tr" id="S6.T9.6.12.11"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.12.11.1" style="padding-left:5.5pt;padding-right:5.5pt;">ConvNeXt v1-B <html><body><p>( <strong>A convnet for the 2020s.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.12.11.2" style="padding-left:5.5pt;padding-right:5.5pt;">Sup IN-1k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.12.11.3" style="padding-left:5.5pt;padding-right:5.5pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T9.6.12.11.4" style="padding-left:5.5pt;padding-right:5.5pt;">49.9</td></tr><tr class="ltx_tr" id="S6.T9.6.13.12"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.13.12.1" style="padding-left:5.5pt;padding-right:5.5pt;">ConvNeXt v2-B <html><body><p>( <strong>Convnext v2: Co-designing and scaling convnets with maskedautoencoders.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.13.12.2" style="padding-left:5.5pt;padding-right:5.5pt;">Sup IN-1k</th><td class="ltx_td ltx_align_center" id="S6.T9.6.13.12.3" style="padding-left:5.5pt;padding-right:5.5pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T9.6.13.12.4" style="padding-left:5.5pt;padding-right:5.5pt;">50.5</td></tr><tr class="ltx_tr" id="S6.T9.6.14.13"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.14.13.1" style="padding-left:5.5pt;padding-right:5.5pt;">ConvNeXt v2-B <html><body><p>( <strong>Convnext v2: Co-designing and scaling convnets with maskedautoencoders.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T9.6.14.13.2" style="padding-left:5.5pt;padding-right:5.5pt;">FCMAE</th><td class="ltx_td ltx_align_center" id="S6.T9.6.14.13.3" style="padding-left:5.5pt;padding-right:5.5pt;">-</td><td class="ltx_td ltx_align_center" id="S6.T9.6.14.13.4" style="padding-left:5.5pt;padding-right:5.5pt;">52.1</td></tr><tr class="ltx_tr" id="S6.T9.6.15.14" style="background-color:#E6E6E6;"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T9.6.15.14.1" style="padding-left:5.5pt;padding-right:5.5pt;">DaViT-B <html><body><p>( <strong>Davit: Dual attention vision transformers.</strong> )</p></body></html></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T9.6.15.14.2" style="padding-left:5.5pt;padding-right:5.5pt;"><em class="ltx_emph ltx_font_italic" id="S6.T9.6.15.14.2.1" style="font-size:90%;background-color:#E6E6E6;">Florence-2</em></th><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T9.6.15.14.3" style="padding-left:5.5pt;padding-right:5.5pt;">54.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T9.6.15.14.4" style="padding-left:5.5pt;padding-right:5.5pt;">55.5</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 9:  ADE20K semantic segmentation results using UperNet. | ✅ Table 9:  ADE20K semantic segmentation results 使用 UperNet。 |
| ✅ The input size is  $512\times 512$  for all the entries, except for models with BEiT pre-trained, which use the input size of  $640\times 640$  . | ✅ 所有条目的输入大小均为  $512\times 512$ ，但经过 BEiT 预训练的模型除外，这些模型使用  $640\times 640$  的输入大小。 |

### 6.5 Ablation Studies

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x12.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 7:  Multitask transfer. We conduct experiments with three different versions of Florence-2 models, each trained on a different level of image annotation: image level, image and region level, and image, region, and pixel level. We then evaluate the transfer learning performance of these models on four downstream tasks: COCO caption, COCO object detection, Flickr30k grounding, and Refcoco referring segmentation. | ✅ Figure 7:  Multitask transfer. We conduct experiments with three different versions of Florence-2 models, each trained on a different level of image annotation: image level, image and region level, and image, region, and pixel level. We then evaluate the transfer learning performance of these models on four downstream tasks: COCO caption, COCO object detection, Flickr30k grounding, and Refcoco referring segmentation. |

#### 6.5.1 Multitask transfer.

| 【第6.5.1节，第1段】原文 | 【第6.5.1节，第1段】翻译 |
| ---- | ---- |
| ✅ In this study, we aimed to identify the most effective pre-trained model for transfer learning across various downstream tasks in computer vision. | ✅ 在这项研究中，我们的目标是找到最有效的预训练模型，用于计算机视觉中各个下游任务的迁移学习。 |
| ✅ We compared three different models, each pre-trained on a different combination of tasks: | ✅ 我们比较了三种不同的模型，每种模型都针对不同的任务组合进行了预训练： |

| 【第6.5.1节，第2段】原文 | 【第6.5.1节，第2段】翻译 |
| ---- | ---- |
| ✅ Image-level Model: pre-trained on image-level tasks only | ✅ 图像级模型：仅在图像级任务上进行预训练 |

| 【第6.5.1节，第3段】原文 | 【第6.5.1节，第3段】翻译 |
| ---- | ---- |
| ✅ Image-Region Model: pre-trained on image-level and region-level tasks | ✅ 图像区域模型：针对图像级和区域级任务进行预训练 |

| 【第6.5.1节，第4段】原文 | 【第6.5.1节，第4段】翻译 |
| ---- | ---- |
| ✅ Image-Region-Pixel Model: pre-trained on image-level, region-level, and pixel-level tasks | ✅ 图像-区域-像素模型：针对图像级、区域级和像素级任务进行预训练 |

| 【第6.5.1节，第5段】原文 | 【第6.5.1节，第5段】翻译 |
| ---- | ---- |
| ✅ For pre-training, we optimize all models for the same number of effective samples (72M) on a subset of our FLD-5B dataset. | ✅ 对于预训练，我们在 FLD-5B 数据集的子集上针对相同数量的有效样本（72M）优化所有模型。 |

| 【第6.5.1节，第6段】原文 | 【第6.5.1节，第6段】翻译 |
| ---- | ---- |
| ✅ These models are then transferred to a combined dataset with four downstream tasks, each representing a different level of task granularity: COCO caption (image-level task), COCO object detection (region-level task), Flickr30k grounding (region-level task), RefCOCO referring segmentation (pixel-level task). | ✅ 然后将这些模型转移到具有四个下游任务的组合数据集，每个任务代表不同级别的任务粒度：COCO 标题（图像级任务）、COCO 对象检测（区域级任务）、Flickr30k 接地（区域级任务）、RefCOCO 参考分割（像素级任务）。 |

| 【第6.5.1节，第7段】原文 | 【第6.5.1节，第7段】翻译 |
| ---- | ---- |
| ✅ The results are shown in Figure 7. | ✅ 结果显示在Figure 7中。 |
| ✅ The results demonstrate that Image-Region-Pixel Model, pre-trained on all three levels of tasks, consistently demonstrated competitive performance across the four downstream tasks. | ✅ 结果表明，在所有三个级别的任务上进行预训练的图像区域像素模型在四个下游任务中始终表现出有竞争力的性能。 |

| 【第6.5.1节，第8段】原文 | 【第6.5.1节，第8段】翻译 |
| ---- | ---- |
| ✅ For the COCO caption task, Image-Region-Pixel Model initially performs worse than Image-level Model and Image-Region Model but eventually achieve a final performance (133.4 CIDEr) that is only slightly worse than the other models (134.6 CIDEr). | ✅ 对于 COCO 字幕任务，图像区域像素模型最初的表现比图像级模型和图像区域模型差，但最终实现了最终性能（133.4 CIDEr），仅比其他模型（134.6 CIDEr）稍差。 |

| 【第6.5.1节，第9段】原文 | 【第6.5.1节，第9段】翻译 |
| ---- | ---- |
| ✅ For the COCO object detection task, Image-Region-Pixel Model outperforms Image-level Model by a significant margin (28.3 vs. | ✅ 对于 COCO 对象检测任务，图像区域像素模型的表现明显优于图像级模型（28.3 vs. |
| ✅ 0.1) and was only slightly worse than Image-Region Model (29.7). | ✅ 0.1)，仅比图像区域模型 (29.7) 稍差。 |

| 【第6.5.1节，第10段】原文 | 【第6.5.1节，第10段】翻译 |
| ---- | ---- |
| ✅ For the Flickr30k grounding task, Image-Region-Pixel Model shows strong performance (78.1 recall@1), comparable to Image-Region Model (79.1 recall@1) and significantly better than Image-level Model (62.0 recall@1). | ✅ 对于 Flickr30k 基础任务，图像区域像素模型表现出色（78.1 召回率@1），与图像区域模型（79.1 召回率@1）相当，并且明显优于图像级模型（62.0 召回率@1）。 |

| 【第6.5.1节，第11段】原文 | 【第6.5.1节，第11段】翻译 |
| ---- | ---- |
| ✅ For the RefCOCO referring segmentation task, Image-Region-Pixel Model clearly outperforms both Image-level Model and Image-Region Model, achieving the highest performance (31.6 mIoU) compared to the other models (28.4 and 18.2 mIoU). | ✅ 对于 RefCOCO 参照分割任务，图像区域像素模型明显优于图像级模型和图像区域模型，与其他模型（28.4 和 18.2 mIoU）相比实现了最高性能（31.6 mIoU）。 |

| 【第6.5.1节，第12段】原文 | 【第6.5.1节，第12段】翻译 |
| ---- | ---- |
| ✅ Our findings suggest that the Image-Region-Pixel Model, which is pre-trained on tasks at the image, region, and pixel levels, is the most effective base model for transfer learning across various computer vision tasks. | ✅ 我们的研究结果表明，在图像、区域和像素级别的任务上进行预训练的图像区域像素模型是跨各种计算机视觉任务进行迁移学习的最有效的基础模型。 |
| ✅ This model shows strong performance on all four downstream tasks we evaluated, and consistently outperforms the Image-level Model and matches or exceeds the Image-Region Model in performance. | ✅ 该模型在我们评估的所有四个下游任务中都表现出了强劲的性能，并且始终优于图像级模型，并且在性能上达到或超过了图像区域模型。 |
| ✅ By pre-training a model on tasks at different levels of granularity, we can ensure that the base model is better prepared to handle a diverse range of downstream tasks, offering a versatile and robust solution for transfer learning in computer vision. | ✅ 通过对不同粒度级别的任务进行模型预训练，我们可以确保基础模型能够更好地处理各种下游任务，为计算机视觉中的迁移学习提供多功能且强大的解决方案。 |

#### 6.5.2 Model scaling.

| 【第6.5.2节，第1段】原文 | 【第6.5.2节，第1段】翻译 |
| ---- | ---- |
| ✅ We aimed to investigate the impact of increasing model capacity on zero-shot performance on various downstream tasks in computer vision. | ✅ 我们的目的是研究增加模型容量对计算机视觉中各种下游任务的零样本性能的影响。 |
| ✅ We compared two models: Florence-2-B and Florence-2-L , which have 232M and 771M parameters, respectively. | ✅ 我们比较了两个模型：Florence-2-B 和 Florence-2-L，它们分别有 232M 和 771M 个参数。 |
| ✅ The model architectures are described in Table 15. | ✅ 模型架构在Table 15中描述。 |
| ✅ We show the zero-shot performance on four downstream tasks in Table 10. | ✅ 我们展示了 Table 10 中四个下游任务的零样本性能。 |
| ✅ The large model clearly outperforms the base model across various downstream tasks. | ✅ 大型模型在各种下游任务中的表现明显优于基础模型。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T10.2"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T10.2.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T10.2.1.1.1" rowspan="2" style="padding-left:3.3pt;padding-right:3.3pt;">Model</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T10.2.1.1.2" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T10.2.1.1.3" style="padding-left:3.3pt;padding-right:3.3pt;">Caption</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T10.2.1.1.4" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T10.2.1.1.5" style="padding-left:3.3pt;padding-right:3.3pt;">Detection</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T10.2.1.1.6" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T10.2.1.1.7" style="padding-left:3.3pt;padding-right:3.3pt;">Grounding</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T10.2.1.1.8" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="2" id="S6.T10.2.1.1.9" style="padding-left:3.3pt;padding-right:3.3pt;">RES</th></tr><tr class="ltx_tr" id="S6.T10.2.2.2"><td class="ltx_td" id="S6.T10.2.2.2.1" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T10.2.2.2.2" style="padding-left:3.3pt;padding-right:3.3pt;">CIDEr</td><td class="ltx_td" id="S6.T10.2.2.2.3" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T10.2.2.2.4" style="padding-left:3.3pt;padding-right:3.3pt;">AP</td><td class="ltx_td" id="S6.T10.2.2.2.5" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T10.2.2.2.6" style="padding-left:3.3pt;padding-right:3.3pt;">Recall@1</td><td class="ltx_td" id="S6.T10.2.2.2.7" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T10.2.2.2.8" style="padding-left:3.3pt;padding-right:3.3pt;">mIOU</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T10.2.2.2.9" style="padding-left:3.3pt;padding-right:3.3pt;">oIOU</td></tr><tr class="ltx_tr" id="S6.T10.2.3.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" id="S6.T10.2.3.3.1" style="padding-left:3.3pt;padding-right:3.3pt;">Base</th><th class="ltx_td ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.2" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.3" style="padding-left:3.3pt;padding-right:3.3pt;">118.7</th><th class="ltx_td ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.4" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.5" style="padding-left:3.3pt;padding-right:3.3pt;">19.7</th><th class="ltx_td ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.6" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.7" style="padding-left:3.3pt;padding-right:3.3pt;">76.3</th><th class="ltx_td ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.8" style="padding-left:3.3pt;padding-right:3.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.9" style="padding-left:3.3pt;padding-right:3.3pt;">18.6</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T10.2.3.3.10" style="padding-left:3.3pt;padding-right:3.3pt;">17.8</th></tr><tr class="ltx_tr" id="S6.T10.2.4.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T10.2.4.4.1" style="padding-left:3.3pt;padding-right:3.3pt;">Large</th><td class="ltx_td ltx_border_bb" id="S6.T10.2.4.4.2" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T10.2.4.4.3" style="padding-left:3.3pt;padding-right:3.3pt;">124.4</td><td class="ltx_td ltx_border_bb" id="S6.T10.2.4.4.4" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T10.2.4.4.5" style="padding-left:3.3pt;padding-right:3.3pt;">22.6</td><td class="ltx_td ltx_border_bb" id="S6.T10.2.4.4.6" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T10.2.4.4.7" style="padding-left:3.3pt;padding-right:3.3pt;">78.2</td><td class="ltx_td ltx_border_bb" id="S6.T10.2.4.4.8" style="padding-left:3.3pt;padding-right:3.3pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T10.2.4.4.9" style="padding-left:3.3pt;padding-right:3.3pt;">21.5</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T10.2.4.4.10" style="padding-left:3.3pt;padding-right:3.3pt;">19.1</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 10:  Model scaling. Zero-shot performance on COCO caption and COCO object detection, Flickr30k grounding, RefCOCO referring expression segmentation(RES). | ✅ Table 10:  Model scaling. 在 COCO 标题和 COCO 对象检测上的零样本性能，Flickr30k 基础，RefCOCO 参考表情分割（RES）。 |

#### 6.5.3 Data scaling.

| 【第6.5.3节，第1段】原文 | 【第6.5.3节，第1段】翻译 |
| ---- | ---- |
| ✅ We conducted experiments to study how zero-shot performance on various computer vision tasks is affected by the scale of pre-training data. | ✅ 我们进行了实验，研究预训练数据规模如何影响各种计算机视觉任务的零样本性能。 |
| ✅ We used four different data sizes for pre-training: 0.12M, 0.36M, 1.2M, and 12M images. | ✅ 我们使用四种不同的数据大小进行预训练：0.12M、0.36M、1.2M 和 12M 图像。 |
| ✅ All models were trained with the same effective sample size (72M) on a subset of FLD-5B data. | ✅ 所有模型均在 FLD-5B 数据子集上使用相同的有效样本量（72M）进行训练。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T11.2"><thead class="ltx_thead"><tr class="ltx_tr" id="S6.T11.2.1.1"><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S6.T11.2.1.1.1" style="padding-left:3.1pt;padding-right:3.1pt;">Data</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T11.2.1.1.2" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T11.2.1.1.3" style="padding-left:3.1pt;padding-right:3.1pt;">Caption</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T11.2.1.1.4" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T11.2.1.1.5" style="padding-left:3.1pt;padding-right:3.1pt;">Detection</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T11.2.1.1.6" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S6.T11.2.1.1.7" style="padding-left:3.1pt;padding-right:3.1pt;">Grounding</th><th class="ltx_td ltx_th ltx_th_column ltx_border_tt" id="S6.T11.2.1.1.8" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="2" id="S6.T11.2.1.1.9" style="padding-left:3.1pt;padding-right:3.1pt;">RES</th></tr><tr class="ltx_tr" id="S6.T11.2.2.2"><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_th_row ltx_border_r" id="S6.T11.2.2.2.1" style="padding-left:3.1pt;padding-right:3.1pt;">size</th><th class="ltx_td ltx_th ltx_th_column" id="S6.T11.2.2.2.2" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T11.2.2.2.3" style="padding-left:3.1pt;padding-right:3.1pt;">CIDEr</th><th class="ltx_td ltx_th ltx_th_column" id="S6.T11.2.2.2.4" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T11.2.2.2.5" style="padding-left:3.1pt;padding-right:3.1pt;">AP</th><th class="ltx_td ltx_th ltx_th_column" id="S6.T11.2.2.2.6" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T11.2.2.2.7" style="padding-left:3.1pt;padding-right:3.1pt;">Recall@1</th><th class="ltx_td ltx_th ltx_th_column" id="S6.T11.2.2.2.8" style="padding-left:3.1pt;padding-right:3.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T11.2.2.2.9" style="padding-left:3.1pt;padding-right:3.1pt;">mIOU</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="S6.T11.2.2.2.10" style="padding-left:3.1pt;padding-right:3.1pt;">oIOU</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T11.2.3.1"><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T11.2.3.1.1" style="padding-left:3.1pt;padding-right:3.1pt;">0.12M</th><td class="ltx_td ltx_border_t" id="S6.T11.2.3.1.2" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T11.2.3.1.3" style="padding-left:3.1pt;padding-right:3.1pt;">102.8</td><td class="ltx_td ltx_border_t" id="S6.T11.2.3.1.4" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T11.2.3.1.5" style="padding-left:3.1pt;padding-right:3.1pt;">16.1</td><td class="ltx_td ltx_border_t" id="S6.T11.2.3.1.6" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T11.2.3.1.7" style="padding-left:3.1pt;padding-right:3.1pt;">74.0</td><td class="ltx_td ltx_border_t" id="S6.T11.2.3.1.8" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T11.2.3.1.9" style="padding-left:3.1pt;padding-right:3.1pt;">15.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T11.2.3.1.10" style="padding-left:3.1pt;padding-right:3.1pt;">16.6</td></tr><tr class="ltx_tr" id="S6.T11.2.4.2"><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T11.2.4.2.1" style="padding-left:3.1pt;padding-right:3.1pt;">0.36M</th><td class="ltx_td" id="S6.T11.2.4.2.2" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.4.2.3" style="padding-left:3.1pt;padding-right:3.1pt;">114.3</td><td class="ltx_td" id="S6.T11.2.4.2.4" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.4.2.5" style="padding-left:3.1pt;padding-right:3.1pt;">18.7</td><td class="ltx_td" id="S6.T11.2.4.2.6" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.4.2.7" style="padding-left:3.1pt;padding-right:3.1pt;">75.8</td><td class="ltx_td" id="S6.T11.2.4.2.8" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.4.2.9" style="padding-left:3.1pt;padding-right:3.1pt;">16.6</td><td class="ltx_td ltx_align_center" id="S6.T11.2.4.2.10" style="padding-left:3.1pt;padding-right:3.1pt;">16.4</td></tr><tr class="ltx_tr" id="S6.T11.2.5.3"><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S6.T11.2.5.3.1" style="padding-left:3.1pt;padding-right:3.1pt;">1.2M</th><td class="ltx_td" id="S6.T11.2.5.3.2" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.5.3.3" style="padding-left:3.1pt;padding-right:3.1pt;">118.1</td><td class="ltx_td" id="S6.T11.2.5.3.4" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.5.3.5" style="padding-left:3.1pt;padding-right:3.1pt;">18.9</td><td class="ltx_td" id="S6.T11.2.5.3.6" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.5.3.7" style="padding-left:3.1pt;padding-right:3.1pt;">76.3</td><td class="ltx_td" id="S6.T11.2.5.3.8" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T11.2.5.3.9" style="padding-left:3.1pt;padding-right:3.1pt;">19.3</td><td class="ltx_td ltx_align_center" id="S6.T11.2.5.3.10" style="padding-left:3.1pt;padding-right:3.1pt;">18.4</td></tr><tr class="ltx_tr" id="S6.T11.2.6.4"><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T11.2.6.4.1" style="padding-left:3.1pt;padding-right:3.1pt;">12M</th><td class="ltx_td ltx_border_bb" id="S6.T11.2.6.4.2" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T11.2.6.4.3" style="padding-left:3.1pt;padding-right:3.1pt;">118.7</td><td class="ltx_td ltx_border_bb" id="S6.T11.2.6.4.4" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T11.2.6.4.5" style="padding-left:3.1pt;padding-right:3.1pt;">19.7</td><td class="ltx_td ltx_border_bb" id="S6.T11.2.6.4.6" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T11.2.6.4.7" style="padding-left:3.1pt;padding-right:3.1pt;">76.3</td><td class="ltx_td ltx_border_bb" id="S6.T11.2.6.4.8" style="padding-left:3.1pt;padding-right:3.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T11.2.6.4.9" style="padding-left:3.1pt;padding-right:3.1pt;">18.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T11.2.6.4.10" style="padding-left:3.1pt;padding-right:3.1pt;">17.8</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 11:  Data scaling. Zero-shot performance on COCO caption, COCO object detection, Flickr30k grounding, COCORef referring segmentation. | ✅ Table 11:  Data scaling. 在 COCO 标题、COCO 对象检测、Flickr30k 基础、COCORef 引用分割上的零样本性能。 |

| 【第6.5.3节，第2段】原文 | 【第6.5.3节，第2段】翻译 |
| ---- | ---- |
| ✅ Table 11 presents the zero-shot performance results on COCO caption, COCO object detection, Flickr30k grounding, and RefCoco referring segmentation (RES) tasks. | ✅ Table 11 展示了 COCO 标题、COCO 对象检测、Flickr30k 基础和 RefCoco 引用分割 (RES) 任务的零样本性能结果。 |
| ✅ We can observe a trend of improved zero-shot performance on the downstream tasks as the pre-training data size increases (except for RES, 1.2M data has slightly better performance compared to 12M). | ✅ 我们可以观察到，随着预训练数据大小的增加，下游任务的零样本性能呈现提高的趋势（RES 除外，1.2M 数据的性能略优于 12M）。 |

| 【第6.5.3节，第3段】原文 | 【第6.5.3节，第3段】翻译 |
| ---- | ---- |
| ✅ Our experiments on data scaling demonstrate that larger pre-training data sizes generally lead to improved zero-shot performance across a variety of downstream tasks in computer vision. | ✅ 我们进行数据扩展的实验表明，更大的预训练数据量通常会提高计算机视觉中各种下游任务的零样本性能。 |
| ✅ This finding suggests that investing in larger pre-training datasets can provide a more effective and versatile foundation for handling a wide range of downstream tasks. | ✅ 这一发现表明，投资更大的预训练数据集可以为处理广泛的下游任务提供更有效、更通用的基础。 |

| 【第6.5.3节，第4段】原文 | 【第6.5.3节，第4段】翻译 |
| ---- | ---- |
| ✅ Our approach to scaling data is significantly more efficient than relying solely on human annotations, as most of the annotation generation is performed using model inference. | ✅ 我们扩展数据的方法比单纯依赖人工注释要高效得多，因为大多数注释生成都是使用模型推理执行的。 |
| ✅ By leveraging specialist models to generate annotations, we can substantially reduce the time and cost associated with manual annotation efforts, which often involve labor-intensive processes and may be subject to human errors or inconsistencies. | ✅ 通过利用专业模型来生成注释，我们可以大大减少与手动注释工作相关的时间和成本，手动注释工作通常涉及劳动密集型过程，并且可能受到人为错误或不一致的影响。 |

| 【第6.5.3节，第5段】原文 | 【第6.5.3节，第5段】翻译 |
| ---- | ---- |
| ✅ Furthermore, utilizing model-generated annotations enables us to scale the pre-training datasets more rapidly and efficiently, allowing us to explore the impact of larger data sizes on model performance across various downstream tasks in computer vision. | ✅ 此外，利用模型生成的注释使我们能够更快、更有效地扩展预训练数据集，从而使我们能够探索更大的数据量对计算机视觉中各种下游任务的模型性能的影响。 |
| ✅ This not only facilitates the development of more effective and versatile foundation models but also ensures that the annotation process remains sustainable and scalable as the need for high-quality labeled data continues to grow. | ✅ 这不仅有利于开发更有效、更通用的基础模型，而且还确保注释过程在对高质量标记数据的需求不断增长的情况下保持可持续性和可扩展性。 |

| 【第6.5.3节，第6段】原文 | 【第6.5.3节，第6段】翻译 |
| ---- | ---- |
| ✅ In summary, our data scaling approach offers a more efficient alternative to traditional human annotation methods by harnessing the power of specialist models for annotation generation. | ✅ 总之，我们的数据扩展方法利用专业模型的功能进行注释生成，为传统人工注释方法提供了更有效的替代方案。 |
| ✅ This strategy enables us to accelerate the pre-training process, optimize model performance, and effectively manage the ever-increasing demand for labeled data in the field of computer vision. | ✅ 这一策略使我们能够加速预训练过程，优化模型性能，并有效管理计算机视觉领域对标记数据不断增长的需求。 |

#### 6.5.4 Training settings.

| 【第6.5.4节，第1段】原文 | 【第6.5.4节，第1段】翻译 |
| ---- | ---- |
| ✅ We analyze the basic model training settings for the two primary components of our model, namely the vision encoder and the multi-modality encoder-decoder. | ✅ 我们分析了模型的两个主要组件，即视觉编码器和多模态编码器-解码器的基本模型训练设置。 |
| ✅ The experiment results are presented in Table 12 | ✅ 实验结果呈现在Table 12中 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S6.T12.2"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S6.T12.2.1.1"><th class="ltx_td ltx_th ltx_th_row ltx_border_tt" id="S6.T12.2.1.1.1" style="padding-left:2.1pt;padding-right:2.1pt;"></th><th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="S6.T12.2.1.1.2" style="padding-left:2.1pt;padding-right:2.1pt;"></th><td class="ltx_td ltx_border_tt" id="S6.T12.2.1.1.3" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T12.2.1.1.4" style="padding-left:2.1pt;padding-right:2.1pt;">Caption</td><td class="ltx_td ltx_border_tt" id="S6.T12.2.1.1.5" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T12.2.1.1.6" style="padding-left:2.1pt;padding-right:2.1pt;">Detection</td><td class="ltx_td ltx_border_tt" id="S6.T12.2.1.1.7" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" id="S6.T12.2.1.1.8" style="padding-left:2.1pt;padding-right:2.1pt;">Grounding</td><td class="ltx_td ltx_border_tt" id="S6.T12.2.1.1.9" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_tt" colspan="2" id="S6.T12.2.1.1.10" style="padding-left:2.1pt;padding-right:2.1pt;">RES</td></tr><tr class="ltx_tr" id="S6.T12.2.2.2"><th class="ltx_td ltx_align_center ltx_th ltx_th_row" id="S6.T12.2.2.2.1" style="padding-left:2.1pt;padding-right:2.1pt;">V Pre</th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" id="S6.T12.2.2.2.2" style="padding-left:2.1pt;padding-right:2.1pt;">L Pre</th><td class="ltx_td" id="S6.T12.2.2.2.3" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.2.2.4" style="padding-left:2.1pt;padding-right:2.1pt;">CIDEr</td><td class="ltx_td" id="S6.T12.2.2.2.5" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.2.2.6" style="padding-left:2.1pt;padding-right:2.1pt;">AP</td><td class="ltx_td" id="S6.T12.2.2.2.7" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.2.2.8" style="padding-left:2.1pt;padding-right:2.1pt;">Recall@1</td><td class="ltx_td" id="S6.T12.2.2.2.9" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.2.2.10" style="padding-left:2.1pt;padding-right:2.1pt;">mIOU</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.2.2.11" style="padding-left:2.1pt;padding-right:2.1pt;">oIOU</td></tr><tr class="ltx_tr" id="S6.T12.2.3.3"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" colspan="11" id="S6.T12.2.3.3.1" style="padding-left:2.1pt;padding-right:2.1pt;">Freeze Vision Encoder</th></tr><tr class="ltx_tr" id="S6.T12.2.4.4"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" id="S6.T12.2.4.4.1" style="padding-left:2.1pt;padding-right:2.1pt;">✓</th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T12.2.4.4.2" style="padding-left:2.1pt;padding-right:2.1pt;">✓</th><td class="ltx_td ltx_border_t" id="S6.T12.2.4.4.3" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.4.4.4" style="padding-left:2.1pt;padding-right:2.1pt;">120.0</td><td class="ltx_td ltx_border_t" id="S6.T12.2.4.4.5" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.4.4.6" style="padding-left:2.1pt;padding-right:2.1pt;">6.9</td><td class="ltx_td ltx_border_t" id="S6.T12.2.4.4.7" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.4.4.8" style="padding-left:2.1pt;padding-right:2.1pt;">66.3</td><td class="ltx_td ltx_border_t" id="S6.T12.2.4.4.9" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.4.4.10" style="padding-left:2.1pt;padding-right:2.1pt;">9.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.4.4.11" style="padding-left:2.1pt;padding-right:2.1pt;">13.6</td></tr><tr class="ltx_tr" id="S6.T12.2.5.5"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" colspan="11" id="S6.T12.2.5.5.1" style="padding-left:2.1pt;padding-right:2.1pt;">Unfreeze Vision Encoder</th></tr><tr class="ltx_tr" id="S6.T12.2.6.6"><th class="ltx_td ltx_th ltx_th_row ltx_border_t" id="S6.T12.2.6.6.1" style="padding-left:2.1pt;padding-right:2.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S6.T12.2.6.6.2" style="padding-left:2.1pt;padding-right:2.1pt;">✓</th><td class="ltx_td ltx_border_t" id="S6.T12.2.6.6.3" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.6.6.4" style="padding-left:2.1pt;padding-right:2.1pt;">81.3</td><td class="ltx_td ltx_border_t" id="S6.T12.2.6.6.5" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.6.6.6" style="padding-left:2.1pt;padding-right:2.1pt;">4.9</td><td class="ltx_td ltx_border_t" id="S6.T12.2.6.6.7" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.6.6.8" style="padding-left:2.1pt;padding-right:2.1pt;">69.0</td><td class="ltx_td ltx_border_t" id="S6.T12.2.6.6.9" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.6.6.10" style="padding-left:2.1pt;padding-right:2.1pt;">15.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S6.T12.2.6.6.11" style="padding-left:2.1pt;padding-right:2.1pt;">15.6</td></tr><tr class="ltx_tr" id="S6.T12.2.7.7"><th class="ltx_td ltx_align_center ltx_th ltx_th_row" id="S6.T12.2.7.7.1" style="padding-left:2.1pt;padding-right:2.1pt;">✓</th><th class="ltx_td ltx_th ltx_th_row ltx_border_r" id="S6.T12.2.7.7.2" style="padding-left:2.1pt;padding-right:2.1pt;"></th><td class="ltx_td" id="S6.T12.2.7.7.3" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T12.2.7.7.4" style="padding-left:2.1pt;padding-right:2.1pt;">117.4</td><td class="ltx_td" id="S6.T12.2.7.7.5" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T12.2.7.7.6" style="padding-left:2.1pt;padding-right:2.1pt;">19.6</td><td class="ltx_td" id="S6.T12.2.7.7.7" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T12.2.7.7.8" style="padding-left:2.1pt;padding-right:2.1pt;">75.2</td><td class="ltx_td" id="S6.T12.2.7.7.9" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center" id="S6.T12.2.7.7.10" style="padding-left:2.1pt;padding-right:2.1pt;">21.5</td><td class="ltx_td ltx_align_center" id="S6.T12.2.7.7.11" style="padding-left:2.1pt;padding-right:2.1pt;">19.3</td></tr><tr class="ltx_tr" id="S6.T12.2.8.8"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb" id="S6.T12.2.8.8.1" style="padding-left:2.1pt;padding-right:2.1pt;">✓</th><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S6.T12.2.8.8.2" style="padding-left:2.1pt;padding-right:2.1pt;">✓</th><td class="ltx_td ltx_border_bb" id="S6.T12.2.8.8.3" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T12.2.8.8.4" style="padding-left:2.1pt;padding-right:2.1pt;">118.7</td><td class="ltx_td ltx_border_bb" id="S6.T12.2.8.8.5" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T12.2.8.8.6" style="padding-left:2.1pt;padding-right:2.1pt;">19.7</td><td class="ltx_td ltx_border_bb" id="S6.T12.2.8.8.7" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T12.2.8.8.8" style="padding-left:2.1pt;padding-right:2.1pt;">76.3</td><td class="ltx_td ltx_border_bb" id="S6.T12.2.8.8.9" style="padding-left:2.1pt;padding-right:2.1pt;"></td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T12.2.8.8.10" style="padding-left:2.1pt;padding-right:2.1pt;">18.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S6.T12.2.8.8.11" style="padding-left:2.1pt;padding-right:2.1pt;">17.8</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 12:  Basic components. Zero-shot performance on COCO caption, COCO object detection, Flickr30k grounding, and COCORef referring segmentation. | ✅ Table 12:  Basic components. 在 COCO 标题、COCO 对象检测、Flickr30k 基础和 COCORef 引用分割上的零样本性能。 |
| ✅ V Pre and L Pre indicate that using vision and language pre-training initialization, respectively. | ✅ V Pre和L Pre分别表示使用视觉和语言预训练初始化。 |

| 【第6.5.4节，第2段】原文 | 【第6.5.4节，第2段】翻译 |
| ---- | ---- |
| ✅ We observe that freezing the vision encoders does not affect the performance on tasks that require image-level understanding, but it significantly degrades the performance on tasks that require region-level or pixel-level understanding (e.g., AP on COCO object detection drops from 19.7 to 6.9). | ✅ 我们观察到，冻结视觉编码器不会影响需要图像级理解的任务的性能，但会显著降低需要区域级或像素级理解的任务的性能（例如，COCO 对象检测的 AP 从 19.7 下降到 6.9）。 |
| ✅ Previous methods for pre-training vision foundation models mainly focus on image-level tasks (e.g., image classification ( **1. Imagenet classification with deep convolutional neural networks.** ｜ **2. Deep residual learning for image recognition.** ) , image-text contrastive learning ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Florence: A new foundation model for computer vision.** ) ), which may not provide them with sufficient region-level and pixel-level skills for downstream tasks. | ✅ 先前的视觉基础模型预训练方法主要侧重于图像级任务（例如图像分类 ( **1. Imagenet classification with deep convolutional neural networks.** ｜ **2. Deep residual learning for image recognition.** )、图像文本对比学习 ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Florence: A new foundation model for computer vision.** )），这可能无法为下游任务提供足够的区域级和像素级技能。 |
| ✅ Therefore, it is important to unfreeze the vision backbone, enabling it to learn region-level and pixel-level features for various downstream tasks. | ✅ 因此，解冻视觉主干非常重要，使其能够学习区域级和像素级特征以用于各种下游任务。 |

| 【第6.5.4节，第3段】原文 | 【第6.5.4节，第3段】翻译 |
| ---- | ---- |
| ✅ The effect of language pre-training weights on multi-modal encoder-decoder tasks varies depending on the task. | ✅ 语言预训练权重对多模态编码器-解码器任务的影响因任务而异。 |
| ✅ Tasks that require more text understanding, such as captioning and grounding, benefit slightly from using language pre-training weights (e.g., COCO caption, Flickr30k grounding). | ✅ 对于需要更多文本理解的任务（例如字幕和基础）来说，使用语言预训练权重（例如 COCO 字幕、Flickr30k 基础）会略有好处。 |
| ✅ Tasks that are mostly vision-focused, such as object detection and region segmentation, do not gain much from using language pre-training weights (for COCO object detection, the gain is only 0.1; for RES tasks, which use only localization tokens, the drop is 2.91 mIOU). | ✅ 对于主要以视觉为中心的任务（例如对象检测和区域分割），使用语言预训练权重不会带来太大的收益（对于 COCO 对象检测，收益仅为 0.1；对于仅使用定位标记的 RES 任务，下降幅度为 2.91 mIOU）。 |

| 【第6.5.4节，第4段】原文 | 【第6.5.4节，第4段】翻译 |
| ---- | ---- |
| ✅ We investigate the effects of different training configurations on the performance of a foundation model in region-level and pixel-level tasks. | ✅ 我们研究了不同的训练配置对基础模型在区域级和像素级任务中性能的影响。 |
| ✅ We find that unfreezing the vision backbone is crucial for enhancing the model’s ability to learn from regions and pixels, which is beneficial for transferring to various downstream tasks. | ✅ 我们发现解冻视觉主干对于增强模型从区域和像素学习的能力至关重要，这有利于转移到各种下游任务。 |
| ✅ Moreover, we observe that using language pre-training weights can help the model in tasks that require text understanding, but have less impact on tasks that are purely vision-based. | ✅ 此外，我们观察到使用语言预训练权重可以帮助模型完成需要文本理解的任务，但对纯粹基于视觉的任务影响较小。 |
| ✅ These results offer useful guidance for choosing the best training settings for different computer vision tasks. | ✅ 这些结果为选择不同计算机视觉任务的最佳训练设置提供了有用的指导。 |

## 7 Related Works

### 7.1 Vision-Language Foundation Models

| 【第7.1节，第1段】原文 | 【第7.1节，第1段】翻译 |
| ---- | ---- |
| ✅ Recent vision-language pre-training models ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ｜ **3. Florence: A new foundation model for computer vision.** ) have demonstrated impressive zero-shot transfer abilities to vision-language alignment and image classification tasks, thanks to the alignment of vision and text embeddings extracted from respective encoders through contrastive learning objectives ( **1. Improved deep metric learning with multi-class n-pair loss objective.** ｜ **2. Representation learning with contrastive predictive coding.** ). | ✅ 最近的视觉语言预训练模型 ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ｜ **3. Florence: A new foundation model for computer vision.** ) 已经展示了令人印象深刻的零样本迁移能力，可用于视觉语言对齐和图像分类任务，这要归功于通过对比学习目标 ( **1. Improved deep metric learning with multi-class n-pair loss objective.** ｜ **2. Representation learning with contrastive predictive coding.** ) 从各自的编码器提取的视觉和文本嵌入的对齐。 |
| ✅ These models ( e.g. | ✅ 这些模型（e.g. |
| ✅  , ( **Florence: A new foundation model for computer vision.** ) ), trained on weakly large-scale image-text data, have been further extended to more downstream tasks such as object detection, achieving state-of-the-art performance with task-specific adaptation heads. | ✅ 、( **Florence: A new foundation model for computer vision.** )）在弱大规模图像文本数据上进行训练，并进一步扩展到对象检测等更下游的任务，并通过特定于任务的自适应头实现了最佳性能。 |

| 【第7.1节，第2段】原文 | 【第7.1节，第2段】翻译 |
| ---- | ---- |
| ✅ In contrast, other studies ( **1. Coca: Contrastive captioners are image-text foundation models, 2022.** ｜ **2. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ｜ **3. Git: A generative image-to-text transformer for vision and language, 2022.** ｜ **4. Flamingo: a visual language model for few-shot learning.** ) propose using a multi-modality decoder to predict text in an autoregressive manner with language modeling pre-training objectives. | ✅ 相比之下，其他研究 ( **1. Coca: Contrastive captioners are image-text foundation models, 2022.** ｜ **2. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ｜ **3. Git: A generative image-to-text transformer for vision and language, 2022.** ｜ **4. Flamingo: a visual language model for few-shot learning.** ) 提出使用多模态解码器以语言建模预训练目标以自回归方式预测文本。 |
| ✅ Techniques for fusing vision and language embeddings vary: GIT ( **Git: A generative image-to-text transformer for vision and language, 2022.** ) concatenates vision and text tokens as decoder input and designs a casual attention mask, CoCa ( **Coca: Contrastive captioners are image-text foundation models, 2022.** ) uses attentional poolers with learnable queries to select task-specific vision representations which are then cross-attended via the decoder, and Flamingo ( **Flamingo: a visual language model for few-shot learning.** ) pools a fixed number of vision tokens with a Perceiver Resampler and adds new learnable cross-attention layers to the decoder while freezing the pre-trained vision encoder and text decoder. | ✅ 融合视觉和语言嵌入的技术各不相同：GIT ( **Git: A generative image-to-text transformer for vision and language, 2022.** ) 将视觉和文本标记连接起来作为解码器输入并设计一个随意注意掩码，CoCa ( **Coca: Contrastive captioners are image-text foundation models, 2022.** ) 使用具有可学习查询的注意力池来选择特定于任务的视觉表示，然后通过解码器进行交叉注意，而 Flamingo ( **Flamingo: a visual language model for few-shot learning.** ) 将固定数量的视觉标记与感知器重采样器池化，并向解码器添加新的可学习的交叉注意层，同时冻结预先训练的视觉编码器和文本解码器。 |

| 【第7.1节，第3段】原文 | 【第7.1节，第3段】翻译 |
| ---- | ---- |
| ✅ Beyond image captioning pre-training task, some research ( **1. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **2. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **3. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** ) attempts to formulate more vision tasks in a unified sequence-to-sequence learning paradigm, including object detection and image segmentation. | ✅ 除了图像字幕预训练任务之外，一些研究 ( **1. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **2. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **3. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** ) 尝试在统一的序列到序列学习范式中制定更多的视觉任务，包括对象检测和图像分割。 |
| ✅ Customized special tokens accommodate representations beyond pure text, such as bounding boxes ( **1. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **2. Pix2seq: A language modeling framework for object detection, 2022.** ｜ **3. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** ). | ✅ 定制的特殊标记可适应纯文本以外的表示，例如边界框 ( **1. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ｜ **2. Pix2seq: A language modeling framework for object detection, 2022.** ｜ **3. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.** )。 |
| ✅ This approach uses the same architecture for pre-training and downstream tasks, potentially using the same set of weights for all tasks. | ✅ 该方法对预训练和下游任务使用相同的架构，可能对所有任务使用相同的权重集。 |
| ✅ Our method, which falls into this category, aims to obtain foundation models that understand dense information beyond simple image-level captions. | ✅ 我们的方法属于这一类，旨在获得能够理解简单图像级标题之外的密集信息的基础模型。 |
| ✅ It shares the same encoder-decoder design as other multi-modality encoder-decoder models ( **1. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **2. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ) adapted for sequence-to-sequence learning, but uses our built large-scale comprehensive annotation data instead of combining existing sparse annotated data. | ✅ 它与其他适用于序列到序列学习的多模态编码器-解码器模型 ( **1. Pali: A jointly-scaled multilingual language-image model, 2022.** ｜ **2. Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.** ) 具有相同的编码器-解码器设计，但使用我们构建的大规模综合注释数据，而不是结合现有的稀疏注释数据。 |

### 7.2 Vision Datasets

#### 7.2.1 Comprehensive annotations.

| 【第7.2.1节，第1段】原文 | 【第7.2.1节，第1段】翻译 |
| ---- | ---- |
| ✅ The quest for comprehensive understanding of visual scenes, the holy grail of computer vision ( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ) , has evolved from focusing on individual datasets each targeting a single perspective, e.g. | ✅ 对视觉场景进行全面理解的追求，即计算机视觉的终极目标 ( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** )，已经从关注每个针对单一视角的单个数据集 e.g 发展而来。 |
| ✅  , image classification ( **Imagenet: A large-scale hierarchical image database.** ) , to providing multi-perspective ( **1. Microsoft coco: Common objects in context.** ｜ **2. Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ｜ **3. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.** ) , comprehensive annotations for every visual data point. | ✅ 、图像分类( **Imagenet: A large-scale hierarchical image database.** )、为每个视觉数据点提供多视角( **1. Microsoft coco: Common objects in context.** ｜ **2. Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ｜ **3. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.** )、全面的注释。 |
| ✅ Notable datasets like MS-COCO ( **1. Microsoft coco: Common objects in context.** ｜ **2. Microsoft coco captions: Data collection and evaluation server.** ) and Visual Genome ( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ) integrate various types of annotations, enabling richer understanding in spatial and semantic granularities and better model interactions across annotations. | ✅ 著名的数据集如 MS-COCO ( **1. Microsoft coco: Common objects in context.** ｜ **2. Microsoft coco captions: Data collection and evaluation server.** ) 和 Visual Genome ( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ) 集成了各种类型的注释，从而能够更丰富地理解空间和语义粒度，并实现跨注释的更好的模型交互。 |
| ✅ However, due to the high cost of human verification, these annotations are limited in size. | ✅ 但由于人工验证的成本较高，这些注释的大小受到限制。 |
| ✅ Our datasets, while large-scale, maintain comprehensive annotations covering text, region-text pairs, and text-phrase-region triplets, with reduced human involvement. | ✅ 我们的数据集虽然规模很大，但仍保持了涵盖文本、区域-文本对和文本-短语-区域三元组的全面注释，同时减少了人工参与。 |

#### 7.2.2 Scalable annotations.

| 【第7.2.2节，第1段】原文 | 【第7.2.2节，第1段】翻译 |
| ---- | ---- |
| ✅ : Over the past decade, vision datasets have rapidly scaled up from thousands ( **1. Mnist handwritten digit database.** ｜ **2. Learning multiple layers of features from tiny images.** ) to billion examples ( **1. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ｜ **2. Scaling vision transformers.** ) to encompass more visual concepts for better generalization. | ✅ ：在过去十年中，视觉数据集已从数千个 ( **1. Mnist handwritten digit database.** ｜ **2. Learning multiple layers of features from tiny images.** ) 迅速扩大到十亿个示例 ( **1. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ｜ **2. Scaling vision transformers.** )，以涵盖更多的视觉概念，从而实现更好的泛化。 |
| ✅ This shift is evident in recent foundation models that employ massive quantities of data ( **On the opportunities and risks of foundation models.** ). | ✅ 这种转变在最近采用大量数据 ( **On the opportunities and risks of foundation models.** ) 的基础模型中显而易见。 |
| ✅ These large datasets typically collect images from the web and parse noisy annotations from the corresponding metadata, such as category label from query ( **1. Revisiting unreasonable effectiveness of data in deep learning era.** ｜ **2. Scaling vision transformers.** ) , short description from alt-text ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ) , as well as detailed description from interleaved text ( **1. Flamingo: a visual language model for few-shot learning.** ｜ **2. Obelisc: An open web-scale filtered dataset of interleaved image-text documents.** ). | ✅ 这些大型数据集通常从网络上收集图像，并从相应的元数据中解析噪声注释，例如来自查询 ( **1. Revisiting unreasonable effectiveness of data in deep learning era.** ｜ **2. Scaling vision transformers.** ) 的类别标签、来自替代文本 ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision, 2021.** ) 的简短描述，以及来自交错文本 ( **1. Flamingo: a visual language model for few-shot learning.** ｜ **2. Obelisc: An open web-scale filtered dataset of interleaved image-text documents.** ) 的详细描述。 |
| ✅ Despite their diversity, these annotations suffer from randomness and limited types ( i.e. | ✅ 尽管这些注释具有多样性，但它们却具有随机性和有限类型（i.e）。 |
| ✅  , texts only). | ✅ ，仅限文本）。 |
| ✅ Some works ( **1. Segment anything.** ｜ **2. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ) attempt to scale up annotations using pseudo-label generation with iteratively trained models, which offer higher quality without significant diversity loss. | ✅ 一些研究 ( **1. Segment anything.** ｜ **2. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.** ) 尝试使用经过迭代训练的模型生成伪标签来扩大注释，这可以提供更高的质量而不会造成明显的多样性损失。 |
| ✅ Our data pipeline extends these large-scale, web-crawled noisy annotations with higher-quality, autonomous annotations generated from multiple specialist models. | ✅ 我们的数据管道利用由多个专家模型生成的更高质量、自主的注释来扩展这些大规模、网络爬行的噪声注释。 |
| ✅ The pipeline iteratively refines labels and completes missing pieces, resulting in a scalable and comprehensive dataset for learning a unified visual representation. | ✅ 该管道迭代地细化标签并完成缺失的部分，从而产生一个可扩展且全面的数据集，用于学习统一的视觉表示。 |

## 8 Conclusion

| 【第8节，第1段】原文 | 【第8节，第1段】翻译 |
| ---- | ---- |
| ✅ The Florence Project endeavors to develop a foundational vision model endowed with a diverse array of perceptual capabilities, encompassing spatial hierarchy and semantic granularity. | ✅ 佛罗伦萨项目致力于开发一种具有多种感知能力的基础视觉模型，涵盖空间层次和语义粒度。 |
| ✅ To this end, we construct FLD-5B dataset containing an extensive collection of 126M images paired with 5B comprehensive annotations, which are collected by the Florence data engine. | ✅ 为此，我们构建了 FLD-5B 数据集，其中包含 126M 张图像和 5B 份综合注释，由 Florence 数据引擎收集。 |
| ✅ Subsequently, we pre-train Florence-2 on this rich dataset through comprehensive multitask learning in a unified manner. | ✅ 随后，我们在这个丰富的数据集上通过统一的方式进行综合多任务学习对Florence-2进行预训练。 |
| ✅ Florence-2 has exhibited remarkable zero-shot capabilities that extend across a wide spectrum of visual tasks, such as captioning, object detection, visual grounding, and referring segmentation, among others. | ✅ Florence-2 表现出卓越的零样本能力，可涵盖广泛的视觉任务，例如字幕、对象检测、视觉基础和指称分割等。 |
| ✅ The experimental findings underscore the potency of the universal representation pre-trained by Florence-2 , revealing its substantial contributions to the enhancement of a multitude of downstream tasks. | ✅ 实验结果强调了 Florence-2 预训练的通用表示的效力，揭示了其对增强大量下游任务的重大贡献。 |

#### 8.1 Acknowledgment.

| 【第8.1节，第1段】原文 | 【第8.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We would like to express our heartfelt gratitude to all the contributors from the Azure AI team who worked on the Florence project. | ✅ 我们想向 Azure AI 团队所有参与 Florence 项目的贡献者表示衷心的感谢。 |
| ✅ We sincerely appreciate Misha Bilenko for the invaluable guidance and support. | ✅ 我们真诚感谢 Misha Bilenko 的宝贵指导和支持。 |
| ✅ Our thanks are extended to Yi-Ling Chen, Mengchen Liu, Yen-Chun Chen and Dongdong Chen for engaging in helpful discussions and to Yunsheng Li for their assistance with segmentation annotations. | ✅ 我们感谢 Yi-Ling Chen、Mengchen Liu、Yen-Chun Chen 和 Dongdong Chen 参与的有益讨论，以及感谢 Yunsheng Li 对分割注释的帮助。 |
| ✅ Deep appreciation is also expressed to Qingfen Lin, Ryan Menezes, Kuan Lu, Gabe Blanco, Shohei Ono, Ping Jin, Jiahe Zhou, Xiong Qiao, Tong Bai, Xingchao Peng, Pei Guo, Lihang Li for providing valuable feedback in downstream applications discussions. | ✅ 同时，我们也对青芬林 (Qingfen Lin)、瑞安梅内泽斯 (Ryan Menezes)、宽 (Kuan Lu)、加贝布兰科 (Gabe Blanco)、小野翔平 (Shohei Ono)、金平 (Ping Jin)、周嘉禾 (Jiahe Zhou)、乔雄 (Xiong Qiao)、白桐 (Tong Bai)、彭兴超 (Xingchao Peng)、郭培 (Pei Guo)、李航 (Lihang Li) 在下游应用讨论中提供的宝贵反馈表示深深的感谢。 |
| ✅ Special thanks to Cha Zhang, Jinyu Li, Min Gao, Christina Sun, Oliver Ernst, Kevin Pan, Mei Gao for their work on data annotation support and insightful discussions in data pipeline. | ✅ 特别感谢 Cha Zhang、Jinyu Li、Min Gao、Christina Sun、Oliver Ernst、Kevin Pan 和 Mei Gao 在数据注释支持方面所做的工作以及在数据管道中的深刻讨论。 |
| ✅ Furthermore, we would like to thank Thomas Soemo, Nguyen Bach for their constructive feedback. | ✅ 此外，我们还要感谢 Thomas Soemo 和 Nguyen Bach 的建设性反馈。 |

## 9 References

- 1
  - Azure ai services.
  - **https://azure.microsoft.com/en-us/products/ai-services?activetab=pivot:azureopenaiservicetab.**
  - Accessed: 2023-10-13.

- 2
  - Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al.
  - **Flamingo: a visual language model for few-shot learning.**
  - Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

- 3
  - Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton.
  - **Layer normalization, 2016.**

- 4
  - Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei.
  - **BEiT: BERT pre-training of image transformers.**
  - In International Conference on Learning Representations, 2022.

- 5
  - Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al.
  - **On the opportunities and risks of foundation models.**
  - arXiv preprint arXiv:2108.07258, 2021.

- 6
  - Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.
  - **Language models are few-shot learners.**
  - In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020.

- 7
  - Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.
  - **End-to-end object detection with transformers.**
  - In European conference on computer vision, pages 213–229. Springer, 2020.

- 8
  - Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin.
  - **Unsupervised learning of visual features by contrasting cluster assignments.**
  - In Advances in Neural Information Processing Systems, volume 33, 2020.

- 9
  - Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
  - **A simple framework for contrastive learning of visual representations.**
  - In International conference on machine learning, pages 1597–1607. PMLR, 2020.

- 10
  - Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, and Geoffrey Hinton.
  - **Pix2seq: A language modeling framework for object detection, 2022.**

- 11
  - Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J Fleet, and Geoffrey E Hinton.
  - **A unified sequence interface for vision tasks.**
  - Advances in Neural Information Processing Systems, 35:31333–31346, 2022.

- 12
  - Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al.
  - **Pali-x: On scaling up a multilingual vision and language model.**
  - arXiv preprint arXiv:2305.18565, 2023.

- 13
  - Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick.
  - **Microsoft coco captions: Data collection and evaluation server.**
  - arXiv preprint arXiv:1504.00325, 2015.

- 14
  - Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Soricut.
  - **Pali-3 vision language models: Smaller, faster, stronger, 2023.**

- 15
  - Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut.
  - **Pali: A jointly-scaled multilingual language-image model, 2022.**

- 16
  - Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar.
  - **Masked-attention mask transformer for universal image segmentation.**
  - 2022.

- 17
  - Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio.
  - **Learning phrase representations using rnn encoder-decoder for statistical machine translation.**
  - arXiv preprint arXiv:1406.1078, 2014.

- 18
  - Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.
  - **Imagenet: A large-scale hierarchical image database.**
  - In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

- 19
  - Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
  - **Bert: Pre-training of deep bidirectional transformers for language understanding, 2019.**

- 20
  - Mingyu Ding, Bin Xiao, Noel Codella, Ping Luo, Jingdong Wang, and Lu Yuan.
  - **Davit: Dual attention vision transformers.**
  - In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIV, pages 74–92. Springer, 2022.

- 21
  - Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.
  - **An image is worth 16x16 words: Transformers for image recognition at scale, 2021.**

- 22
  - Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh.
  - **Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering.**
  - In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

- 23
  - Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham.
  - **Vizwiz grand challenge: Answering visual questions from blind people.**
  - In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018.

- 24
  - Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick.
  - **Masked autoencoders are scalable vision learners.**
  - In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

- 25
  - Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick.
  - **Momentum contrast for unsupervised visual representation learning.**
  - In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

- 26
  - Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.
  - **Mask r-cnn.**
  - In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.

- 27
  - Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  - **Deep residual learning for image recognition.**
  - In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

- 28
  - Matthew Honnibal, Ines Montani, Sofie Van Landeghem, Adriane Boyd, et al.
  - **spacy: Industrial-strength natural language processing in python.**
  - 2020.

- 29
  - Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig.
  - **Scaling up visual and vision-language representation learning with noisy text supervision, 2021.**

- 30
  - Andrej Karpathy and Li Fei-Fei.
  - **Deep visual-semantic alignments for generating image descriptions.**
  - 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3128–3137, 2014.

- 31
  - Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg.
  - **Referitgame: Referring to objects in photographs of natural scenes.**
  - In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.

- 32
  - Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.
  - **Segment anything.**
  - arXiv preprint arXiv:2304.02643, 2023.

- 33
  - Aniket Kittur, Ed Chi, Bryan A Pendleton, Bongwon Suh, and Todd Mytkowicz.
  - **Power of the few vs. wisdom of the crowd: Wikipedia and the rise of the bourgeoisie.**
  - World wide web, 1(2):19, 2007.

- 34
  - Jonathan Krause, Justin Johnson, Ranjay Krishna, and Li Fei-Fei.
  - **A hierarchical approach for generating descriptive image paragraphs.**
  - In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 317–325, 2017.

- 35
  - Jonathan Krause, Justin Johnson, Ranjay Krishna, and Li Fei-Fei.
  - **A hierarchical approach for generating descriptive image paragraphs.**
  - In Computer Vision and Patterm Recognition (CVPR), 2017.

- 36
  - Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al.
  - **Visual genome: Connecting language and vision using crowdsourced dense image annotations.**
  - International journal of computer vision, 123:32–73, 2017.

- 37
  - Alex Krizhevsky, Geoffrey Hinton, et al.
  - **Learning multiple layers of features from tiny images.**
  - 2009.

- 38
  - Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
  - **Imagenet classification with deep convolutional neural networks.**
  - In Advances in neural information processing systems, pages 1097–1105, 2012.

- 39
  - Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari.
  - **The open images dataset v4.**
  - International Journal of Computer Vision, 128(7):1956–1981, mar 2020.

- 40
  - Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al.
  - **The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale.**
  - International Journal of Computer Vision, 128(7):1956–1981, 2020.

- 41
  - Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M Rush, Douwe Kiela, et al.
  - **Obelisc: An open web-scale filtered dataset of interleaved image-text documents.**
  - arXiv preprint arXiv:2306.16527, 2023.

- 42
  - Yann LeCun, Corinna Cortes, and CJ Burges.
  - **Mnist handwritten digit database.**
  - ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

- 43
  - Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer.
  - **Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.**

- 44
  - Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
  - **Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.**
  - arXiv preprint arXiv:2301.12597, 2023.

- 45
  - Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.
  - **Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.**
  - In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.

- 46
  - Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He.
  - **Exploring plain vision transformer backbones for object detection.**
  - In European Conference on Computer Vision, pages 280–296. Springer, 2022.

- 47
  - Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár.
  - **Microsoft coco: Common objects in context, 2015.**

- 48
  - Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick.
  - **Microsoft coco: Common objects in context.**
  - In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

- 49
  - Jiang Liu, Hui Ding, Zhaowei Cai, Yuting Zhang, Ravi Kumar Satzoda, Vijay Mahadevan, and R Manmatha.
  - **Polyformer: Referring image segmentation as sequential polygon generation.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18653–18663, 2023.

- 50
  - Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al.
  - **Grounding dino: Marrying dino with grounded pre-training for open-set object detection.**
  - arXiv preprint arXiv:2303.05499, 2023.

- 51
  - Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo.
  - **Swin transformer: Hierarchical vision transformer using shifted windows, 2021.**

- 52
  - Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.
  - **A convnet for the 2020s.**
  - In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

- 53
  - Ilya Loshchilov and Frank Hutter.
  - **Sgdr: Stochastic gradient descent with warm restarts, 2017.**

- 54
  - Ilya Loshchilov and Frank Hutter.
  - **Decoupled weight decay regularization, 2019.**

- 55
  - Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi.
  - **Unified-io: A unified model for vision, language, and multi-modal tasks, 2022.**

- 56
  - Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy.
  - **Generation and comprehension of unambiguous object descriptions.**
  - In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016.

- 57
  - Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi.
  - **Ok-vqa: A visual question answering benchmark requiring external knowledge, 2019.**

- 58
  - Aaron van den Oord, Yazhe Li, and Oriol Vinyals.
  - **Representation learning with contrastive predictive coding.**
  - arXiv preprint arXiv:1807.03748, 2018.

- 59
  - Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei.
  - **BEiT v2: Masked image modeling with vector-quantized visual tokenizers.**
  - 2022.

- 60
  - Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei.
  - **Kosmos-2: Grounding multimodal large language models to the world.**
  - arXiv preprint arXiv:2306.14824, 2023.

- 61
  - Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik.
  - **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.**
  - In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015.

- 62
  - Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari.
  - **Connecting vision and language with localized narratives.**
  - In ECCV, 2020.

- 63
  - Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan.
  - **Filtering, distillation, and hard negatives for vision-language pre-training.**
  - arXiv preprint arXiv:2301.02280, 2023.

- 64
  - Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
  - **Learning transferable visual models from natural language supervision.**
  - In International conference on machine learning, pages 8748–8763. PMLR, 2021.

- 65
  - Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.
  - **Language models are unsupervised multitask learners.**
  - 2019.

- 66
  - Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.
  - **Exploring the limits of transfer learning with a unified text-to-text transformer.**
  - The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

- 67
  - Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He.
  - **Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.**
  - In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.

- 68
  - Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki.
  - **Laion-400m: Open dataset of clip-filtered 400 million image-text pairs.**
  - arXiv preprint arXiv:2111.02114, 2021.

- 69
  - Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi.
  - **A-okvqa: A benchmark for visual question answering using world knowledge, 2022.**

- 70
  - Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun.
  - **Objects365: A large-scale, high-quality dataset for object detection.**
  - In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019.

- 71
  - Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut.
  - **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.**
  - In Proceedings of ACL, 2018.

- 72
  - Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh.
  - **Textcaps: a dataset for image captioning with reading comprehension, 2020.**

- 73
  - Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach.
  - **Towards vqa models that can read.**
  - In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.

- 74
  - Kihyuk Sohn.
  - **Improved deep metric learning with multi-class n-pair loss objective.**
  - Advances in neural information processing systems, 29, 2016.

- 75
  - Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta.
  - **Revisiting unreasonable effectiveness of data in deep learning era.**
  - In Proceedings of the IEEE international conference on computer vision, pages 843–852, 2017.

- 76
  - Ilya Sutskever, Oriol Vinyals, and Quoc V Le.
  - **Sequence to sequence learning with neural networks.**
  - Advances in neural information processing systems, 27, 2014.

- 77
  - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
  - **Attention is all you need.**
  - In Advances in neural information processing systems, pages 5998–6008, 2017.

- 78
  - Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang.
  - **Git: A generative image-to-text transformer for vision and language, 2022.**

- 79
  - Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang.
  - **Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework, 2022.**

- 80
  - Nic M Weststrate, Susan Bluck, and Judith Glück.
  - **Wisdom of the crowd.**
  - The Cambridge handbook of wisdom, pages 97–121, 2019.

- 81
  - Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie.
  - **Convnext v2: Co-designing and scaling convnets with masked autoencoders.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16133–16142, 2023.

- 82
  - Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun.
  - **Unified perceptual parsing for scene understanding.**
  - In Proceedings of the European conference on computer vision (ECCV), pages 418–434, 2018.

- 83
  - Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu.
  - **Simmim: A simple framework for masked image modeling.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9653–9663, 2022.

- 84
  - Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu.
  - **Universal instance perception as object discovery and retrieval.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15325–15336, 2023.

- 85
  - Jianwei Yang, Chunyuan Li, Xiyang Dai, and Jianfeng Gao.
  - **Focal modulation networks.**
  - Advances in Neural Information Processing Systems, 35:4203–4217, 2022.

- 86
  - Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, and Jianfeng Gao.
  - **Focal self-attention for local-global interactions in vision transformers.**
  - arXiv preprint arXiv:2107.00641, 2021.

- 87
  - Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao.
  - **Unified contrastive learning in image-text-label space, 2022.**

- 88
  - Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Faisal Ahmed, Zicheng Liu, Yumao Lu, and Lijuan Wang.
  - **Unitab: Unifying text and box outputs for grounded vision-language modeling.**
  - In European Conference on Computer Vision, pages 521–539. Springer, 2022.

- 89
  - Sheng Kung Michael Yi, Mark Steyvers, Michael D Lee, and Matthew J Dry.
  - **The wisdom of the crowd in combinatorial problems.**
  - Cognitive science, 36(3):452–470, 2012.

- 90
  - Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang.
  - **Ferret: Refer and ground anything anywhere at any granularity, 2023.**

- 91
  - Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier.
  - **From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions.**
  - Transactions of the Association for Computational Linguistics, 2:67–78, 2014.

- 92
  - Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu.
  - **Coca: Contrastive captioners are image-text foundation models, 2022.**

- 93
  - Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg.
  - **Modeling context in referring expressions.**
  - In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016.

- 94
  - Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg.
  - **Modeling context in referring expressions.**
  - In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision – ECCV 2016, pages 69–85, Cham, 2016. Springer International Publishing.

- 95
  - Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu, Zicheng Liu, Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao, Zhen Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, and Pengchuan Zhang.
  - **Florence: A new foundation model for computer vision.**
  - arXiv preprint arXiv:2111.11432, 2021.

- 96
  - Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer.
  - **Scaling vision transformers.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12104–12113, 2022.

- 97
  - Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum.
  - **Dino: Detr with improved denoising anchor boxes for end-to-end object detection.**
  - arXiv preprint arXiv:2203.03605, 2022.

- 98
  - Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba.
  - **Scene parsing through ade20k dataset.**
  - In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017.

- 99
  - Chaoyang Zhu, Yiyi Zhou, Yunhang Shen, Gen Luo, Xingjia Pan, Mingbao Lin, Chao Chen, Liujuan Cao, Xiaoshuai Sun, and Rongrong Ji.
  - **Seqtr: A simple yet universal network for visual grounding.**
  - In European Conference on Computer Vision, pages 598–615. Springer, 2022.

- 100
  - Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai.
  - **Deformable detr: Deformable transformers for end-to-end object detection.**
  - arXiv preprint arXiv:2010.04159, 2020.

## 10 Appendix A Supported Tasks and Annotations in Florence-2

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="A1.T13.2"><thead class="ltx_thead"><tr class="ltx_tr" id="A1.T13.2.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="A1.T13.2.1.1.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Task</th><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="A1.T13.2.1.1.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Annotation Type</th><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="A1.T13.2.1.1.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Prompt Input</th><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt" id="A1.T13.2.1.1.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Output</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="A1.T13.2.2.1"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A1.T13.2.2.1.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Caption</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A1.T13.2.2.1.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Text</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A1.T13.2.2.1.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left ltx_border_t" id="A1.T13.2.2.1.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text</td></tr><tr class="ltx_tr" id="A1.T13.2.3.2"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.3.2.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Detailed caption</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.3.2.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.3.2.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.3.2.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text</td></tr><tr class="ltx_tr" id="A1.T13.2.4.3"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.4.3.1" style="padding-left:18.0pt;padding-right:18.0pt;">      More detailed caption</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.4.3.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.4.3.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.4.3.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text</td></tr><tr class="ltx_tr" id="A1.T13.2.5.4"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.5.4.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Region proposal</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.5.4.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.5.4.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.5.4.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Region</td></tr><tr class="ltx_tr" id="A1.T13.2.6.5"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.6.5.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Object detection</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.6.5.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region-Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.6.5.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.6.5.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text, region</td></tr><tr class="ltx_tr" id="A1.T13.2.7.6"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.7.6.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Dense region caption</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.7.6.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region-Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.7.6.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.7.6.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text, region</td></tr><tr class="ltx_tr" id="A1.T13.2.8.7"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.8.7.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Phrase grounding</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.8.7.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Text-Phrase-Region</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.8.7.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.8.7.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text, region</td></tr><tr class="ltx_tr" id="A1.T13.2.9.8"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.9.8.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Referring expression comprehension</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.9.8.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region-Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.9.8.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.9.8.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text, region</td></tr><tr class="ltx_tr" id="A1.T13.2.10.9"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.10.9.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Open vocabulary detection</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.10.9.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region-Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.10.9.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.10.9.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text, region</td></tr><tr class="ltx_tr" id="A1.T13.2.11.10"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.11.10.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Referring segmentation</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.11.10.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region-Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.11.10.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left" id="A1.T13.2.11.10.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text, region</td></tr><tr class="ltx_tr" id="A1.T13.2.12.11"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.12.11.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Region to text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.12.11.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region-Text</td><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T13.2.12.11.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text, region</td><td class="ltx_td ltx_align_left" id="A1.T13.2.12.11.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text</td></tr><tr class="ltx_tr" id="A1.T13.2.13.12"><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="A1.T13.2.13.12.1" style="padding-left:18.0pt;padding-right:18.0pt;">      Text detection and recognition</td><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="A1.T13.2.13.12.2" style="padding-left:18.0pt;padding-right:18.0pt;">      Region-Text</td><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="A1.T13.2.13.12.3" style="padding-left:18.0pt;padding-right:18.0pt;">      Image, text</td><td class="ltx_td ltx_align_left ltx_border_bb" id="A1.T13.2.13.12.4" style="padding-left:18.0pt;padding-right:18.0pt;">      Text, region</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 13:  Supported Tasks and annotations used for Florence-2 pretraining. | ✅ Table 13:  Supported Tasks and annotations used for Florence-2 pretraining. |

## 11 Appendix B Supervised Data Collection for Generalist Model Fine-tuning

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="A2.T14.6"><tbody class="ltx_tbody"><tr class="ltx_tr" id="A2.T14.6.7.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="A2.T14.6.7.1.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Task</th><td class="ltx_td ltx_align_left ltx_border_tt" id="A2.T14.6.7.1.2" style="padding-left:20.0pt;padding-right:20.0pt;">      Dataset</td></tr><tr class="ltx_tr" id="A2.T14.6.8.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="A2.T14.6.8.2.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Caption</th><td class="ltx_td ltx_align_left ltx_border_t" id="A2.T14.6.8.2.2" style="padding-left:20.0pt;padding-right:20.0pt;">      COCO <html><body><p>( <strong>Microsoft coco captions: Data collection and evaluation server.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.9.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.9.3.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Text Caption</th><td class="ltx_td ltx_align_left" id="A2.T14.6.9.3.2" style="padding-left:20.0pt;padding-right:20.0pt;">      TextCaps <html><body><p>( <strong>Textcaps: a dataset for image captioning with reading comprehension,2020.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.10.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.10.4.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Paragraph caption</th><td class="ltx_td ltx_align_left" id="A2.T14.6.10.4.2" style="padding-left:20.0pt;padding-right:20.0pt;">      Standford Paragraph Caption <html><body><p>( <strong>A hierarchical approach for generating descriptive image paragraphs.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.11.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.11.5.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Detailed caption</th><td class="ltx_td ltx_align_left" id="A2.T14.6.11.5.2" style="padding-left:20.0pt;padding-right:20.0pt;">      Localized Narratives <html><body><p>( <strong>Connecting vision and language with localized narratives.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.2.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.2.2.3" style="padding-left:20.0pt;padding-right:20.0pt;">      Detection</th><td class="ltx_td ltx_align_left" id="A2.T14.2.2.2" style="padding-left:20.0pt;padding-right:20.0pt;">      COCO <html><body><p>( <strong>Microsoft coco: Common objects in context, 2015.</strong> )</p></body></html>, Object365<sup class="ltx_sup" id="A2.T14.2.2.2.5">∗</sup> <html><body><p>( <strong>Objects365: A large-scale, high-quality dataset for object detection.</strong> )</p></body></html>, Open Images<sup class="ltx_sup" id="A2.T14.2.2.2.10">∗</sup> <html><body><p>( <strong>The open images dataset v4.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.4.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.4.4.3" style="padding-left:20.0pt;padding-right:20.0pt;">      Phrase Grounding</th><td class="ltx_td ltx_align_left" id="A2.T14.4.4.2" style="padding-left:20.0pt;padding-right:20.0pt;">      Flickr30k, Object365<sup class="ltx_sup" id="A2.T14.4.4.2.2">∗</sup> <html><body><p>( <strong>Objects365: A large-scale, high-quality dataset for object detection.</strong> )</p></body></html>, Open Images<sup class="ltx_sup" id="A2.T14.4.4.2.7">∗</sup> <html><body><p>( <strong>The open images dataset v4.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.12.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.12.6.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Referring expression</th><td class="ltx_td ltx_align_left" id="A2.T14.6.12.6.2" style="padding-left:20.0pt;padding-right:20.0pt;">      RefCOCO-mix (RefCOCO, RefCOCO+, RefCOCOg) <html><body><p>( <strong>1. Referitgame: Referring to objects in photographs of natural scenes.</strong> ｜ <strong>2. Modeling context in referring expressions.</strong> ｜ <strong>3. Generation and comprehension of unambiguous object descriptions.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.13.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.13.7.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Referring expression segmentation</th><td class="ltx_td ltx_align_left" id="A2.T14.6.13.7.2" style="padding-left:20.0pt;padding-right:20.0pt;">      RefCOCO-mix (RefCOCO, RefCOCO+, RefCOCOg) <html><body><p>( <strong>1. Referitgame: Referring to objects in photographs of natural scenes.</strong> ｜ <strong>2. Modeling context in referring expressions.</strong> ｜ <strong>3. Generation and comprehension of unambiguous object descriptions.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.6.3" style="padding-left:20.0pt;padding-right:20.0pt;">      Region to category</th><td class="ltx_td ltx_align_left" id="A2.T14.6.6.2" style="padding-left:20.0pt;padding-right:20.0pt;">      COCO <html><body><p>( <strong>Microsoft coco: Common objects in context, 2015.</strong> )</p></body></html>, Object365<sup class="ltx_sup" id="A2.T14.6.6.2.5">∗</sup> <html><body><p>( <strong>Objects365: A large-scale, high-quality dataset for object detection.</strong> )</p></body></html>, Open Images<sup class="ltx_sup" id="A2.T14.6.6.2.10">∗</sup> <html><body><p>( <strong>The open images dataset v4.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.14.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.14.8.1" style="padding-left:20.0pt;padding-right:20.0pt;">      Region to polygon</th><td class="ltx_td ltx_align_left" id="A2.T14.6.14.8.2" style="padding-left:20.0pt;padding-right:20.0pt;">      COCO <html><body><p>( <strong>Microsoft coco: Common objects in context, 2015.</strong> )</p></body></html> (after deduplicating RefCOCO-mix val)</td></tr><tr class="ltx_tr" id="A2.T14.6.15.9"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="A2.T14.6.15.9.1" style="padding-left:20.0pt;padding-right:20.0pt;">      VQA</th><td class="ltx_td ltx_align_left" id="A2.T14.6.15.9.2" style="padding-left:20.0pt;padding-right:20.0pt;">      VQAv2 <html><body><p>( <strong>Making the V in VQA matter: Elevating the role of imageunderstanding in Visual Question Answering.</strong> )</p></body></html>, OKVQA <html><body><p>( <strong>Ok-vqa: A visual question answering benchmark requiring externalknowledge, 2019.</strong> )</p></body></html>, AOKVQA <html><body><p>( <strong>A-okvqa: A benchmark for visual question answering using worldknowledge, 2022.</strong> )</p></body></html>, TextVQA <html><body><p>( <strong>Towards vqa models that can read.</strong> )</p></body></html>, ViZWiz VQA <html><body><p>( <strong>Vizwiz grand challenge: Answering visual questions from blind people.</strong> )</p></body></html></td></tr><tr class="ltx_tr" id="A2.T14.6.16.10"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="A2.T14.6.16.10.1" style="padding-left:20.0pt;padding-right:20.0pt;">      OCR</th><td class="ltx_td ltx_align_left ltx_border_bb" id="A2.T14.6.16.10.2" style="padding-left:20.0pt;padding-right:20.0pt;">      Subset from <em class="ltx_emph ltx_font_italic" id="A2.T14.6.16.10.2.2" style="font-size:90%;">FLD-5B</em> OCR (2 millon samples)</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 14:  Collection of dataset for finetuning one single generalist model for downstream tasks evaluation. | ✅ Table 14:  用于微调单一通用模型以供下游任务评估的数据集集合。 |
| ✅ ∗ indicates using the annotations from FLD-5B , which merges original annotations with ours. | ✅ ∗ 表示使用来自 FLD-5B 的注释，它将原始注释与我们的注释合并。 |

## 12 Appendix C Model Configuration

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="A3.T15.2"><thead class="ltx_thead"><tr class="ltx_tr" id="A3.T15.2.1.1"><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="A3.T15.2.1.1.1" rowspan="2" style="padding-left:3.7pt;padding-right:3.7pt;">Model</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt" colspan="4" id="A3.T15.2.1.1.2" style="padding-left:3.7pt;padding-right:3.7pt;">Image Encoder (DaViT)</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="4" id="A3.T15.2.1.1.3" style="padding-left:3.7pt;padding-right:3.7pt;">Encoder-Decoder (Transformer)</th></tr><tr class="ltx_tr" id="A3.T15.2.2.2"><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="A3.T15.2.2.2.1" style="padding-left:3.7pt;padding-right:3.7pt;">dimensions</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="A3.T15.2.2.2.2" style="padding-left:3.7pt;padding-right:3.7pt;">blocks</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="A3.T15.2.2.2.3" style="padding-left:3.7pt;padding-right:3.7pt;">heads/groups</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r" id="A3.T15.2.2.2.4" style="padding-left:3.7pt;padding-right:3.7pt;">#params</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="A3.T15.2.2.2.5" style="padding-left:3.7pt;padding-right:3.7pt;">encoder layers</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="A3.T15.2.2.2.6" style="padding-left:3.7pt;padding-right:3.7pt;">decoder layers</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="A3.T15.2.2.2.7" style="padding-left:3.7pt;padding-right:3.7pt;">dimensions</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="A3.T15.2.2.2.8" style="padding-left:3.7pt;padding-right:3.7pt;">#params</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="A3.T15.2.3.1"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" id="A3.T15.2.3.1.1" style="padding-left:3.7pt;padding-right:3.7pt;"><em class="ltx_emph ltx_font_italic" id="A3.T15.2.3.1.1.1" style="font-size:90%;">Florence-2-B</em></th><td class="ltx_td ltx_align_center ltx_border_t" id="A3.T15.2.3.1.2" style="padding-left:3.7pt;padding-right:3.7pt;">[128, 256, 512, 1024]</td><td class="ltx_td ltx_align_center ltx_border_t" id="A3.T15.2.3.1.3" style="padding-left:3.7pt;padding-right:3.7pt;">[1, 1, 9, 1]</td><td class="ltx_td ltx_align_center ltx_border_t" id="A3.T15.2.3.1.4" style="padding-left:3.7pt;padding-right:3.7pt;">[4, 8, 16, 32]</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="A3.T15.2.3.1.5" style="padding-left:3.7pt;padding-right:3.7pt;">90M</td><td class="ltx_td ltx_align_center ltx_border_t" id="A3.T15.2.3.1.6" style="padding-left:3.7pt;padding-right:3.7pt;">6</td><td class="ltx_td ltx_align_center ltx_border_t" id="A3.T15.2.3.1.7" style="padding-left:3.7pt;padding-right:3.7pt;">6</td><td class="ltx_td ltx_align_center ltx_border_t" id="A3.T15.2.3.1.8" style="padding-left:3.7pt;padding-right:3.7pt;">768</td><td class="ltx_td ltx_align_center ltx_border_t" id="A3.T15.2.3.1.9" style="padding-left:3.7pt;padding-right:3.7pt;">140M</td></tr><tr class="ltx_tr" id="A3.T15.2.4.2"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="A3.T15.2.4.2.1" style="padding-left:3.7pt;padding-right:3.7pt;"><em class="ltx_emph ltx_font_italic" id="A3.T15.2.4.2.1.1" style="font-size:90%;">Florence-2-L</em></th><td class="ltx_td ltx_align_center ltx_border_bb" id="A3.T15.2.4.2.2" style="padding-left:3.7pt;padding-right:3.7pt;">[256, 512, 1024, 2048]</td><td class="ltx_td ltx_align_center ltx_border_bb" id="A3.T15.2.4.2.3" style="padding-left:3.7pt;padding-right:3.7pt;">[1, 1, 9, 1]</td><td class="ltx_td ltx_align_center ltx_border_bb" id="A3.T15.2.4.2.4" style="padding-left:3.7pt;padding-right:3.7pt;">[8, 16, 32, 64]</td><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r" id="A3.T15.2.4.2.5" style="padding-left:3.7pt;padding-right:3.7pt;">360M</td><td class="ltx_td ltx_align_center ltx_border_bb" id="A3.T15.2.4.2.6" style="padding-left:3.7pt;padding-right:3.7pt;">12</td><td class="ltx_td ltx_align_center ltx_border_bb" id="A3.T15.2.4.2.7" style="padding-left:3.7pt;padding-right:3.7pt;">12</td><td class="ltx_td ltx_align_center ltx_border_bb" id="A3.T15.2.4.2.8" style="padding-left:3.7pt;padding-right:3.7pt;">1024</td><td class="ltx_td ltx_align_center ltx_border_bb" id="A3.T15.2.4.2.9" style="padding-left:3.7pt;padding-right:3.7pt;">410M</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 15:  Model configuration of different size. | ✅ Table 15:  不同尺寸的模型配置。 |

## 13 Appendix D More Examples of Annotations in FLD-5B

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x13.png)



| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 8:  Examples of annotations in FLD-5B. | ✅ Figure 8:  Examples of annotations in FLD-5B. |

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x14.png)



| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 9:  Examples of annotations in FLD-5B (continued). | ✅ Figure 9:  Examples of annotations in FLD-5B (continued). |

## 14 Appendix E Qualitative Evaluation and Visualization Results

### 14.1 E.1 Detailed Image Caption

Detailed Image Caption

Prompt

: Describe with a paragraph what is shown in the image.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_1.jpg)




The image is a hand-drawn illustration of a glass of gin and tonic. The glass is filled with ice cubes, lime slices, and a slice of lemon. There is a straw sticking out of the top of the glass. The background is a light green color with a grungy texture. On the right side of the image, there is a list of ingredients written in black ink. The text reads "Gin & Tonic" and "2oz gin, 5oz tonic water, 2-4 ice cubes lime slice for garnish".


![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_2.jpg)




The image shows the interior of a grocery store with a large variety of fruits and vegetables on display. The store has a high ceiling with green and white striped awnings, and the floor is made of wood. There are rows of wooden crates filled with different types of fruits, including apples, oranges, lemons, limes, and limes. The crates are arranged in neat rows, and there are price tags attached to each crate. In the background, there are shelves stocked with various items such as fruits, vegetables, and other grocery items. The overall atmosphere of the store is bright and colorful, with a sense of freshness and abundance.


![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_3.jpg)




The image is a movie poster for the film "Harry Potter and the Sorcerer’s Stone" starring Ron Weasley. The poster features a young boy, Ron Weasley, standing in front of a dark background with a castle-like structure in the background. He is wearing a school uniform with a Gryffindor crest on his jacket and a red tie. He has blonde hair and is looking directly at the camera with a serious expression on his face. The title of the film is written in white text at the top of the poster, with the tagline "20 years of movie magic" written in smaller text below.


![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_4.jpg)




The image is a digital illustration of a girl hugging a white cat. The girl is wearing a pink sweater and has long brown hair. She is sitting on a green surface with several potted plants and flowers around her. The plants have green leaves and pink and white flowers. There are also two butterflies fluttering around the scene. The background is white. The overall style of the illustration is cartoon-like and playful.


| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 10:  Detailed captioning prediction results. | ✅ Figure 10:  Detailed captioning prediction results. |

### 14.2 E.2 Visual Grounding

Visual Grounding

Prompt

: Locate the phrases in the caption: {caption}

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_6.png)




               The image shows a group of five cartoon monsters. On the left side, there is 

a brown monster

1

 with horns and a big smile on its face. Next to it, there are two 

smaller monsters

2

, one black and one green. 

The black monster

3

 has two large horns on its head and is standing in the center of the group. 

The green monster

4

 on the right side is a green monster with big eyes and a long antennae. It is standing on its hind legs with its arms stretched out to the sides. In the middle of the image, there appears to be 

a small blue monster

5

 with a round head and two antennae on its back. The background is light beige with small green circles scattered around.


![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_7.png)




                 The image shows a cluttered room with a black and white checkered 

floor

1

. On the right side of the image, there is 

a small white cabinet

2

 with a 

television

3

 on top of it. Next to the cabinet, there are 

several items

4

 scattered on the floor, including a red 

blanket

5

, 

a wooden stool

6

, and a pile of trash. On top of the cabinet is 

a picture frame

7

 and a 

hat

8

. In the center of the room is 

a white refrigerator

9

 with a few items on top. 

The walls

10

 are painted white and there are 

a few clothes

11

 hanging on a 

rack

12

 on the left wall. The room appears to be in disarray, with some items strewn about and others scattered around.


![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_8.png)




           The image shows a kitchen countertop with various kitchen items on it. On the left side of the countertop, there is a microscope with a black body and a white 

lens

1

. Next to the microscope, there are two bottles of 

condiments

2

 - one with 

a red label

3

4

 and the other with green. On top of the microscope is 

a yellow banana

5

, 

a blue spatula

6

, 

a red plate

7

, and 

a yellow corn

8

9

 on the cob. In the center of the image, there appears to be 

a frying pan

10

 with a 

fried egg

11

 on it, and on the right side is 

a white sink

12

 with a white 

faucet

13

. 

The countertop

14

 is made of wood and has a gray tile backsplash.


| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 11:  Visual grounding prediction results. | ✅ Figure 11:  Visual grounding prediction results. |

Visual Grounding

Prompt

: Locate the phrases in the caption: {caption}

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_1.png)




   The image is a flat lay of various food items arranged on a white marble countertop. On the left side of the image, there is 

a piece of salmon

1

. Next to it, there are 

slices of cheese

2

, 

a glass of oil

3

, 

coffee beans

4

, 

a zucchini

5

, a bunch of 

strawberries

6

, two 

chicken breasts

7

, 

a avocado

8

 and 

a few whole spinach leaves

9

. In the center of the table, there appears to be  

a pile of ground beef

10

 on 

paper

11

, two 

eggs

12

, two 

orange bell peppers

13

, and 

some dark chocolate bars

14

. The items are arranged in a way that suggests they are being prepared for a meal.



![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_2.png)




             The image shows a modern kitchen with a large window on the left side. 

The window

1

 has a view of trees and greenery outside. On the left side of the image, there is 

a blue sofa

2

 with a wooden coffee table in front of it. Above the table, there are 

three copper pendant lights

3

 hanging from the ceiling. There is 

a large island

4

 with a white countertop. There are 

two bar stools

5

 next to the table. In the center of the kitchen, there is 

a bottle green plants

6

 on the table. 

The floor

7

 is made of light-colored wood and 

the walls

8

 are painted in a dark blue color.


%


![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_5.png)




         The image shows a 

man

1

 standing in a kitchen with a small dog. 

The man

1

 is wearing a plaid 

shirt

2

 and 

jeans

3

 and is holding a red 

cup

4

 in his hand. 

The dog

5

 is a light brown color and is standing on a tiled 

floor

6

. 

The kitchen

7

 has wooden 

cabinets

8

 and a 

countertop

9

 with various kitchen utensils hanging on the wall. There is 

a window

10

 with yellow 

curtains

11

 in the background. On the right side of the image, there is 

a wooden cutting board

12

 and a wooden 

stool

13

.


| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 12:  Visual grounding prediction results. (continued) | ✅ Figure 12:  Visual grounding prediction results. (continued) |

### 14.3 E.3 Dense Region Caption

Dense Region Caption

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_1.png)



![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_2.png)



![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_3.png)



![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_4.png)



![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_5.png)



![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_6.png)



| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 13:  Dense region caption prediction results. | ✅ Figure 13:  Dense region caption prediction results. |

### 14.4 E.4 Open Vocabulary Detection

Open Vocabulary Object Detection

Prompt

: Locate 

Five Alive juice box

 $\langle$ 

and

 $\rangle$ 

Colgate toothpaste

 in the image.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_1.png)



Prompt

: Locate 

Chewbacca

 in the image.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_2.png)



Prompt

:
Locate 

giraffe

 in the image.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_3.png)



Prompt

:
Locate 

Mercedes-Benz

 $\langle$ 

and

 $\rangle$ 

M2

 $\langle$ 

and

 $\rangle$ 

Audi

 in the image.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_4.png)



Prompt

: Locate the 

objects with category name

 in the image.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_5.png)



Prompt

: Locate the 

objects with category name

 in the image.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_6.png)



| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 14:  Open vocabulary object detection prediction results. | ✅ Figure 14:  Open vocabulary object detection prediction results. |

### 14.5 E.5 OCR

Ocr with region

Prompt

: What is the text in the image, with regions?

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/ocr_1.png)



Easy Stroganoff

1

Brown - 1 lb. ground beef in skillet

2

Add - 1 can beef broth

3

1 can cream of mushroom soup

4

Cut in squares & 2dld to above -

5

1/ Boz pkg. cream cheese

6

Simmer - 20-30 min.

7

Serve over hotrice /noodles.

8

Vintage. Recipes/Easy-Stroganof

9

Charlotte Miller

10

Tulsa

11

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/ocr_4.png)



COFFEE+TEA

1

BLENDED

2

$1.69/$1.89/$2.09

3

$3.49/$3.99

4

Hot Coffee/Tea

5

Taro

6

Iced Coffee/ Tea

7

Mango

8

Hot Chocolate

9

Honeydew

10

$3,49/$ 3.99

11

Strawberry

12

Mocha

14

Thai Iced Tea / Coffee

13

Caramel

15

$1,99/$2,29/$2:59

16

SPECIALTY Brew !!

17

Jasmine GreenTea

18

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/ocr_2.png)



LEONARDO

1

DiCAPRIO

2

ROBERT

3

DE NIRO

4

LILY

5

GLADSTONE

6

A MARTIN SCORSESE PICTURE

7

KILLERS

8

OF

9

FLOWER

10

MOON

11

SCREENLY ERIC ROTH AND MARTIIN SCORSESE DIRECTED BYMARTIN SORSESE

12

ONLY IN THEATRES OCTOBER 20

13

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 15:  OCR with region prediction results. | ✅ Figure 15:  OCR with region prediction results. |

### 14.6 E.6 Region to segmentation

Region to Segmentation

Prompt

:
What is the polygon mask of region


 $\langle$ 

loc_586

 $\rangle$ 

 $\langle$ 

loc_294

 $\rangle$ 

 $\langle$ 

loc_929

 $\rangle$ 

 $\langle$ 

loc_814

 $\rangle$ 

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_1.png)



Prompt

:
What is the polygon mask of region 

 $\langle$ 

loc_317

 $\rangle$ 

 $\langle$ 

loc_314

 $\rangle$ 

 $\langle$ 

loc_893

 $\rangle$ 

 $\langle$ 

loc_904

 $\rangle$ 

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_2.png)



Prompt

:
What is the polygon mask of region


 $\langle$ 

loc_541

 $\rangle$ 

 $\langle$ 

loc_266

 $\rangle$ 

 $\langle$ 

loc_692

 $\rangle$ 

 $\langle$ 

loc_627

 $\rangle$ 

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_3.png)



Prompt

:
What is the polygon mask of region


 $\langle$ 

loc_583

 $\rangle$ 

 $\langle$ 

loc_66

 $\rangle$ 

 $\langle$ 

loc_794

 $\rangle$ 

 $\langle$ 

loc_331

 $\rangle$ 

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_4.png)



Prompt

:
What is the polygon mask of region 

 $\langle$ 

loc_386

 $\rangle$ 

 $\langle$ 

loc_53

 $\rangle$ 

 $\langle$ 

loc_759

 $\rangle$ 

 $\langle$ 

loc_998

 $\rangle$ 

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_5.png)



Prompt

:
What is the polygon mask of region 

 $\langle$ 

loc_102

 $\rangle$ 

 $\langle$ 

loc_7

 $\rangle$ 

 $\langle$ 

loc_375

 $\rangle$ 

 $\langle$ 

loc_648

 $\rangle$ 

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_6.png)



| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 16:  Region to segmentation prediction results. | ✅ Figure 16:  Region to segmentation prediction results. |

## 15 Appendix F Comparision with LMMs on Detailed Caption Task

Comparison with LMMs on Detailed Image Caption Task

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x15.jpg)



Florence-2

: The image is a photograph of a handwritten note from a person named Angelica. The note is written in black ink on a white background. The handwriting is neat and legible, and it appears to be a handwritten letter. The person’s name, Jonathan, is written at the top of the note in a cursive font. Below the name, there is a note that reads, "I’ve got to admit, London’s lights have nothing on our family get together. I miss that, I miss you guys too. Let’s not get too happy about it though I’m keeping my fingers crossed for a trip back stateside. Give a big hug to Amelia for me. Will you?"


GPT-4V

1

1

1

https://chat.openai.com

: The image shows a handwritten note addressed to Jonathan. The note conveys a personal message from the sender, Angelica, who admits to missing family gatherings and the feeling of home, contrasting it with the lights of London. Angelica expresses a longing to return stateside and asks Jonathan to give Amelia a big hug on her behalf. The tone of the note is casual and warm, with a sense of fondness and a hint of homesickness. The note is written informally, suggesting a close relationship between the sender and the recipients.


LLaVA

2

2

2

https://llava.hliu.cc

:

 The image is a handwritten letter on a piece of paper, written by a person named Jonathan. The letter is addressed to someone named Angela, and it appears to be a heartfelt message expressing love and affection. The letter is written in cursive, giving it a personal and intimate touch.


MiniGPT-4

3

3

3

https://huggingface.co/spaces/Vision-CAIR/minigpt4

:

 The image is a handwritten letter from an unknown person to someone named Angel. The letter is written in black ink on a piece of white paper. The letter is addressed to Angel and begins with the words, “Dear Angel,”. The letter goes on to describe the writer’s feelings about a recent event or situation, but the details are unclear. The letter ends with the words, “Sincerely, [Writer’s Name].”



| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 17:  Comparison with LMMs on detailed caption task. | ✅ Figure 17:  Comparison with LMMs on detailed caption task. |

Comparison with LMMs on Detailed Image Caption Task

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/com_5.jpg)



Florence-2

: The image is a close-up of a red apple with a green label on it. The label has the text "FUJI 94131 USA" and a barcode on it, indicating that the apple is from the United States. The apple appears to be fresh and has a shiny texture. The background is white, making the apple stand out.


GPT-4V

4

4

4

https://chat.openai.com

: The image displays a Fuji apple with a sticker indicating it is organic, along with the identifying number 94131, and specifying that it is from the USA. The apple has a reddish-pink skin with some yellow and green coloration, which is characteristic of Fuji apples. The apple is placed on a white background, which appears to be a textured fabric.


LLaVA

5

5

5

https://llava.hliu.cc

:

 The image features a large, fresh, and ripe red apple sitting on a white surface. The apple has a sticker on it, which is likely a barcode or a label providing information about the product. The apple is prominently displayed, taking up a significant portion of the image.


MiniGPT-4

6

6

6

https://huggingface.co/spaces/Vision-CAIR/minigpt4

:

 The image shows a red apple with a label on it that reads “Organic” in white letters. The apple is sitting on a white surface.


| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 18:  Comparison with LMMs on detailed caption task (continued). | ✅ Figure 18:  Comparison with LMMs on detailed caption task (continued). |

Comparison with LMMs on Detailed Image Caption Task

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/com_2.jpg)



Florence-2

: The image shows a highway with multiple lanes. There are several vehicles on the road, including a black van and a red truck. On the left side of the image, there is a white semi-truck with the word "Martini" written on it. Above the highway, there are several bridges with cars and trucks on them. The sky is blue and there are trees and grass on the right side. In the center of the highway is a green sign that reads "Exit 94, Chamblee Tucker Rd, Mercer Univ."


GPT-4V

7

7

7

https://chat.openai.com

: The image shows a highway scene with several lanes of traffic. There are overpasses with cars and trucks traveling over them, and the highway itself appears to be busy with vehicles, including a prominent black van in the foreground.

A road sign indicates that Exit 94 for Chamblee-Tucker Rd and Mercer University is 1 1/4 miles ahead. The presence of the sign suggests that drivers in the right lanes will soon have the opportunity to take the exit to reach these destinations.

The infrastructure suggests a well-traveled interstate or highway system, possibly in an urban or suburban area given the overpass complexity and road signage for local roads and institutions. The vehicles on the road are diverse, indicating a mix of commercial and personal transport.


LLaVA

8

8

8

https://llava.hliu.cc

:

 The image depicts a busy highway with multiple lanes of traffic, including cars and trucks. There is a large truck driving on the highway, and several other trucks are also visible in the scene. The highway is filled with cars, and there is a traffic sign above the lanes, indicating the direction to Tucker Road. The scene captures the hustle and bustle of a typical day on a highway filled with vehicles.


MiniGPT-4

9

9

9

https://huggingface.co/spaces/Vision-CAIR/minigpt4

:

 The image shows a highway with multiple lanes of traffic moving in both directions. There are several large bridges spanning the highway, and in the distance, there are mountains and a blue sky. There are also several large trucks and cars on the highway, including a black van in the center of the image. The road surface is smooth and well-maintained.


| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 19:  Comparison with LMMs on detailed caption task (continued). | ✅ Figure 19:  Comparison with LMMs on detailed caption task (continued). |

Comparison with LMMs on Detailed Image Caption Task

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/com_3.jpg)



Florence-2

: The image shows a kitchen countertop with a laptop on it. The laptop is open and the screen is displaying a blue abstract design. On the countertop, there are various kitchen utensils hanging on a metal rack. The rack is attached to the wall above the sink and has a variety of kitchen tools hanging on it, including a grater, a spatula, a ladle, a spoon, and a whisk. There are also a few other kitchen items scattered around the counter. The countertop is made of wood and there is a wire rack at the bottom of the image.


GPT-4V

10

10

10

https://chat.openai.com

: The image shows a modern kitchen setting where culinary tools are organized neatly on a rack against the wall. These tools include a variety of utensils such as a grater, tongs, a whisk, ladles, and spatulas. On the counter, there is an open laptop with a vibrant blue wallpaper on the screen, which suggests that someone might be using it for a recipe or as entertainment while cooking.
In front of the laptop, there appears to be a cutting board with some chopped ingredients and a knife, indicating that food preparation is underway or has recently taken place. There are also some dried chilies and a plastic container that might contain other ingredients or spices, further suggesting cooking activities.
The counter also has an electrical outlet and some covered sockets, and below the main counter, there are storage bins that likely hold additional kitchen supplies. The environment looks lived-in and practical, with a mix of technology and traditional cooking tools.


LLaVA

11

11

11

https://llava.hliu.cc

:

 The image features a wooden kitchen counter with a laptop computer sitting on top of it. The laptop is open and displaying a picture of a cake. The counter is also cluttered with various kitchen utensils, including multiple spoons and knives, as well as a bowl. The scene suggests that someone might be using the laptop for cooking or baking purposes, possibly following a recipe or watching a tutorial.


MiniGPT-4

12

12

12

https://huggingface.co/spaces/Vision-CAIR/minigpt4

:

 The image shows a kitchen counter with a laptop computer on it, a set of cooking utensils hanging from a rack above it, and a knife block on the counter. The laptop appears to be open, with a blue screen displaying an image of a kitchen. The utensils hanging from the rack include a spatula, a ladle, a whisk, and a cutting board. The knife block contains several knives, including a chef’s knife, a paring knife, and a bread knife.


| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 20:  Comparison with LMMs on detailed caption task (continued). | ✅ Figure 20:  Comparison with LMMs on detailed caption task (continued). |

## 16 Appendix G Comarison with Kosmos-2 on Detailed Caption and Grounding Tasks

Comparison on detailed caption and grounding tasks.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/ouput_1_kosmos2.jpg)



 The image features a home office with 

two white desks

, one on the left and the other on the right. The desks are connected by  

a white metal frame

, creating a functional and stylish workspace. 

A laptop

  is placed on the desk on the far left, and   

a keyboard

  is visible on the other desk.   

Two chairs

  are placed in front of the desks, one closer to the left desk and the second one on its right side.


(a) Result from Kosmos-2.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output1_fld2.jpg)




      The image shows a modern home office setup with two white 

desks

1

 and 

two black chairs

2

. 

The desks

3

 are arranged in a corner of the room with a large window on the left side. On the right side of the desk, there is 

a computer monitor

4

, a 

keyboard

5

, a mouse, 

a printer

6

, and a 

laptop

7

. Above the computer monitor and keyboard, there are 

two framed pictures

8

 hanging on the wall. 

The walls

9

 are painted in a light green color and 

the floor

10

 is made of light-colored wood. 

The floor

11

 is covered with a beige area rug with a geometric pattern. The overall style of the space is minimal and contemporary.


(b) Result from 

Florence-2

.

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 21:  Systematic comparison with Kosmos-2 [60] on detailed caption and grounding tasks. The models generate both the detailed caption and grounding results. The results of Kosmos-2 are from https://huggingface.co/spaces/ydshieh/Kosmos-2. | ✅ Figure 21:  Systematic comparison with Kosmos-2 [60] on detailed caption and grounding tasks. The models generate both the detailed caption and grounding results. The results of Kosmos-2 are from https://huggingface.co/spaces/ydshieh/Kosmos-2. |

Comparison on detailed caption and grounding tasks.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output_2_kosmos2.jpg)



 The image features a white dining table with a variety of food items on it. On the table, there is a bowl of bread, a bowl with a salad, and a plate with a side of yogurt. In addition to the food, there are   

two cups

  placed on the table. One cup is located near the left side of the table and the other cup is situated on the right side. The table is set with a fork and a knife, ready for a meal.


(c) Result from Kosmos-2.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output2_fld2.jpg)




  The image shows a table with a basket of bread and a plate of 

salad

1

. 

The basket

2

 is made of woven straw and has several slices of bread in it. Next to the basket, there is 

a small bowl

1

 of salad with a variety of vegetables, including 

lettuce

3

, 

tomatoes

4

, 

cucumbers

5

, and 

feta cheese

6

. There are also 

two salt

7

 and pepper 

shakers

7

 on the table. On the right side of the table, there are 

two white plates

8

9

 with a dollop of white sauce on them. The table is covered with a white tablecloth and there are 

a few other dishes

9

 and 

utensils

10

 scattered around. 

A person"s hand

11

 can be seen in the top right corner of the image.


(d) Result from 

Florence-2

.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output_3_kosmos2.jpg)



  The image features a family of   

elephants

  walking together in a lush green forest. The  

elephants

 are walking in a line, with one of them walking in front of the rest of the family. The family consists of a mother and her two calves, with the mother and the two calves walking side by side. The forest is filled with green trees, adding to the natural beauty of the scene.


(e) Result from Kosmos-2.

![figure](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output3_fld2.jpg)




  The image shows a group of three elephants standing in a dirt field with trees and bushes in the background. 

The elephants

1

 are standing close together, with the largest elephant in the center and two smaller ones on either side. 

The largest elephant

1

2

  on the left is standing with its 

trunk

3

 extended, while 

the smaller one

4

 is standing next to it. 

All three elephants

1

 have 

tusks

5

 and appear to be in their natural habitat. 

The ground

6

 is covered in dirt and there is 

a small pile of dirt

7

 in front of them. The overall mood of the image is peaceful and serene.


(f) Result from 

Florence-2

.

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 22:  Systematic comparison with Kosmos-2 [60] on detailed caption and grounding tasks. The models generate both the detailed caption and grounding results. The results of Kosmos-2 are from https://huggingface.co/spaces/ydshieh/Kosmos-2. (continued) | ✅ Figure 22:  Systematic comparison with Kosmos-2 [60] on detailed caption and grounding tasks. The models generate both the detailed caption and grounding results. The results of Kosmos-2 are from https://huggingface.co/spaces/ydshieh/Kosmos-2. (continued) |