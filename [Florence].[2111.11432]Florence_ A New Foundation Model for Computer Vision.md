# Florence: A New Foundation Model for Computer Vision

## 0. Abstract

| 【概述】原文 | 【概述】翻译 |
| ---- | ---- |
| ✅ Automated visual understanding of our diverse and open world demands computer vision models to generalize well with minimal customization for specific tasks, similar to human vision. | ✅ 要实现对我们多样化和开放的世界的自动化视觉理解，就需要计算机视觉模型能够很好地概括，并针对特定任务进行最少的定制，类似于人类视觉。 |
| ✅ Computer vision foundation models, which are trained on diverse, large-scale dataset and can be adapted to a wide range of downstream tasks, are critical for this mission to solve real-world computer vision applications. | ✅ 计算机视觉基础模型在多样化、大规模数据集上进行训练，可适应广泛的下游任务，对于解决现实世界的计算机视觉应用至关重要。 |
| ✅ While existing vision foundation models such as CLIP (  ) , ALIGN (  ) , and Wu Dao 2.0 (  ) focus mainly on mapping images and textual representations to a cross-modal shared representation, we introduce a new computer vision foundation model, Florence , to expand the representations from coarse (scene) to fine (object), from static (images) to dynamic (videos), and from RGB to multiple modalities (caption, depth). | ✅ 虽然现有的视觉基础模型（例如 CLIP (  )、ALIGN (  ) 和 Wu Dao 2.0 (  )）主要侧重于将图像和文本表示映射到跨模态共享表示，但我们引入了一种新的计算机视觉基础模型 Florence，将表示从粗糙（场景）扩展到精细（对象）、从静态（图像）到动态（视频），从 RGB 扩展到多模态（标题、深度）。 |
| ✅ By incorporating universal visual-language representations from Web-scale image-text data, our Florence model can be easily adapted for various computer vision tasks, such as classification, retrieval, object detection, VQA, image caption, video retrieval and action recognition. | ✅ 通过结合来自 Web 规模图像文本数据的通用视觉语言表示，我们的 Florence 模型可以轻松适应各种计算机视觉任务，例如分类、检索、对象检测、VQA、图像标题、视频检索和动作识别。 |
| ✅ Moreover, Florence demonstrates outstanding performance in many types of transfer learning: fully sampled fine-tuning, linear probing, few-shot transfer and zero-shot transfer for novel images and objects. | ✅ 此外，Florence 在许多类型的迁移学习中都表现出色：针对新图像和物体的完全采样微调、线性探测、小样本迁移和零样本迁移。 |
| ✅ All of these properties are critical for our vision foundation model to serve general purpose vision tasks. | ✅ 所有这些属性对于我们的视觉基础模型服务通用视觉任务都至关重要。 |
| ✅ Florence achieves new state-of-the-art results in majority of $44$ representative benchmarks, e.g. | ✅ Florence 在大多数 $44$ 代表性基准测试中取得了新的最佳结果，例如 |
| ✅ ImageNet-1K zero-shot classification with top-1 accuracy of ${83.74}$ and the top-5 accuracy of ${97.18}$ , ${62.4}$ mAP on COCO fine tuning, ${80.36}$ on VQA, and ${87.8}$ on Kinetics-600. | ✅ ImageNet-1K 零样本分类，top-1 准确率 ${83.74}$，top-5 准确率 ${97.18}$，COCO 微调上 mAP ${62.4}$，VQA 上 mAP ${80.36}$，Kinetics-600 上 mAP ${87.8}$。 |

## 1 Introduction

| 【第1节，第1段】原文 | 【第1节，第1段】翻译 |
| ---- | ---- |
| ✅ Human-like AI is not achieved by designing specific models to solve specific problems, but by holistic, joint models that can simultaneously solve diverse, real-world problems without too much human involvement. | ✅ 类人人工智能不是通过设计特定的模型来解决特定的问题来实现的，而是通过整体的、联合的模型来实现的，这些模型可以同时解决现实世界中的各种问题，而不需要太多的人为参与。 |
| ✅ It is thus desirable to have new AI architectures that learn joint, fundamental representations to support a broad range of downstream AI tasks with limited additional domain knowledge, similar to what humans would do. | ✅ 因此，希望有新的人工智能架构能够学习联合的基本表示，以有限的额外领域知识来支持广泛的下游人工智能任务，类似于人类所做的。 |
| ✅ One such proposal is XYZ-code ( **A holistic representation toward integrative ai.** ) , where monolingual text (X), audio and visual sensory signals (Y), and multilingual (Z) are organically integrated to create AI models that can speak, hear, see, and understand. | ✅ 其中一个提案是 XYZ 代码 ( **A holistic representation toward integrative ai.** )，其中单语文本（X）、音频和视觉感官信号（Y）和多语言（Z）有机地结合在一起，以创建能够说、听、看和理解的人工智能模型。 |
| ✅ Another approach is Pathways ( **Introducing pathways: A next-generation ai architecture.** ) , a single model that can generalize across millions of tasks. | ✅ 另一种方法是 Pathways ( **Introducing pathways: A next-generation ai architecture.** )，这是一个可以推广到数百万个任务的单一模型。 |

![figure](https://ar5iv.labs.arxiv.org/html/2111.11432/assets/x1.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 1:  Common computer vision tasks are mapped to a Space-Time-Modality space. | ✅ Figure 1:  常见的计算机视觉任务被映射到空间-时间-模态空间。 |
| ✅ A computer vision foundation model should serve as general purpose vision system for all of these tasks. | ✅ 计算机视觉基础模型应该作为所有这些任务的通用视觉系统。 |

![figure](https://ar5iv.labs.arxiv.org/html/2111.11432/assets/x2.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 2:  Overview of building Florence. | ✅ Figure 2:  佛罗伦萨建筑概览。 |
| ✅ Our workflow consists of data curation, unified learning, Transformer architectures and adaption. | ✅ 我们的工作流程包括数据管理、统一学习、Transformer 架构和适应。 |
| ✅ It shows the foundation model can be adapted to various downstream tasks and finally integrated into modern computer vision system to power real-world vision and multimedia applications. | ✅ 它表明基础模型可以适应各种下游任务，并最终集成到现代计算机视觉系统中，为现实世界的视觉和多媒体应用提供支持。 |
| ✅ Compared with existing image-text pretraining models ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision.** ｜ **3. https://gpt3demo.com/apps/wu-dao-20.** ) , mainly limited on cross-modal shared representation for classification and retrieval (illustrated by light-green adaptation module), Florence expands the representation to support object level, multiple modality, and videos respectively. | ✅ 与现有的图像文本预训练模型 ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision.** ｜ **3. https://gpt3demo.com/apps/wu-dao-20.** ) 相比，后者主要局限于分类和检索的跨模态共享表征（浅绿色适配模块所示），而 Florence 扩展了表征以分别支持对象级、多模态和视频。 |

| 【第1节，第2段】原文 | 【第1节，第2段】翻译 |
| ---- | ---- |
| ✅ A concrete step towards this direction is the development of foundation models. | ✅ 朝着这个方向迈出的具体一步是基础模型的开发。 |
| ✅ The term of foundation model was first introduced in ( **On the opportunities and risks of foundation models.** ) to refer to any model that is trained from broad data at scale that is capable of being adapted ( e.g. | ✅ 基础模型这个术语最早是在 ( **On the opportunities and risks of foundation models.** ) 中引入的，指的是任何从大规模的广泛数据中训练出来的、能够适应的模型（例如 |
| ✅ fine-tuned) to a wide range of downstream tasks. | ✅ 经过微调后，可以应用于广泛的下游任务。 |
| ✅ Foundation models become promising due to their impressive performance and generalization capabilities. | ✅ 基础模型由于其出色的性能和泛化能力而变得有前景。 |
| ✅ They are quickly integrated and deployed into real-world AI systems by many researchers and developers. | ✅ 它们被许多研究人员和开发人员迅速集成并部署到现实世界的人工智能系统中。 |

| 【第1节，第3段】原文 | 【第1节，第3段】翻译 |
| ---- | ---- |
| ✅ Although foundation models have already demonstrated huge impact in NLP, e.g. | ✅ 尽管基础模型已经在 NLP 中展现出巨大的影响，例如 |
| ✅ , BERT ( **Bert: Pre-training of deep bidirectional transformers for language understanding.** ) , GPT-3 ( **Language models are few-shot learners.** ) , in computer vision it is still standard practice to pre-train models on labeled data sets such as ImageNet ( **Imagenet: A large-scale hierarchical image database.** ). | ✅ 、BERT ( **Bert: Pre-training of deep bidirectional transformers for language understanding.** )、GPT-3 ( **Language models are few-shot learners.** )，在计算机视觉领域，在标记数据集（例如 ImageNet ( **Imagenet: A large-scale hierarchical image database.** )）上预训练模型仍然是标准做法。 |
| ✅ More recently, large-scale pre-training methods such as CLIP ( **Learning transferable visual models from natural language supervision.** ) , ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) , and Wu Dao 2.0 ( **https://gpt3demo.com/apps/wu-dao-20.** ) , which learn directly from Web-scale image-text pairs, show very encouraging progress for efficient transfer learning, and zero-shot capability. | ✅ 最近，CLIP ( **Learning transferable visual models from natural language supervision.** )、ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) 和 Wu Dao 2.0 ( **https://gpt3demo.com/apps/wu-dao-20.** ) 等大规模预训练方法直接从 Web 规模的图像-文本对中学习，在高效迁移学习和零样本能力方面取得了非常令人鼓舞的进展。 |
| ✅ However, such models are restricted to image to text mapping only tasks such as classification, retrieval, and tagging. | ✅ 然而，此类模型仅限于图像到文本的映射任务，例如分类、检索和标记。 |

| 【第1节，第4段】原文 | 【第1节，第4段】翻译 |
| ---- | ---- |
| ✅ We raise the question: “ What is the foundation model for computer vision ?”. | ✅ 我们提出一个问题：“计算机视觉的基础模型是什么？”。 |
| ✅ But first, in order to better define what “foundation” means in computer vision, we capture the spectrum of tasks in a problem space (Figure 1 ) with three orthogonal axes: 1) Space : from coarse ( e.g. | ✅ 但首先，为了更好地定义计算机视觉中“基础”的含义，我们用三个正交轴捕获问题空间中的任务范围（图 1）：1）空间：从粗糙（例如 |
| ✅ scene-level classification) to fine-grained ( e.g. | ✅ 场景级分类）到细粒度（例如 |
| ✅ object detection), 2) Time : from static ( e.g. | ✅ 物体检测），2）时间：从静态（例如 |
| ✅ images) to dynamic ( e.g. | ✅ 图像）转换为动态（例如 |
| ✅ videos), and 3) Modality : from RGB only to multiple senses ( e.g. | ✅ 视频）和 3）模态：从仅 RGB 到多种感官（例如 |
| ✅ captioning and depth). | ✅ 字幕和深度）。 |
| ✅ Due to the diversity nature of visual understanding, we redefine foundation models for computer vision to be a pre-trained model and its adapters for solving all vision tasks in this Space-Time-Modality space, with transferability such as zero-/few-shot learning and fully fine tuning, etc. | ✅ 由于视觉理解的多样性，我们将计算机视觉的基础模型重新定义为预训练模型及其适配器，用于解决该时空模态空间中的所有视觉任务，具有零/小样本学习和完全微调等可转移性。 |
| ✅ The adaptation for transferability is restricted to minimum customization for the pre-trained foundation models, such as continuing training, few epochs or few layers for fine tuning without significantly increasing or changing model parameters. | ✅ 可转移性的适应性仅限于对预先训练的基础模型进行最低限度的定制，例如持续训练、少数几个时期或少数层进行微调，而不会显著增加或改变模型参数。 |

| 【第1节，第5段】原文 | 【第1节，第5段】翻译 |
| ---- | ---- |
| ✅ In this paper, we present an emerging paradigm for building a vision foundation model , called Florence. | ✅ 在本文中，我们提出了一个用于构建视觉基础模型的新兴范例，称为 Florence。 |
| ✅ We use the name of Florence as the origin of the trail for exploring vision foundation models, as well as the birthplace of Renaissance. | ✅ 我们以佛罗伦萨这个名字，作为探索视觉基础模式之路的起源，也是文艺复兴的发源地。 |
| ✅ Florence is trained on noisy Web-scale data end-to-end with a unifying objective, allowing the model to achieve best-in-class performance across a wide range of benchmarks. | ✅ Florence 采用端到端的方式在嘈杂的 Web 规模数据上进行训练，具有统一的目标，从而使模型在广泛的基准测试中实现一流的性能。 |

| 【第1节，第6段】原文 | 【第1节，第6段】翻译 |
| ---- | ---- |
| ✅ The ecosystem of constructing Florence consists of data curation , model pretraining , task adaptations and training infrascturue , as shown in Figure 2 . | ✅ 构建 Florence 的生态系统由数据管理、模型预训练、任务适配和训练基础设施组成，如图 2 所示。 |

| 【第1节，第7段】原文 | 【第1节，第7段】翻译 |
| ---- | ---- |
| ✅ Data curation. | ✅ 数据管理。 |
| ✅ Diverse, large-scale data is the lifeblood of foundation models. | ✅ 多样化、大规模的数据是基础模型的命脉。 |
| ✅ Enabled by large amounts of publicly available images on the Internet with natural language weak supervision, we curate a new dataset of $900$ million image-text pairs for training. | ✅ 利用互联网上大量公开的图像和自然语言弱监督，我们整理出一个包含 $900$ 百万个图像-文本对的新数据集用于训练。 |
| ✅ As Web-crawled data is usually noisy free-form texts ( e.g. | ✅ 由于 Web 抓取的数据通常是嘈杂的自由格式文本（例如 |
| ✅ , word, phrase or sentence), to attain more effective learning, we consider UniCL , a unified image-text contrastive learning objective recently proposed in ( **Unified contrastive learning in image-text-label space.** ) , which has demonstrated improvements over contrastive and supervised learning approaches. | ✅ 在本文中，我们将讨论如何在图像文本（包括单词、短语或句子）中进行对比学习，以实现更有效的学习，我们考虑使用 UniCL，这是最近在 ( **Unified contrastive learning in image-text-label space.** ) 中提出的统一图像文本对比学习目标，它已证明比对比和监督学习方法有改进。 |

| 【第1节，第8段】原文 | 【第1节，第8段】翻译 |
| ---- | ---- |
| ✅ Model pretraining (representation learning). | ✅ 模型预训练（表征学习）。 |
| ✅ To learn a good representation from image-text pairs, we used a two-tower architecture including an image encoder and a language encoder, as commonly used in CLIP ( **Learning transferable visual models from natural language supervision.** ) and ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ). | ✅ 为了从图像-文本对中学习良好的表示，我们使用了双塔架构，包括图像编码器和语言编码器，就像在 CLIP ( **Learning transferable visual models from natural language supervision.** ) 和 ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) 中常用的那样。 |
| ✅ For the image encoder, we chose hierarchical Vision Transformers ( e.g. | ✅ 对于图像编码器，我们选择了分层视觉变换器（例如 |
| ✅ , Swin ( **Swin transformer: Hierarchical vision transformer using shifted windows.** ) , CvT ( **Cvt: Introducing convolutions to vision transformers.** ) , Vision Longformer ( **Multi-scale vision longformer: A new vision transformer for high-resolution image encoding.** ) , Focal Transformer ( **Focal self-attention for local-global interactions in vision transformers.** ) , and CSwin ( **Cswin transformer: A general vision transformer backbone with cross-shaped windows.** ) ). | ✅ 、Swin ( **Swin transformer: Hierarchical vision transformer using shifted windows.** )、CvT ( **Cvt: Introducing convolutions to vision transformers.** )、Vision Longformer ( **Multi-scale vision longformer: A new vision transformer for high-resolution image encoding.** )、Focal Transformer ( **Focal self-attention for local-global interactions in vision transformers.** ) 和 CSwin ( **Cswin transformer: A general vision transformer backbone with cross-shaped windows.** )）。 |
| ✅ While inheriting performance benefits of the transformer self-attention operations ( **An image is worth 16x16 words: Transformers for image recognition at scale.** ) , these hierarchical architectures model the scale invariance nature of images and have linear computational complexity with respect to image size, a property that is essential to dense prediction tasks such as object detection and segmentation. | ✅ 在继承了 Transformer 自注意力操作 ( **An image is worth 16x16 words: Transformers for image recognition at scale.** ) 的性能优势的同时，这些分层架构还对图像的尺度不变性进行了建模，并且具有与图像大小相关的线性计算复杂度，这一特性对于对象检测和分割等密集预测任务至关重要。 |

| 【第1节，第9段】原文 | 【第1节，第9段】翻译 |
| ---- | ---- |
| ✅ Task adaptations. | ✅ 任务调整。 |
| ✅ As we have defined computer vision foundation models to adapt to various downstream tasks, it is vital for Florence to be extensible and transferable for this purpose. | ✅ 由于我们已经定义了计算机视觉基础模型来适应各种下游任务，因此 Florence 的可扩展性和可转移性对于此目的至关重要。 |
| ✅ We extended the learned feature representation along space (from scene to objects) using the dynamic head adapter ( **Dynamic head: Unifying object detection heads with attentions.** ) , time (from static image to videos) via proposed video CoSwin adapter, and modality (from images to language) via METER adapter ( **An empirical study of training end-to-end vision-and-language transformers.** ). | ✅ 我们使用动态头适配器 ( **Dynamic head: Unifying object detection heads with attentions.** ) 沿空间（从场景到物体）扩展了学习到的特征表示，通过提出的视频 CoSwin 适配器扩展了时间（从静态图像到视频），并通过 METER 适配器 ( **An empirical study of training end-to-end vision-and-language transformers.** ) 扩展了模态（从图像到语言）。 |
| ✅ Florence is designed to effectively adapted in the open world via few-shot and zero-shot transfer learning, with the ability of efficient deployment by extra training with few epochs ( e.g. | ✅ Florence 的设计目标是通过少样本和零样本迁移学习有效适应开放世界，并能够通过少量阶段的额外训练实现有效部署（例如 |
| ✅ in retrieval). | ✅ 正在检索）。 |
| ✅ Our model can be customized for various domains that application-developers can use. | ✅ 我们的模型可以针对应用程序开发人员可以使用的各个领域进行定制。 |

| 【第1节，第10段】原文 | 【第1节，第10段】翻译 |
| ---- | ---- |
| ✅ Training infrastructure. | ✅ 培训基础设施。 |
| ✅ For both energy and cost concerns, it is critical to build foundation models with as low cost as possible. | ✅ 出于能源和成本方面的考虑，以尽可能低的成本构建基础模型至关重要。 |
| ✅ We developed scalable training infrastructure to improve training efficiency. | ✅ 我们开发了可扩展的训练基础设施来提高训练效率。 |
| ✅ It consists of several key techniques such as ZeRO ( **Zero: Memory optimization towards training A trillion parameter models.** ) , activation checkpointing, mixed-precision training, gradient cache ( **Scaling deep contrastive learning batch size under memory limited setup.** ) to greatly reduce the memory consumption and thus improves the training throughput. | ✅ 它由 ZeRO ( **Zero: Memory optimization towards training A trillion parameter models.** )、激活检查点、混合精度训练、梯度缓存 ( **Scaling deep contrastive learning batch size under memory limited setup.** ) 等几项关键技术组成，大大减少了内存消耗，从而提高了训练吞吐量。 |

| 【第1节，第11段】原文 | 【第1节，第11段】翻译 |
| ---- | ---- |
| ✅ Florence significantly outperforms previous large-scale pre-training methods and achieves new state-of-the-art results on a wide range of vision and vision-language benchmarks. | ✅ Florence 的表现显著优于之前的大规模预训练方法，并在广泛的视觉和视觉语言基准上取得了新的最先进的成果。 |
| ✅ It showed strength in zero-shot transfer in $12$ classification downstream tasks (win $9/12$ , SOTA in ImageNet-1K zero-shot with top-1 accuracy of ${83.74}$ and the top-5 accuracy of ${97.18}$ ), linear probe in $11$ classification downstream tasks (win $9/11$ ), image retrieval zero-shot ( ${90.9}/{76.7}$ R $@1$ on Flickr30K image-to-text / text-to-image, ${64.7}/{47.2}$ R $@1$ on MSCOCO image-to-text / text-to-image) and fine-tuning ( ${97.2}/{87.9}$ R $@1$ on Flickr30K image-to-text / text-to-image, ${81.8}/{63.2}$ R $@1$ on MSCOCO image-to-text/ text-to-image), object detection ( ${62.4}$ mAP on COCO, ${39.3}$ mAP on Object365, ${16.2}$ AP50 on Visual Genome), VQA ( ${80.36}$ ), text-to-video retrieval zero-shot ( ${37.6}$ R $@1$ on MSR-VTT), and video action recognition (top-1 accuracy ${86.5}/{87.8}$ on Kinetics-400 / Kinetics-600). | ✅ 它在 $12$ 分类下游任务的零样本迁移（win $9/12$、ImageNet-1K 零样本中的 SOTA，${83.74}$ 的 top-1 准确率和 ${97.18}$ 的 top-5 准确率）中表现出色，在 $11$ 分类下游任务中的线性探测（win $9/11$）、图像检索零样本（Flickr30K 图像到文本/文本到图像上的 ${90.9}/{76.7}$ R $@1$、MSCOCO 图像到文本/文本到图像上的 ${64.7}/{47.2}$ R $@1$）和微调（Flickr30K 图像到文本/文本到图像上的 ${97.2}/{87.9}$ R $@1$、MSCOCO 图像到文本/文本到图像上的 ${81.8}/{63.2}$ R $@1$）、对象检测（COCO 上的 ${62.4}$ mAP、Object365 上的 ${39.3}$ mAP、Visual Genome 上的 ${16.2}$ AP50）、VQA（${80.36}$）、文本到视频检索零样本（${37.6}$ R $@1$我们的目标是在 MSR-VTT 上实现高性能计算（MSR-VTT 上达到 top-1 准确率 ${86.5}/{87.8}$）以及视频动作识别（Kinetics-400 / Kinetics-600 上达到 top-1 准确率 ${86.5}/{87.8}$）。 |

## 2 Approach

### 2.1 Dataset Curation

| 【第2.1节，第1段】原文 | 【第2.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We leverage large quantities of image-text data available publicly on the internet. | ✅ 我们利用互联网上公开的大量图像文本数据。 |
| ✅ Specifically, we construct a $900$ million image-text-pair dataset, called FLD-900M (FLD stands for FL orence D ataset), using a programmatic data curation pipeline that processes around $3$ billion Internet images and their raw descriptions in parallel. | ✅ 具体来说，我们构建了一个 $900$ 百万个图像文本对数据集，称为 FLD-900M（FLD 代表 FL orence D ataset），使用程序化数据管理管道，并行处理大约 $3$ 亿个互联网图像及其原始描述。 |
| ✅ Selection and post-filtering is employed to ensure data relevance and quality while respecting legal and ethical constraints. | ✅ 采用选择和后过滤来确保数据的相关性和质量，同时遵守法律和道德约束。 |
| ✅ To improve data quality, we performed rigorous data filtering, similar to ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) , including a simple hash-based near-duplicate image removal, small-size image removal, image-text relevance, etc. | ✅ 为了提高数据质量，我们进行了类似于 ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) 的严格数据过滤，包括简单的基于哈希的近似重复图像删除、小尺寸图像删除、图像文本相关性等。 |
| ✅ In addition, we follow the sampling strategy introduced in ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Zero-shot text-to-image generation.** ) with the goal of achieving improved balance, informativeness, and learnability of the sampled dataset. | ✅ 此外，我们遵循( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Zero-shot text-to-image generation.** )中引入的采样策略，目标是提高采样数据集的平衡性、信息量和可学习性。 |
| ✅ The final form of the FLD-900M dataset consists of $900M$ images with $900M$ free-form texts (ranging from one word, phase to sentences), $9.7M$ unique queries, and $7.5B$ tokens in total. | ✅ FLD-900M 数据集的最终形式由 $900M$ 图像、$900M$ 自由形式文本（从一个单词、阶段到句子）、$9.7M$ 唯一查询和 $7.5B$ 标记组成。 |

### 2.2 Unified Image-Text Contrastive Learning

| 【第2.2节，第1段】原文 | 【第2.2节，第1段】翻译 |
| ---- | ---- |
| ✅ CLIP ( **Learning transferable visual models from natural language supervision.** ) implicitly assumes that each image-text pair has its unique caption, which allows other captions to be considered negative examples. | ✅ CLIP ( **Learning transferable visual models from natural language supervision.** ) 隐式假设每个图像-文本对都有其独特的标题，这使得其他标题被视为反面例子。 |
| ✅ However, in web-scale data, multiple images can be associated with identical captions. | ✅ 然而，在网络规模数据中，多幅图像可以与相同的标题相关联。 |
| ✅ For example, in FLD-900M, there are $350M$ image-text pairs where there are more than one images corresponding to one identical text, and all images associated with the same text can be treated as positive pairs in contrastive learning. | ✅ 例如，在 FLD-900M 中，有 $350M$ 图像-文本对，其中有多幅图像对应一个相同的文本，并且与同一文本相关的所有图像都可以在对比学习中被视为正对。 |

| 【第2.2节，第2段】原文 | 【第2.2节，第2段】翻译 |
| ---- | ---- |
| ✅ To address this issue, we utilize a unified image-text contrastive learning ( UniCL ) ( **Unified contrastive learning in image-text-label space.** ) , where Florence is pre-trained in an image-label-description space. | ✅ 为了解决这个问题，我们利用统一的图像文本对比学习（UniCL）( **Unified contrastive learning in image-text-label space.** )，其中Florence在图像标签描述空间中进行了预训练。 |
| ✅ Given an image-text pair, we generate a triplet $(\boldsymbol{x},\boldsymbol{t},\boldsymbol{y})$ via a text hash-table, where $\boldsymbol{x}$ is the image, $\boldsymbol{t}$ is the language description ( i.e. | ✅ 给定一个图像-文本对，我们通过文本哈希表生成一个三元组 $(\boldsymbol{x},\boldsymbol{t},\boldsymbol{y})$，其中 $\boldsymbol{x}$ 是图像，$\boldsymbol{t}$ 是语言描述（即 |
| ✅ , hash value), and $\boldsymbol{y}$ is the language label ( i.e. | ✅ ，哈希值），$\boldsymbol{y}$ 是语言标签（即 |
| ✅ , hash key) indicating the index of unique language description in the dataset. | ✅ ，hash key）表示数据集中唯一语言描述的索引。 |
| ✅ Note that we only map identical language description to the same hash key, i.e. | ✅ 请注意，我们只将相同的语言描述映射到相同的哈希键，即 |
| ✅ , language label. | ✅ 、语言标签。 |
| ✅ Thus, all image-text pairs mapped to the same label $\boldsymbol{y}$ are regarded as positive in our universal image-text contrastive learning. | ✅ 因此，在我们的通用图像-文本对比学习中，映射到相同标签 $\boldsymbol{y}$ 的所有图像-文本对都被视为积极的。 |
| ✅ Others are still regarded as negative. | ✅ 其他人仍然被视为负面的。 |
| ✅ The unified learning objective in the common image-label-description space unifies two popular learning paradigms – mapping images to the label for learning discriminative representations ( i.e. | ✅ 通用图像标签描述空间中的统一学习目标统一了两种流行的学习范式——将图像映射到标签以学习判别表示（即。 |
| ✅ , supervised learning) and assigning each description with a unique label for language-image pre-training ( i.e. | ✅ ，监督学习）并为每个描述分配一个唯一的标签，以进行语言图像预训练（即 |
| ✅ , contrastive learning). | ✅ 、对比学习）。 |

| 【第2.2节，第3段】原文 | 【第2.2节，第3段】翻译 |
| ---- | ---- |
| ✅ Our empirical experiments indicate that long language descriptions with rich content would be more beneficial for image-text representation learning than short descriptions ( e.g. | ✅ 我们的实证实验表明，内容丰富的长语言描述比短描述（例如 |
| ✅ , one or two words). | ✅ ，一个或两个词）。 |
| ✅ We have to enrich the short description by generating prompt templates such as “A photo of the [WORD] ”, “A cropped photo of [WORD] ”, as data augmentation. | ✅ 我们必须通过生成提示模板（例如“[WORD] 的照片”、“[WORD] 的裁剪照片”）来丰富简短描述，作为数据增强。 |
| ✅ During training, we randomly select one template to generate $\boldsymbol{t}$ for each short language description. | ✅ 在训练期间，我们随机选择一个模板为每个简短语言描述生成 $\boldsymbol{t}$。 |

| 【第2.2节，第4段】原文 | 【第2.2节，第4段】翻译 |
| ---- | ---- |
| ✅ Following UniCL ( **Unified contrastive learning in image-text-label space.** ) , we denote $f_{\theta}$ and $f_{\phi}$ as the image encoder and text encoder, respectively. | ✅ 按照 UniCL ( **Unified contrastive learning in image-text-label space.** )，我们分别将 $f_{\theta}$ 和 $f_{\phi}$ 表示为图像编码器和文本编码器。 |
| ✅ $\boldsymbol{u}$ and $\boldsymbol{v}$ are the normalized visual feature vector and language feature vector, respectively, where $\boldsymbol{u}=\frac{f_{\theta}(\boldsymbol{x})}{\ \vert f_{\theta}(\boldsymbol{x})\ \vert }$ , and $\boldsymbol{v}=\frac{f_{\phi}(\boldsymbol{t})}{\ \vert f_{\phi}(\boldsymbol{t})\ \vert }$. | ✅ $\boldsymbol{u}$ 和 $\boldsymbol{v}$ 分别是归一化的视觉特征向量和语言特征向量，其中 $\boldsymbol{u}=\frac{f_{\theta}(\boldsymbol{x})}{\ \vert f_{\theta}(\boldsymbol{x})\ \vert }$ 和 $\boldsymbol{v}=\frac{f_{\phi}(\boldsymbol{t})}{\ \vert f_{\phi}(\boldsymbol{t})\ \vert }$。 |
| ✅ $\tau$ is a learnable temperature. | ✅ $\tau$是可学习的温度。 |
| ✅ Given a mini-batch $\mathcal{B}$ , we use a bi-directional supervised contrastive learning objective between images and language descriptions to train the model as: | ✅ 给定一个小批量 $\mathcal{B}$，我们使用图像和语言描述之间的双向监督对比学习目标来训练模型： |

**公式(1):** 
$$ \mathcal{L}=\mathcal{L}_{i2t}+\mathcal{L}_{t2i} $$

| 【第2.2节，第5段】原文 | 【第2.2节，第5段】翻译 |
| ---- | ---- |
| ✅ This objective contains two contrastive terms: the supervised image-to-language contrastive loss | ✅ 该目标包含两个对比项：监督图像到语言对比损失 |

**公式(2):** 
$$ \displaystyle\mathcal{L}_{i2t}= \displaystyle-\sum_{i\in\mathcal{B}}\frac{1}{|\mathcal{P}(i)|}\sum_{k\in\mathcal{P}(i)}\log\frac{\exp(\tau\boldsymbol{u}_{i}\boldsymbol{v}_{k})}{\sum_{j\in\mathcal{B}}\exp(\tau\boldsymbol{u}_{i}\boldsymbol{v}_{j})} $$

| 【第2.2节，第6段】原文 | 【第2.2节，第6段】翻译 |
| ---- | ---- |
| ✅ where $k\in\mathcal{P}(i)=\{k \vert k\in\mathcal{B},y_{k}=y_{i}\}$ , and the supervised language-to-image contrastive loss | ✅ 其中 $k\in\mathcal{P}(i)=\{k \vert k\in\mathcal{B},y_{k}=y_{i}\}$ 和监督语言到图像对比损失 |

**公式(3):** 
$$ \displaystyle\mathcal{L}_{t2i}= \displaystyle-\sum_{j\in\mathcal{B}}\frac{1}{|\mathcal{Q}(j)|}\sum_{k\in\mathcal{Q}(j)}\log\frac{\exp(\tau\boldsymbol{u}_{k}\boldsymbol{v}_{j})}{\sum_{i\in\mathcal{B}}\exp(\tau\boldsymbol{u}_{i}\boldsymbol{v}_{j})} $$

| 【第2.2节，第7段】原文 | 【第2.2节，第7段】翻译 |
| ---- | ---- |
| ✅ where $k\in\mathcal{Q}(j)=\{k \vert k\in\mathcal{B},y_{k}=y_{j}\}$ . | ✅ 其中 $k\in\mathcal{Q}(j)=\{k \vert k\in\mathcal{B},y_{k}=y_{j}\}$ 。 |

| 【第2.2节，第8段】原文 | 【第2.2节，第8段】翻译 |
| ---- | ---- |
| ✅ The generated language prompt is not a precise description of an image, typically not as informative as the associated text descriptions from the Internet. | ✅ 生成的语言提示不是图像的精确描述，通常不如来自互联网的相关文本描述那么具有信息量。 |
| ✅ Although including generated language prompt might not affect classification accuracy, it hurts the performance in retrieval and vision-language tasks. | ✅ 虽然包含生成的语言提示可能不会影响分类准确性，但它会损害检索和视觉语言任务的性能。 |
| ✅ To mitigate the negative effect from augmented prompts, our training is separated into two stages. | ✅ 为了减轻增强提示带来的负面影响，我们的训练分为两个阶段。 |
| ✅ In the first stage, we use all data including augmented texts for training; while in the second stage, we exclude all augmented data for continuing training. | ✅ 在第一阶段，我们使用包括增强文本在内的所有数据进行训练；而在第二阶段，我们排除所有增强数据进行继续训练。 |
| ✅ We trained $1M$ iterations in the first stage, and continuously trained $180K$ iterations in the second stage. | ✅ 我们在第一阶段训练了$1M$次迭代，在第二阶段持续训练了$180K$次迭代。 |
| ✅ The Adam optimizer with decoupled weight decay regularization is utilized for model training. | ✅ 采用具有解耦权重衰减正则化的 Adam 优化器进行模型训练。 |
| ✅ The image size is $224\times 224$ and the maximum language description length is truncated at $76$. | ✅ 图像大小为$224\times 224$，最大语言描述长度在$76$处被截断。 |
| ✅ The batch size is $24,576$. | ✅ 批次大小为$24,576$。 |
| ✅ We further trained $80K$ iterations at a higher resolution of $384\times 384$ to boost the performance, which follows existing pre-training approaches. | ✅ 我们进一步以比 $384\times 384$ 更高的分辨率训练了 $80K$ 迭代以提高性能，这遵循了现有的预训练方法。 |

### 2.3 Transformer-based Florence Pretrained Models

| 【第2.3节，第1段】原文 | 【第2.3节，第1段】翻译 |
| ---- | ---- |
| ✅ Our Florence pretrained model uses a two-tower architecture: a 12-layer transformer ( **Attention is all you need.** ) as language encoder, similar to CLIP ( **Learning transferable visual models from natural language supervision.** ) , and a hierarchical Vision Transformer as the image encoder. | ✅ 我们的 Florence 预训练模型采用双塔架构：一个 12 层变压器 ( **Attention is all you need.** ) 作为语言编码器，类似于 CLIP ( **Learning transferable visual models from natural language supervision.** )，以及一个分层 Vision Transformer 作为图像编码器。 |
| ✅ The hierarchical Vision Transformer is a modified Swin Transformer ( **Swin transformer: Hierarchical vision transformer using shifted windows.** ) with convolutional embedding, called CoSwin Transformer. | ✅ 分层 Vision Transformer 是具有卷积嵌入的改进型 Swin Transformer ( **Swin transformer: Hierarchical vision transformer using shifted windows.** )，称为 CoSwin Transformer。 |
| ✅ Specifically, we replace the patch embedding and patch merging modules in the Swin Transformer ( **Swin transformer: Hierarchical vision transformer using shifted windows.** ) with the convolutional embedding layers as described in CvT ( **Cvt: Introducing convolutions to vision transformers.** ). | ✅ 具体来说，我们用 CvT ( **Cvt: Introducing convolutions to vision transformers.** ) 中描述的卷积嵌入层替换 Swin Transformer ( **Swin transformer: Hierarchical vision transformer using shifted windows.** ) 中的补丁嵌入和补丁合并模块。 |
| ✅ We use the CoSwin Transformer with global average pooling to extract image features. | ✅ 我们使用具有全局平均池化的 CoSwin Transformer 来提取图像特征。 |
| ✅ Two linear projection layers are added on top of the image encoder and language encoder to match the dimensions of image and language features. | ✅ 在图像编码器和语言编码器的顶部添加了两个线性投影层，以匹配图像和语言特征的维度。 |
| ✅ Our Florence pretrained model has in total $893M$ parameters, including the language transformer with $256M$ parameters and the CoSwin -H transformer with $637M$ parameters. | ✅ 我们的 Florence 预训练模型总共有 $893M$ 个参数，其中包括具有 $256M$ 参数的语言转换器和具有 $637M$ 参数的 CoSwin -H 转换器。 |
| ✅ The model takes $10$ days to train on $512$ NVIDIA-A100 GPUs with 40GB memory per GPU. | ✅ 该模型在 $512$ NVIDIA-A100 GPU 上花费 $10$ 天进行训练，每个 GPU 配备 40GB 内存。 |

### 2.4 Object-level Visual Representation Learning

| 【第2.4节，第1段】原文 | 【第2.4节，第1段】翻译 |
| ---- | ---- |
| ✅ We extend the Florence pretrained model to learn fine-grained ( i.e. | ✅ 我们扩展了 Florence 预训练模型来学习细粒度（即 |
| ✅ , object-level) representation, which is fundamental to dense prediction tasks such as object detection. | ✅ ，对象级）表示，这对于诸如对象检测之类的密集预测任务至关重要。 |
| ✅ For this goal, we add an adaptor Dynamic Head ( **Dynamic head: Unifying object detection heads with attentions.** ) (or Dynamic DETR ( **Dynamic detr: End-to-end object detection with dynamic attention.** ) ), a unified attention mechanism for the detection head, to the pretrained image encoder ( i.e. | ✅ 为了实现这一目标，我们在预训练图像编码器（即）中添加了一个适配器动态头 ( **Dynamic head: Unifying object detection heads with attentions.** )（或动态 DETR ( **Dynamic detr: End-to-end object detection with dynamic attention.** )），即检测头的统一注意机制。 |
| ✅ , CoSwin ). | ✅ 、CoSwin）。 |
| ✅ We can continue visual representation learning from coarse (scene) to fine (object). | ✅ 我们可以继续从粗糙（场景）到精细（对象）的视觉表征学习。 |

| 【第2.4节，第2段】原文 | 【第2.4节，第2段】翻译 |
| ---- | ---- |
| ✅ Based on the hierarchical structure of the image encoder CoSwin -H, we can get the output feature pyramids from the different scale levels. | ✅ 基于图像编码器CoSwin-H的层次结构，我们可以从不同的尺度层次得到输出特征金字塔。 |
| ✅ The feature pyramid scale levels can be concatenated and scaled-down or scaled-up into a 3-dimensional tensor with dimensions $level\times space\times channel$. | ✅ 特征金字塔尺度级别可以连接并缩小或放大为维度为 $level\times space\times channel$ 的三维张量。 |
| ✅ The key idea of Dynamic Head ( **Dynamic head: Unifying object detection heads with attentions.** ) is to deploy three attention mechanisms, each on one of the orthogonal dimensions of the tensor, i.e. | ✅ Dynamic Head ( **Dynamic head: Unifying object detection heads with attentions.** ) 的关键思想是部署三种注意机制，每种机制位于张量的正交维度之一上，即 |
| ✅ , level-wise, spatial-wise, and channel-wise. | ✅ 、级别、空间和通道。 |
| ✅ Compared with building a single self-attention mechanism over this tensor, Dynamic Head makes the computation more affordable and enables more efficient learning. | ✅ 与在该张量上构建单一的自注意力机制相比，Dynamic Head 使得计算更加廉价，并且能够实现更高效的学习。 |
| ✅ The above three attention mechanisms are applied sequentially, and we can effectively stack multiple blocks consisting of such three attention layers together. | ✅ 上述三种注意力机制被依次应用，并且我们可以有效地将由这三种注意力层组成的多个块堆叠在一起。 |
| ✅ Figure 3 shows the Dynamic Head building blocks. | ✅ 图 3 显示了动态头构建块。 |
| ✅ In this work, Dynamic Head is trained with the one-stage ATSS framework and losses. | ✅ 在这项工作中，Dynamic Head 采用单阶段 ATSS 框架和损失进行训练。 |

![figure](https://ar5iv.labs.arxiv.org/html/2111.11432/assets/x3.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 3:  Dynamic Head ( **Dynamic head: Unifying object detection heads with attentions.** ) adapter is used for object-level visual representation learning. | ✅ Figure 3: 动态头( **Dynamic head: Unifying object detection heads with attentions.** )适配器用于对象级视觉表征学习。 |

| 【第2.4节，第3段】原文 | 【第2.4节，第3段】翻译 |
| ---- | ---- |
| ✅ We have constructed a large-scale object detection dataset, called FLOD-9M (for FL orence O bject detection D ataset), for object detection pre-training. | ✅ 我们构建了一个大规模物体检测数据集，称为 FLOD-9M（FL orence O bject detection D ataset），用于物体检测预训练。 |
| ✅ We merge several well-known object detection datasets, including COCO ( **Microsoft COCO:: Common objects in context, 2015.** ) , LVIS ( **Lvis: A dataset for large vocabulary instance segmentation.** ) , OpenImages ( **Openimages: A public dataset for large-scale multi-label and multi-class image classification.** ) , Object365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ). | ✅ 我们合并了几个著名的物体检测数据集，包括 COCO ( **Microsoft COCO:: Common objects in context, 2015.** )、LVIS ( **Lvis: A dataset for large vocabulary instance segmentation.** )、OpenImages ( **Openimages: A public dataset for large-scale multi-label and multi-class image classification.** )、Object365 ( **Objects365: A large-scale, high-quality dataset for object detection.** )。 |
| ✅ In addition, we generate pseudo bounding boxes on ImageNet-22K dataset ( **Imagenet: A large-scale hierarchical image database.** ) by following ( **Rethinking pre-training and self-training.** ) , which further enlarges our data. | ✅ 此外，我们按照( **Rethinking pre-training and self-training.** )在ImageNet-22K数据集( **Imagenet: A large-scale hierarchical image database.** )上生成伪边界框，进一步扩大了我们的数据。 |
| ✅ In the end, FLOD-9M consists of $8,967,286$ images, $25,190$ object categories, and $33,408,237$ bounding boxes including annotations and pseudo labels. | ✅ 最后，FLOD-9M 由 $8,967,286$ 图像、$25,190$ 对象类别和包括注释和伪标签的 $33,408,237$ 边界框组成。 |
| ✅ We then pre-train our Dynamic Head model for $12$ epochs with batch size $128$ , which takes $7$ days on $128$ NVIDIA-A100 GPUs. | ✅ 然后，我们对 $12$ 时期的动态头部模型进行预训练，批次大小为 $128$，在 $128$ NVIDIA-A100 GPU 上需要 $7$ 天。 |

### 2.5 Fine-Grained V+L Representation Learning

| 【第2.5节，第1段】原文 | 【第2.5节，第1段】翻译 |
| ---- | ---- |
| ✅ We use METER ( **An empirical study of training end-to-end vision-and-language transformers.** ) adapter to expand to fine-grained vision-language representation. | ✅ 我们使用 METER ( **An empirical study of training end-to-end vision-and-language transformers.** ) 适配器来扩展到细粒度的视觉语言表示。 |
| ✅ In the vision-language area, e.g. | ✅ 在视觉语言领域，例如 |
| ✅ visual question answering (VQA) and image captioning, fine-grained representation ( i.e. | ✅ 视觉问答（VQA）和图像字幕、细粒度表示（即 |
| ✅ , object-level) is indispensable. | ✅ （1）、对象级）是不可缺少的。 |
| ✅ Thus, the object detector has been a de facto tool for image feature extraction, followed by a fusion network for prediction in many works ( **1. Bottom-up and top-down attention for image captioning and visual question answering.** ｜ **2. Oscar: Object-semantics aligned pre-training for vision-language tasks.** ｜ **3. Vinvl: Revisiting visual representations in vision-language models.** ｜ **4. Minivlm: A smaller and faster vision-language model.** ｜ **5. Compressing visual-linguistic model via knowledge distillation.** ｜ **6. Uniter: Universal image-text representation learning.** ). | ✅ 因此，在许多工作 ( **1. Bottom-up and top-down attention for image captioning and visual question answering.** ｜ **2. Oscar: Object-semantics aligned pre-training for vision-language tasks.** ｜ **3. Vinvl: Revisiting visual representations in vision-language models.** ｜ **4. Minivlm: A smaller and faster vision-language model.** ｜ **5. Compressing visual-linguistic model via knowledge distillation.** ｜ **6. Uniter: Universal image-text representation learning.** ) 中，对象检测器已成为图像特征提取的事实上的工具，随后使用融合网络进行预测。 |
| ✅ Recently, there is an increasing trend ( **1. Seeing out of the box: End-to-end pre-training for vision-language representation learning.** ｜ **2. Probing inter-modality: Visual parsing with self-attention for vision-language pre-training.** ｜ **3. Simvlm: Simple visual language model pretraining with weak supervision.** ｜ **4. Vilt: Vision-and-language transformer without convolution or region supervision.** ｜ **5. An empirical study of training end-to-end vision-and-language transformers.** ) of end-to-end approaches to reduce dependency on the object bounding box, which instead consider grid-based feature representations as the fine-grained features for V+L tasks. | ✅ 最近，端到端方法的趋势 ( **1. Seeing out of the box: End-to-end pre-training for vision-language representation learning.** ｜ **2. Probing inter-modality: Visual parsing with self-attention for vision-language pre-training.** ｜ **3. Simvlm: Simple visual language model pretraining with weak supervision.** ｜ **4. Vilt: Vision-and-language transformer without convolution or region supervision.** ｜ **5. An empirical study of training end-to-end vision-and-language transformers.** ) 日益增多，以减少对对象边界框的依赖，而是将基于网格的特征表示视为 V+L 任务的细粒度特征。 |

| 【第2.5节，第2段】原文 | 【第2.5节，第2段】翻译 |
| ---- | ---- |
| ✅ In the Florence V+L adaptation model, we replace the image encoder of METER ( **An empirical study of training end-to-end vision-and-language transformers.** ) with Florence pretrained model CoSwin , and use a pretrained Roberta ( **RoBERTa: A robustly optimized bert pretraining approach.** ) as the language encoder, shown in Figure 4. | ✅ 在 Florence V+L 适配模型中，我们用 Florence 预训练模型 CoSwin 替换了 METER ( **An empirical study of training end-to-end vision-and-language transformers.** ) 的图像编码器，并使用预训练的 Roberta ( **RoBERTa: A robustly optimized bert pretraining approach.** ) 作为语言编码器，如图 4 所示。 |
| ✅ The Florence pretrained language encoder can be used for this adapter as it utilizes BERT-based architecture. | ✅ 由于 Florence 预训练语言编码器采用基于 BERT 的架构，因此可用于此适配器。 |
| ✅ Then, the two modalities are fused together to learn the contextual representation with a transformer network based on co-attention. | ✅ 然后，将两种模态融合在一起，利用基于共同注意的变换器网络来学习上下文表示。 |
| ✅ The co-attention model (Figure 4 ) allows feeding the text and visual features to two $M_{co}$ -layer transformers separately, and each top transformer encoding layer consists of one self-attention block, one cross-attention block, and one feed-forward network block. | ✅ 共同注意模型（图 4）允许将文本和视觉特征分别输入到两个 $M_{co}$ 层变压器，并且每个顶部变压器编码层由一个自注意块、一个交叉注意块和一个前馈网络块组成。 |
| ✅ We first train the model with the image-text matching loss and the masked-language modeling loss. | ✅ 我们首先使用图像文本匹配损失和掩蔽语言建模损失来训练模型。 |
| ✅ Then, we fine-tune the model on the downstream task, such as VQA ( **Making the V in VQA matter: Elevating the role of image understanding in visual question answering.** ) task. | ✅ 然后，我们在下游任务上对模型进行微调，例如 VQA ( **Making the V in VQA matter: Elevating the role of image understanding in visual question answering.** ) 任务。 |

![figure](https://ar5iv.labs.arxiv.org/html/2111.11432/assets/x4.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 4:  METER ( **An empirical study of training end-to-end vision-and-language transformers.** ) is used as Florence V+L adaptation model, trained with the image-text matching (ITM) loss and the masked language modeling (MLM) loss. | ✅ Figure 4:  METER ( **An empirical study of training end-to-end vision-and-language transformers.** ) 用作 Florence V+L 适配模型，使用图像文本匹配 (ITM) 损失和掩码语言建模 (MLM) 损失进行训练。 |

### 2.6 Adaption to Video Recognition

| 【第2.6节，第1段】原文 | 【第2.6节，第1段】翻译 |
| ---- | ---- |
| ✅ The self-attention based design in Transformer makes it possible to unify the systems of image and video recognition. | ✅ Transformer中基于自注意力的设计使得统一图像和视频识别系统成为可能。 |
| ✅ Our Video CoSwin adapter can borrow the image encoder from CoSwin for the video domain with minimum changes, similar to prior work ( **Video swin transformer.** ). | ✅ 我们的视频 CoSwin 适配器可以借用 CoSwin 的图像编码器用于视频领域，只需进行最少的更改，类似于之前的工作 ( **Video swin transformer.** )。 |
| ✅ First, the image tokenization layer is replaced with a video tokenization layer. | ✅ 首先，将图像标记层替换为视频标记层。 |
| ✅ Accordingly, video CoSwin replaces the tokenization layer of CoSwin (in Section 2.3 ) from 2D convolutional layers to 3D convolutional layers, which converts each 3D tube into one token. | ✅ 相应地，视频 CoSwin 将 CoSwin 的标记层（见第 2.3 节）从 2D 卷积层替换为 3D 卷积层，将每个 3D 管转换为一个标记。 |
| ✅ As the initialization to 3D convolutional weights, the pre-trained 2D convolutional weights of CoSwin are duplicated along the temporal dimension and divided by the temporal kernel size to keep the mean and variance of the output unchanged. | ✅ 作为对 3D 卷积权重的初始化，CoSwin 的预训练 2D 卷积权重沿时间维度复制并除以时间核大小，以保持输出的均值和方差不变。 |
| ✅ Second, video CoSwin uses the 3D convolution-based patch merging operator instead of the 2D patch merging operator used in ( **Video swin transformer.** ). | ✅ 其次，视频CoSwin采用基于3D卷积的面片合并算子，替代( **Video swin transformer.** )中使用的2D面片合并算子。 |
| ✅ Such overlapped token merging can enhance spatial and temporal interactions among tokens. | ✅ 这种重叠的标记合并可以增强标记之间的空间和时间交互。 |
| ✅ Third, we follow prior work ( **Video swin transformer.** ) to replace the 2D shifted window design with 3D shifted local windows in self-attention layers. | ✅ 第三，我们遵循之前的工作 ( **Video swin transformer.** )，在自注意层中用 3D 移位局部窗口取代 2D 移位窗口设计。 |
| ✅ We duplicate the 2D relative positional embedding matrix from the pre-trained CoSwin along the temporal dimension to initialize the 3D positional embedding matrix. | ✅ 我们沿时间维度从预先训练的 CoSwin 中复制二维相对位置嵌入矩阵，以初始化三维位置嵌入矩阵。 |
| ✅ In this way, the 2D relative positional embedding is the same for each temporal shift. | ✅ 这样，二维相对位置嵌入对于每次时间移位都是相同的。 |
| ✅ In addition, all other layers and weights (including self-attention, FFN) can be inherited directly from the pre-trained CoSwin. | ✅ 此外，所有其他层和权重（包括自我注意力、FFN）都可以直接从预先训练的 CoSwin 继承。 |
| ✅ To mitigate memory issues in the video training, we adopt the dynamic window size strategy, i.e., a relatively small window size in early stages of CoSwin , and large window sizes in its later stages. | ✅ 为了缓解视频训练中的内存问题，我们采用动态窗口大小策略，即在 CoSwin 的早期阶段使用相对较小的窗口大小，在后期阶段使用较大的窗口大小。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S2.T1.2"><thead class="ltx_thead"><tr class="ltx_tr" id="S2.T1.2.3.1"><th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S2.T1.2.3.1.1" style="padding:1.6pt 7.1pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.2" style="padding:1.6pt 7.1pt;">Food101</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.3" style="padding:1.6pt 7.1pt;">CIFAR10</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.4" style="padding:1.6pt 7.1pt;">CIFAR100</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.5" style="padding:1.6pt 7.1pt;">SUN397</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.6" style="padding:1.6pt 7.1pt;">Stanford Cars</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.7" style="padding:1.6pt 7.1pt;">FGVC Aircraft</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.8" style="padding:1.6pt 7.1pt;">VOC2007</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.9" style="padding:1.6pt 7.1pt;">DTD</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.10" style="padding:1.6pt 7.1pt;">Oxford Pets</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.11" style="padding:1.6pt 7.1pt;">Caltech101</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.12" style="padding:1.6pt 7.1pt;">Flowers102</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S2.T1.2.3.1.13" style="padding:1.6pt 7.1pt;">ImageNet</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S2.T1.2.4.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S2.T1.2.4.1.1" style="padding:1.6pt 7.1pt;">CLIP-ResNet-50x64</th><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.2" style="padding:1.6pt 7.1pt;">91.8</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.3" style="padding:1.6pt 7.1pt;">86.8</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.4" style="padding:1.6pt 7.1pt;">61.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.5" style="padding:1.6pt 7.1pt;">48.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.6" style="padding:1.6pt 7.1pt;">76.0</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.7" style="padding:1.6pt 7.1pt;">35.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.8" style="padding:1.6pt 7.1pt;">83.8</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.9" style="padding:1.6pt 7.1pt;">53.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.10" style="padding:1.6pt 7.1pt;">93.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.11" style="padding:1.6pt 7.1pt;">90.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.12" style="padding:1.6pt 7.1pt;">77.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S2.T1.2.4.1.13" style="padding:1.6pt 7.1pt;">73.6</td></tr><tr class="ltx_tr" id="S2.T1.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S2.T1.1.1.1" style="padding:1.6pt 7.1pt;">CLIP-ViT-L/14 (<math alttext="@336" class="ltx_Math" display="inline" id="S2.T1.1.1.1.m1.1"><semantics id="S2.T1.1.1.1.m1.1a"><mrow id="S2.T1.1.1.1.m1.1.1" xref="S2.T1.1.1.1.m1.1.1.cmml"><mi id="S2.T1.1.1.1.m1.1.1.2" mathsize="90%" mathvariant="normal" xref="S2.T1.1.1.1.m1.1.1.2.cmml">@</mi><mo id="S2.T1.1.1.1.m1.1.1.1" lspace="0em" rspace="0em" xref="S2.T1.1.1.1.m1.1.1.1.cmml">​</mo><mn id="S2.T1.1.1.1.m1.1.1.3" mathsize="90%" xref="S2.T1.1.1.1.m1.1.1.3.cmml">336</mn></mrow><annotation-xml encoding="MathML-Content" id="S2.T1.1.1.1.m1.1b"><apply id="S2.T1.1.1.1.m1.1.1.cmml" xref="S2.T1.1.1.1.m1.1.1"><times id="S2.T1.1.1.1.m1.1.1.1.cmml" xref="S2.T1.1.1.1.m1.1.1.1"></times><ci id="S2.T1.1.1.1.m1.1.1.2.cmml" xref="S2.T1.1.1.1.m1.1.1.2">@</ci><cn id="S2.T1.1.1.1.m1.1.1.3.cmml" type="integer" xref="S2.T1.1.1.1.m1.1.1.3">336</cn></apply></annotation-xml><annotation encoding="application/x-tex" id="S2.T1.1.1.1.m1.1c">@336</annotation></semantics></math>pix)</th><td class="ltx_td ltx_align_center" id="S2.T1.1.1.2" style="padding:1.6pt 7.1pt;">93.8</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.3" style="padding:1.6pt 7.1pt;">95.7</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.4" style="padding:1.6pt 7.1pt;">77.5</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.5" style="padding:1.6pt 7.1pt;">68.4</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.6" style="padding:1.6pt 7.1pt;">78.8</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.7" style="padding:1.6pt 7.1pt;">37.2</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.8" style="padding:1.6pt 7.1pt;">84.3</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.9" style="padding:1.6pt 7.1pt;">55.7</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.10" style="padding:1.6pt 7.1pt;">93.5</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.11" style="padding:1.6pt 7.1pt;">92.8</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.12" style="padding:1.6pt 7.1pt;">78.3</td><td class="ltx_td ltx_align_center" id="S2.T1.1.1.13" style="padding:1.6pt 7.1pt;">76.2</td></tr><tr class="ltx_tr" id="S2.T1.2.5.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S2.T1.2.5.2.1" style="padding:1.6pt 7.1pt;">FLIP-ViT-L/14</th><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.2" style="padding:1.6pt 7.1pt;">92.2</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.3" style="padding:1.6pt 7.1pt;">95.7</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.4" style="padding:1.6pt 7.1pt;">75.3</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.5" style="padding:1.6pt 7.1pt;">73.1</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.6" style="padding:1.6pt 7.1pt;">70.8</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.7" style="padding:1.6pt 7.1pt;">60.2</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.8" style="padding:1.6pt 7.1pt;">-</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.9" style="padding:1.6pt 7.1pt;">60.7</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.10" style="padding:1.6pt 7.1pt;">92.0</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.11" style="padding:1.6pt 7.1pt;">93.0</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.12" style="padding:1.6pt 7.1pt;">90.1</td><td class="ltx_td ltx_align_center" id="S2.T1.2.5.2.13" style="padding:1.6pt 7.1pt;">78.3</td></tr><tr class="ltx_tr" id="S2.T1.2.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S2.T1.2.2.1" style="padding:1.6pt 7.1pt;">Florence-CoSwin-H (<math alttext="@384" class="ltx_Math" display="inline" id="S2.T1.2.2.1.m1.1"><semantics id="S2.T1.2.2.1.m1.1a"><mrow id="S2.T1.2.2.1.m1.1.1" xref="S2.T1.2.2.1.m1.1.1.cmml"><mi id="S2.T1.2.2.1.m1.1.1.2" mathsize="90%" mathvariant="normal" xref="S2.T1.2.2.1.m1.1.1.2.cmml">@</mi><mo id="S2.T1.2.2.1.m1.1.1.1" lspace="0em" rspace="0em" xref="S2.T1.2.2.1.m1.1.1.1.cmml">​</mo><mn id="S2.T1.2.2.1.m1.1.1.3" mathsize="90%" xref="S2.T1.2.2.1.m1.1.1.3.cmml">384</mn></mrow><annotation-xml encoding="MathML-Content" id="S2.T1.2.2.1.m1.1b"><apply id="S2.T1.2.2.1.m1.1.1.cmml" xref="S2.T1.2.2.1.m1.1.1"><times id="S2.T1.2.2.1.m1.1.1.1.cmml" xref="S2.T1.2.2.1.m1.1.1.1"></times><ci id="S2.T1.2.2.1.m1.1.1.2.cmml" xref="S2.T1.2.2.1.m1.1.1.2">@</ci><cn id="S2.T1.2.2.1.m1.1.1.3.cmml" type="integer" xref="S2.T1.2.2.1.m1.1.1.3">384</cn></apply></annotation-xml><annotation encoding="application/x-tex" id="S2.T1.2.2.1.m1.1c">@384</annotation></semantics></math>pix)</th><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.2" style="padding:1.6pt 7.1pt;">95.1</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.3" style="padding:1.6pt 7.1pt;">94.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.4" style="padding:1.6pt 7.1pt;">77.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.5" style="padding:1.6pt 7.1pt;">77.0</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.6" style="padding:1.6pt 7.1pt;">93.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.7" style="padding:1.6pt 7.1pt;">55.5</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.8" style="padding:1.6pt 7.1pt;">85.5</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.9" style="padding:1.6pt 7.1pt;">66.4</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.10" style="padding:1.6pt 7.1pt;">95.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.11" style="padding:1.6pt 7.1pt;">94.7</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.12" style="padding:1.6pt 7.1pt;">86.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S2.T1.2.2.13" style="padding:1.6pt 7.1pt;">83.7</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 1:  Zero-shot transfer of image classification comparisons on 12 datasets: CLIP-ResNet-50x64 ( **Learning transferable visual models from natural language supervision.** ) , FLIP-ViT-L/14 ( **Filip: Fine-grained interactive language-image pre-training.** ) . | ✅ Table 1:  在 12 个数据集上进行图像分类比较的零样本迁移：CLIP-ResNet-50x64 ( **Learning transferable visual models from natural language supervision.** )、FLIP-ViT-L/14 ( **Filip: Fine-grained interactive language-image pre-training.** )。 |

### 2.7 Scalable Training Infrastructure

![figure](https://ar5iv.labs.arxiv.org/html/2111.11432/assets/x5.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 5:  GPU memory reduction for various batch sizes. | ✅ 针对不同批次大小的 Figure 5:  GPU 内存减少量。 |
| ✅ We compared the profiling between Torch (w/o optimization) and Florence (w/ optimization) on various number of GPUs. | ✅ 我们比较了 Torch（未进行优化）和 Florence（进行优化）在不同数量的 GPU 上的分析。 |

| 【第2.7节，第1段】原文 | 【第2.7节，第1段】翻译 |
| ---- | ---- |
| ✅ To train the Florence model on our large-scale dataset, our scalable training infrastructure faces two main challenges: reducing memory cost on each GPU and increasing the throughput. | ✅ 为了在我们的大规模数据集上训练 Florence 模型，我们的可扩展训练基础设施面临两个主要挑战：降低每个 GPU 上的内存成本并提高吞吐量。 |
| ✅ Reducing the memory cost allows us to feed more data into each GPU and use a larger batch size, which has been proved to be effective for contrastive learning. | ✅ 降低内存成本使我们能够向每个 GPU 输入更多数据并使用更大的批量大小，这已被证明对于对比学习是有效的。 |
| ✅ Increasing the throughput can significantly speed up the whole training process and thus reduce carbon emissions. | ✅ 提高吞吐量可以显著加快整个训练过程，从而减少碳排放。 |
| ✅ We have developed several techniques that can be combined to achieve the two goals: | ✅ 我们开发了几种可以结合起来实现两个目标的技术： |

| 【第2.7节，第2段】原文 | 【第2.7节，第2段】翻译 |
| ---- | ---- |
| ✅ The ZeRO technique ( **Zero: Memory optimization towards training A trillion parameter models.** ) partitions the optimizer states, gradients and parameters across the GPUs and each partition is only updated locally. | ✅ ZeRO 技术 ( **Zero: Memory optimization towards training A trillion parameter models.** ) 在 GPU 上对优化器状态、梯度和参数进行分区，并且每个分区仅在本地更新。 |
| ✅ Thus, the memory consumption is largely reduced. | ✅ 因此，内存消耗大大减少。 |

| 【第2.7节，第3段】原文 | 【第2.7节，第3段】翻译 |
| ---- | ---- |
| ✅ For a checkpointed model component, e.g. | ✅ 对于检查点模型组件，例如 |
| ✅ , multi-head attention, it reruns a forward pass during backward pass. | ✅ ，多头注意力，它在后向传递过程中重新运行前向传递。 |
| ✅ In this way, the internal gradients in the component do not need to be stored in the forward pass and then reduce the memory cost in the training. | ✅ 这样，组件中的内部梯度就不需要存储在前向传递中，从而减少训练中的内存成本。 |

| 【第2.7节，第4段】原文 | 【第2.7节，第4段】翻译 |
| ---- | ---- |
| ✅ In mixed-precision training, various operations are trained with different numerical precision ( i.e. | ✅ 在混合精度训练中，各种操作以不同的数值精度进行训练（即 |
| ✅ , float-32 or float-16). | ✅ 、float-32 或 float-16）。 |
| ✅ Float-32 is used for numerically less stable operations, such as layer normalization; while float-16 is used for the other operations. | ✅ Float-32 用于数值上不太稳定的操作，例如层规范化；而 float-16 用于其他操作。 |
| ✅ Such a combination improves the training throughput and maintains the model performance. | ✅ 这样的组合提高了训练吞吐量并保持了模型性能。 |

| 【第2.7节，第5段】原文 | 【第2.7节，第5段】翻译 |
| ---- | ---- |
| ✅ The gradient cache technique ( **Scaling deep contrastive learning batch size under memory limited setup.** ) is able to increase the total batch size in a training step. | ✅ 梯度缓存技术 ( **Scaling deep contrastive learning batch size under memory limited setup.** ) 能够增加训练步骤中的总批次大小。 |
| ✅ A large batch size is shown to be beneficial to learn better representations in previous works. | ✅ 先前的研究证明，较大的批量有利于学习更好的表征。 |
| ✅ However, it is bounded by available GPU memory. | ✅ 然而，它受到可用 GPU 内存的限制。 |
| ✅ To resolve this problem, we factor the contrastive loss by breaking the large batch gradient update into several sub-updates that can fit into GPU memory. | ✅ 为了解决这个问题，我们将大批量梯度更新分解为几个可以放入 GPU 内存的子更新，从而计算对比损失。 |
| ✅ It enables us to train big models with a large batch size. | ✅ 它使我们能够训练具有大批量的大型模型。 |

| 【第2.7节，第6段】原文 | 【第2.7节，第6段】翻译 |
| ---- | ---- |
| ✅ Thanks to these above optimizations, we can achieve consistent improvement in reducing GPU memory for variable batch sizes on various numbers of NVIDIA-A100s, shown in Figure 5 . | ✅ 通过上述优化，我们可以在不同数量的 NVIDIA-A100 上针对不同批次大小的 GPU 内存减少方面实现持续改进，如图 5 所示。 |

## 3 Experiments

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S3.T2.4"><thead class="ltx_thead"><tr class="ltx_tr" id="S3.T2.4.5.1"><th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S3.T2.4.5.1.1" style="padding:1.6pt 8.3pt;"></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.2" style="padding:1.6pt 8.3pt;">Food101</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.3" style="padding:1.6pt 8.3pt;">CIFAR10</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.4" style="padding:1.6pt 8.3pt;">CIFAR100</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.5" style="padding:1.6pt 8.3pt;">SUN397</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.6" style="padding:1.6pt 8.3pt;">Stanford Cars</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.7" style="padding:1.6pt 8.3pt;">FGVC Aircraft</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.8" style="padding:1.6pt 8.3pt;">VOC2007</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.9" style="padding:1.6pt 8.3pt;">DTD</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.10" style="padding:1.6pt 8.3pt;">Oxford Pets</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.11" style="padding:1.6pt 8.3pt;">Caltech101</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T2.4.5.1.12" style="padding:1.6pt 8.3pt;">Flowers102</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T2.4.6.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T2.4.6.1.1" style="padding:1.6pt 8.3pt;">SimCLRv2-ResNet-152x3</th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.2" style="padding:1.6pt 8.3pt;">83.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.3" style="padding:1.6pt 8.3pt;">96.8</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.4" style="padding:1.6pt 8.3pt;">84.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.5" style="padding:1.6pt 8.3pt;">69.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.6" style="padding:1.6pt 8.3pt;">68.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.7" style="padding:1.6pt 8.3pt;">63.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.8" style="padding:1.6pt 8.3pt;">86.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.9" style="padding:1.6pt 8.3pt;">80.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.10" style="padding:1.6pt 8.3pt;">92.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.11" style="padding:1.6pt 8.3pt;">94.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T2.4.6.1.12" style="padding:1.6pt 8.3pt;">96.3</td></tr><tr class="ltx_tr" id="S3.T2.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T2.1.1.1" style="padding:1.6pt 8.3pt;">ViT-L/16 (<math alttext="@384" class="ltx_Math" display="inline" id="S3.T2.1.1.1.m1.1"><semantics id="S3.T2.1.1.1.m1.1a"><mrow id="S3.T2.1.1.1.m1.1.1" xref="S3.T2.1.1.1.m1.1.1.cmml"><mi id="S3.T2.1.1.1.m1.1.1.2" mathsize="90%" mathvariant="normal" xref="S3.T2.1.1.1.m1.1.1.2.cmml">@</mi><mo id="S3.T2.1.1.1.m1.1.1.1" lspace="0em" rspace="0em" xref="S3.T2.1.1.1.m1.1.1.1.cmml">​</mo><mn id="S3.T2.1.1.1.m1.1.1.3" mathsize="90%" xref="S3.T2.1.1.1.m1.1.1.3.cmml">384</mn></mrow><annotation-xml encoding="MathML-Content" id="S3.T2.1.1.1.m1.1b"><apply id="S3.T2.1.1.1.m1.1.1.cmml" xref="S3.T2.1.1.1.m1.1.1"><times id="S3.T2.1.1.1.m1.1.1.1.cmml" xref="S3.T2.1.1.1.m1.1.1.1"></times><ci id="S3.T2.1.1.1.m1.1.1.2.cmml" xref="S3.T2.1.1.1.m1.1.1.2">@</ci><cn id="S3.T2.1.1.1.m1.1.1.3.cmml" type="integer" xref="S3.T2.1.1.1.m1.1.1.3">384</cn></apply></annotation-xml><annotation encoding="application/x-tex" id="S3.T2.1.1.1.m1.1c">@384</annotation></semantics></math>pix)</th><td class="ltx_td ltx_align_center" id="S3.T2.1.1.2" style="padding:1.6pt 8.3pt;">87.4</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.3" style="padding:1.6pt 8.3pt;">97.9</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.4" style="padding:1.6pt 8.3pt;">89.0</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.5" style="padding:1.6pt 8.3pt;">74.9</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.6" style="padding:1.6pt 8.3pt;">62.5</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.7" style="padding:1.6pt 8.3pt;">52.2</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.8" style="padding:1.6pt 8.3pt;">86.1</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.9" style="padding:1.6pt 8.3pt;">75.0</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.10" style="padding:1.6pt 8.3pt;">92.9</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.11" style="padding:1.6pt 8.3pt;">94.7</td><td class="ltx_td ltx_align_center" id="S3.T2.1.1.12" style="padding:1.6pt 8.3pt;">99.3</td></tr><tr class="ltx_tr" id="S3.T2.2.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T2.2.2.1" style="padding:1.6pt 8.3pt;">EfficientNet-L2 (<math alttext="@800" class="ltx_Math" display="inline" id="S3.T2.2.2.1.m1.1"><semantics id="S3.T2.2.2.1.m1.1a"><mrow id="S3.T2.2.2.1.m1.1.1" xref="S3.T2.2.2.1.m1.1.1.cmml"><mi id="S3.T2.2.2.1.m1.1.1.2" mathsize="90%" mathvariant="normal" xref="S3.T2.2.2.1.m1.1.1.2.cmml">@</mi><mo id="S3.T2.2.2.1.m1.1.1.1" lspace="0em" rspace="0em" xref="S3.T2.2.2.1.m1.1.1.1.cmml">​</mo><mn id="S3.T2.2.2.1.m1.1.1.3" mathsize="90%" xref="S3.T2.2.2.1.m1.1.1.3.cmml">800</mn></mrow><annotation-xml encoding="MathML-Content" id="S3.T2.2.2.1.m1.1b"><apply id="S3.T2.2.2.1.m1.1.1.cmml" xref="S3.T2.2.2.1.m1.1.1"><times id="S3.T2.2.2.1.m1.1.1.1.cmml" xref="S3.T2.2.2.1.m1.1.1.1"></times><ci id="S3.T2.2.2.1.m1.1.1.2.cmml" xref="S3.T2.2.2.1.m1.1.1.2">@</ci><cn id="S3.T2.2.2.1.m1.1.1.3.cmml" type="integer" xref="S3.T2.2.2.1.m1.1.1.3">800</cn></apply></annotation-xml><annotation encoding="application/x-tex" id="S3.T2.2.2.1.m1.1c">@800</annotation></semantics></math>pix)</th><td class="ltx_td ltx_align_center" id="S3.T2.2.2.2" style="padding:1.6pt 8.3pt;">92.0</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.3" style="padding:1.6pt 8.3pt;">98.7</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.4" style="padding:1.6pt 8.3pt;">89.0</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.5" style="padding:1.6pt 8.3pt;">75.7</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.6" style="padding:1.6pt 8.3pt;">75.5</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.7" style="padding:1.6pt 8.3pt;">68.4</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.8" style="padding:1.6pt 8.3pt;">89.4</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.9" style="padding:1.6pt 8.3pt;">82.5</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.10" style="padding:1.6pt 8.3pt;">95.6</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.11" style="padding:1.6pt 8.3pt;">94.7</td><td class="ltx_td ltx_align_center" id="S3.T2.2.2.12" style="padding:1.6pt 8.3pt;">97.9</td></tr><tr class="ltx_tr" id="S3.T2.4.7.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T2.4.7.2.1" style="padding:1.6pt 8.3pt;">CLIP-ResNet-50x64</th><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.2" style="padding:1.6pt 8.3pt;">94.8</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.3" style="padding:1.6pt 8.3pt;">94.1</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.4" style="padding:1.6pt 8.3pt;">78.6</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.5" style="padding:1.6pt 8.3pt;">81.1</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.6" style="padding:1.6pt 8.3pt;">90.5</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.7" style="padding:1.6pt 8.3pt;">67.7</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.8" style="padding:1.6pt 8.3pt;">88.9</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.9" style="padding:1.6pt 8.3pt;">82.0</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.10" style="padding:1.6pt 8.3pt;">94.5</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.11" style="padding:1.6pt 8.3pt;">95.4</td><td class="ltx_td ltx_align_center" id="S3.T2.4.7.2.12" style="padding:1.6pt 8.3pt;">98.9</td></tr><tr class="ltx_tr" id="S3.T2.3.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T2.3.3.1" style="padding:1.6pt 8.3pt;">CLIP-ViT-L/14 (<math alttext="@336" class="ltx_Math" display="inline" id="S3.T2.3.3.1.m1.1"><semantics id="S3.T2.3.3.1.m1.1a"><mrow id="S3.T2.3.3.1.m1.1.1" xref="S3.T2.3.3.1.m1.1.1.cmml"><mi id="S3.T2.3.3.1.m1.1.1.2" mathsize="90%" mathvariant="normal" xref="S3.T2.3.3.1.m1.1.1.2.cmml">@</mi><mo id="S3.T2.3.3.1.m1.1.1.1" lspace="0em" rspace="0em" xref="S3.T2.3.3.1.m1.1.1.1.cmml">​</mo><mn id="S3.T2.3.3.1.m1.1.1.3" mathsize="90%" xref="S3.T2.3.3.1.m1.1.1.3.cmml">336</mn></mrow><annotation-xml encoding="MathML-Content" id="S3.T2.3.3.1.m1.1b"><apply id="S3.T2.3.3.1.m1.1.1.cmml" xref="S3.T2.3.3.1.m1.1.1"><times id="S3.T2.3.3.1.m1.1.1.1.cmml" xref="S3.T2.3.3.1.m1.1.1.1"></times><ci id="S3.T2.3.3.1.m1.1.1.2.cmml" xref="S3.T2.3.3.1.m1.1.1.2">@</ci><cn id="S3.T2.3.3.1.m1.1.1.3.cmml" type="integer" xref="S3.T2.3.3.1.m1.1.1.3">336</cn></apply></annotation-xml><annotation encoding="application/x-tex" id="S3.T2.3.3.1.m1.1c">@336</annotation></semantics></math>pix)</th><td class="ltx_td ltx_align_center" id="S3.T2.3.3.2" style="padding:1.6pt 8.3pt;">95.9</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.3" style="padding:1.6pt 8.3pt;">97.9</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.4" style="padding:1.6pt 8.3pt;">87.4</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.5" style="padding:1.6pt 8.3pt;">82.2</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.6" style="padding:1.6pt 8.3pt;">91.5</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.7" style="padding:1.6pt 8.3pt;">71.6</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.8" style="padding:1.6pt 8.3pt;">89.9</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.9" style="padding:1.6pt 8.3pt;">83.0</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.10" style="padding:1.6pt 8.3pt;">95.1</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.11" style="padding:1.6pt 8.3pt;">96.0</td><td class="ltx_td ltx_align_center" id="S3.T2.3.3.12" style="padding:1.6pt 8.3pt;">99.2</td></tr><tr class="ltx_tr" id="S3.T2.4.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S3.T2.4.4.1" style="padding:1.6pt 8.3pt;">Florence-CoSwin-H (<math alttext="@384" class="ltx_Math" display="inline" id="S3.T2.4.4.1.m1.1"><semantics id="S3.T2.4.4.1.m1.1a"><mrow id="S3.T2.4.4.1.m1.1.1" xref="S3.T2.4.4.1.m1.1.1.cmml"><mi id="S3.T2.4.4.1.m1.1.1.2" mathsize="90%" mathvariant="normal" xref="S3.T2.4.4.1.m1.1.1.2.cmml">@</mi><mo id="S3.T2.4.4.1.m1.1.1.1" lspace="0em" rspace="0em" xref="S3.T2.4.4.1.m1.1.1.1.cmml">​</mo><mn id="S3.T2.4.4.1.m1.1.1.3" mathsize="90%" xref="S3.T2.4.4.1.m1.1.1.3.cmml">384</mn></mrow><annotation-xml encoding="MathML-Content" id="S3.T2.4.4.1.m1.1b"><apply id="S3.T2.4.4.1.m1.1.1.cmml" xref="S3.T2.4.4.1.m1.1.1"><times id="S3.T2.4.4.1.m1.1.1.1.cmml" xref="S3.T2.4.4.1.m1.1.1.1"></times><ci id="S3.T2.4.4.1.m1.1.1.2.cmml" xref="S3.T2.4.4.1.m1.1.1.2">@</ci><cn id="S3.T2.4.4.1.m1.1.1.3.cmml" type="integer" xref="S3.T2.4.4.1.m1.1.1.3">384</cn></apply></annotation-xml><annotation encoding="application/x-tex" id="S3.T2.4.4.1.m1.1c">@384</annotation></semantics></math>pix)</th><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.2" style="padding:1.6pt 8.3pt;">96.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.3" style="padding:1.6pt 8.3pt;">97.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.4" style="padding:1.6pt 8.3pt;">87.1</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.5" style="padding:1.6pt 8.3pt;">84.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.6" style="padding:1.6pt 8.3pt;">95.7</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.7" style="padding:1.6pt 8.3pt;">83.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.8" style="padding:1.6pt 8.3pt;">90.5</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.9" style="padding:1.6pt 8.3pt;">86.0</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.10" style="padding:1.6pt 8.3pt;">96.4</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.11" style="padding:1.6pt 8.3pt;">96.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T2.4.4.12" style="padding:1.6pt 8.3pt;">99.7</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 2:  Comparisons of image classification linear probing on 11 datasets with existing state-of-the-art models, including SimCLRv2 ( **Big self-supervised models are strong semi-supervised learners.** ) , ViT ( **An image is worth 16x16 words: Transformers for image recognition at scale.** ) , EfficientNet ( **Self-training with noisy student improves imagenet classification.** ) , and CLIP ( **Learning transferable visual models from natural language supervision.** ) . | ✅ Table 2:  在 11 个数据集上进行图像分类线性探测与现有的最先进模型的比较，包括 SimCLRv2 ( **Big self-supervised models are strong semi-supervised learners.** )、ViT ( **An image is worth 16x16 words: Transformers for image recognition at scale.** )、EfficientNet ( **Self-training with noisy student improves imagenet classification.** ) 和 CLIP ( **Learning transferable visual models from natural language supervision.** )。 |

### 3.1 Zero-shot Transfer in Classification

| 【第3.1节，第1段】原文 | 【第3.1节，第1段】翻译 |
| ---- | ---- |
| ✅ In computer vision, zero-shot learning usually refers to the study of predicting classes that are defined via descriptive text. | ✅ 在计算机视觉中，零样本学习通常指研究预测通过描述性文本定义的类别。 |
| ✅ As a vision foundation model, Florence can be directly used to predict if an image and a text snippet are semantically matched together in the task dataset. | ✅ Florence 作为视觉基础模型，可以直接用于预测任务数据集中图像与文本片段是否在语义上匹配。 |
| ✅ We follow the same method of CLIP ( **Learning transferable visual models from natural language supervision.** ) to perform zero-shot classification. | ✅ 我们遵循与 CLIP ( **Learning transferable visual models from natural language supervision.** ) 相同的方法进行零样本分类。 |
| ✅ For each dataset, we use the names of all the classes in the dataset as the set of potential text pairings and predict the most probable (image, text) pair according to Florence. | ✅ 对于每个数据集，我们使用数据集中所有类的名称作为潜在文本配对的集合，并根据 Florence 预测最可能的（图像，文本）对。 |
| ✅ We compute the feature embedding of the image for CoSwin and the feature embedding of the set of possible texts by the language encoder. | ✅ 我们计算 CoSwin 的图像特征嵌入以及语言编码器可能的文本集的特征嵌入。 |
| ✅ The cosine similarities among these embeddings are then calculated, and then we rank the similarity scores over all the classes to select the Top-1 or Top-5 classes as the predicted classes. | ✅ 然后计算这些嵌入之间的余弦相似度，然后我们对所有类别的相似度得分进行排序，以选择 Top-1 或 Top-5 类别作为预测类别。 |
| ✅ Here, we do not need to compute the normalized cosine similarity as done in ( **Learning transferable visual models from natural language supervision.** ) , since it won’t affect the ranking order of final results. | ✅ 这里我们不需要像( **Learning transferable visual models from natural language supervision.** )那样计算归一化余弦相似度，因为它不会影响最终结果的排名顺序。 |

| 【第3.1节，第2段】原文 | 【第3.1节，第2段】翻译 |
| ---- | ---- |
| ✅ We evaluate our Florence model on the ImageNet-1K dataset and 11 downstream datasets from the well-studied evaluation suit introduced by ( **Do better imagenet models transfer better?** ). | ✅ 我们在 ImageNet-1K 数据集和 ( **Do better imagenet models transfer better?** ) 引入的经过充分研究的评估套件中的 11 个下游数据集上评估了我们的 Florence 模型。 |
| ✅ Note that our benchmarks exclude the Birdsnap ( **Birdsnap: Large-scale fine-grained visual categorization of birds.** ) dataset from 12 original classification datasets introduced in ( **Do better imagenet models transfer better?** ) , because $20\%$ of the image URLs provided by the authors are invalid. | ✅ 请注意，我们的基准测试将 Birdsnap ( **Birdsnap: Large-scale fine-grained visual categorization of birds.** ) 数据集从 ( **Do better imagenet models transfer better?** ) 中引入的 12 个原始分类数据集中排除，因为作者提供的图像 URL 的 $20\%$ 无效。 |
| ✅ We follow the same prompt templates and engineering, and ensembling as previously proposed in ( **Learning transferable visual models from natural language supervision.** ) for evaluating zero-shot performance. | ✅ 我们遵循相同的提示模板和工程，并按照之前在 ( **Learning transferable visual models from natural language supervision.** ) 中提出的组合来评估零样本性能。 |
| ✅ For all zero-shot tasks in this paper, we follow the setup in CLIP ( **Learning transferable visual models from natural language supervision.** ) and ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) to remove near-duplicate test images from our training data. | ✅ 对于本文中的所有零样本任务，我们按照 CLIP ( **Learning transferable visual models from natural language supervision.** ) 和 ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) 中的设置从训练数据中删除近似重复的测试图像。 |
| ✅ Table 1 shows the results over these 12 datasets, in comparison with the best performance achieved by both CLIP ResNet and Vision Transformer models, and the concurrent work FILIP ( **Filip: Fine-grained interactive language-image pre-training.** ). | ✅ 表 1 展示了这 12 个数据集的结果，并与 CLIP ResNet 和 Vision Transformer 模型以及并行工作 FILIP ( **Filip: Fine-grained interactive language-image pre-training.** ) 所取得的最佳性能进行了比较。 |
| ✅ Florence outperforms on $9/12$ tasks compared with state-of-the-art methods. | ✅ 与最先进的方法相比，Florence 在 $9/12$ 任务上的表现更佳。 |
| ✅ We achieved a remarkable improvement in the zero-shot transfer on ImageNet-1K – the top-1 accuracy of $83.74\%$ ( $+5.6\%$ over SOTA result), and the top-5 accuracy of $97.18\%$ . | ✅ 我们在 ImageNet-1K 上的零样本迁移中取得了显著的提升——$83.74\%$ 的准确率达到了 top-1（$+5.6\%$ 超过 SOTA 结果），$97.18\%$ 的准确率达到了 top-5。 |

### 3.2 Linear Probe in Classification

| 【第3.2节，第1段】原文 | 【第3.2节，第1段】翻译 |
| ---- | ---- |
| ✅ Linear probe as another main metric for evaluating representation quality has been used in most recent studies, including self-supervised learning ( **1. A simple framework for contrastive learning of visual representations.** ｜ **2. Big self-supervised models are strong semi-supervised learners.** ) , self-training with noisy student ( **Self-training with noisy student improves imagenet classification.** ) and contrastive learning ( **Learning transferable visual models from natural language supervision.** ). | ✅ 线性探测作为评估表征质量的另一个主要指标，已被用于最近的大多数研究，包括自监督学习 ( **1. A simple framework for contrastive learning of visual representations.** ｜ **2. Big self-supervised models are strong semi-supervised learners.** )、带噪声学生的自训练 ( **Self-training with noisy student improves imagenet classification.** ) 和对比学习 ( **Learning transferable visual models from natural language supervision.** )。 |
| ✅ We follow the same setting and implementation of CLIP ( **Learning transferable visual models from natural language supervision.** ) for linear evaluation, where the image encoder (or vision backbone) is frozen, and only the appended linear layers can be fine-tuned on the downstream datasets. | ✅ 我们遵循与 CLIP ( **Learning transferable visual models from natural language supervision.** ) 相同的设置和实现进行线性评估，其中图像编码器（或视觉主干）被冻结，并且只有附加的线性层可以在下游数据集上进行微调。 |
| ✅ We use public available models (shown in Table 10 ( **Learning transferable visual models from natural language supervision.** ) ) to verify the correctness of our own implementation. | ✅ 我们使用公开可用的模型（如表 10 ( **Learning transferable visual models from natural language supervision.** ) 所示）来验证我们自己实现的正确性。 |
| ✅ The variance between our reproduced results and their reported results is $\pm 0.1$ for each task. | ✅ 对于每个任务，我们重现的结果和他们的报告结果之间的差异是 $\pm 0.1$。 |
| ✅ Our linear evaluation considers 11 classification benchmarks which are also used for our zero-shot transfer of classification. | ✅ 我们的线性评估考虑了 11 个分类基准，这些基准也用于我们的零样本分类转移。 |
| ✅ We compared our results with state-of-the-art methods with their best performance models, including SimCLRv2 ( **Big self-supervised models are strong semi-supervised learners.** ) , ViT ( **An image is worth 16x16 words: Transformers for image recognition at scale.** ) , Noisy Student ( **Self-training with noisy student improves imagenet classification.** ) and CLIP ( **Learning transferable visual models from natural language supervision.** ) on Table 2. | ✅ 我们将我们的结果与最先进的方法及其最佳性能模型进行了比较，包括表 2 中的 SimCLRv2 ( **Big self-supervised models are strong semi-supervised learners.** )、ViT ( **An image is worth 16x16 words: Transformers for image recognition at scale.** )、Noisy Student ( **Self-training with noisy student improves imagenet classification.** ) 和 CLIP ( **Learning transferable visual models from natural language supervision.** )。 |
| ✅ Our results are consistently better than existing state-of-the-art results, expect for two datasets: CIFAR10, CIFAR100. | ✅ 我们的结果始终比现有的最先进结果更好，除了两个数据集：CIFAR10、CIFAR100。 |
| ✅ On the two datasets, the input image resolution is quite low ( i.e. | ✅ 在这两个数据集上，输入图像分辨率相当低（即 |
| ✅ , $32\times 32$ ). | ✅ ，$32\times 32$）。 |
| ✅ Training with higher resolution definitely boosts the performance,such as Efficient-L2 ( **Self-training with noisy student improves imagenet classification.** ) which achieves the best accuracy compared with all other approaches trained on lower-resolution images. | ✅ 使用更高分辨率进行训练无疑会提高性能，例如 Efficient-L2 ( **Self-training with noisy student improves imagenet classification.** ) 与在低分辨率图像上训练的所有其他方法相比实现了最佳准确率。 |

### 3.3 ImageNet-1K Fine-tune Evaluation

| 【第3.3节，第1段】原文 | 【第3.3节，第1段】翻译 |
| ---- | ---- |
| ✅ Florence can be easily adapted to support continual fine-tuning on target classification tasks. | ✅ Florence 可以轻松适应以支持目标分类任务的持续微调。 |
| ✅ We do not change or add anything into our architecture, but continue the training on task-specific data using the same pre-training loss (shown in Equation 1 ). | ✅ 我们不会改变或在我们的架构中添加任何东西，而是使用相同的预训练损失继续对特定任务的数据进行训练（如公式 1 所示）。 |
| ✅ We feed the class name to the text encoder of Florence to get the text feature embedding. | ✅ 我们将类名输入到 Florence 的文本编码器以获取文本特征嵌入。 |
| ✅ We use the same prompt templates as in ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision.** ) to expand the descriptions of ImageNet ( **Imagenet: A large-scale hierarchical image database.** ) class names. | ✅ 我们使用与 ( **1. Learning transferable visual models from natural language supervision.** ｜ **2. Scaling up visual and vision-language representation learning with noisy text supervision.** ) 相同的提示模板来扩展 ImageNet ( **Imagenet: A large-scale hierarchical image database.** ) 类名的描述。 |

| 【第3.3节，第2段】原文 | 【第3.3节，第2段】翻译 |
| ---- | ---- |
| ✅ We evaluate the performance of continual fine-tuning on ImageNet ILSVRC-2012 benchmark ( **Imagenet: A large-scale hierarchical image database.** ). | ✅ 我们对 ImageNet ILSVRC-2012 基准 ( **Imagenet: A large-scale hierarchical image database.** ) 上的持续微调的性能进行了评估。 |
| ✅ Our image encoder CoSwin -H is fine-tuned at the resolution of $512\times 512$ with a batch size of $8,192$ for $10$ epochs. | ✅ 我们的图像编码器 CoSwin -H 在 $512\times 512$ 的分辨率下进行了微调，在 $10$ 时期的批次大小为 $8,192$。 |
| ✅ We use a cosine learning rate decay scheduler with $500$ warmup steps and a peak learning rate of $0.00002$. | ✅ 我们使用余弦学习率衰减调度程序，其中预热步骤为 $500$，峰值学习率为 $0.00002$。 |
| ✅ The comparisons with state-of-the-art results are shown in Table 3. | ✅ 与最先进结果的比较如表 3 所示。 |
| ✅ Our model outperforms BiT ( **Big transfer (bit): General visual representation learning.** ) with larger model size and ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) trained from more data in terms of Top-1 and Top-5 accuracy. | ✅ 在 Top-1 和 Top-5 准确度方面，我们的模型优于模型尺寸更大的 BiT ( **Big transfer (bit): General visual representation learning.** ) 和使用更多数据训练的 ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** )。 |
| ✅ Our result is slightly worse than SOTA ( **Coatnet: Marrying convolution and attention for all data sizes.** ) , but their model and data scale are both $3\times$ larger. | ✅ 我们的结果比SOTA ( **Coatnet: Marrying convolution and attention for all data sizes.** )稍差，但是他们的模型和数据规模都比$3\times$大。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S3.T3.1"><thead class="ltx_thead"><tr class="ltx_tr" id="S3.T3.1.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S3.T3.1.1.1.1" rowspan="2" style="padding:1.6pt 6.3pt;">Model</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt" id="S3.T3.1.1.1.2" rowspan="2" style="padding:1.6pt 6.3pt;">Params</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt" id="S3.T3.1.1.1.3" rowspan="2" style="padding:1.6pt 6.3pt;">Data</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" colspan="2" id="S3.T3.1.1.1.4" style="padding:1.6pt 6.3pt;">Accuracy</th></tr><tr class="ltx_tr" id="S3.T3.1.2.2"><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="S3.T3.1.2.2.1" style="padding:1.6pt 6.3pt;">Top-1</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="S3.T3.1.2.2.2" style="padding:1.6pt 6.3pt;">Top-5</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T3.1.3.1"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S3.T3.1.3.1.1" style="padding:1.6pt 6.3pt;">BiT-L-ResNet152x4</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T3.1.3.1.2" style="padding:1.6pt 6.3pt;">928M</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T3.1.3.1.3" style="padding:1.6pt 6.3pt;">300M</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T3.1.3.1.4" style="padding:1.6pt 6.3pt;">87.54</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T3.1.3.1.5" style="padding:1.6pt 6.3pt;">98.46</td></tr><tr class="ltx_tr" id="S3.T3.1.4.2"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T3.1.4.2.1" style="padding:1.6pt 6.3pt;">ALIGN-Efficient-L2</td><td class="ltx_td ltx_align_right" id="S3.T3.1.4.2.2" style="padding:1.6pt 6.3pt;">480M</td><td class="ltx_td ltx_align_right" id="S3.T3.1.4.2.3" style="padding:1.6pt 6.3pt;">1800M</td><td class="ltx_td ltx_align_center" id="S3.T3.1.4.2.4" style="padding:1.6pt 6.3pt;">88.64</td><td class="ltx_td ltx_align_center" id="S3.T3.1.4.2.5" style="padding:1.6pt 6.3pt;">98.67</td></tr><tr class="ltx_tr" id="S3.T3.1.5.3"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T3.1.5.3.1" style="padding:1.6pt 6.3pt;">ViT-G/14</td><td class="ltx_td ltx_align_right" id="S3.T3.1.5.3.2" style="padding:1.6pt 6.3pt;">1843M</td><td class="ltx_td ltx_align_right" id="S3.T3.1.5.3.3" style="padding:1.6pt 6.3pt;">3000M</td><td class="ltx_td ltx_align_center" id="S3.T3.1.5.3.4" style="padding:1.6pt 6.3pt;">90.45</td><td class="ltx_td ltx_align_center" id="S3.T3.1.5.3.5" style="padding:1.6pt 6.3pt;">-</td></tr><tr class="ltx_tr" id="S3.T3.1.6.4"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T3.1.6.4.1" style="padding:1.6pt 6.3pt;">CoAtNet-7</td><td class="ltx_td ltx_align_right" id="S3.T3.1.6.4.2" style="padding:1.6pt 6.3pt;">2440M</td><td class="ltx_td ltx_align_right" id="S3.T3.1.6.4.3" style="padding:1.6pt 6.3pt;">3000M</td><td class="ltx_td ltx_align_center" id="S3.T3.1.6.4.4" style="padding:1.6pt 6.3pt;">90.88</td><td class="ltx_td ltx_align_center" id="S3.T3.1.6.4.5" style="padding:1.6pt 6.3pt;">-</td></tr><tr class="ltx_tr" id="S3.T3.1.7.5"><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="S3.T3.1.7.5.1" style="padding:1.6pt 6.3pt;">Florence-CoSwin-H</td><td class="ltx_td ltx_align_right ltx_border_bb" id="S3.T3.1.7.5.2" style="padding:1.6pt 6.3pt;">637M</td><td class="ltx_td ltx_align_right ltx_border_bb" id="S3.T3.1.7.5.3" style="padding:1.6pt 6.3pt;">900M</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T3.1.7.5.4" style="padding:1.6pt 6.3pt;">90.05</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T3.1.7.5.5" style="padding:1.6pt 6.3pt;">99.02</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 3:  Classification fine tuning on ImageNet-1K. | ✅ 在 ImageNet-1K 上对 Table 3:  分类进行微调。 |
| ✅ Florence is compared with: BiT-L-ResNet152x4 ( **Big transfer (bit): General visual representation learning.** ) , ALIGN-Efficient-L2 ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) , ViT-G/14 ( **Scaling vision transformers.** ) , CoAtNet-7 ( **Coatnet: Marrying convolution and attention for all data sizes.** ) in terms of model scale, data scale and Top-1/Top-5 accuracy. | ✅ Florence在模型规模、数据规模、Top-1/Top-5准确率等方面与BiT-L-ResNet152x4 ( **Big transfer (bit): General visual representation learning.** )、ALIGN-Efficient-L2 ( **Scaling up visual and vision-language representation learning with noisy text supervision.** )、ViT-G/14 ( **Scaling vision transformers.** )、CoAtNet-7 ( **Coatnet: Marrying convolution and attention for all data sizes.** )进行了对比。 |

### 3.4 Few-shot Cross-domain Classification

| 【第3.4节，第1段】原文 | 【第3.4节，第1段】翻译 |
| ---- | ---- |
| ✅ The Cross-Domain Few-Shot learning benchmark ( **A new benchmark for evaluation of cross-domain few-shot learning.** ) is used to measure an algorithm’s capability to adapt to downstream few-shot target tasks, containing domains with varying levels of dissimilarity to typical consumer photographs. | ✅ 跨领域小样本学习基准 ( **A new benchmark for evaluation of cross-domain few-shot learning.** ) 用于衡量算法适应下游小样本目标任务的能力，其中包含与典型消费者照片有不同程度差异的领域。 |
| ✅ The datasets in the benchmark include: CropDisease ( **Using deep learning for image-based plant disease detection.** ) (plant leaf images, 38 disease states over 14 plant species), EuroSAT ( **Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification.** ) (RGB satellite images, 10 categories), ISIC 2018 ( **1. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC).** ｜ **2. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.** ) (dermoscopic images of skin lesions, 7 disease states), and ChestX ( **Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases.** ) (Chest X-rays, 16 conditions). | ✅ 基准中的数据集包括：CropDisease ( **Using deep learning for image-based plant disease detection.** )（植物叶片图像，14 种植物的 38 种疾病状态）、EuroSAT ( **Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification.** )（RGB 卫星图像，10 个类别）、ISIC 2018 ( **1. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC).** ｜ **2. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.** )（皮肤病变的皮肤镜图像，7 种疾病状态）和 ChestX ( **Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases.** )（胸部 X 光片，16 种情况）。 |
| ✅ Exemplar image for each dataset is shown on the top of Table 4. | ✅ 每个数据集的示例图像显示在表 4 的顶部。 |
| ✅ The evaluation protocol involves 5-way classification across 5-shot, 20-shot, and 50-shot. | ✅ 评估协议涉及 5 次、20 次和 50 次的 5 路分类。 |
| ✅ The classes and shots are randomly sampled for each episode, for 600 episodes per way and shot. | ✅ 每集随机抽取类别和镜头，每种方式和镜头 600 集。 |
| ✅ Average accuracy over all episodes is reported. | ✅ 报告的是所有事件的平均准确率。 |

| 【第3.4节，第2段】原文 | 【第3.4节，第2段】翻译 |
| ---- | ---- |
| ✅ To predict the class, we append a single linear layer as an adapter head to our image encoder CoSwin. | ✅ 为了预测类别，我们将单个线性层作为适配器头附加到图像编码器 CoSwin。 |
| ✅ Training occurs over 100 epochs per episode. | ✅ 每集训练超过 100 个时期。 |
| ✅ We use SGD with momentum, with learning rate and momentum values of $0.9/0.0002$ , respectively, for CoSwin , and $0.99/0.01$ , respectively, for the adapter head. | ✅ 我们使用带动量的 SGD，对于 CoSwin ，学习率和动量值分别为 $0.9/0.0002$ ，对于适配器头，学习率和动量值分别为 $0.99/0.01$ 。 |
| ✅ Horizontal data flip augmentation is used for training and test, and dropout of $0.5$ is used between the image encoder and the classifier head. | ✅ 使用水平数据翻转增强进行训练和测试，并在图像编码器和分类器头之间使用$0.5$的dropout。 |

| 【第3.4节，第3段】原文 | 【第3.4节，第3段】翻译 |
| ---- | ---- |
| ✅ Table 4 shows the results of adapting our model to the CD-FSL benchmark, in comparison to the winner of the challenge benchmark ( **Feature transformation ensemble model with batch spectral regularization for cross-domain few-shot classification.** ) , which employs ensembes and transductive learning. | ✅ 表 4 展示了我们的模型适应 CD-FSL 基准的结果，并与采用集成和传导学习的挑战基准 ( **Feature transformation ensemble model with batch spectral regularization for cross-domain few-shot classification.** ) 的获胜者进行了比较。 |
| ✅ By comparison, we employ a single model and no transduction on the test data is performed, yet we achieve higher results without any “bells and whistles”. | ✅ 相比之下，我们采用单一模型，并且不对测试数据进行任何转换，但我们却无需任何“花哨”就能取得更高的结果。 |

<table class="ltx_tabular ltx_centering ltx_figure_panel ltx_guessed_headers ltx_align_middle" id="S3.T4.6"><thead class="ltx_thead"><tr class="ltx_tr" id="S3.T4.6.1.1"><th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S3.T4.6.1.1.1" style="padding:1.6pt 3.8pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S3.T4.6.1.1.2" style="padding:1.6pt 3.8pt;">Model</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T4.6.1.1.3" style="padding:1.6pt 3.8pt;">ISIC</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T4.6.1.1.4" style="padding:1.6pt 3.8pt;">EuroSAT</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T4.6.1.1.5" style="padding:1.6pt 3.8pt;">CropD</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S3.T4.6.1.1.6" style="padding:1.6pt 3.8pt;">ChestX</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T4.6.1.1.7" style="padding:1.6pt 3.8pt;">mean</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T4.6.2.1"><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T4.6.2.1.1" rowspan="2" style="padding:1.6pt 3.8pt;">5-shot</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T4.6.2.1.2" style="padding:1.6pt 3.8pt;">CW</th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.2.1.3" style="padding:1.6pt 3.8pt;">57.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.2.1.4" style="padding:1.6pt 3.8pt;">88.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.2.1.5" style="padding:1.6pt 3.8pt;">96.6</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T4.6.2.1.6" style="padding:1.6pt 3.8pt;">29.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.2.1.7" style="padding:1.6pt 3.8pt;">68.0</td></tr><tr class="ltx_tr" id="S3.T4.6.3.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T4.6.3.2.1" style="padding:1.6pt 3.8pt;">Florence</th><td class="ltx_td ltx_align_center" id="S3.T4.6.3.2.2" style="padding:1.6pt 3.8pt;">57.1</td><td class="ltx_td ltx_align_center" id="S3.T4.6.3.2.3" style="padding:1.6pt 3.8pt;">90.0</td><td class="ltx_td ltx_align_center" id="S3.T4.6.3.2.4" style="padding:1.6pt 3.8pt;">97.7</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T4.6.3.2.5" style="padding:1.6pt 3.8pt;">29.3</td><td class="ltx_td ltx_align_center" id="S3.T4.6.3.2.6" style="padding:1.6pt 3.8pt;">68.5</td></tr><tr class="ltx_tr" id="S3.T4.6.4.3"><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T4.6.4.3.1" rowspan="2" style="padding:1.6pt 3.8pt;">20-shot</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T4.6.4.3.2" style="padding:1.6pt 3.8pt;">CW</th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.4.3.3" style="padding:1.6pt 3.8pt;">68.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.4.3.4" style="padding:1.6pt 3.8pt;">94.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.4.3.5" style="padding:1.6pt 3.8pt;">99.2</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T4.6.4.3.6" style="padding:1.6pt 3.8pt;">38.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.4.3.7" style="padding:1.6pt 3.8pt;">75.1</td></tr><tr class="ltx_tr" id="S3.T4.6.5.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T4.6.5.4.1" style="padding:1.6pt 3.8pt;">Florence</th><td class="ltx_td ltx_align_center" id="S3.T4.6.5.4.2" style="padding:1.6pt 3.8pt;">72.9</td><td class="ltx_td ltx_align_center" id="S3.T4.6.5.4.3" style="padding:1.6pt 3.8pt;">95.8</td><td class="ltx_td ltx_align_center" id="S3.T4.6.5.4.4" style="padding:1.6pt 3.8pt;">99.3</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T4.6.5.4.5" style="padding:1.6pt 3.8pt;">37.5</td><td class="ltx_td ltx_align_center" id="S3.T4.6.5.4.6" style="padding:1.6pt 3.8pt;">76.4</td></tr><tr class="ltx_tr" id="S3.T4.6.6.5"><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t" id="S3.T4.6.6.5.1" rowspan="2" style="padding:1.6pt 3.8pt;">50-shot</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T4.6.6.5.2" style="padding:1.6pt 3.8pt;">CW</th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.6.5.3" style="padding:1.6pt 3.8pt;">74.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.6.5.4" style="padding:1.6pt 3.8pt;">96.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.6.5.5" style="padding:1.6pt 3.8pt;">99.7</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T4.6.6.5.6" style="padding:1.6pt 3.8pt;">44.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T4.6.6.5.7" style="padding:1.6pt 3.8pt;">78.8</td></tr><tr class="ltx_tr" id="S3.T4.6.7.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S3.T4.6.7.6.1" style="padding:1.6pt 3.8pt;">Florence</th><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T4.6.7.6.2" style="padding:1.6pt 3.8pt;">78.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T4.6.7.6.3" style="padding:1.6pt 3.8pt;">97.1</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T4.6.7.6.4" style="padding:1.6pt 3.8pt;">99.6</td><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r" id="S3.T4.6.7.6.5" style="padding:1.6pt 3.8pt;">42.8</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T4.6.7.6.6" style="padding:1.6pt 3.8pt;">79.5</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 4:  Comparison with CW ( **Feature transformation ensemble model with batch spectral regularization for cross-domain few-shot classification.** ) (CD-FSL Challenge 2020 Winner) on CD-FSL benchmark. | ✅ Table 4:  与 CW ( **Feature transformation ensemble model with batch spectral regularization for cross-domain few-shot classification.** )（CD-FSL 挑战赛 2020 获胜者）在 CD-FSL 基准上的比较。 |
| ✅ The average result comparison is $74.8$ (Florence) vs. | ✅ 平均结果比较是$74.8$（佛罗伦萨）vs。 |
| ✅ $73.9$ (CW). | ✅ $73.9$（连续波）。 |

### 3.5 Image-Text Retrieval

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S3.T5.6"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T5.2.2"><th class="ltx_td ltx_th ltx_th_row ltx_border_tt" id="S3.T5.2.2.3" style="padding:1.6pt 9.5pt;"></th><th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="S3.T5.2.2.4" style="padding:1.6pt 9.5pt;"></th><td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt" colspan="4" id="S3.T5.1.1.1" style="padding:1.6pt 9.5pt;">Flickr30K (<math alttext="1K" class="ltx_Math" display="inline" id="S3.T5.1.1.1.m1.1"><semantics id="S3.T5.1.1.1.m1.1a"><mrow id="S3.T5.1.1.1.m1.1.1" xref="S3.T5.1.1.1.m1.1.1.cmml"><mn id="S3.T5.1.1.1.m1.1.1.2" mathsize="90%" xref="S3.T5.1.1.1.m1.1.1.2.cmml">1</mn><mo id="S3.T5.1.1.1.m1.1.1.1" lspace="0em" rspace="0em" xref="S3.T5.1.1.1.m1.1.1.1.cmml">​</mo><mi id="S3.T5.1.1.1.m1.1.1.3" mathsize="90%" xref="S3.T5.1.1.1.m1.1.1.3.cmml">K</mi></mrow><annotation-xml encoding="MathML-Content" id="S3.T5.1.1.1.m1.1b"><apply id="S3.T5.1.1.1.m1.1.1.cmml" xref="S3.T5.1.1.1.m1.1.1"><times id="S3.T5.1.1.1.m1.1.1.1.cmml" xref="S3.T5.1.1.1.m1.1.1.1"></times><cn id="S3.T5.1.1.1.m1.1.1.2.cmml" type="integer" xref="S3.T5.1.1.1.m1.1.1.2">1</cn><ci id="S3.T5.1.1.1.m1.1.1.3.cmml" xref="S3.T5.1.1.1.m1.1.1.3">𝐾</ci></apply></annotation-xml><annotation encoding="application/x-tex" id="S3.T5.1.1.1.m1.1c">1K</annotation></semantics></math> test set)</td><td class="ltx_td ltx_align_center ltx_border_tt" colspan="4" id="S3.T5.2.2.2" style="padding:1.6pt 9.5pt;">MSCOCO (<math alttext="5K" class="ltx_Math" display="inline" id="S3.T5.2.2.2.m1.1"><semantics id="S3.T5.2.2.2.m1.1a"><mrow id="S3.T5.2.2.2.m1.1.1" xref="S3.T5.2.2.2.m1.1.1.cmml"><mn id="S3.T5.2.2.2.m1.1.1.2" mathsize="90%" xref="S3.T5.2.2.2.m1.1.1.2.cmml">5</mn><mo id="S3.T5.2.2.2.m1.1.1.1" lspace="0em" rspace="0em" xref="S3.T5.2.2.2.m1.1.1.1.cmml">​</mo><mi id="S3.T5.2.2.2.m1.1.1.3" mathsize="90%" xref="S3.T5.2.2.2.m1.1.1.3.cmml">K</mi></mrow><annotation-xml encoding="MathML-Content" id="S3.T5.2.2.2.m1.1b"><apply id="S3.T5.2.2.2.m1.1.1.cmml" xref="S3.T5.2.2.2.m1.1.1"><times id="S3.T5.2.2.2.m1.1.1.1.cmml" xref="S3.T5.2.2.2.m1.1.1.1"></times><cn id="S3.T5.2.2.2.m1.1.1.2.cmml" type="integer" xref="S3.T5.2.2.2.m1.1.1.2">5</cn><ci id="S3.T5.2.2.2.m1.1.1.3.cmml" xref="S3.T5.2.2.2.m1.1.1.3">𝐾</ci></apply></annotation-xml><annotation encoding="application/x-tex" id="S3.T5.2.2.2.m1.1c">5K</annotation></semantics></math> test set)</td></tr><tr class="ltx_tr" id="S3.T5.6.6"><th class="ltx_td ltx_th ltx_th_row" id="S3.T5.6.6.5" style="padding:1.6pt 9.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.6.6" style="padding:1.6pt 9.5pt;">Method</th><td class="ltx_td ltx_align_center" colspan="2" id="S3.T5.3.3.1" style="padding:1.6pt 9.5pt;">Image <math alttext="\to" class="ltx_Math" display="inline" id="S3.T5.3.3.1.m1.1"><semantics id="S3.T5.3.3.1.m1.1a"><mo id="S3.T5.3.3.1.m1.1.1" mathsize="90%" stretchy="false" xref="S3.T5.3.3.1.m1.1.1.cmml">→</mo><annotation-xml encoding="MathML-Content" id="S3.T5.3.3.1.m1.1b"><ci id="S3.T5.3.3.1.m1.1.1.cmml" xref="S3.T5.3.3.1.m1.1.1">→</ci></annotation-xml><annotation encoding="application/x-tex" id="S3.T5.3.3.1.m1.1c">\to</annotation></semantics></math> Text</td><td class="ltx_td ltx_align_center ltx_border_r" colspan="2" id="S3.T5.4.4.2" style="padding:1.6pt 9.5pt;">Text <math alttext="\to" class="ltx_Math" display="inline" id="S3.T5.4.4.2.m1.1"><semantics id="S3.T5.4.4.2.m1.1a"><mo id="S3.T5.4.4.2.m1.1.1" mathsize="90%" stretchy="false" xref="S3.T5.4.4.2.m1.1.1.cmml">→</mo><annotation-xml encoding="MathML-Content" id="S3.T5.4.4.2.m1.1b"><ci id="S3.T5.4.4.2.m1.1.1.cmml" xref="S3.T5.4.4.2.m1.1.1">→</ci></annotation-xml><annotation encoding="application/x-tex" id="S3.T5.4.4.2.m1.1c">\to</annotation></semantics></math> Image</td><td class="ltx_td ltx_align_center" colspan="2" id="S3.T5.5.5.3" style="padding:1.6pt 9.5pt;">Image <math alttext="\to" class="ltx_Math" display="inline" id="S3.T5.5.5.3.m1.1"><semantics id="S3.T5.5.5.3.m1.1a"><mo id="S3.T5.5.5.3.m1.1.1" mathsize="90%" stretchy="false" xref="S3.T5.5.5.3.m1.1.1.cmml">→</mo><annotation-xml encoding="MathML-Content" id="S3.T5.5.5.3.m1.1b"><ci id="S3.T5.5.5.3.m1.1.1.cmml" xref="S3.T5.5.5.3.m1.1.1">→</ci></annotation-xml><annotation encoding="application/x-tex" id="S3.T5.5.5.3.m1.1c">\to</annotation></semantics></math> Text</td><td class="ltx_td ltx_align_center" colspan="2" id="S3.T5.6.6.4" style="padding:1.6pt 9.5pt;">Text <math alttext="\to" class="ltx_Math" display="inline" id="S3.T5.6.6.4.m1.1"><semantics id="S3.T5.6.6.4.m1.1a"><mo id="S3.T5.6.6.4.m1.1.1" mathsize="90%" stretchy="false" xref="S3.T5.6.6.4.m1.1.1.cmml">→</mo><annotation-xml encoding="MathML-Content" id="S3.T5.6.6.4.m1.1b"><ci id="S3.T5.6.6.4.m1.1.1.cmml" xref="S3.T5.6.6.4.m1.1.1">→</ci></annotation-xml><annotation encoding="application/x-tex" id="S3.T5.6.6.4.m1.1c">\to</annotation></semantics></math> Image</td></tr><tr class="ltx_tr" id="S3.T5.6.7.1"><th class="ltx_td ltx_th ltx_th_row" id="S3.T5.6.7.1.1" style="padding:1.6pt 9.5pt;"></th><th class="ltx_td ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.7.1.2" style="padding:1.6pt 9.5pt;"></th><td class="ltx_td ltx_align_center" id="S3.T5.6.7.1.3" style="padding:1.6pt 9.5pt;">R@1</td><td class="ltx_td ltx_align_center" id="S3.T5.6.7.1.4" style="padding:1.6pt 9.5pt;">R@5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.7.1.5" style="padding:1.6pt 9.5pt;">R@1</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.7.1.6" style="padding:1.6pt 9.5pt;">R@5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.7.1.7" style="padding:1.6pt 9.5pt;">R@1</td><td class="ltx_td ltx_align_center" id="S3.T5.6.7.1.8" style="padding:1.6pt 9.5pt;">R@5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.7.1.9" style="padding:1.6pt 9.5pt;">R@1</td><td class="ltx_td ltx_align_center" id="S3.T5.6.7.1.10" style="padding:1.6pt 9.5pt;">R@5</td></tr><tr class="ltx_tr" id="S3.T5.6.8.2"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" id="S3.T5.6.8.2.1" rowspan="5" style="padding:1.6pt 9.5pt;">Zero-shot</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T5.6.8.2.2" style="padding:1.6pt 9.5pt;">ImageBERT <html><body><p>( <strong>Imagebert: Cross-modal pre-training with large-scale weak-supervisedimage-text data.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.8.2.3" style="padding:1.6pt 9.5pt;">70.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.8.2.4" style="padding:1.6pt 9.5pt;">90.2</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.8.2.5" style="padding:1.6pt 9.5pt;">54.3</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T5.6.8.2.6" style="padding:1.6pt 9.5pt;">79.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.8.2.7" style="padding:1.6pt 9.5pt;">44.0</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.8.2.8" style="padding:1.6pt 9.5pt;">71.2</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.8.2.9" style="padding:1.6pt 9.5pt;">32.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.8.2.10" style="padding:1.6pt 9.5pt;">59.0</td></tr><tr class="ltx_tr" id="S3.T5.6.9.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.9.3.1" style="padding:1.6pt 9.5pt;">UNITER <html><body><p>( <strong>Uniter: Universal image-text representation learning.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.9.3.2" style="padding:1.6pt 9.5pt;">83.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.9.3.3" style="padding:1.6pt 9.5pt;">95.7</td><td class="ltx_td ltx_align_center" id="S3.T5.6.9.3.4" style="padding:1.6pt 9.5pt;">68.7</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.9.3.5" style="padding:1.6pt 9.5pt;">89.2</td><td class="ltx_td ltx_align_center" id="S3.T5.6.9.3.6" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.9.3.7" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.9.3.8" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.9.3.9" style="padding:1.6pt 9.5pt;">-</td></tr><tr class="ltx_tr" id="S3.T5.6.10.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.10.4.1" style="padding:1.6pt 9.5pt;">CLIP <html><body><p>( <strong>Learning transferable visual models from natural languagesupervision.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.10.4.2" style="padding:1.6pt 9.5pt;">88.0</td><td class="ltx_td ltx_align_center" id="S3.T5.6.10.4.3" style="padding:1.6pt 9.5pt;">98.7</td><td class="ltx_td ltx_align_center" id="S3.T5.6.10.4.4" style="padding:1.6pt 9.5pt;">68.7</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.10.4.5" style="padding:1.6pt 9.5pt;">90.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.10.4.6" style="padding:1.6pt 9.5pt;">58.4</td><td class="ltx_td ltx_align_center" id="S3.T5.6.10.4.7" style="padding:1.6pt 9.5pt;">81.5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.10.4.8" style="padding:1.6pt 9.5pt;">37.8</td><td class="ltx_td ltx_align_center" id="S3.T5.6.10.4.9" style="padding:1.6pt 9.5pt;">62.4</td></tr><tr class="ltx_tr" id="S3.T5.6.11.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.11.5.1" style="padding:1.6pt 9.5pt;">ALIGN <html><body><p>( <strong>Scaling up visual and vision-language representation learning withnoisy text supervision.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.11.5.2" style="padding:1.6pt 9.5pt;">88.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.11.5.3" style="padding:1.6pt 9.5pt;">98.7</td><td class="ltx_td ltx_align_center" id="S3.T5.6.11.5.4" style="padding:1.6pt 9.5pt;">75.7</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.11.5.5" style="padding:1.6pt 9.5pt;">93.8</td><td class="ltx_td ltx_align_center" id="S3.T5.6.11.5.6" style="padding:1.6pt 9.5pt;">58.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.11.5.7" style="padding:1.6pt 9.5pt;">83.0</td><td class="ltx_td ltx_align_center" id="S3.T5.6.11.5.8" style="padding:1.6pt 9.5pt;">45.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.11.5.9" style="padding:1.6pt 9.5pt;">69.8</td></tr><tr class="ltx_tr" id="S3.T5.6.12.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.12.6.1" style="padding:1.6pt 9.5pt;">FLIP <html><body><p>( <strong>Filip: Fine-grained interactive language-image pre-training.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.12.6.2" style="padding:1.6pt 9.5pt;">89.8</td><td class="ltx_td ltx_align_center" id="S3.T5.6.12.6.3" style="padding:1.6pt 9.5pt;">99.2</td><td class="ltx_td ltx_align_center" id="S3.T5.6.12.6.4" style="padding:1.6pt 9.5pt;">75.0</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.12.6.5" style="padding:1.6pt 9.5pt;">93.4</td><td class="ltx_td ltx_align_center" id="S3.T5.6.12.6.6" style="padding:1.6pt 9.5pt;">61.3</td><td class="ltx_td ltx_align_center" id="S3.T5.6.12.6.7" style="padding:1.6pt 9.5pt;">84.3</td><td class="ltx_td ltx_align_center" id="S3.T5.6.12.6.8" style="padding:1.6pt 9.5pt;">45.9</td><td class="ltx_td ltx_align_center" id="S3.T5.6.12.6.9" style="padding:1.6pt 9.5pt;">70.6</td></tr><tr class="ltx_tr" id="S3.T5.6.13.7"><th class="ltx_td ltx_th ltx_th_row" id="S3.T5.6.13.7.1" style="padding:1.6pt 9.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.13.7.2" style="padding:1.6pt 9.5pt;">Florence</th><td class="ltx_td ltx_align_center" id="S3.T5.6.13.7.3" style="padding:1.6pt 9.5pt;">90.9</td><td class="ltx_td ltx_align_center" id="S3.T5.6.13.7.4" style="padding:1.6pt 9.5pt;">99.1</td><td class="ltx_td ltx_align_center" id="S3.T5.6.13.7.5" style="padding:1.6pt 9.5pt;">76.7</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.13.7.6" style="padding:1.6pt 9.5pt;">93.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.13.7.7" style="padding:1.6pt 9.5pt;">64.7</td><td class="ltx_td ltx_align_center" id="S3.T5.6.13.7.8" style="padding:1.6pt 9.5pt;">85.9</td><td class="ltx_td ltx_align_center" id="S3.T5.6.13.7.9" style="padding:1.6pt 9.5pt;">47.2</td><td class="ltx_td ltx_align_center" id="S3.T5.6.13.7.10" style="padding:1.6pt 9.5pt;">71.4</td></tr><tr class="ltx_tr" id="S3.T5.6.14.8"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" id="S3.T5.6.14.8.1" rowspan="7" style="padding:1.6pt 9.5pt;">Fine-tuned</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T5.6.14.8.2" style="padding:1.6pt 9.5pt;">GPO <html><body><p>( <strong>Learning the best pooling strategy for visual semantic embedding.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.14.8.3" style="padding:1.6pt 9.5pt;">88.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.14.8.4" style="padding:1.6pt 9.5pt;">98.9</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.14.8.5" style="padding:1.6pt 9.5pt;">76.1</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T5.6.14.8.6" style="padding:1.6pt 9.5pt;">94.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.14.8.7" style="padding:1.6pt 9.5pt;">68.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.14.8.8" style="padding:1.6pt 9.5pt;">90.2</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.14.8.9" style="padding:1.6pt 9.5pt;">52.7</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T5.6.14.8.10" style="padding:1.6pt 9.5pt;">80.2</td></tr><tr class="ltx_tr" id="S3.T5.6.15.9"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.15.9.1" style="padding:1.6pt 9.5pt;">UNITER <html><body><p>( <strong>Uniter: Universal image-text representation learning.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.15.9.2" style="padding:1.6pt 9.5pt;">87.3</td><td class="ltx_td ltx_align_center" id="S3.T5.6.15.9.3" style="padding:1.6pt 9.5pt;">98.0</td><td class="ltx_td ltx_align_center" id="S3.T5.6.15.9.4" style="padding:1.6pt 9.5pt;">75.6</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.15.9.5" style="padding:1.6pt 9.5pt;">94.1</td><td class="ltx_td ltx_align_center" id="S3.T5.6.15.9.6" style="padding:1.6pt 9.5pt;">65.7</td><td class="ltx_td ltx_align_center" id="S3.T5.6.15.9.7" style="padding:1.6pt 9.5pt;">88.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.15.9.8" style="padding:1.6pt 9.5pt;">52.9</td><td class="ltx_td ltx_align_center" id="S3.T5.6.15.9.9" style="padding:1.6pt 9.5pt;">79.9</td></tr><tr class="ltx_tr" id="S3.T5.6.16.10"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.16.10.1" style="padding:1.6pt 9.5pt;">ERNIE-ViL <html><body><p>( <strong>Ernie-vil: Knowledge enhanced vision-language representations throughscene graph.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.16.10.2" style="padding:1.6pt 9.5pt;">88.1</td><td class="ltx_td ltx_align_center" id="S3.T5.6.16.10.3" style="padding:1.6pt 9.5pt;">98.0</td><td class="ltx_td ltx_align_center" id="S3.T5.6.16.10.4" style="padding:1.6pt 9.5pt;">76.7</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.16.10.5" style="padding:1.6pt 9.5pt;">93.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.16.10.6" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.16.10.7" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.16.10.8" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.16.10.9" style="padding:1.6pt 9.5pt;">-</td></tr><tr class="ltx_tr" id="S3.T5.6.17.11"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.17.11.1" style="padding:1.6pt 9.5pt;">VILLA <html><body><p>( <strong>Large-scale adversarial training for vision-and-languagerepresentation learning.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.17.11.2" style="padding:1.6pt 9.5pt;">87.9</td><td class="ltx_td ltx_align_center" id="S3.T5.6.17.11.3" style="padding:1.6pt 9.5pt;">97.5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.17.11.4" style="padding:1.6pt 9.5pt;">76.3</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.17.11.5" style="padding:1.6pt 9.5pt;">94.2</td><td class="ltx_td ltx_align_center" id="S3.T5.6.17.11.6" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.17.11.7" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.17.11.8" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.17.11.9" style="padding:1.6pt 9.5pt;">-</td></tr><tr class="ltx_tr" id="S3.T5.6.18.12"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.18.12.1" style="padding:1.6pt 9.5pt;">Oscar <html><body><p>( <strong>Oscar: Object-semantics aligned pre-training for vision-languagetasks.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.18.12.2" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.18.12.3" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.18.12.4" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.18.12.5" style="padding:1.6pt 9.5pt;">-</td><td class="ltx_td ltx_align_center" id="S3.T5.6.18.12.6" style="padding:1.6pt 9.5pt;">73.5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.18.12.7" style="padding:1.6pt 9.5pt;">92.2</td><td class="ltx_td ltx_align_center" id="S3.T5.6.18.12.8" style="padding:1.6pt 9.5pt;">57.5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.18.12.9" style="padding:1.6pt 9.5pt;">82.8</td></tr><tr class="ltx_tr" id="S3.T5.6.19.13"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.19.13.1" style="padding:1.6pt 9.5pt;">ALIGN <html><body><p>( <strong>Scaling up visual and vision-language representation learning withnoisy text supervision.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.19.13.2" style="padding:1.6pt 9.5pt;">95.3</td><td class="ltx_td ltx_align_center" id="S3.T5.6.19.13.3" style="padding:1.6pt 9.5pt;">99.8</td><td class="ltx_td ltx_align_center" id="S3.T5.6.19.13.4" style="padding:1.6pt 9.5pt;">84.9</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.19.13.5" style="padding:1.6pt 9.5pt;">97.4</td><td class="ltx_td ltx_align_center" id="S3.T5.6.19.13.6" style="padding:1.6pt 9.5pt;">77.0</td><td class="ltx_td ltx_align_center" id="S3.T5.6.19.13.7" style="padding:1.6pt 9.5pt;">93.5</td><td class="ltx_td ltx_align_center" id="S3.T5.6.19.13.8" style="padding:1.6pt 9.5pt;">59.9</td><td class="ltx_td ltx_align_center" id="S3.T5.6.19.13.9" style="padding:1.6pt 9.5pt;">83.3</td></tr><tr class="ltx_tr" id="S3.T5.6.20.14"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T5.6.20.14.1" style="padding:1.6pt 9.5pt;">FLIP <html><body><p>( <strong>Filip: Fine-grained interactive language-image pre-training.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T5.6.20.14.2" style="padding:1.6pt 9.5pt;">96.6</td><td class="ltx_td ltx_align_center" id="S3.T5.6.20.14.3" style="padding:1.6pt 9.5pt;">100.0</td><td class="ltx_td ltx_align_center" id="S3.T5.6.20.14.4" style="padding:1.6pt 9.5pt;">87.1</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T5.6.20.14.5" style="padding:1.6pt 9.5pt;">97.7</td><td class="ltx_td ltx_align_center" id="S3.T5.6.20.14.6" style="padding:1.6pt 9.5pt;">78.9</td><td class="ltx_td ltx_align_center" id="S3.T5.6.20.14.7" style="padding:1.6pt 9.5pt;">94.4</td><td class="ltx_td ltx_align_center" id="S3.T5.6.20.14.8" style="padding:1.6pt 9.5pt;">61.2</td><td class="ltx_td ltx_align_center" id="S3.T5.6.20.14.9" style="padding:1.6pt 9.5pt;">84.3</td></tr><tr class="ltx_tr" id="S3.T5.6.21.15"><th class="ltx_td ltx_th ltx_th_row ltx_border_bb" id="S3.T5.6.21.15.1" style="padding:1.6pt 9.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S3.T5.6.21.15.2" style="padding:1.6pt 9.5pt;">Florence</th><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T5.6.21.15.3" style="padding:1.6pt 9.5pt;">97.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T5.6.21.15.4" style="padding:1.6pt 9.5pt;">99.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T5.6.21.15.5" style="padding:1.6pt 9.5pt;">87.9</td><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r" id="S3.T5.6.21.15.6" style="padding:1.6pt 9.5pt;">98.1</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T5.6.21.15.7" style="padding:1.6pt 9.5pt;">81.8</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T5.6.21.15.8" style="padding:1.6pt 9.5pt;">95.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T5.6.21.15.9" style="padding:1.6pt 9.5pt;">63.2</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T5.6.21.15.10" style="padding:1.6pt 9.5pt;">85.7</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 5:  Image-text retrieval comparisons on Flickr30K and MSCOCO datasets (zero-shot and fine-tuned). | ✅ Table 5:  在 Flickr30K 和 MSCOCO 数据集 (零样本和微调) 上进行图像文本检索比较。 |

| 【第3.5节，第1段】原文 | 【第3.5节，第1段】翻译 |
| ---- | ---- |
| ✅ Table 5 presents the zero-shot transfer and fine-tuning performance of Florence for both text and image retrieval on the Flickr30k ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** ) and MSCOCO ( **Microsoft COCO:: Common objects in context, 2015.** ) datasets. | ✅ 表 5 展示了 Florence 在 Flickr30k ( **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.** ) 和 MSCOCO ( **Microsoft COCO:: Common objects in context, 2015.** ) 数据集上进行文本和图像检索的零样本传输和微调性能。 |

| 【第3.5节，第2段】原文 | 【第3.5节，第2段】翻译 |
| ---- | ---- |
| ✅ For zero-shot retrieval, we feed the input text (or image) to the language (or image) encoder of Florence to get the feature embeddings, and also compute the feature embeddings of the set of possible images (or texts) by the image (or language) encoder. | ✅ 对于零样本检索，我们将输入文本（或图像）提供给 Florence 的语言（或图像）编码器以获取特征嵌入，并通过图像（或语言）编码器计算可能的图像（或文本）集合的特征嵌入。 |
| ✅ Then we compute cosine similarity of these embeddings and rank the similarity scores over the testing set to select the Top-1 or Top-5 results. | ✅ 然后，我们计算这些嵌入的余弦相似度，并对测试集的相似度得分进行排序，以选择 Top-1 或 Top-5 结果。 |
| ✅ Zero-shot Florence matches or outperforms all prior zero-shot results on these two datasets. | ✅ 零样本佛罗伦萨在这两个数据集上匹配或超越了所有先前的零样本结果。 |

| 【第3.5节，第3段】原文 | 【第3.5节，第3段】翻译 |
| ---- | ---- |
| ✅ For fine-tuning retrieval, we continuously train our language and text encoders on the target image-text pair data, as well as classification fine-tuning (shown in Section 3.3 ). | ✅ 对于微调检索，我们在目标图像-文本对数据上不断训练我们的语言和文本编码器，以及分类微调（如第 3.3 节所示）。 |
| ✅ We fine-tune our model with a batch size of $3,072$ for $12$ epochs. | ✅ 我们使用 $3,072$ 的批量大小对 $12$ 个时期的模型进行微调。 |
| ✅ We use the cosine learning rate decay scheduler with $200$ warmup steps and a peak learning rate of $0.00002$. | ✅ 我们使用余弦学习率衰减调度程序，其中 $200$ 预热步骤和 $0.00002$ 峰值学习率。 |
| ✅ Our results are superior to all previous fine-tuning results on the two datasets. | ✅ 我们的结果优于之前对这两个数据集的所有微调结果。 |
| ✅ Moreover, our fine tuning on retrieval is more efficient, with only roughly $6\%$ and $8\%$ fine-tuning epochs of ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) on Flickr30k and MSCOCO respectively. | ✅ 此外，我们对检索的微调更加高效，在 Flickr30k 和 MSCOCO 上分别仅对 ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) 进行大约 $6\%$ 和 $8\%$ 的微调。 |

### 3.6 Object Detection and Zero-shot Transfer

| 【第3.6节，第1段】原文 | 【第3.6节，第1段】翻译 |
| ---- | ---- |
| ✅ Object detection is one of the most prominent applications in computer vision. | ✅ 物体检测是计算机视觉中最突出的应用之一。 |
| ✅ Compared with existing large-scale pre-trained models ( e.g. | ✅ 与现有的大规模预训练模型（例如 |
| ✅ , CLIP ( **Learning transferable visual models from natural language supervision.** ) , ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** ) , Wu Dao 2.0 ( **https://gpt3demo.com/apps/wu-dao-20.** ) ), Florence is more desirable for object detection since its adaptation helps learn visual representation at the object level. | ✅ 、CLIP ( **Learning transferable visual models from natural language supervision.** )、ALIGN ( **Scaling up visual and vision-language representation learning with noisy text supervision.** )、Wu Dao 2.0 ( **https://gpt3demo.com/apps/wu-dao-20.** )），Florence 更适合用于物体检测，因为它的适应性有助于学习物体级别的视觉表征。 |
| ✅ We evaluate its performance of object-level visual representations via fine-tuned object detection and zero-shot transfer tasks. | ✅ 我们通过微调的对象检测和零样本传输任务来评估其对象级视觉表征的性能。 |

#### 3.6.1 Fine-tuning

| 【第3.6.1节，第1段】原文 | 【第3.6.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We evaluate fine-tuning on three popular object detection datasets: COCO ( **Microsoft COCO:: Common objects in context, 2015.** ) , Object365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ) , and Visual Genome ( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ). | ✅ 我们对三个流行的对象检测数据集进行了微调评估：COCO ( **Microsoft COCO:: Common objects in context, 2015.** )、Object365 ( **Objects365: A large-scale, high-quality dataset for object detection.** ) 和 Visual Genome ( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** )。 |
| ✅ For COCO, we increase the maximum image side to $2,500$ and fine-tune with multi-scale training for $12$ epochs. | ✅ 对于 COCO，我们将最大图像边增加到 $2,500$，并使用多尺度训练对 $12$ 时期进行微调。 |
| ✅ We follow the same multi-scale testing strategy widely used in existing state-of-the-art approaches. | ✅ 我们遵循现有最先进方法中广泛使用的相同多尺度测试策略。 |
| ✅ For Object365, we use the same input resolution of images ( i.e. | ✅ 对于 Object365，我们使用相同的图像输入分辨率（即 |
| ✅ , the maximum image side $1,333$ ) as the Multi-dataset Detection 111This work was ranked 1-stin the object detection track of ECCV 2020 Robust Vision Challenge. ( **Simple multi-dataset detection.** ) for fine-tuning. | ✅ ，最大图像边$1,333$）作为多数据集检测111This work was ranked 1-stin the object detection track of ECCV 2020 Robust Vision Challenge. ( **Simple multi-dataset detection.** )进行微调。 |
| ✅ For Visual Genome, we increase the maximum side of input resolution to $3,000$ and fine-tune with multi-scale training for $24$ epochs. | ✅ 对于 Visual Genome，我们将输入分辨率的最大边增加到 $3,000$，并使用多尺度训练对 $24$ 时期进行微调。 |
| ✅ To leverage attributes annotations in Visual Genome, we insert an $1\times 1$ ROI pool on the final stage of CoSwin backbone to extract features for attribute learning, which allows the object detection adapter being optimized for multi-task learning. | ✅ 为了利用 Visual Genome 中的属性注释，我们在 CoSwin 主干的最后阶段插入了一个 $1\times 1$ ROI 池来提取属性学习的特征，这使得对象检测适配器可以针对多任务学习进行优化。 |

<table class="ltx_tabular ltx_centering ltx_align_middle" id="S3.T6.1"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T6.1.1.1"><td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt" id="S3.T6.1.1.1.1" style="padding:1.6pt 12.5pt;">Benchmark</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt" id="S3.T6.1.1.1.2" style="padding:1.6pt 12.5pt;">Model</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T6.1.1.1.3" style="padding:1.6pt 12.5pt;">AP</td></tr><tr class="ltx_tr" id="S3.T6.1.2.2"><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T6.1.2.2.1" rowspan="3" style="padding:1.6pt 12.5pt;">COCO miniVal</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S3.T6.1.2.2.2" style="padding:1.6pt 12.5pt;">DyHead</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T6.1.2.2.3" style="padding:1.6pt 12.5pt;">60.3</td></tr><tr class="ltx_tr" id="S3.T6.1.3.3"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T6.1.3.3.1" style="padding:1.6pt 12.5pt;">Soft Teacher</td><td class="ltx_td ltx_align_center" id="S3.T6.1.3.3.2" style="padding:1.6pt 12.5pt;">60.7</td></tr><tr class="ltx_tr" id="S3.T6.1.4.4"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T6.1.4.4.1" style="padding:1.6pt 12.5pt;">Florence</td><td class="ltx_td ltx_align_center" id="S3.T6.1.4.4.2" style="padding:1.6pt 12.5pt;">62.0</td></tr><tr class="ltx_tr" id="S3.T6.1.5.5"><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T6.1.5.5.1" rowspan="3" style="padding:1.6pt 12.5pt;">COCO test-Dev</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S3.T6.1.5.5.2" style="padding:1.6pt 12.5pt;">DyHead</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T6.1.5.5.3" style="padding:1.6pt 12.5pt;">60.6</td></tr><tr class="ltx_tr" id="S3.T6.1.6.6"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T6.1.6.6.1" style="padding:1.6pt 12.5pt;">Soft Teacher</td><td class="ltx_td ltx_align_center" id="S3.T6.1.6.6.2" style="padding:1.6pt 12.5pt;">61.3</td></tr><tr class="ltx_tr" id="S3.T6.1.7.7"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T6.1.7.7.1" style="padding:1.6pt 12.5pt;">Florence</td><td class="ltx_td ltx_align_center" id="S3.T6.1.7.7.2" style="padding:1.6pt 12.5pt;">62.4</td></tr><tr class="ltx_tr" id="S3.T6.1.8.8"><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T6.1.8.8.1" rowspan="2" style="padding:1.6pt 12.5pt;">Object365</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S3.T6.1.8.8.2" style="padding:1.6pt 12.5pt;">Multi-dataset Detection</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T6.1.8.8.3" style="padding:1.6pt 12.5pt;">33.7</td></tr><tr class="ltx_tr" id="S3.T6.1.9.9"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T6.1.9.9.1" style="padding:1.6pt 12.5pt;">Florence</td><td class="ltx_td ltx_align_center" id="S3.T6.1.9.9.2" style="padding:1.6pt 12.5pt;">39.3</td></tr><tr class="ltx_tr" id="S3.T6.1.10.10"><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t" id="S3.T6.1.10.10.1" rowspan="2" style="padding:1.6pt 12.5pt;">Visual Genome</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S3.T6.1.10.10.2" style="padding:1.6pt 12.5pt;">VinVL</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T6.1.10.10.3" style="padding:1.6pt 12.5pt;">13.8</td></tr><tr class="ltx_tr" id="S3.T6.1.11.11"><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="S3.T6.1.11.11.1" style="padding:1.6pt 12.5pt;">Florence</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T6.1.11.11.2" style="padding:1.6pt 12.5pt;">16.2</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 6:  Object detection fine tuning comparisons with state-of-the-art methods, including DyHead ( **Dynamic head: Unifying object detection heads with attentions.** ) , Soft Teacher ( **End-to-end semi-supervised object detection with soft teacher.** ) , Multi-dataset Detection ( **Simple multi-dataset detection.** ) , VinVL ( **Vinvl: Revisiting visual representations in vision-language models.** ) . | ✅ Table 6:  对象检测微调与最先进方法的比较，包括 DyHead ( **Dynamic head: Unifying object detection heads with attentions.** )、Soft Teacher ( **End-to-end semi-supervised object detection with soft teacher.** )、多数据集检测 ( **Simple multi-dataset detection.** )、VinVL ( **Vinvl: Revisiting visual representations in vision-language models.** )。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S3.T7.1"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T7.1.1.1"><th class="ltx_td ltx_th ltx_th_row ltx_border_tt" id="S3.T7.1.1.1.1" style="padding:1.6pt 6.5pt;"></th><th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="S3.T7.1.1.1.2" style="padding:1.6pt 6.5pt;"></th><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.3" style="padding:1.6pt 6.5pt;">Aquarium</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.4" style="padding:1.6pt 6.5pt;">BCCD</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.5" style="padding:1.6pt 6.5pt;">Chess Pieces</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.6" style="padding:1.6pt 6.5pt;">Mask Wearing</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.7" style="padding:1.6pt 6.5pt;">Oxford Pets</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.8" style="padding:1.6pt 6.5pt;">Packages</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.9" style="padding:1.6pt 6.5pt;">Pistols</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.10" style="padding:1.6pt 6.5pt;">PKLot</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.11" style="padding:1.6pt 6.5pt;">Pothole</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.12" style="padding:1.6pt 6.5pt;">Thermal</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.1.1.13" style="padding:1.6pt 6.5pt;">Wildfire Smoke</td></tr><tr class="ltx_tr" id="S3.T7.1.2.2"><th class="ltx_td ltx_th ltx_th_row ltx_border_t" id="S3.T7.1.2.2.1" style="padding:1.6pt 6.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T7.1.2.2.2" style="padding:1.6pt 6.5pt;">Images</th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.3" style="padding:1.6pt 6.5pt;">638</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.4" style="padding:1.6pt 6.5pt;">364</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.5" style="padding:1.6pt 6.5pt;">292</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.6" style="padding:1.6pt 6.5pt;">149</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.7" style="padding:1.6pt 6.5pt;">3680</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.8" style="padding:1.6pt 6.5pt;">26</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.9" style="padding:1.6pt 6.5pt;">2986</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.10" style="padding:1.6pt 6.5pt;">12416</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.11" style="padding:1.6pt 6.5pt;">665</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.12" style="padding:1.6pt 6.5pt;">203</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.2.2.13" style="padding:1.6pt 6.5pt;">737</td></tr><tr class="ltx_tr" id="S3.T7.1.3.3"><th class="ltx_td ltx_th ltx_th_row" id="S3.T7.1.3.3.1" style="padding:1.6pt 6.5pt;"></th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T7.1.3.3.2" style="padding:1.6pt 6.5pt;">Categories</th><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.3" style="padding:1.6pt 6.5pt;">7</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.4" style="padding:1.6pt 6.5pt;">3</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.5" style="padding:1.6pt 6.5pt;">12</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.6" style="padding:1.6pt 6.5pt;">2</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.7" style="padding:1.6pt 6.5pt;">37</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.8" style="padding:1.6pt 6.5pt;">1</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.9" style="padding:1.6pt 6.5pt;">1</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.10" style="padding:1.6pt 6.5pt;">2</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.11" style="padding:1.6pt 6.5pt;">1</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.12" style="padding:1.6pt 6.5pt;">2</td><td class="ltx_td ltx_align_center" id="S3.T7.1.3.3.13" style="padding:1.6pt 6.5pt;">1</td></tr><tr class="ltx_tr" id="S3.T7.1.4.4"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt" id="S3.T7.1.4.4.1" rowspan="2" style="padding:1.6pt 6.5pt;">Fine-tuned</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt" id="S3.T7.1.4.4.2" style="padding:1.6pt 6.5pt;">DyHead-Swin-L (full)</th><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.3" style="padding:1.6pt 6.5pt;">53.1</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.4" style="padding:1.6pt 6.5pt;">62.6</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.5" style="padding:1.6pt 6.5pt;">80.7</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.6" style="padding:1.6pt 6.5pt;">52.0</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.7" style="padding:1.6pt 6.5pt;">85.9</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.8" style="padding:1.6pt 6.5pt;">52.0</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.9" style="padding:1.6pt 6.5pt;">74.4</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.10" style="padding:1.6pt 6.5pt;">98.0</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.11" style="padding:1.6pt 6.5pt;">61.8</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.12" style="padding:1.6pt 6.5pt;">75.9</td><td class="ltx_td ltx_align_center ltx_border_tt" id="S3.T7.1.4.4.13" style="padding:1.6pt 6.5pt;">58.7</td></tr><tr class="ltx_tr" id="S3.T7.1.5.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T7.1.5.5.1" style="padding:1.6pt 6.5pt;">DyHead-Swin-L (5-shot)</th><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.2" style="padding:1.6pt 6.5pt;">39.0</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.3" style="padding:1.6pt 6.5pt;">40.6</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.4" style="padding:1.6pt 6.5pt;">57.3</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.5" style="padding:1.6pt 6.5pt;">26.8</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.6" style="padding:1.6pt 6.5pt;">47.5</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.7" style="padding:1.6pt 6.5pt;">32.8</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.8" style="padding:1.6pt 6.5pt;">20.0</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.9" style="padding:1.6pt 6.5pt;">22.1</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.10" style="padding:1.6pt 6.5pt;">10.8</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.11" style="padding:1.6pt 6.5pt;">54.9</td><td class="ltx_td ltx_align_center" id="S3.T7.1.5.5.12" style="padding:1.6pt 6.5pt;">14.2</td></tr><tr class="ltx_tr" id="S3.T7.1.6.6"><th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t" id="S3.T7.1.6.6.1" rowspan="2" style="padding:1.6pt 6.5pt;">Zero-shot</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T7.1.6.6.2" style="padding:1.6pt 6.5pt;">ZSD</th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.3" style="padding:1.6pt 6.5pt;">16.0</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.4" style="padding:1.6pt 6.5pt;">1.2</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.5" style="padding:1.6pt 6.5pt;">0.1</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.6" style="padding:1.6pt 6.5pt;">0.6</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.7" style="padding:1.6pt 6.5pt;">0.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.8" style="padding:1.6pt 6.5pt;">58.3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.9" style="padding:1.6pt 6.5pt;">31.5</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.10" style="padding:1.6pt 6.5pt;">0.2</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.11" style="padding:1.6pt 6.5pt;">2.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.12" style="padding:1.6pt 6.5pt;">37.4</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T7.1.6.6.13" style="padding:1.6pt 6.5pt;">0.002</td></tr><tr class="ltx_tr" id="S3.T7.1.7.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S3.T7.1.7.7.1" style="padding:1.6pt 6.5pt;">Florence</th><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.2" style="padding:1.6pt 6.5pt;">43.1</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.3" style="padding:1.6pt 6.5pt;">15.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.4" style="padding:1.6pt 6.5pt;">13.4</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.5" style="padding:1.6pt 6.5pt;">15.0</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.6" style="padding:1.6pt 6.5pt;">68.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.7" style="padding:1.6pt 6.5pt;">79.6</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.8" style="padding:1.6pt 6.5pt;">41.4</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.9" style="padding:1.6pt 6.5pt;">31.4</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.10" style="padding:1.6pt 6.5pt;">53.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.11" style="padding:1.6pt 6.5pt;">46.9</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T7.1.7.7.12" style="padding:1.6pt 6.5pt;">48.7</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 7:  Zero-shot transfer in object detection, in comparison with previous state-of-the-art model DyHead ( **Dynamic head: Unifying object detection heads with attentions.** ) (on COCO) fine tuning results on full-set or 5-shot respectively and zero-shot detection baseline model ZSD ( **Zero-shot object detection.** ) . | ✅ Table 7:  在物体检测中的零样本迁移，与之前最先进的模型 DyHead ( **Dynamic head: Unifying object detection heads with attentions.** )（在 COCO 上）分别在全套或 5 样本上进行微调的结果以及零样本检测基线模型 ZSD ( **Zero-shot object detection.** ) 相比。 |

![figure](https://ar5iv.labs.arxiv.org/html/2111.11432/assets/x7.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 6:  Our fine-tuned detection results on COCO (sparse object boxes), Object365 (dense object boxes), Visual Genome (w/ object attributes), and zero-shot transfer results on 11 downstream detection tasks. | ✅ Figure 6:  我们对 COCO（稀疏对象框）、Object365（密集对象框）、Visual Genome（带对象属性）的微调检测结果以及 11 个下游检测任务的零样本传输结果。 |
| ✅ Boxes with different colors denote different object categories. | ✅ 不同颜色的框表示不同的对象类别。 |

| 【第3.6.1节，第2段】原文 | 【第3.6.1节，第2段】翻译 |
| ---- | ---- |
| ✅ We compare Florence with state-of-the-art results on these three benchmarks in Table 6. | ✅ 我们在表 6 中将 Florence 与这三个基准上的最新结果进行了比较。 |
| ✅ In object detection, the standard mean average precision (AP) metric is used to report results under different IoU thresholds and object scales for all datasets. | ✅ 在对象检测中，标准平均精度（AP）指标用于报告所有数据集在不同 IoU 阈值和对象尺度下的结果。 |
| ✅ We follow the metrics used in existing state-of-the-art methods. | ✅ 我们遵循现有最先进方法中使用的指标。 |
| ✅ For COCO, Object365 and zero-shot transfer benchmarks, we use mAP, i.e. | ✅ 对于 COCO、Object365 和零样本传输基准，我们使用 mAP，即 |
| ✅ , average over multiple IoUs ( $0.5:0.05:0.95$ ). | ✅ ，多个 IoU 的平均值（$0.5:0.05:0.95$）。 |
| ✅ For Visual Genome, we use AP50 at IoU threshold $0.5$. | ✅ 对于 Visual Genome，我们在 IoU 阈值 $0.5$ 处使用 AP50。 |
| ✅ As we can see, Florence establishes new results in these main benchmarks of object detection. | ✅ 我们可以看到，Florence 在这些主要的物体检测基准测试中都取得了新的成果。 |

#### 3.6.2 Zero-shot Transfer

| 【第3.6.2节，第1段】原文 | 【第3.6.2节，第1段】翻译 |
| ---- | ---- |
| ✅ Zero-shot object detection is more challenging than zero-shot classification, since neither object proposal classification nor location ( i.e. | ✅ 零样本物体检测比零样本分类更具挑战性，因为无论是物体提议分类还是位置（即 |
| ✅ , bounding box regression) in downstream tasks is seen during training. | ✅ 在训练过程中可以看到下游任务中的预测模型（例如边界框回归）。 |
| ✅ In our zero-shot transfer setting, object proposal and object classification are decoupled into two tasks. | ✅ 在我们的零样本传输设置中，对象提议和对象分类被分解为两个任务。 |
| ✅ Object proposal discriminates object from background, ignoring semantics of object categories. | ✅ 对象提议将对象与背景区分开来，忽略对象类别的语义。 |
| ✅ Classification, on the other hand, focuses on object semantics for each bounding box proposal. | ✅ 另一方面，分类关注每个边界框提议的对象语义。 |
| ✅ In spirit, this setup is similar to the behavior of R-CNN model ( **Rich feature hierarchies for accurate object detection and semantic segmentation.** ) which has been widely used for object detection before. | ✅ 从本质上讲，这种设置类似于之前广泛用于对象检测的 R-CNN 模型 ( **Rich feature hierarchies for accurate object detection and semantic segmentation.** ) 的行为。 |
| ✅ Using this approach, we can follow existing work on zero-shot image classification to zero-shot transfer in object detection, to evaluate the Florence for novel object recognition. | ✅ 使用这种方法，我们可以遵循现有的零样本图像分类工作，将零样本迁移到物体检测中，以评估 Florence 对新物体识别的能力。 |
| ✅ As mentioned in ZSD ( **Zero-shot object detection.** ) , it more approaches real world settings. | ✅ 正如 ZSD ( **Zero-shot object detection.** ) 中提到的，它更接近现实世界的设置。 |

| 【第3.6.2节，第2段】原文 | 【第3.6.2节，第2段】翻译 |
| ---- | ---- |
| ✅ For zero-shot transfer, the training of the detection adapter can be different from fine-tuning. | ✅ 对于零样本迁移，检测适配器的训练可以与微调不同。 |
| ✅ Specifically, we freeze the CoSwin backbones and pre-train the Dynamic Head on FLOD-9M by neglecting semantics from each object bounding box. | ✅ 具体来说，我们冻结 CoSwin 主干，并通过忽略每个对象边界框的语义在 FLOD-9M 上预训练动态头。 |
| ✅ We treat the object detection pre-training as general-purpose object proposal training. | ✅ 我们将对象检测预训练视为通用对象提议训练。 |
| ✅ Note that the detection pre-training only updates the object adapter, and does not affect the fused feature representations learned from large-scale image-text pairs. | ✅ 请注意，检测预训练仅更新对象适配器，并且不会影响从大规模图像 - 文本对中学习到的融合特征表示。 |
| ✅ In inference, we apply the pre-trained CoSwin and Dynamic Head on downstream datasets, and obtain the object proposals for every image. | ✅ 在推理中，我们将预先训练的 CoSwin 和 Dynamic Head 应用于下游数据集，并获得每个图像的对象建议。 |
| ✅ For each object proposal, we apply zero-shot classification, as described in Section 3.1 . | ✅ 对于每个对象提议，我们应用零样本分类，如第 3.1 节所述。 |

| 【第3.6.2节，第3段】原文 | 【第3.6.2节，第3段】翻译 |
| ---- | ---- |
| ✅ To evaluate Florence ’s transferability to novel, diverse and application-oriented tasks, following ( **Grounded language-image pre-training.** ) , we curate an “open-set oject detection benchmark” which aggregates $11$ public datasets from Roboflow 222https://public.roboflow.com/object-detection , spanning scenarios including fine-grained fishes/chess detection, drone-view detection, and thermal object detection. | ✅ 为了评估 Florence 对新颖、多样化和面向应用的任务的可转移性，继 ( **Grounded language-image pre-training.** ) 之后，我们策划了一个“开放集对象检测基准”，它汇总了来自 Roboflow 222https://public.roboflow.com/object-detection 的 $11$ 公共数据集，涵盖了细粒度鱼类/国际象棋检测、无人机视图检测和热物体检测等场景。 |
| ✅ We use their split test datasets for evaluation. | ✅ 我们使用他们的分割测试数据集进行评估。 |
| ✅ Table 7 shows that our Florence model effectively zero-shot transfers to these tasks. | ✅ 表 7 表明我们的 Florence 模型有效地对这些任务进行了零样本迁移。 |
| ✅ We use the results of the baseline approach ZSD ( **Zero-shot object detection.** ) , which considers a similar setting, for reference. | ✅ 我们使用考虑类似设置的基线方法 ZSD ( **Zero-shot object detection.** ) 的结果作为参考。 |
| ✅ In our implementation 333We refer to (Li et al., 2021b) for details. , we replace their supervised object detector FasterRCNN with the recent SOTA detector ( **Dynamic head: Unifying object detection heads with attentions.** ) and use pre-trained BERT as the language encoder. | ✅ 在我们的实现 333We refer to (Li et al., 2021b) for details. 中，我们用最近的 SOTA 检测器 ( **Dynamic head: Unifying object detection heads with attentions.** ) 替换他们的监督对象检测器 FasterRCNN，并使用预训练的 BERT 作为语言编码器。 |
| ✅ Both are pre-trained end-to-end on the Objects365 dataset. | ✅ 两者均在 Objects365 数据集上进行端到端的预训练。 |
| ✅ Thanks to large-scale image-text pretraining, Florence shows remarkable gains on all tasks. | ✅ 得益于大规模图像文本预训练，Florence 在所有任务上都表现出了显著的进步。 |
| ✅ Zero-shot in object detection still has a long way to be applied to real-world tasks. | ✅ 物体检测中的零样本方法距离应用于实际任务还有很长的路要走。 |
| ✅ We further compare Florence zero-shot with previous state-of-the-art detector 444It is pre-trained on ImageNet and COCO in supervised way. ( **Dynamic head: Unifying object detection heads with attentions.** ) (on COCO) fine-tunning on these tasks. | ✅ 我们进一步比较了 Florence zero-shot 与之前最先进的检测器 444It is pre-trained on ImageNet and COCO in supervised way. ( **Dynamic head: Unifying object detection heads with attentions.** )（在 COCO 上）在这些任务上的微调。 |
| ✅ We can observe noticeable performance gap between zero-shot and supervised learning, especially for novel scenarios whose concepts/classes may not be covered by the pre-training dataset, such as “BCCD” (blood cells photos), “Chess Pieces” (Chess board photos and various pieces). | ✅ 我们可以观察到零样本学习和监督学习之间存在明显的性能差距，特别是对于那些概念/类别可能未被预训练数据集涵盖的新场景，例如“BCCD”（血细胞照片）、“棋子”（棋盘照片和各种棋子）。 |
| ✅ However, the results are encouraging when compared with few-shot fine-tuning results. | ✅ 然而，与少量微调结果相比，结果是令人鼓舞的。 |
| ✅ Florence outperforms in $7/11$ tasks over 5-shot fine tuning, and outperforms full-set fine-tuning on the “Packages” dataset, consisting of only $26$ images for training. | ✅ Florence 在 $7/11$ 任务中的表现优于 5 次微调，并且在仅由 $26$ 图像组成的“Packages”数据集上优于全集微调。 |
| ✅ It demonstrates the foundation models’ great potential of improving data efficiency and reducing deployment cost for new tasks or domains. | ✅ 它展示了基础模型在提高数据效率和降低新任务或领域部署成本方面的巨大潜力。 |

### 3.7 V+L Representation Learning

| 【第3.7节，第1段】原文 | 【第3.7节，第1段】翻译 |
| ---- | ---- |
| ✅ The vision-langauge pretraining (VLP) is performed on MSCOCO ( **Microsoft COCO:: Common objects in context, 2015.** ) , Conceptual Captions (CC) ( **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.** ) , CC12M ( **Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts.** ) , SBU ( **Im2text: Describing images using 1 million captioned photographs.** ) , and Visual Genome (VG) ( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ). | ✅ 视觉语言预训练（VLP）在 MSCOCO ( **Microsoft COCO:: Common objects in context, 2015.** )、概念字幕（CC）( **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.** )、CC12M ( **Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts.** )、SBU ( **Im2text: Describing images using 1 million captioned photographs.** ) 和视觉基因组（VG）( **Visual genome: Connecting language and vision using crowdsourced dense image annotations.** ) 上进行。 |
| ✅ These datasets result in $14$ million images with $20$ million associated captions. | ✅ 这些数据集产生了 $14$ 百万张图像和 $20$ 百万张相关标题。 |
| ✅ Beyond replacing the image encoder with CoSwin -H of our Florence model on ( **An empirical study of training end-to-end vision-and-language transformers.** ) , we remove the weight decay on the text embedding layer and the modality-specific embedding. | ✅ 除了用 ( **An empirical study of training end-to-end vision-and-language transformers.** ) 上的 Florence 模型的 CoSwin -H 替换图像编码器之外，我们还删除了文本嵌入层和特定于模态的嵌入上的权重衰减。 |
| ✅ ITM and MLM are applied for VLP with $43$ epochs with the image input size as $384$ . | ✅ ITM 和 MLM 应用于 VLP，时期 $43$，图像输入大小为 $384$。 |

| 【第3.7节，第2段】原文 | 【第3.7节，第2段】翻译 |
| ---- | ---- |
| ✅ To evaluate the performance, we fine-tune the pre-trained model on the challenging VQA ( **Making the V in VQA matter: Elevating the role of image understanding in visual question answering.** ) task, which is to answer a question based on the image context. | ✅ 为了评估性能，我们在具有挑战性的 VQA ( **Making the V in VQA matter: Elevating the role of image understanding in visual question answering.** ) 任务上对预训练模型进行了微调，即根据图像上下文回答问题。 |
| ✅ The dataset consists of $82$ K training images and $41$ K validation images. | ✅ 该数据集由 $82$ K 训练图像和 $41$ K 验证图像组成。 |
| ✅ Only $1$ K validation images are reserved and the rest are merged with the training data for fine-tuning. | ✅ 仅保留 $1$ K 验证图像，其余图像与训练数据合并进行微调。 |
| ✅ As a common practice, the problem is cast as a classification task where each class corresponds to an answer. | ✅ 按照常见的做法，该问题被视为分类任务，其中每个类别对应一个答案。 |
| ✅ The final pooling representations are fed into a randomly-initialized multilayer perceptron (MLP) network to predict the answer over $3,129$ answers. | ✅ 最终的池化表示被输入到随机初始化的多层感知器 (MLP) 网络中，以预测 $3,129$ 答案的答案。 |
| ✅ The loss is the binary cross-entropy loss, and the inference is to select the answer with the highest confidence. | ✅ 损失是二元交叉熵损失，推理是选择置信度最高的答案。 |
| ✅ The model is fine-tuned for $10$ epochs with the learning rate as $8e-6$ and is evaluated on the test-dev and test-std. | ✅ 该模型针对 $10$ 时期进行了微调，学习率为 $8e-6$，并在 test-dev 和 test-std 上进行了评估。 |
| ✅ The final accuracy is calculated on the public server 555http://evalai.com . | ✅ 最终准确率在公共服务器555http://evalai.com上计算。 |

| 【第3.7节，第3段】原文 | 【第3.7节，第3段】翻译 |
| ---- | ---- |
| ✅ Figure 8 shows the comparison results with the existing methods. | ✅ 图8显示了与现有方法的比较结果。 |
| ✅ As we can see, we achieve the new state-of-the-art performance. | ✅ 我们可以看到，我们达到了新的最先进的性能。 |
| ✅ Compared with SimVLM ( **Simvlm: Simple visual language model pretraining with weak supervision.** ) , which uses $1.8$ B image-text pairs, we only use $900$ M data to pre-train the image encoder and $20$ M for VLP, but achieve better results. | ✅ 与使用 $1.8$ B 图像-文本对的 SimVLM ( **Simvlm: Simple visual language model pretraining with weak supervision.** ) 相比，我们仅使用 $900$ M 数据对图像编码器进行预训练，并使用 $20$ M 进行 VLP 训练，但取得了更好的效果。 |
| ✅ This also demonstrates the data efficiency of our approach. | ✅ 这也证明了我们方法的数据效率。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S3.T8.1"><thead class="ltx_thead"><tr class="ltx_tr" id="S3.T8.1.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S3.T8.1.1.1.1" style="padding:1.6pt 10.5pt;">Model</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T8.1.1.1.2" style="padding:1.6pt 10.5pt;">test-dev</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T8.1.1.1.3" style="padding:1.6pt 10.5pt;">test-std</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T8.1.2.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T8.1.2.1.1" style="padding:1.6pt 10.5pt;">UNITER <html><body><p>( <strong>Uniter: Universal image-text representation learning.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T8.1.2.1.2" style="padding:1.6pt 10.5pt;">73.82</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T8.1.2.1.3" style="padding:1.6pt 10.5pt;">74.02</td></tr><tr class="ltx_tr" id="S3.T8.1.3.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.3.2.1" style="padding:1.6pt 10.5pt;">Visual Parsing <html><body><p>( <strong>Probing inter-modality: Visual parsing with self-attention forvision-language pre-training.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.3.2.2" style="padding:1.6pt 10.5pt;">74.00</td><td class="ltx_td ltx_align_center" id="S3.T8.1.3.2.3" style="padding:1.6pt 10.5pt;">74.17</td></tr><tr class="ltx_tr" id="S3.T8.1.4.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.4.3.1" style="padding:1.6pt 10.5pt;">PixelBERT <html><body><p>( <strong>Pixel-BERT: Aligning image pixels with text by deep multi-modaltransformers.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.4.3.2" style="padding:1.6pt 10.5pt;">74.45</td><td class="ltx_td ltx_align_center" id="S3.T8.1.4.3.3" style="padding:1.6pt 10.5pt;">74.55</td></tr><tr class="ltx_tr" id="S3.T8.1.5.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.5.4.1" style="padding:1.6pt 10.5pt;">VILLA <html><body><p>( <strong>Large-scale adversarial training for vision-and-languagerepresentation learning.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.5.4.2" style="padding:1.6pt 10.5pt;">74.69</td><td class="ltx_td ltx_align_center" id="S3.T8.1.5.4.3" style="padding:1.6pt 10.5pt;">74.87</td></tr><tr class="ltx_tr" id="S3.T8.1.6.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.6.5.1" style="padding:1.6pt 10.5pt;">UNIMO <html><body><p>( <strong>Unimo: Towards unified-modal understanding and generation viacross-modal contrastive learning.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.6.5.2" style="padding:1.6pt 10.5pt;">75.06</td><td class="ltx_td ltx_align_center" id="S3.T8.1.6.5.3" style="padding:1.6pt 10.5pt;">75.27</td></tr><tr class="ltx_tr" id="S3.T8.1.7.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.7.6.1" style="padding:1.6pt 10.5pt;">ALBEF <html><body><p>( <strong>Align before fuse: Vision and language representation learning withmomentum distillation.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.7.6.2" style="padding:1.6pt 10.5pt;">75.84</td><td class="ltx_td ltx_align_center" id="S3.T8.1.7.6.3" style="padding:1.6pt 10.5pt;">76.04</td></tr><tr class="ltx_tr" id="S3.T8.1.8.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.8.7.1" style="padding:1.6pt 10.5pt;">VinVL <html><body><p>( <strong>Vinvl: Revisiting visual representations in vision-language models.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.8.7.2" style="padding:1.6pt 10.5pt;">76.52</td><td class="ltx_td ltx_align_center" id="S3.T8.1.8.7.3" style="padding:1.6pt 10.5pt;">76.60</td></tr><tr class="ltx_tr" id="S3.T8.1.9.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.9.8.1" style="padding:1.6pt 10.5pt;">CLIP-ViL <html><body><p>( <strong>How much can clip benefit vision-and-language tasks?</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.9.8.2" style="padding:1.6pt 10.5pt;">76.48</td><td class="ltx_td ltx_align_center" id="S3.T8.1.9.8.3" style="padding:1.6pt 10.5pt;">76.70</td></tr><tr class="ltx_tr" id="S3.T8.1.10.9"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.10.9.1" style="padding:1.6pt 10.5pt;">METER <html><body><p>( <strong>An empirical study of training end-to-end vision-and-languagetransformers.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.10.9.2" style="padding:1.6pt 10.5pt;">77.68</td><td class="ltx_td ltx_align_center" id="S3.T8.1.10.9.3" style="padding:1.6pt 10.5pt;">77.64</td></tr><tr class="ltx_tr" id="S3.T8.1.11.10"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T8.1.11.10.1" style="padding:1.6pt 10.5pt;">SimVLM <html><body><p>( <strong>Simvlm: Simple visual language model pretraining with weaksupervision.</strong> )</p></body></html></th><td class="ltx_td ltx_align_center" id="S3.T8.1.11.10.2" style="padding:1.6pt 10.5pt;">80.03</td><td class="ltx_td ltx_align_center" id="S3.T8.1.11.10.3" style="padding:1.6pt 10.5pt;">80.34</td></tr><tr class="ltx_tr" id="S3.T8.1.12.11"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t" id="S3.T8.1.12.11.1" style="padding:1.6pt 10.5pt;">Florence</th><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" id="S3.T8.1.12.11.2" style="padding:1.6pt 10.5pt;">80.16</td><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" id="S3.T8.1.12.11.3" style="padding:1.6pt 10.5pt;">80.36</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 8:  Compare our model with the existing state-of-the-art methods on VQA. | ✅ Table 8:  将我们的模型与 VQA 上现有的最先进方法进行比较。 |

<table class="ltx_tabular ltx_centering ltx_align_middle" id="S3.T9.1"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T9.1.2.1"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt" id="S3.T9.1.2.1.1" style="padding:1.6pt 10.5pt;">Method</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_tt" id="S3.T9.1.2.1.2" style="padding:1.6pt 10.5pt;">Pre-training Type</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_tt" id="S3.T9.1.2.1.3" style="padding:1.6pt 10.5pt;">Pre-training Data</td><td class="ltx_td ltx_align_right ltx_border_tt" id="S3.T9.1.2.1.4" style="padding:1.6pt 10.5pt;">R@1</td><td class="ltx_td ltx_align_right ltx_border_tt" id="S3.T9.1.2.1.5" style="padding:1.6pt 10.5pt;">R@5</td><td class="ltx_td ltx_align_right ltx_border_tt" id="S3.T9.1.2.1.6" style="padding:1.6pt 10.5pt;">R@10</td></tr><tr class="ltx_tr" id="S3.T9.1.3.2"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S3.T9.1.3.2.1" style="padding:1.6pt 10.5pt;">MIL-NCE <html><body><p>( <strong>End-to-end learning of visual representations from uncuratedinstructional videos.</strong> )</p></body></html></td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S3.T9.1.3.2.2" style="padding:1.6pt 10.5pt;">Video</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S3.T9.1.3.2.3" style="padding:1.6pt 10.5pt;">HowTo100M</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T9.1.3.2.4" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T9.1.3.2.5" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T9.1.3.2.6" style="padding:1.6pt 10.5pt;">32.4</td></tr><tr class="ltx_tr" id="S3.T9.1.4.3"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T9.1.4.3.1" style="padding:1.6pt 10.5pt;">MMV <html><body><p>( <strong>Self-supervised multimodal versatile networks.</strong> )</p></body></html></td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.4.3.2" style="padding:1.6pt 10.5pt;">Video</td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.4.3.3" style="padding:1.6pt 10.5pt;">HowTo100M, AudioSet</td><td class="ltx_td ltx_align_right" id="S3.T9.1.4.3.4" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right" id="S3.T9.1.4.3.5" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right" id="S3.T9.1.4.3.6" style="padding:1.6pt 10.5pt;">31.1</td></tr><tr class="ltx_tr" id="S3.T9.1.1"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T9.1.1.2" style="padding:1.6pt 10.5pt;">VideoCLIP <html><body><p>( <strong>Videoclip: Contrastive pre-training for zero-shot video-textunderstanding.</strong> )</p></body></html></td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.1.1" style="padding:1.6pt 10.5pt;">Video<sup class="ltx_sup" id="S3.T9.1.1.1.2">∗</sup></td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.1.3" style="padding:1.6pt 10.5pt;">HowTo100M</td><td class="ltx_td ltx_align_right" id="S3.T9.1.1.4" style="padding:1.6pt 10.5pt;">10.4</td><td class="ltx_td ltx_align_right" id="S3.T9.1.1.5" style="padding:1.6pt 10.5pt;">22.2</td><td class="ltx_td ltx_align_right" id="S3.T9.1.1.6" style="padding:1.6pt 10.5pt;">30.0</td></tr><tr class="ltx_tr" id="S3.T9.1.5.4"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T9.1.5.4.1" style="padding:1.6pt 10.5pt;">VATT <html><body><p>( <strong>Vatt: Transformers for multimodal self-supervised learning from rawvideo, audio and text.</strong> )</p></body></html></td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.5.4.2" style="padding:1.6pt 10.5pt;">Video</td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.5.4.3" style="padding:1.6pt 10.5pt;">HowTo100M, AudioSet</td><td class="ltx_td ltx_align_right" id="S3.T9.1.5.4.4" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right" id="S3.T9.1.5.4.5" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right" id="S3.T9.1.5.4.6" style="padding:1.6pt 10.5pt;">29.7</td></tr><tr class="ltx_tr" id="S3.T9.1.6.5"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T9.1.6.5.1" style="padding:1.6pt 10.5pt;">MCN <html><body><p>( <strong>Multimodal clustering networks for self-supervised learning fromunlabeled videos.</strong> )</p></body></html></td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.6.5.2" style="padding:1.6pt 10.5pt;">Image and Video</td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.6.5.3" style="padding:1.6pt 10.5pt;">HowTo100M</td><td class="ltx_td ltx_align_right" id="S3.T9.1.6.5.4" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right" id="S3.T9.1.6.5.5" style="padding:1.6pt 10.5pt;">-</td><td class="ltx_td ltx_align_right" id="S3.T9.1.6.5.6" style="padding:1.6pt 10.5pt;">33.8</td></tr><tr class="ltx_tr" id="S3.T9.1.7.6"><td class="ltx_td ltx_align_left ltx_border_r" id="S3.T9.1.7.6.1" style="padding:1.6pt 10.5pt;">Frozen-in-Time <html><body><p>( <strong>Frozen in time: A joint video and image encoder for end-to-endretrieval.</strong> )</p></body></html></td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.7.6.2" style="padding:1.6pt 10.5pt;">Image and Video</td><td class="ltx_td ltx_align_right ltx_border_r" id="S3.T9.1.7.6.3" style="padding:1.6pt 10.5pt;">ImageNet,CC, WebVid-2M</td><td class="ltx_td ltx_align_right" id="S3.T9.1.7.6.4" style="padding:1.6pt 10.5pt;">18.7</td><td class="ltx_td ltx_align_right" id="S3.T9.1.7.6.5" style="padding:1.6pt 10.5pt;">39.5</td><td class="ltx_td ltx_align_right" id="S3.T9.1.7.6.6" style="padding:1.6pt 10.5pt;">51.6</td></tr><tr class="ltx_tr" id="S3.T9.1.8.7"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="S3.T9.1.8.7.1" style="padding:1.6pt 10.5pt;">CLIP-ViT-B/16 <html><body><p>( <strong>Learning transferable visual models from natural languagesupervision.</strong> )</p></body></html></td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S3.T9.1.8.7.2" style="padding:1.6pt 10.5pt;">Image</td><td class="ltx_td ltx_align_right ltx_border_r ltx_border_t" id="S3.T9.1.8.7.3" style="padding:1.6pt 10.5pt;">WIT400M</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T9.1.8.7.4" style="padding:1.6pt 10.5pt;">26.0</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T9.1.8.7.5" style="padding:1.6pt 10.5pt;">49.4</td><td class="ltx_td ltx_align_right ltx_border_t" id="S3.T9.1.8.7.6" style="padding:1.6pt 10.5pt;">60.7</td></tr><tr class="ltx_tr" id="S3.T9.1.9.8"><td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r" id="S3.T9.1.9.8.1" style="padding:1.6pt 10.5pt;">Florence</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S3.T9.1.9.8.2" style="padding:1.6pt 10.5pt;">Image</td><td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r" id="S3.T9.1.9.8.3" style="padding:1.6pt 10.5pt;">FLD-900M</td><td class="ltx_td ltx_align_right ltx_border_bb" id="S3.T9.1.9.8.4" style="padding:1.6pt 10.5pt;">37.6</td><td class="ltx_td ltx_align_right ltx_border_bb" id="S3.T9.1.9.8.5" style="padding:1.6pt 10.5pt;">63.8</td><td class="ltx_td ltx_align_right ltx_border_bb" id="S3.T9.1.9.8.6" style="padding:1.6pt 10.5pt;">72.6</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 9:  Zero-shot text-to-video retrieval results on MSR-VTT 1K-A test set. | ✅ Table 9:  在 MSR-VTT 1K-A 测试集上的零样本文本转视频检索结果。 |
| ✅ ( ∗ : Feature extracted from the pre-trained model ( **End-to-end learning of visual representations from uncurated instructional videos.** ) , followed by another stage of video-and-language pre-training) The pretraining data used in these existing methods include HowTo100M ( **Howto100m: Learning a text-video embedding by watching hundred million narrated video clips.** ) , AudioSet ( **Audio set: An ontology and human-labeled dataset for audio events.** ) , ImageNet ( **Imagenet: A large-scale hierarchical image database.** ) , CC ( **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.** ) , WebVid-2M ( **Frozen in time: A joint video and image encoder for end-to-end retrieval.** ) , WIT400M ( **Learning transferable visual models from natural language supervision.** ) | ✅ （∗：从预训练模型 ( **End-to-end learning of visual representations from uncurated instructional videos.** ) 中提取的特征，然后进行另一阶段的视频和语言预训练）这些现有方法中使用的预训练数据包括 HowTo100M ( **Howto100m: Learning a text-video embedding by watching hundred million narrated video clips.** )、AudioSet ( **Audio set: An ontology and human-labeled dataset for audio events.** )、ImageNet ( **Imagenet: A large-scale hierarchical image database.** )、CC ( **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.** )、WebVid-2M ( **Frozen in time: A joint video and image encoder for end-to-end retrieval.** )、WIT400M ( **Learning transferable visual models from natural language supervision.** ) |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S3.T10.5"><thead class="ltx_thead"><tr class="ltx_tr" id="S3.T10.5.6.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S3.T10.5.6.1.1" rowspan="2" style="padding:1.6pt 12.0pt;">Method</th><th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt" id="S3.T10.5.6.1.2" rowspan="2" style="padding:1.6pt 12.0pt;">Pretraining Data</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt" colspan="2" id="S3.T10.5.6.1.3" style="padding:1.6pt 12.0pt;">Kinetics-400</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt" colspan="2" id="S3.T10.5.6.1.4" style="padding:1.6pt 12.0pt;">Kinetics-600</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt" id="S3.T10.5.6.1.5" rowspan="2" style="padding:1.6pt 12.0pt;">Views</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" id="S3.T10.5.6.1.6" rowspan="2" style="padding:1.6pt 12.0pt;">Params</th></tr><tr class="ltx_tr" id="S3.T10.5.7.2"><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="S3.T10.5.7.2.1" style="padding:1.6pt 12.0pt;">Top-1</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r" id="S3.T10.5.7.2.2" style="padding:1.6pt 12.0pt;">Top-5</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column" id="S3.T10.5.7.2.3" style="padding:1.6pt 12.0pt;">Top-1</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r" id="S3.T10.5.7.2.4" style="padding:1.6pt 12.0pt;">Top-5</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="S3.T10.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T10.1.1.2" style="padding:1.6pt 12.0pt;">ViViT-H/16x2</th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S3.T10.1.1.3" style="padding:1.6pt 12.0pt;">JFT-300M</th><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T10.1.1.4" style="padding:1.6pt 12.0pt;">84.8</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T10.1.1.5" style="padding:1.6pt 12.0pt;">95.8</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T10.1.1.6" style="padding:1.6pt 12.0pt;">85.8</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T10.1.1.7" style="padding:1.6pt 12.0pt;">96.5</td><td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" id="S3.T10.1.1.1" style="padding:1.6pt 12.0pt;">4 <math alttext="\times" class="ltx_Math" display="inline" id="S3.T10.1.1.1.m1.1"><semantics id="S3.T10.1.1.1.m1.1a"><mo id="S3.T10.1.1.1.m1.1.1" mathsize="90%" xref="S3.T10.1.1.1.m1.1.1.cmml">×</mo><annotation-xml encoding="MathML-Content" id="S3.T10.1.1.1.m1.1b"><times id="S3.T10.1.1.1.m1.1.1.cmml" xref="S3.T10.1.1.1.m1.1.1"></times></annotation-xml><annotation encoding="application/x-tex" id="S3.T10.1.1.1.m1.1c">\times</annotation></semantics></math> 3</td><td class="ltx_td ltx_align_center ltx_border_t" id="S3.T10.1.1.8" style="padding:1.6pt 12.0pt;">648M</td></tr><tr class="ltx_tr" id="S3.T10.2.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T10.2.2.2" style="padding:1.6pt 12.0pt;">VideoSwin-L</th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S3.T10.2.2.3" style="padding:1.6pt 12.0pt;">ImageNet-22K</th><td class="ltx_td ltx_align_center" id="S3.T10.2.2.4" style="padding:1.6pt 12.0pt;">84.6</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.2.2.5" style="padding:1.6pt 12.0pt;">96.5</td><td class="ltx_td ltx_align_center" id="S3.T10.2.2.6" style="padding:1.6pt 12.0pt;">85.9</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.2.2.7" style="padding:1.6pt 12.0pt;">97.1</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.2.2.1" style="padding:1.6pt 12.0pt;">4 <math alttext="\times" class="ltx_Math" display="inline" id="S3.T10.2.2.1.m1.1"><semantics id="S3.T10.2.2.1.m1.1a"><mo id="S3.T10.2.2.1.m1.1.1" mathsize="90%" xref="S3.T10.2.2.1.m1.1.1.cmml">×</mo><annotation-xml encoding="MathML-Content" id="S3.T10.2.2.1.m1.1b"><times id="S3.T10.2.2.1.m1.1.1.cmml" xref="S3.T10.2.2.1.m1.1.1"></times></annotation-xml><annotation encoding="application/x-tex" id="S3.T10.2.2.1.m1.1c">\times</annotation></semantics></math> 3</td><td class="ltx_td ltx_align_center" id="S3.T10.2.2.8" style="padding:1.6pt 12.0pt;">200M</td></tr><tr class="ltx_tr" id="S3.T10.3.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T10.3.3.2" style="padding:1.6pt 12.0pt;">VideoSwin-L</th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S3.T10.3.3.3" style="padding:1.6pt 12.0pt;">ImageNet-22K</th><td class="ltx_td ltx_align_center" id="S3.T10.3.3.4" style="padding:1.6pt 12.0pt;">84.9</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.3.3.5" style="padding:1.6pt 12.0pt;">96.7</td><td class="ltx_td ltx_align_center" id="S3.T10.3.3.6" style="padding:1.6pt 12.0pt;">86.1</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.3.3.7" style="padding:1.6pt 12.0pt;">97.3</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.3.3.1" style="padding:1.6pt 12.0pt;">10 <math alttext="\times" class="ltx_Math" display="inline" id="S3.T10.3.3.1.m1.1"><semantics id="S3.T10.3.3.1.m1.1a"><mo id="S3.T10.3.3.1.m1.1.1" mathsize="90%" xref="S3.T10.3.3.1.m1.1.1.cmml">×</mo><annotation-xml encoding="MathML-Content" id="S3.T10.3.3.1.m1.1b"><times id="S3.T10.3.3.1.m1.1.1.cmml" xref="S3.T10.3.3.1.m1.1.1"></times></annotation-xml><annotation encoding="application/x-tex" id="S3.T10.3.3.1.m1.1c">\times</annotation></semantics></math> 5</td><td class="ltx_td ltx_align_center" id="S3.T10.3.3.8" style="padding:1.6pt 12.0pt;">200M</td></tr><tr class="ltx_tr" id="S3.T10.4.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S3.T10.4.4.2" style="padding:1.6pt 12.0pt;">TokenLearner 16at18+L/10</th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_r" id="S3.T10.4.4.3" style="padding:1.6pt 12.0pt;">JFT-300M</th><td class="ltx_td ltx_align_center" id="S3.T10.4.4.4" style="padding:1.6pt 12.0pt;">85.4</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.4.4.5" style="padding:1.6pt 12.0pt;">96.3</td><td class="ltx_td ltx_align_center" id="S3.T10.4.4.6" style="padding:1.6pt 12.0pt;">86.3</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.4.4.7" style="padding:1.6pt 12.0pt;">97.0</td><td class="ltx_td ltx_align_center ltx_border_r" id="S3.T10.4.4.1" style="padding:1.6pt 12.0pt;">4<math alttext="\times" class="ltx_Math" display="inline" id="S3.T10.4.4.1.m1.1"><semantics id="S3.T10.4.4.1.m1.1a"><mo id="S3.T10.4.4.1.m1.1.1" mathsize="90%" xref="S3.T10.4.4.1.m1.1.1.cmml">×</mo><annotation-xml encoding="MathML-Content" id="S3.T10.4.4.1.m1.1b"><times id="S3.T10.4.4.1.m1.1.1.cmml" xref="S3.T10.4.4.1.m1.1.1"></times></annotation-xml><annotation encoding="application/x-tex" id="S3.T10.4.4.1.m1.1c">\times</annotation></semantics></math> 3</td><td class="ltx_td ltx_align_center" id="S3.T10.4.4.8" style="padding:1.6pt 12.0pt;">460M</td></tr><tr class="ltx_tr" id="S3.T10.5.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S3.T10.5.5.2" style="padding:1.6pt 12.0pt;">Florence</th><th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_bb ltx_border_r" id="S3.T10.5.5.3" style="padding:1.6pt 12.0pt;">FLD-900M</th><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T10.5.5.4" style="padding:1.6pt 12.0pt;">86.5</td><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r" id="S3.T10.5.5.5" style="padding:1.6pt 12.0pt;">97.3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T10.5.5.6" style="padding:1.6pt 12.0pt;">87.8</td><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r" id="S3.T10.5.5.7" style="padding:1.6pt 12.0pt;">97.8</td><td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r" id="S3.T10.5.5.1" style="padding:1.6pt 12.0pt;">4 <math alttext="\times" class="ltx_Math" display="inline" id="S3.T10.5.5.1.m1.1"><semantics id="S3.T10.5.5.1.m1.1a"><mo id="S3.T10.5.5.1.m1.1.1" mathsize="90%" xref="S3.T10.5.5.1.m1.1.1.cmml">×</mo><annotation-xml encoding="MathML-Content" id="S3.T10.5.5.1.m1.1b"><times id="S3.T10.5.5.1.m1.1.1.cmml" xref="S3.T10.5.5.1.m1.1.1"></times></annotation-xml><annotation encoding="application/x-tex" id="S3.T10.5.5.1.m1.1c">\times</annotation></semantics></math> 3</td><td class="ltx_td ltx_align_center ltx_border_bb" id="S3.T10.5.5.8" style="padding:1.6pt 12.0pt;">647M</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 10:  Comparison to state-of-the-art methods, including ViViT ( **Vivit: A video vision transformer.** ) , VideoSwin ( **Video swin transformer.** ) , TokenLearner ( **Tokenlearner: What can 8 learned tokens do for images and videos?** ) , on Kinetics-400 and Kinetics-600. | ✅ Table 10:  与 Kinetics-400 和 Kinetics-600 上最先进的方法进行比较，包括 ViViT ( **Vivit: A video vision transformer.** )、VideoSwin ( **Video swin transformer.** )、TokenLearner ( **Tokenlearner: What can 8 learned tokens do for images and videos?** )。 |
| ✅ Views indicate $\#temporal\;clip\times\#spatial\;crop$ . | ✅ 视图表明$\#temporal\;clip\times\#spatial\;crop$。 |

### 3.8 Zero-Shot Text-to-Video Retrieval

| 【第3.8节，第1段】原文 | 【第3.8节，第1段】翻译 |
| ---- | ---- |
| ✅ Although Florence is pre-trained on image-text pairs, it can be easily adapted to video tasks (shown in Section 2.6 ), such as text-video retrieval. | ✅ 尽管 Florence 是在图像-文本对上进行过预训练的，但它可以轻松适应视频任务（如第 2.6 节所示），例如文本-视频检索。 |
| ✅ We expand the input 2D patch embeddings and positional embeddings to 3D so that the encoder can process video inputs, following ( **Vivit: A video vision transformer.** ). | ✅ 我们将输入的 2D 块嵌入和位置嵌入扩展为 3D，以便编码器可以处理视频输入，遵循 ( **Vivit: A video vision transformer.** )。 |
| ✅ Then, we perform zero-shot text-to-video evaluation on the MSR-VTT ( **Msr-vtt: A large video description dataset for bridging video and language.** ) dataset. | ✅ 然后，我们在 MSR-VTT ( **Msr-vtt: A large video description dataset for bridging video and language.** ) 数据集上执行零样本文本到视频评估。 |
| ✅ We report results on the 1K-A test ( **A joint sequence fusion model for video question answering and retrieval.** ) , which contains 1K video and caption pairs. | ✅ 我们报告了 1K-A 测试 ( **A joint sequence fusion model for video question answering and retrieval.** ) 的结果，其中包含 1K 视频和字幕对。 |
| ✅ We use the standard recall metrics for evaluation and compare with existing state-of-the-art methods in Table 9. | ✅ 我们使用标准召回率指标进行评估，并与表 9 中的现有最先进方法进行比较。 |
| ✅ As we can see, these two image-text pre-trained models CLIP 666We use a public available CLIP checkpoint forcomparison ( **Learning transferable visual models from natural language supervision.** ) and Florence outperform all the state-of-the-art methods by a large margin in terms of the $R@1$ metric. | ✅ 我们可以看出，这两个图像文本预训练模型 CLIP 666We use a public available CLIP checkpoint forcomparison ( **Learning transferable visual models from natural language supervision.** ) 和 Florence 在 $R@1$ 指标方面远远优于所有最先进的方法。 |
| ✅ It reveals that the video data used for pretraining in these state-of-the-art methods may not be so rich or diverse as image-text data used in Florence or CLIP. | ✅ 它表明，这些最先进的方法中用于预训练的视频数据可能不像 Florence 或 CLIP 中使用的图像文本数据那样丰富或多样化。 |

### 3.9 Video Action Recognition

| 【第3.9节，第1段】原文 | 【第3.9节，第1段】翻译 |
| ---- | ---- |
| ✅ We evaluate Florence on fine-tuned video action recognition tasks. | ✅ 我们在微调的视频动作识别任务上对 Florence 进行了评估。 |
| ✅ On the Kinectics-400 and Kinectics-600 datasets, we follow the typical fine-tuning setting ( **Video swin transformer.** ) and fine tune the model (Section 2.6 ) with $384\times 384$ resolution for $30$ epochs. | ✅ 在 Kinectics-400 和 Kinectics-600 数据集上，我们遵循典型的微调设置 ( **Video swin transformer.** )，并使用 $384\times 384$ 分辨率对 $30$ 时期的模型进行微调（第 2.6 节）。 |
| ✅ We use the label smoothing, rand augmentation, a small learning rate $0.0002$ and a relatively large drop path rate $0.5$ to avoid over-fitting the target video datasets. | ✅ 我们使用标签平滑、随机增强、较小的学习率 $0.0002$ 和相对较大的丢弃路径率 $0.5$ 来避免过度拟合目标视频数据集。 |
| ✅ We compare with existing state-of-the-art methods in Table 10. | ✅ 我们在表 10 中与现有的最先进方法进行了比较。 |
| ✅ Our results are better than the state-of-the-art by $1.1\%$ and $1.5\%$ on Kinectics-400 and Kinectics-600, respectively. | ✅ 我们的结果分别优于 Kinectics-400 和 Kinectics-600 上的 $1.1\%$ 和 $1.5\%$ 的最新成果。 |

## 4 Conclusion and Future Work

| 【第4节，第1段】原文 | 【第4节，第1段】翻译 |
| ---- | ---- |
| ✅ In this paper we investigated a new paradigm of building a computer vision foundation model, Florence , as a general-purpose vision system. | ✅ 在本文中，我们研究了构建计算机视觉基础模型 Florence 作为通用视觉系统的新范式。 |
| ✅ Our attempt is a step towards building XYZ-code ( **A holistic representation toward integrative ai.** ) , an integrative AI system that makes progress toward human-like AI. | ✅ 我们的尝试是朝着构建 XYZ-code ( **A holistic representation toward integrative ai.** ) 迈出的一步，这是一个朝着类人 AI 发展的综合 AI 系统。 |
| ✅ Although the model size is still below several other existing billion-scale models, Florence successfully extends to different tasks along space, time, and modality, with great transferbility, and achieves new SOTA results on a wide range of vision benchmarks. | ✅ 尽管模型大小仍低于其他几个现有的十亿级模型，但 Florence 成功扩展到空间、时间和模态的不同任务，具有很强的可迁移性，并在广泛的视觉基准上取得了新的 SOTA 结果。 |

| 【第4节，第2段】原文 | 【第4节，第2段】翻译 |
| ---- | ---- |
| ✅ For the future work, we plan to include more vision tasks and applications, such as depth/flow estimation, tracking, and additional vision+language tasks. | ✅ 对于未来的工作，我们计划纳入更多的视觉任务和应用，例如深度/流估计、跟踪和额外的视觉+语言任务。 |
| ✅ Florence is designed to pave the way for building vision foundation models to power millions of real-world vision tasks and applications. | ✅ Florence 旨在为构建视觉基础模型铺平道路，为数百万个现实世界的视觉任务和应用程序提供支持。 |
| ✅ In addition, the preliminary progress on zero-shot classification and object detection may motivate more research to close the performance gap to supervised learning. | ✅ 此外，零样本分类和目标检测方面的初步进展可能会激发更多的研究，以缩小与监督学习的性能差距。 |

#### 4.1 Acknowledgment

| 【第4.1节，第1段】原文 | 【第4.1节，第1段】翻译 |
| ---- | ---- |

| 【第4.1节，第2段】原文 | 【第4.1节，第2段】翻译 |
| ---- | ---- |
| ✅ We would like to thank the following people involved in the discussion for their valuable feedback including Xiaowei Hu, Yen-Chun Chen, Lin Liang, Yinpeng Chen, Li Dong, Furu Wei, Han Hu, Yue Cao, Zheng Zhang, Hao Yang, Jianmin Bao, Dong Chen, Fang Wen, Jianlong Fu, Houwen Peng, Chong Luo, Baining Guo. | ✅ We would like to thank the following people involved in the discussion for their valuable feedback including Xiaowei Hu, Yen-Chun Chen, Lin Liang, Yinpeng Chen, Li Dong, Furu Wei, Han Hu, Yue Cao, Zheng Zhang, Hao Yang, Jianmin Bao, Dong Chen, Fang Wen, Jianlong Fu, Houwen Peng, Chong Luo, Baining Guo. |
| ✅ We would also thank Qingfen Lin, Cha Zhang for their thoughtful feedback on the broader impacts of the paper. | ✅ 我们还要感谢青芬林、张查对本文更广泛影响提出的深思熟虑的反馈。 |
| ✅ Thanks Mei Gao, Ping Jin for helping run evaluations on benchmark infrastructure. | ✅ 感谢高梅、金平帮助对基准基础设施进行评估。 |
| ✅ We are also grateful to the developers of software toolkits used throughout this project, including Liyang Lu, Robert Gmyr, Felipe Cruz Salinas, Canrun Li, Steven Tsai, Min Gao, Kevin Pan, Shohei Ono, Christina Sun. | ✅ 我们还要感谢在整个项目中使用的软件工具包的开发人员，包括 Liyang Lu、Robert Gmyr、Felipe Cruz Salinas、Canrun Li、Steven Tsai、Min Gao、Kevin Pan、Shohei Ono 和 Christina Sun。 |
| ✅ Additionally, we would like to thank the entire Deepspeed, AI Frameworks, and ITP teams for making it possible to train models at this scale. | ✅ 此外，我们还要感谢整个 Deepspeed、AI Frameworks 和 ITP 团队，使得我们能够以这种规模训练模型。 |

## 5 References

- 1
  - Wu dao 2.0.
  - **https://gpt3demo.com/apps/wu-dao-20.**

- 2
  - Akbari, H., Yuan, L., Qian, R., Chuang, W.-H., Chang, S.-F., Cui, Y., and Gong, B.
  - **Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text.**
  - In NeurIPS, 2021.

- 3
  - Alayrac, J.-B., Recasens, A., Schneider, R., Arandjelovic, R., Ramapuram, J., De Fauw, J., Smaira, L., Dieleman, S., and Zisserman, A.
  - **Self-supervised multimodal versatile networks.**
  - In NeurIPS, volume 2, pp.  7, 2020.

- 4
  - Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., and Zhang, L.
  - **Bottom-up and top-down attention for image captioning and visual question answering.**
  - In CVPR, 2018.

- 5
  - Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., and Schmid, C.
  - **Vivit: A video vision transformer.**
  - In ICCV, 2021.

- 6
  - Bain, M., Nagrani, A., Varol, G., and Zisserman, A.
  - **Frozen in time: A joint video and image encoder for end-to-end retrieval.**
  - In ICCV, 2021.

- 7
  - Bansal, A., Sikka, K., Sharma, G., Chellappa, R., and Divakaran, A.
  - **Zero-shot object detection.**
  - In Proceedings of the European Conference on Computer Vision (ECCV), pp.  384–400, 2018.

- 8
  - Berg, T., Liu, J., Lee, S. W., Alexander, M. L., Jacobs, D. W., and Belhumeur, P. N.
  - **Birdsnap: Large-scale fine-grained visual categorization of birds.**
  - In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pp.  2019–2026, 2014.

- 9
  - Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., Brynjolfsson, E., Buch, S., Card, D., Castellon, R., Chatterji, N., Chen, A., Creel, K., Davis, J. Q., Demszky, D., Donahue, C., Doumbouya, M., Durmus, E., Ermon, S., Etchemendy, J., Ethayarajh, K., Fei-Fei, L., Finn, C., Gale, T., Gillespie, L., Goel, K., Goodman, N., Grossman, S., Guha, N., Hashimoto, T., Henderson, P., Hewitt, J., Ho, D. E., Hong, J., Hsu, K., Huang, J., Icard, T., Jain, S., Jurafsky, D., Kalluri, P., Karamcheti, S., Keeling, G., Khani, F., Khattab, O., Koh, P. W., Krass, M., Krishna, R., Kuditipudi, R., Kumar, A., Ladhak, F., Lee, M., Lee, T., Leskovec, J., Levent, I., Li, X. L., Li, X., Ma, T., Malik, A., Manning, C. D., Mirchandani, S., Mitchell, E., Munyikwa, Z., Nair, S., Narayan, A., Narayanan, D., Newman, B., Nie, A., Niebles, J. C., Nilforoshan, H., Nyarko, J., Ogut, G., Orr, L., Papadimitriou, I., Park, J. S., Piech, C., Portelance, E., Potts, C., Raghunathan, A., Reich, R., Ren, H., Rong, F., Roohani, Y., Ruiz, C., Ryan, J., Ré, C., Sadigh, D., Sagawa, S., Santhanam, K., Shih, A., Srinivasan, K., Tamkin, A., Taori, R., Thomas, A. W., Tramèr, F., Wang, R. E., Wang, W., Wu, B., Wu, J., Wu, Y., Xie, S. M., Yasunaga, M., You, J., Zaharia, M., Zhang, M., Zhang, T., Zhang, X., Zhang, Y., Zheng, L., Zhou, K., and Liang, P.
  - **On the opportunities and risks of foundation models.**
  - In arXiv 2108.07258, 2021.

- 10
  - Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D.
  - **Language models are few-shot learners.**
  - In arXiv 2005.14165, 2020.

- 11
  - Changpinyo, S., Sharma, P., Ding, N., and Soricut, R.
  - **Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts.**
  - In CVPR, 2021.

- 12
  - Chen, B., Rouditchenko, A., Duarte, K., Kuehne, H., Thomas, S., Boggust, A., Panda, R., Kingsbury, B., Feris, R., Harwath, D., et al.
  - **Multimodal clustering networks for self-supervised learning from unlabeled videos.**
  - In ICCV, 2021.

- 13
  - Chen, J., Hu, H., Wu, H., Jiang, Y., and Wang, C.
  - **Learning the best pooling strategy for visual semantic embedding.**
  - In arXiv preprint arXiv:2011.04305, 2020a.

- 14
  - Chen, T., Kornblith, S., Norouzi, M., and Hinton, G.
  - **A simple framework for contrastive learning of visual representations.**
  - In Proceedings of the 37th International Conference on Machine Learning, volume 119, pp.  1597–1607, 13–18 Jul 2020b.

- 15
  - Chen, T., Kornblith, S., Swersky, K., Norouzi, M., and Hinton, G.
  - **Big self-supervised models are strong semi-supervised learners.**
  - arXiv preprint arXiv:2006.10029, 2020c.

- 16
  - Chen, Y.-C., Li, L., Yu, L., Kholy, A. E., Ahmed, F., Gan, Z., Cheng, Y., and Liu, J.
  - **Uniter: Universal image-text representation learning.**
  - In Proceedings of European Conference on Computer Vision, 2020d.

- 17
  - Codella, N. C. F., Rotemberg, V., Tschandl, P., Celebi, M. E., Dusza, S. W., Gutman, D. A., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M. A., Kittler, H., and Halpern, A.
  - **Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC).**
  - abs/1902.03368, 2019.

- 18
  - Dai, X., Chen, Y., Xiao, B., Chen, D., Liu, M., Yuan, L., and Zhang, L.
  - **Dynamic head: Unifying object detection heads with attentions.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.  7373–7382, June 2021a.

- 19
  - Dai, X., Chen, Y., Yang, J., Zhang, P., Yuan, L., and Zhang, L.
  - **Dynamic detr: End-to-end object detection with dynamic attention.**
  - In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp.  2988–2997, October 2021b.

- 20
  - Dai, Z., Liu, H., Le, Q. V., and Tan, M.
  - **Coatnet: Marrying convolution and attention for all data sizes.**
  - In arXiv 2106.04803, 2021c.

- 21
  - Dean, J.
  - **Introducing pathways: A next-generation ai architecture.**
  - https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/.

- 22
  - Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L.
  - **Imagenet: A large-scale hierarchical image database.**
  - In 2009 IEEE conference on computer vision and pattern recognition, pp.  248–255. Ieee, 2009.

- 23
  - Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K.
  - **Bert: Pre-training of deep bidirectional transformers for language understanding.**
  - In arXiv 1810.04805, 2019.

- 24
  - Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., and Guo, B.
  - **Cswin transformer: A general vision transformer backbone with cross-shaped windows.**
  - In arXiv 2107.00652, 2021.

- 25
  - Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N.
  - **An image is worth 16x16 words: Transformers for image recognition at scale.**
  - ICLR, 2021a.

- 26
  - Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N.
  - **An image is worth 16x16 words: Transformers for image recognition at scale.**
  - In arXiv 2010.11929, 2021b.

- 27
  - Dou, Z.-Y., Xu, Y., Gan, Z., Wang, J., Wang, S., Wang, L., Zhu, C., Nanyun, Peng, Liu, Z., and Zeng, M.
  - **An empirical study of training end-to-end vision-and-language transformers.**
  - In arXiv 2111.02387, 2021.

- 28
  - Fang, Z., Wang, J., Hu, X., Wang, L., Yang, Y., and Liu, Z.
  - **Compressing visual-linguistic model via knowledge distillation.**
  - In ICCV, 2021.

- 29
  - Gan, Z., Chen, Y.-C., Li, L., Zhu, C., Cheng, Y., and Liu, J.
  - **Large-scale adversarial training for vision-and-language representation learning.**
  - In Proceedings of Neural Information Processing Systems, 2020.

- 30
  - Gao, L., Zhang, Y., Han, J., and Callan, J.
  - **Scaling deep contrastive learning batch size under memory limited setup.**
  - In arXiv 2101.06983, 2021.

- 31
  - Gemmeke, J. F., Ellis, D. P., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., Plakal, M., and Ritter, M.
  - **Audio set: An ontology and human-labeled dataset for audio events.**
  - In ICASSP, pp.  776–780. IEEE, 2017.

- 32
  - Girshick, R., Donahue, J., Darrell, T., and Malik, J.
  - **Rich feature hierarchies for accurate object detection and semantic segmentation.**
  - In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pp.  580–587, 2014.

- 33
  - Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D.
  - **Making the V in VQA matter: Elevating the role of image understanding in visual question answering.**
  - In CVPR, 2017.

- 34
  - Guo, Y., Codella, N. C. F., Karlinsky, L., Smith, J. R., Rosing, T., and Feris, R. S.
  - **A new benchmark for evaluation of cross-domain few-shot learning.**
  - ECCV, 2020.

- 35
  - Gupta, A., Dollar, P., and Girshick, R.
  - **Lvis: A dataset for large vocabulary instance segmentation.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

- 36
  - Helber, P., Bischke, B., Dengel, A., and Borth, D.
  - **Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification.**
  - IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.

- 37
  - Huang, X.
  - **A holistic representation toward integrative ai.**
  - https://www.microsoft.com/en-us/research/blog/a-holistic-representation-toward-integrative-ai/.

- 38
  - Huang, Z., Zeng, Z., Liu, B., Fu, D., and Fu, J.
  - **Pixel-BERT: Aligning image pixels with text by deep multi-modal transformers.**
  - arXiv preprint, 2020.

- 39
  - Huang, Z., Zeng, Z., Huang, Y., Liu, B., Fu, D., and Fu, J.
  - **Seeing out of the box: End-to-end pre-training for vision-language representation learning.**
  - In CVPR, 2021.

- 40
  - Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q. V., Sung, Y., Li, Z., and Duerig, T.
  - **Scaling up visual and vision-language representation learning with noisy text supervision.**
  - In arXiv 2102.05918, 2021.

- 41
  - Kim, W., Son, B., and Kim, I.
  - **Vilt: Vision-and-language transformer without convolution or region supervision.**
  - In Meila, M. and Zhang, T. (eds.), ICML, 2021.

- 42
  - Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., and Houlsby, N.
  - **Big transfer (bit): General visual representation learning.**
  - In arXiv 1912.11370, 2020.

- 43
  - Kornblith, S., Shlens, J., and Le, Q. V.
  - **Do better imagenet models transfer better?**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.  2661–2671, 2019.

- 44
  - Krasin, I., Duerig, T., Alldrin, N., Veit, A., Abu-El-Haija, S., Belongie, S., Cai, D., Feng, Z., Ferrari, V., Gomes, V., Gupta, A., Narayanan, D., Sun, C., Chechik, G., and Murphy, K.
  - **Openimages: A public dataset for large-scale multi-label and multi-class image classification.**
  - Dataset available from https://github.com/openimages, 2016.

- 45
  - Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., Bernstein, M., and Fei-Fei, L.
  - **Visual genome: Connecting language and vision using crowdsourced dense image annotations.**
  - In arXiv 1602.07332, 2016.

- 46
  - Li, J., Selvaraju, R. R., Gotmare, A. D., Joty, S., Xiong, C., and Hoi, S.
  - **Align before fuse: Vision and language representation learning with momentum distillation.**
  - In Conference on Neural Information Processing Systems (NeurIPS), 2021a.

- 47
  - Li, L. H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J.-N., Chang, K.-W., and Gao, J.
  - **Grounded language-image pre-training.**
  - In arXiv In Preparation, 2021b.

- 48
  - Li, W., Gao, C., Niu, G., Xiao, X., Liu, H., Liu, J., Wu, H., and Wang, H.
  - **Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning.**
  - In Annual Meeting of the Association for Computational Linguistics (ACL), 2021c.

- 49
  - Li, X., Yin, X., Li, C., Zhang, P., Hu, X., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., Choi, Y., and Gao, J.
  - **Oscar: Object-semantics aligned pre-training for vision-language tasks.**
  - In Proceedings of European Conference on Computer Vision, 2020.

- 50
  - Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Dollár, P.
  - **Microsoft COCO:: Common objects in context, 2015.**

- 51
  - Liu, B., Zhao, Z., Li, Z., Jiang, J., Guo, Y., and Ye, J.
  - **Feature transformation ensemble model with batch spectral regularization for cross-domain few-shot classification.**
  - 2020.

- 52
  - Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V.
  - **RoBERTa: A robustly optimized bert pretraining approach.**
  - arXiv preprint, 2019.

- 53
  - Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B.
  - **Swin transformer: Hierarchical vision transformer using shifted windows.**
  - International Conference on Computer Vision (ICCV), 2021a.

- 54
  - Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., and Hu, H.
  - **Video swin transformer.**
  - arXiv preprint arXiv:2106.13230, 2021b.

- 55
  - Miech, A., Zhukov, D., Alayrac, J.-B., Tapaswi, M., Laptev, I., and Sivic, J.
  - **Howto100m: Learning a text-video embedding by watching hundred million narrated video clips.**
  - In ICCV, pp.  2630–2640, 2019.

- 56
  - Miech, A., Alayrac, J.-B., Smaira, L., Laptev, I., Sivic, J., and Zisserman, A.
  - **End-to-end learning of visual representations from uncurated instructional videos.**
  - In CVPR, pp.  9879–9889, 2020.

- 57
  - Mohanty, S. P., Hughes, D. P., and Salathe, M.
  - **Using deep learning for image-based plant disease detection.**
  - Front Plant Sci, 7, 2016.

- 58
  - Ordonez, V., Kulkarni, G., and Berg, T. L.
  - **Im2text: Describing images using 1 million captioned photographs.**
  - In NeurIPS, 2011.

- 59
  - Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C., Hockenmaier, J., and Lazebnik, S.
  - **Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models.**
  - In arXiv 1505.04870, 2016.

- 60
  - Qi, D., Su, L., Song, J., Cui, E., Bharti, T., and Sacheti, A.
  - **Imagebert: Cross-modal pre-training with large-scale weak-supervised image-text data.**
  - arXiv preprintarXiv:2001.07966, 2020.

- 61
  - Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I.
  - **Learning transferable visual models from natural language supervision.**
  - In arXiv 2103.00020, 2021.

- 62
  - Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y.
  - **Zero: Memory optimization towards training A trillion parameter models.**
  - CoRR, 2019.

- 63
  - Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I.
  - **Zero-shot text-to-image generation.**
  - In arXiv 2102.12092, 2021.

- 64
  - Ryoo, M. S., Piergiovanni, A., Arnab, A., Dehghani, M., and Angelova, A.
  - **Tokenlearner: What can 8 learned tokens do for images and videos?**
  - In arXiv 2106.11297, 2021.

- 65
  - Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., and Sun, J.
  - **Objects365: A large-scale, high-quality dataset for object detection.**
  - In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

- 66
  - Sharma, P., Ding, N., Goodman, S., and Soricut, R.
  - **Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning.**
  - In ACL, pp.  2556–2565, 2018.

- 67
  - Shen, S., Li, L. H., Tan, H., Bansal, M., Rohrbach, A., Chang, K.-W., Yao, Z., and Keutzer, K.
  - **How much can clip benefit vision-and-language tasks?**
  - arXiv preprint, 2021.

- 68
  - Tschandl, P., Rosendahl, C., and Kittler, H.
  - **The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.**
  - Nature Scientific Data, 5, 2018.

- 69
  - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I.
  - **Attention is all you need.**
  - In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

- 70
  - Wang, J., Hu, X., Zhang, P., Li, X., Wang, L., Zhang, L., Gao, J., and Liu, Z.
  - **Minivlm: A smaller and faster vision-language model.**
  - arXiv preprint arXiv:2012.06946, 2020.

- 71
  - Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., and Summers, R. M.
  - **Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases.**
  - In arXiv 1705.02315, 2017.

- 72
  - Wang, Z., Yu, J., Yu, A. W., Dai, Z., Tsvetkov, Y., and Cao, Y.
  - **Simvlm: Simple visual language model pretraining with weak supervision.**
  - In arXiv 2108.10904, 2021.

- 73
  - Wu, H., Xiao, B., Codella, N., Liu, M., Dai, X., Yuan, L., and Zhang, L.
  - **Cvt: Introducing convolutions to vision transformers.**
  - In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp.  22–31, October 2021.

- 74
  - Xie, Q., Luong, M.-T., Hovy, E., and Le, Q. V.
  - **Self-training with noisy student improves imagenet classification.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

- 75
  - Xu, H., Ghosh, G., Huang, P.-Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., and Feichtenhofer, C.
  - **Videoclip: Contrastive pre-training for zero-shot video-text understanding.**
  - In EMNLP, 2021a.

- 76
  - Xu, J., Mei, T., Yao, T., and Rui, Y.
  - **Msr-vtt: A large video description dataset for bridging video and language.**
  - In CVPR, pp.  5288–5296, 2016.

- 77
  - Xu, M., Zhang, Z., Hu, H., Wang, J., Wang, L., Wei, F., Bai, X., and Liu, Z.
  - **End-to-end semi-supervised object detection with soft teacher.**
  - In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp.  3060–3069, October 2021b.

- 78
  - Xue, H., Huang, Y., Liu, B., Peng, H., Fu, J., Li, H., and Luo, J.
  - **Probing inter-modality: Visual parsing with self-attention for vision-language pre-training.**
  - In NeurIPS, 2021.

- 79
  - Yang, J., Li, C., Zhang, P., Dai, X., Xiao, B., Yuan, L., and Gao, J.
  - **Focal self-attention for local-global interactions in vision transformers.**
  - In arXiv 2107.00641, 2021.

- 80
  - Yang, J., Li, C., Zhang, P., Xiao, B., Liu, C., Yuan, L., and Gao, J.
  - **Unified contrastive learning in image-text-label space.**
  - In arXiv In Preparation, 2022.

- 81
  - Yao, L., Huang, R., Hou, L., Lu, G., Niu, M., Xu, H., Liang, X., Li, Z., Jiang, X., and Xu, C.
  - **Filip: Fine-grained interactive language-image pre-training.**
  - In arXiv 2111.07783, 2021.

- 82
  - Yu, F., Tang, J., Yin, W., Sun, Y., Tian, H., Wu, H., and Wang, H.
  - **Ernie-vil: Knowledge enhanced vision-language representations through scene graph.**
  - arXiv preprint arXiv:2006.16934, 2020.

- 83
  - Yu, Y., Kim, J., and Kim, G.
  - **A joint sequence fusion model for video question answering and retrieval.**
  - In ECCV, pp.  471–487, 2018.

- 84
  - Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L.
  - **Scaling vision transformers.**
  - In arXiv 2106.04560, 2021.

- 85
  - Zhang, P., Dai, X., Yang, J., Xiao, B., Yuan, L., Zhang, L., and Gao, J.
  - **Multi-scale vision longformer: A new vision transformer for high-resolution image encoding.**
  - ICCV 2021, 2021a.

- 86
  - Zhang, P., Li, X., Hu, X., Yang, J., Zhang, L., Wang, L., Choi, Y., and Gao, J.
  - **Vinvl: Revisiting visual representations in vision-language models.**
  - In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.  5579–5588, June 2021b.

- 87
  - Zhou, X., Koltun, V., and Krähenbühl, P.
  - **Simple multi-dataset detection.**
  - In arXiv 2102.13086, 2021.

- 88
  - Zoph, B., Ghiasi, G., Lin, T.-Y., Cui, Y., Liu, H., Cubuk, E. D., and Le, Q.
  - **Rethinking pre-training and self-training.**
  - In NeurIPS, 2020.