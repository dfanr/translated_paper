# Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models

## 0. Abstract

| 【概述】原文 | 【概述】翻译 |
| ---- | ---- |
| ✅ Toon shading is a type of non-photorealistic rendering task of animation. | ✅ 卡通着色是一种非真实感的动画渲染任务。 |
| ✅ Its primary purpose is to render objects with a flat and stylized appearance. | ✅ 其主要目的是渲染具有平面和风格化外观的物体。 |
| ✅ As diffusion models have ascended to the forefront of image synthesis methodologies, this paper delves into an innovative form of toon shading based on diffusion models, aiming to directly render photorealistic videos into anime styles. | ✅ 由于扩散模型已成为图像合成方法的前沿，本文深入研究了基于扩散模型的卡通着色的创新形式，旨在将逼真的视频直接渲染成动漫风格。 |
| ✅ In video stylization, extant methods encounter persistent challenges, notably in maintaining consistency and achieving high visual quality. | ✅ 在视频风格化中，现有的方法面临着持续的挑战，特别是在保持一致性和实现高视觉质量方面。 |
| ✅ In this paper, we model the toon shading problem as four subproblems: stylization, consistency enhancement, structure guidance, and colorization. | ✅ 在本文中，我们将卡通着色问题建模为四个子问题：风格化、一致性增强、结构指导和着色。 |
| ✅ To address the challenges in video stylization, we propose an effective toon shading approach called Diffutoon. | ✅ 为了解决视频风格化方面的挑战，我们提出了一种称为 Diffutoon 的有效卡通着色方法。 |
| ✅ Diffutoon is capable of rendering remarkably detailed, high-resolution, and extended-duration videos in anime style. | ✅ Diffutoon 能够以动漫风格渲染细节丰富、分辨率高且持续时间长的视频。 |
| ✅ It can also edit the content according to prompts via an additional branch. | ✅ 它还可以通过附加分支根据提示来编辑内容。 |
| ✅ The efficacy of Diffutoon is evaluated through quantitive metrics and human evaluation. | ✅ Diffutoon 的功效通过定量指标和人工评估来评估。 |
| ✅ Notably, Diffutoon surpasses both open-source and closed-source baseline approaches in our experiments. | ✅ 值得注意的是，Diffutoon 在我们的实验中超越了开源和闭源基线方法。 |
| ✅ Our work is accompanied by the release of both the source code and example videos on Github 111Project page: https://ecnu-cilab.github.io/DiffutoonProjectPage/ . | ✅ 我们的工作伴随着在 Github 111Project page: https://ecnu-cilab.github.io/DiffutoonProjectPage/ 上发布源代码和示例视频。 |

## 1 Introduction

| 【第1节，第1段】原文 | 【第1节，第1段】翻译 |
| ---- | ---- |
| ✅ Toon shading ( **X-toon: An extended toon shader. In Proceedings of the 4th international symposium on Non-photorealistic animation and rendering. 127–132.** ) is a crucial task within the animation industry, aiming to render 3D computer-generated graphics in a flat style. | ✅ 卡通着色 ( **X-toon: An extended toon shader. In Proceedings of the 4th international symposium on Non-photorealistic animation and rendering. 127–132.** ) 是动画行业中的一项关键任务，旨在以平面风格渲染 3D 计算机生成的图形。 |
| ✅ These techniques are extensively applied across diverse domains, including video game development and animation production ( **2D shading for cel animation. In Proceedings of the Joint Symposium on Computational Aesthetics and Sketch-Based Interfaces and Modeling and Non-Photorealistic Animation and Rendering. 1–12.** ). | ✅ 这些技术广泛应用于各个领域，包括视频游戏开发和动画制作 ( **2D shading for cel animation. In Proceedings of the Joint Symposium on Computational Aesthetics and Sketch-Based Interfaces and Modeling and Non-Photorealistic Animation and Rendering. 1–12.** )。 |
| ✅ As diffusion models ( **Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.** ) achieve impressive performance in image synthesis, we discern their potential in video stylization. | ✅ 由于扩散模型 ( **Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.** ) 在图像合成方面取得了令人印象深刻的性能，我们认识到它们在视频风格化方面的潜力。 |
| ✅ In this paper, we explore a new type of toon shading task, aiming to directly transform photorealistic videos into an animated visual style. | ✅ 在本文中，我们探索一种新型的卡通着色任务，旨在将逼真的视频直接转换为动画视觉风格。 |

| 【第1节，第2段】原文 | 【第1节，第2段】翻译 |
| ---- | ---- |
| ✅ In recent years, Stable Diffusion ( **High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.** ) , a diffusion model pre-trained on large-scale text-image datasets ( **Schuhmann et al. (2022)  Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022.   Laion-5b: An open large-scale dataset for training next generation image-text models.   Advances in Neural Information Processing Systems 35 (2022), 25278–25294.** ) , has emerged as a powerful backbone in text-to-image synthesis. | ✅ 近年来，稳定扩散 ( **High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.** )（一种在大型文本图像数据集 ( **Schuhmann et al. (2022)  Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022.   Laion-5b: An open large-scale dataset for training next generation image-text models.   Advances in Neural Information Processing Systems 35 (2022), 25278–25294.** ) 上进行预训练的扩散模型）已成为文本到图像合成领域的强大支柱。 |
| ✅ In open-source communities, abundant fine-tuned models based on Stable Diffusion are available to handle diverse styles. | ✅ 在开源社区中，有丰富的基于 Stable Diffusion 的微调模型可用于处理多种风格。 |
| ✅ Nevertheless, extending diffusion models to video processing presents many challenges ( **Xing et al. (2023)  Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. 2023.   A survey on video diffusion models.   arXiv preprint arXiv:2310.10647 (2023).** ). | ✅ 然而，将扩散模型扩展到视频处理面临着许多挑战( **Xing et al. (2023)  Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. 2023.   A survey on video diffusion models.   arXiv preprint arXiv:2310.10647 (2023).** )。 |
| ✅ First, there is a lack of controllability. | ✅ 第一，缺乏可控性。 |
| ✅ When applying diffusion models to videos, it is difficult to retain essential information in the original video, such as structure and lighting. | ✅ 当将扩散模型应用于视频时，很难保留原始视频中的必要信息，例如结构和光照。 |
| ✅ Second, the consistency issue is crucial, as independently processing each frame often leads to undesirable flickering. | ✅ 其次，一致性问题至关重要，因为独立处理每一帧通常会导致不必要的闪烁。 |
| ✅ Third, visual quality remains a concern. | ✅ 第三，视觉质量仍然令人担忧。 |
| ✅ While video platforms commonly support resolutions up to 1080P and even 4K, most diffusion models struggle to process high-resolution videos. | ✅ 虽然视频平台通常支持高达 1080P 甚至 4K 的分辨率，但大多数传播模型都难以处理高分辨率视频。 |

| 【第1节，第3段】原文 | 【第1节，第3段】翻译 |
| ---- | ---- |
| ✅ Prior studies have attempted to address these challenges. | ✅ 先前的研究已经尝试解决这些挑战。 |
| ✅ In controllable image synthesis, adapter-type control modules ( **1. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.** ｜ **2. Mou et al. (2023)  Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023.   T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models.   arXiv preprint arXiv:2302.08453 (2023).** ) have already demonstrated the capability for precise control. | ✅ 在可控图像合成方面，适配器型控制模块( **1. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.** ｜ **2. Mou et al. (2023)  Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023.   T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models.   arXiv preprint arXiv:2302.08453 (2023).** )已展现出精准控制的能力。 |
| ✅ However, these modules are limited to processing individual images and cannot handle videos. | ✅ 然而，这些模块仅限于处理单个图像，无法处理视频。 |
| ✅ To improve video consistency, studies on this topic are typically categorized into two types: training-free and training-based approaches. | ✅ 为了提高视频一致性，关于该主题的研究通常分为两种类型：无训练方法和基于训练的方法。 |
| ✅ Training-free methods ( **1. Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ｜ **2. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.** ) align content between frames by constructing specific mechanisms, requiring no training process, but their effectiveness is limited. | ✅ 免训练方法( **1. Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ｜ **2. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.** )通过构建特定的机制来对齐帧之间的内容，不需要训练过程，但其有效性有限。 |
| ✅ On the other hand, training-based methods ( **1. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ｜ **2. Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) can generally achieve better results. | ✅ 另一方面，基于训练的方法( **1. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ｜ **2. Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** )通常可以取得更好的结果。 |
| ✅ However, due to the substantial computational resources required, training diffusion models on lengthy video datasets remains exceedingly challenging. | ✅ 然而，由于需要大量的计算资源，在长视频数据集上训练扩散模型仍然极具挑战性。 |
| ✅ Consequently, most video diffusion models can only handle up to a maximum of 32 continuous frames, leading to inconsistencies in longer videos. | ✅ 因此，大多数视频传播模型最多只能处理 32 个连续帧，从而导致较长的视频出现不一致。 |
| ✅ To achieve better visual quality, super-resolution techniques ( **Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision. 1905–1914.** ) can potentially enhance video resolution, but they may introduce extra issues like over-smoothed information loss ( **Li et al. (2022)  Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. 2022.   Srdiff: Single image super-resolution with diffusion probabilistic models.   Neurocomputing 479 (2022), 47–59.** ) . | ✅ 为了获得更好的视觉质量，超分辨率技术 ( **Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision. 1905–1914.** ) 可以潜在地提高视频分辨率，但它们可能会引入额外的问题，例如过度平滑的信息丢失 ( **Li et al. (2022)  Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. 2022.   Srdiff: Single image super-resolution with diffusion probabilistic models.   Neurocomputing 479 (2022), 47–59.** )。 |

| 【第1节，第4段】原文 | 【第1节，第4段】翻译 |
| ---- | ---- |
| ✅ In this paper, we propose a video processing method specifically designed for toon shading. | ✅ 在本文中，我们提出了一种专门针对卡通着色的视频处理方法。 |
| ✅ We divide the toon shading problem into four subproblems: stylization, consistency enhancement, structure guidance, and colorization. | ✅ 我们将卡通着色问题分为四个子问题：风格化、一致性增强、结构指导和着色。 |
| ✅ For each subproblem, we provide a specific solution. | ✅ 对于每个子问题，我们都提供一个具体的解决方案。 |
| ✅ Our proposed framework consists of a main toon shading pipeline and an editing branch. | ✅ 我们提出的框架由一个主卡通着色管道和一个编辑分支组成。 |
| ✅ In the main toon shading pipeline, we construct a multi-module denoising model based on an anime-style diffusion model. | ✅ 在主卡通着色管道中，我们基于动漫风格扩散模型构建了多模块去噪模型。 |
| ✅ ControlNet ( **Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.** ) and AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) are utilized in the denoising model to address controllability and consistency issues. | ✅ 去噪模型采用 ControlNet ( **Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.** ) 和 AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) 来解决可控性和一致性问题。 |
| ✅ To enable the generation of ultra-high-resolution content in long videos, we depart from the conventional frame-by-frame generation paradigm. | ✅ 为了能够生成长视频中的超高分辨率内容，我们摆脱了传统的逐帧生成范式。 |
| ✅ Instead, we adopt a sliding window approach to iteratively update the latent embedding of each frame. | ✅ 相反，我们采用滑动窗口方法来迭代更新每个帧的潜在嵌入。 |
| ✅ Additionally, our method offers the capability to edit videos through the editing branch, which provides editing signals for the main toon shading pipeline. | ✅ 此外，我们的方法还提供通过编辑分支编辑视频的功能，该分支为主卡通着色管道提供编辑信号。 |
| ✅ To improve the efficiency, we incorporate flash attention ( **Dao et al. (2022)  Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.   Flashattention: Fast and memory-efficient exact attention with io-awareness.   Advances in Neural Information Processing Systems 35 (2022), 16344–16359.** ) into the attention mechanisms, effectively mitigating excessive GPU memory usage. | ✅ 为了提高效率，我们将闪存注意力 ( **Dao et al. (2022)  Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.   Flashattention: Fast and memory-efficient exact attention with io-awareness.   Advances in Neural Information Processing Systems 35 (2022), 16344–16359.** ) 纳入注意力机制，有效缓解过多的 GPU 内存使用。 |
| ✅ Remarkably, our approach can directly handle resolutions of up to  $1536\times 1536$ . | ✅ 值得注意的是，我们的方法可以直接处理高达  $1536\times 1536$  的分辨率。 |
| ✅ In our experiments, we first evaluate Diffutoon in the toon shading task, and then we evaluate the capability of editing some content according to given prompts. | ✅ 在我们的实验中，我们首先在卡通着色任务中评估 Diffutoon，然后评估根据给定的提示编辑某些内容的能力。 |
| ✅ Comparative analyses are conducted with both open-source and closed-source approaches. | ✅ 采用开源和闭源方法进行比较分析。 |
| ✅ Quantitative assessments and human evaluations consistently demonstrate the significant advantages of our approach over other methods. | ✅ 定量评估和人工评价始终证明我们的方法相对于其他方法具有显著的优势。 |
| ✅ The contribution of this paper includes: | ✅ 本文的贡献包括： |

| 【第1节，第5段】原文 | 【第1节，第5段】翻译 |
| ---- | ---- |
| ✅ We introduce an innovative form of toon shading, aiming to release the potential of generative diffusion models in the field of non-photorealistic rendering. | ✅ 我们引入了一种创新形式的卡通着色，旨在释放生成扩散模型在非真实感渲染领域的潜力。 |

| 【第1节，第6段】原文 | 【第1节，第6段】翻译 |
| ---- | ---- |
| ✅ We propose an effective toon shading approach based on diffusion models, making it possible to transform photorealistic videos into an anime style and edit the content according to given prompts if required. | ✅ 我们提出了一种基于扩散模型的有效卡通着色方法，可以将真实感十足的视频转换为动漫风格，并根据需要根据给定的提示编辑内容。 |

| 【第1节，第7段】原文 | 【第1节，第7段】翻译 |
| ---- | ---- |
| ✅ Our implementation presents a robust framework for deploying diffusion models in video processing. | ✅ 我们的实现为在视频处理中部署扩散模型提供了一个强大的框架。 |
| ✅ This framework can achieve very high resolution and is capable of processing long videos. | ✅ 该框架可以实现非常高的分辨率，并且能够处理长视频。 |

## 2 Related Work

### 2.1 Stable Diffusion

| 【第2.1节，第1段】原文 | 【第2.1节，第1段】翻译 |
| ---- | ---- |
| ✅ Stable Diffusion ( **High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.** ) has emerged as a popular foundational backbone within both the academic and open-source communities. | ✅ 稳定扩散 ( **High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.** ) 已经成为学术界和开源社区中流行的基础支柱。 |
| ✅ Its structure includes a text encoder ( **Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.** ) , a UNet ( **U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.** ) , and a VAE ( **Kingma and Welling (2013)  Diederik P Kingma and Max Welling. 2013.   Auto-encoding variational bayes.   arXiv preprint arXiv:1312.6114 (2013).** ). | ✅ 其结构包括文本编码器( **Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.** )、UNet ( **U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.** )和VAE ( **Kingma and Welling (2013)  Diederik P Kingma and Max Welling. 2013.   Auto-encoding variational bayes.   arXiv preprint arXiv:1312.6114 (2013).** )。 |
| ✅ To leverage Stable Diffusion models effectively for toon shading applications, a specialized anime-style image generation model tailored for image-to-image processing is essential. | ✅ 为了有效地利用稳定扩散模型进行卡通着色应用，专门针对图像到图像处理的动漫风格图像生成模型至关重要。 |
| ✅ By employing advanced training methods such as LoRA ( **LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.** ) , Textual Inversion ( **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In The Eleventh International Conference on Learning Representations.** ) , DreamBooth ( **Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22500–22510.** ) , and others, we can easily fine-tune a personalized model. | ✅ 通过采用 LoRA ( **LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.** )、文本反转 ( **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In The Eleventh International Conference on Learning Representations.** )、DreamBooth ( **Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22500–22510.** ) 等先进的训练方法，我们可以轻松地微调个性化模型。 |
| ✅ Additionally, the utilization of prompt engineering techniques ( **BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track. 1–11.** ) allows for the refinement of prompts, thereby enabling the generation of high-aesthetic images. | ✅ 此外，利用提示工程技术( **BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track. 1–11.** )可以对提示进行细化，从而生成高美感的图像。 |

### 2.2 Fast Sampling of Diffusion Models

| 【第2.2节，第1段】原文 | 【第2.2节，第1段】翻译 |
| ---- | ---- |
| ✅ Diffusion models typically require multiple iterative steps to generate clear images, making their generation speed comparatively slower than that of GANs ( **Goodfellow et al. (2014)  Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014.   Generative adversarial nets.   Advances in neural information processing systems 27 (2014).** ). | ✅ 扩散模型通常需要多次迭代步骤才能生成清晰的图像，因此其生成速度比 GANs ( **Goodfellow et al. (2014)  Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014.   Generative adversarial nets.   Advances in neural information processing systems 27 (2014).** ) 相对较慢。 |
| ✅ In video processing, where each frame needs to be processed, the issue of computational efficiency becomes even more significant. | ✅ 在视频处理中，每一帧都需要处理，计算效率的问题变得更加重要。 |
| ✅ Some studies ( **1. Denoising Diffusion Implicit Models. In International Conference on Learning Representations.** ｜ **2. Lu et al. (2022)  Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022.   Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps.   Advances in Neural Information Processing Systems 35 (2022), 5775–5787.** ｜ **3. Optimal Linear Subspace Search: Learning to Construct Fast and High-Quality Schedulers for Diffusion Models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 463–472.** ) have addressed this by introducing schedulers to control the generation process, making it possible to generate clear images in a few steps. | ✅ 一些研究( **1. Denoising Diffusion Implicit Models. In International Conference on Learning Representations.** ｜ **2. Lu et al. (2022)  Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022.   Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps.   Advances in Neural Information Processing Systems 35 (2022), 5775–5787.** ｜ **3. Optimal Linear Subspace Search: Learning to Construct Fast and High-Quality Schedulers for Diffusion Models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 463–472.** )通过引入调度程序来控制生成过程解决了这个问题，使得只需几个步骤就可以生成清晰的图像。 |
| ✅ In high-resolution image generation, although some existing research ( **1. Jin et al. (2023)  Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. 2023.   Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis.   arXiv preprint arXiv:2306.08645 (2023).** ｜ **2. He et al. (2023)  Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. 2023.   ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models.   arXiv preprint arXiv:2310.07702 (2023).** ) has showcased the feasibility of transferring low-resolution models to high-resolution tasks, the computational load of attention layers in high-resolution image generation remains a concern. | ✅ 在高分辨率图像生成中，虽然现有的一些研究( **1. Jin et al. (2023)  Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. 2023.   Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis.   arXiv preprint arXiv:2306.08645 (2023).** ｜ **2. He et al. (2023)  Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. 2023.   ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models.   arXiv preprint arXiv:2310.07702 (2023).** )已经展示了将低分辨率模型转移到高分辨率任务的可行性，但高分辨率图像生成中注意力层的计算负荷仍然是一个问题。 |
| ✅ To alleviate this issue, efficient attention implementations such as flash attention ( **Dao et al. (2022)  Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.   Flashattention: Fast and memory-efficient exact attention with io-awareness.   Advances in Neural Information Processing Systems 35 (2022), 16344–16359.** ) have reduced the memory and time requirements, enabling the processing of high-resolution videos. | ✅ 为了缓解这个问题，高效的注意力实现（例如 flash 注意力 ( **Dao et al. (2022)  Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.   Flashattention: Fast and memory-efficient exact attention with io-awareness.   Advances in Neural Information Processing Systems 35 (2022), 16344–16359.** )）已经减少了内存和时间要求，从而能够处理高分辨率视频。 |

### 2.3 Controllable Image Synthesis

| 【第2.3节，第1段】原文 | 【第2.3节，第1段】翻译 |
| ---- | ---- |
| ✅ To enhance the controllability of the generated results in diffusion models, recent studies such as ControlNet ( **Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.** ) and T2I-Adapter ( **Mou et al. (2023)  Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023.   T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models.   arXiv preprint arXiv:2302.08453 (2023).** ) aim to integrate control signals into the generation process. | ✅ 为了增强扩散模型中生成结果的可控性，最近的研究如 ControlNet ( **Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.** ) 和 T2I-Adapter ( **Mou et al. (2023)  Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023.   T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models.   arXiv preprint arXiv:2302.08453 (2023).** ) 旨在将控制信号集成到生成过程中。 |
| ✅ By connecting controlling modules in the form of adapters to the UNet, we can construct a robust image-to-image pipeline and selectively retain information from the original image. | ✅ 通过将控制模块以适配器的形式连接到 UNet，我们可以构建一个强大的图像到图像管道，并有选择地保留原始图像中的信息。 |
| ✅ The advancements in controllable image-to-image techniques inspired the studies in video-to-video generation. | ✅ 可控图像到图像技术的进步激发了视频到视频生成的研究。 |
| ✅ For instance, Gen-1 ( **Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ) decomposes video information into structural and content components. | ✅ 例如，Gen-1 ( **Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ) 将视频信息分解为结构和内容组件。 |
| ✅ It leverages depth estimation ( **Ranftl et al. (2020)  René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. 2020.   Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer.   IEEE transactions on pattern analysis and machine intelligence 44, 3 (2020), 1623–1637.** ) to represent the structural details and synthesize a stylized video. | ✅ 它利用深度估计 ( **Ranftl et al. (2020)  René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. 2020.   Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer.   IEEE transactions on pattern analysis and machine intelligence 44, 3 (2020), 1623–1637.** ) 来表示结构细节并合成风格化的视频。 |
| ✅ In this paper, we reference and adopt similar controlling strategies in our proposed method. | ✅ 在本文中，我们在提出的方法中参考并采用了类似的控制策略。 |

### 2.4 Temporal Diffusion Models

| 【第2.4节，第1段】原文 | 【第2.4节，第1段】翻译 |
| ---- | ---- |
| ✅ The primary challenge in the application of diffusion models to video processing is consistency. | ✅ 扩散模型在视频处理中的应用面临的主要挑战是一致性。 |
| ✅ The conventional practice of independently processing each frame invariably results in video flickering. | ✅ 独立处理每一帧的传统做法必然会导致视频闪烁。 |
| ✅ Some studies ( **1. Khachatryan et al. (2023)  Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023.   Text2video-zero: Text-to-image diffusion models are zero-shot video generators.   arXiv preprint arXiv:2303.13439 (2023).** ｜ **2. Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ) address this issue by incorporating special mechanisms, such as cross-frame attention, which aligns the content of adjacent frames without the need for training. | ✅ 一些研究 ( **1. Khachatryan et al. (2023)  Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023.   Text2video-zero: Text-to-image diffusion models are zero-shot video generators.   arXiv preprint arXiv:2303.13439 (2023).** ｜ **2. Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ) 通过结合特殊机制来解决这个问题，例如跨帧注意，它可以对齐相邻帧的内容而无需训练。 |
| ✅ Other studies ( **1. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.** ｜ **2. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ｜ **3. Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) tackle the consistency problem by introducing trainable modules and training them on video datasets. | ✅ 其他研究 ( **1. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.** ｜ **2. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ｜ **3. Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) 通过引入可训练模块并在视频数据集上进行训练来解决一致性问题。 |
| ✅ Among these studies, AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) , being compatible with Stable Diffusion architecture, has gained significant popularity within open-source communities. | ✅ 在这些研究中，AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) 与稳定扩散架构兼容，在开源社区中获得了广泛的欢迎。 |
| ✅ In our methodology, we utilize motion modules based on AnimateDiff to enhance the overall coherence. | ✅ 在我们的方法中，我们利用基于 AnimateDiff 的运动模块来增强整体连贯性。 |

### 2.5 Post-Processing Methods

| 【第2.5节，第1段】原文 | 【第2.5节，第1段】翻译 |
| ---- | ---- |
| ✅ Training diffusion models on long videos still faces challenges due to the high computational resource requirements. | ✅ 由于对计算资源的要求较高，在长视频上训练扩散模型仍然面临挑战。 |
| ✅ Some video post-processing approaches can be employed to assist in enhancing the long-term consistency of videos. | ✅ 可以采用一些视频后处理方法来帮助增强视频的长期一致性。 |
| ✅ For instance, CoDeF ( **Ouyang et al. (2023)  Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2023.   Codef: Content deformation fields for temporally consistent video processing.   arXiv preprint arXiv:2308.07926 (2023).** ) , FastBlend ( **Duan et al. (2023b)  Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, Jun Huang, and Mingyi Jin. 2023b.   FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier.   arXiv preprint arXiv:2311.09265 (2023).** ) , and other blind video deflickering algorithms ( **Blind Video Deflickering by Neural Filtering with a Flawed Atlas. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10439–10448.** ). | ✅ 例如，CoDeF ( **Ouyang et al. (2023)  Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2023.   Codef: Content deformation fields for temporally consistent video processing.   arXiv preprint arXiv:2308.07926 (2023).** )、FastBlend ( **Duan et al. (2023b)  Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, Jun Huang, and Mingyi Jin. 2023b.   FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier.   arXiv preprint arXiv:2311.09265 (2023).** )，以及其他盲视频去闪烁算法( **Blind Video Deflickering by Neural Filtering with a Flawed Atlas. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10439–10448.** )。 |
| ✅ While these methods can handle longer videos, they typically encounter issues such as screen tearing and blurring when dealing with high-speed and substantial motion. | ✅ 虽然这些方法可以处理较长的视频，但在处理高速和大量运动时，它们通常会遇到屏幕撕裂和模糊等问题。 |
| ✅ The method proposed in this paper draws inspiration from such approaches to improve long-term consistency. | ✅ 本文提出的方法从此类方法中汲取灵感，以提高长期一致性。 |

## 3 Methodology

![figure](https://ar5iv.labs.arxiv.org/html/2401.16224/assets/x1.png)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 1.  The overall architecture of Diffutoon, where the top part is the main toon shading pipeline, and the bottom part is the editing branch. | ✅ Figure 1.  Diffutoon 的整体架构，其中顶部是主卡通着色管道，底部是编辑分支。 |
| ✅ The editing branch can generate editing signals in the format of color video for the main toon shading pipeline. | ✅ 编辑分支可以为主卡通着色流水线生成彩色视频格式的编辑信号。 |

| 【第3节，第1段】原文 | 【第3节，第1段】翻译 |
| ---- | ---- |
| ✅ The overall architecture of Diffutoon is illustrated in Figure 1. | ✅ Diffutoon 的整体架构如图 1 所示。 |
| ✅ The whole approach consists of a main toon shading pipeline and an editing branch. | ✅ 整个方法由一个主卡通着色管道和一个编辑分支组成。 |
| ✅ The main toon shading pipeline can render the input video in an anime style. | ✅ 主卡通着色管道可以以动漫风格渲染输入视频。 |
| ✅ To enable anime video editing, we designed an additional editing branch to generate an edited color video for the main toon shading pipeline. | ✅ 为了实现动漫视频编辑，我们设计了一个额外的编辑分支，为主卡通着色管道生成编辑后的彩色视频。 |

### 3.1 Toon Shading

| 【第3.1节，第1段】原文 | 【第3.1节，第1段】翻译 |
| ---- | ---- |
| ✅ We divide the toon shading task into four subtasks: stylization, consistency enhancement, structure guidance, and colorization. | ✅ 我们将卡通着色任务分为四个子任务：风格化、一致性增强、结构指导和着色。 |
| ✅ We employ four models to address the four subtasks respectively. | ✅ 我们采用四种模型分别解决四个子任务。 |

| 【第3.1节，第2段】原文 | 【第3.1节，第2段】翻译 |
| ---- | ---- |
| ✅ Stylization : we leverage a personalized Stable Diffusion ( **High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.** ) model for anime stylization. | ✅ Stylization：我们利用个性化的稳定扩散 ( **High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.** ) 模型进行动漫风格化。 |
| ✅ Theoretically, our approach supports every open-sourced diffusion model with such model architecture. | ✅ 理论上，我们的方法支持每个具有这种模型架构的开源传播模型。 |

| 【第3.1节，第3段】原文 | 【第3.1节，第3段】翻译 |
| ---- | ---- |
| ✅ Consistency enhancement : to enhance the temporal consistency, we employ several motion modules in our approach. | ✅ Consistency enhancement：为了增强时间一致性，我们在方法中采用了多个运动模块。 |
| ✅ These modules are based on AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ) , which are inserted into the UNet to keep the content consistent. | ✅ 这些模块基于 AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** )，插入到 UNet 中以保持内容一致。 |

| 【第3.1节，第4段】原文 | 【第3.1节，第4段】翻译 |
| ---- | ---- |
| ✅ Structure guidance : we extract the outline information from the input video and use a ControlNet model to retain the outline information during the generation process. | ✅ Structure guidance：我们从输入视频中提取轮廓信息，并使用ControlNet模型在生成过程中保留轮廓信息。 |
| ✅ Unlike some existing methods ( **Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ) that use depth estimation to represent structural information, we employ outline as structural information, which is more suitable for rendering flat-style animations. | ✅ 与一些现有的使用深度估计来表示结构信息的方法 ( **Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ) 不同，我们使用轮廓作为结构信息，这更适合渲染扁平风格的动画。 |

| 【第3.1节，第5段】原文 | 【第3.1节，第5段】翻译 |
| ---- | ---- |
| ✅ Colorization : we use another ControlNet model for colorization. | ✅ Colorization：我们使用另一个 ControlNet 模型进行着色。 |
| ✅ This model is trained for super-resolution tasks, thus it can improve the overall video quality even if the input video is in low resolution. | ✅ 该模型针对超分辨率任务进行训练，因此即使输入视频分辨率较低，也可以提高整体视频质量。 |
| ✅ This model directly processes the input video in the main toon shading pipeline, and it takes the edited color video as input when the editing branch is enabled. | ✅ 该模型直接在主卡通着色管道中处理输入视频，并在启用编辑分支时将编辑后的彩色视频作为输入。 |

| 【第3.1节，第6段】原文 | 【第3.1节，第6段】翻译 |
| ---- | ---- |
| ✅ As illustrated in the top part of Figure 1 , the main toon shading pipeline involves several key steps. | ✅ 如图 1 顶部所示，主卡通着色管道涉及几个关键步骤。 |
| ✅ Given an input video containing  $N$  frames  $\{\boldsymbol{v}^{1},\boldsymbol{v}^{2},\cdots,\boldsymbol{v}^{N}\}$  , we first generate a structural video and a color video. | ✅ 给定一个包含  $N$  帧  $\{\boldsymbol{v}^{1},\boldsymbol{v}^{2},\cdots,\boldsymbol{v}^{N}\}$  的输入视频，我们首先生成一个结构视频和一个彩色视频。 |
| ✅ The structural video  $\{\boldsymbol{o}^{1},\boldsymbol{o}^{2},\cdots,\boldsymbol{o}^{N}\}$  contains the outline information extracted from the input video, and the color video  $\{\boldsymbol{c}^{1},\boldsymbol{c}^{2},\cdots,\boldsymbol{c}^{N}\}$  is the input video when the editing branch is disabled. | ✅ 结构视频 $\{\boldsymbol{o}^{1},\boldsymbol{o}^{2},\cdots,\boldsymbol{o}^{N}\}$ 包含从输入视频中提取的轮廓信息，彩色视频 $\{\boldsymbol{c}^{1},\boldsymbol{c}^{2},\cdots,\boldsymbol{c}^{N}\}$ 是编辑分支被禁用时的输入视频。 |
| ✅ Subsequently, the two videos serve as inputs to their respective ControlNet models, which in turn produce conditioning signals for the UNet. | ✅ 随后，这两个视频作为各自的 ControlNet 模型的输入，进而为 UNet 产生调节信号。 |
| ✅ Simultaneously, the motion modules generate temporal signals. | ✅ 同时，运动模块产生时间信号。 |
| ✅ These four models constitute a large denoising model  $\mathcal{E}$  , employed iteratively to synthesize a visually consistent video. | ✅ 这四个模型构成一个大型去噪模型  $\mathcal{E}$ ，迭代用于合成视觉一致的视频。 |

| 【第3.1节，第7段】原文 | 【第3.1节，第7段】翻译 |
| ---- | ---- |
| ✅ In the denoising process, initially, the latent embedding of each frame is sampled from a Gaussian distribution. | ✅ 在去噪过程中，最初，从高斯分布中对每个帧的潜在嵌入进行采样。 |

**公式(1):** 
$$ \boldsymbol{x}_{T}=\{\boldsymbol{x}^{i}_{T}\}_{i=1}^{N}\sim\mathcal{N}(\boldsymbol{O},\boldsymbol{I}) $$

| 【第3.1节，第8段】原文 | 【第3.1节，第8段】翻译 |
| ---- | ---- |
| ✅ where  $T$  is the number of iterative steps and each embedding is independent identically distributed. | ✅ 其中  $T$  是迭代步数，并且每个嵌入都是独立同分布的。 |
| ✅ At each denoising step, we use classifier-free guidance ( **Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.** ) to build a textual guidance mechanism, which consists of a positive side and a negative side. | ✅ 在每个去噪步骤中，我们使用无分类器指导 ( **Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.** ) 来构建文本指导机制，该机制由正向指导和负向指导组成。 |
| ✅ On the positive side, we use some empirical keywords (e.g., “best quality”, “perfect anime illustration”) as prompt  $\tau$  for better aesthetics. | ✅ 从积极的一面来看，我们使用一些经验关键词（例如“最佳质量”、“完美的动漫插图”）作为提示  $\tau$ ，以获得更好的美感。 |
| ✅ Note that the motion modules are trained within 32 consecutive frames, we can only use the denoising model  $\mathcal{E}$  in a sliding window with a size no larger than 32. | ✅ 请注意，运动模块是在 32 个连续帧内进行训练的，我们只能在大小不大于 32 的滑动窗口中使用去噪模型  $\mathcal{E}$ 。 |
| ✅ The sliding windows with size  $d$  and stride  $s$  are | ✅ 大小为  $d$ 、步幅为  $s$  的滑动窗口为 |

**公式(2):** 
$$ \mathcal{W}(d,s)=\left\{[i,i+d-1]:1\leq i\leq N,i\equiv 1(\text{mod }s)\right\} $$

| 【第3.1节，第9段】原文 | 【第3.1节，第9段】翻译 |
| ---- | ---- |
| ✅ where  $s<d$  for a smooth transition between different sliding windows. | ✅ 其中 $s<d$ 表示不同滑动窗口之间的平滑过渡。 |
| ✅ In a sliding window  $[l,r]$  , The model output on the positive side is | ✅ 在滑动窗口  $[l,r]$  中，正侧的模型输出为 |

**公式(3):** 
$$ \left\{\boldsymbol{e}_{t,+}(l,i,r)\right\}_{i=l}^{r}=\mathcal{E}\Big{(}\left\{\boldsymbol{x}_{t}^{i}\right\}_{i=l}^{r},\left\{\boldsymbol{o}_{t}^{i}\right\}_{i=l}^{r},\left\{\boldsymbol{c}_{t}^{i}\right\}_{i=l}^{r},t,\tau\Big{)} $$

| 【第3.1节，第10段】原文 | 【第3.1节，第10段】翻译 |
| ---- | ---- |
| ✅ The latent embeddings are initially stored in RAM and will be moved to GPU memory when the sliding window covers them. | ✅ 潜在嵌入最初存储在 RAM 中，当滑动窗口覆盖它们时，它们将被移动到 GPU 内存中。 |
| ✅ We adopt a linear combination of overlapping segments from different sliding windows. | ✅ 我们采用来自不同滑动窗口的重叠段的线性组合。 |

**公式(4):** 
$$ \overline{\boldsymbol{e}}_{t,+}(i)=\sum_{(l,r)\in\mathcal{W}(d,s)}\frac{w(l,i,r)}{\sum_{(l^{\prime},r^{\prime})\in\mathcal{W}(d,s)}w(l^{\prime},i,r^{\prime})}\boldsymbol{e}_{t,+}(l,i,r) $$

| 【第3.1节，第11段】原文 | 【第3.1节，第11段】翻译 |
| ---- | ---- |
| ✅ The weight  $w(l,i,r)$  is formulated as follows: | ✅ 权重 $w(l,i,r)$ 的计算公式如下： |

**公式(5):** 
$$ w(l,i,r)=\begin{cases}1+\epsilon-\left|i-\frac{l+r}{2}\right|/\frac{r-l}{2},&\text{if }l\leq i\leq r,\\
0,&\text{otherwise},\end{cases} $$

| 【第3.1节，第12段】原文 | 【第3.1节，第12段】翻译 |
| ---- | ---- |
| ✅ where  $\epsilon=10^{-2}$  is the minimum weight of tailed frames. | ✅ 其中  $\epsilon=10^{-2}$  是尾部帧的最小权重。 |
| ✅ This allows the information from each frame to be shared with other frames throughout the generation process. | ✅ 这使得整个生成过程中每个帧的信息都可以与其他帧共享。 |
| ✅ This mechanism implicitly implements a large size of sliding window, enhancing the long-term consistency of generated content. | ✅ 该机制隐式实现了大尺寸的滑动窗口，增强了生成内容的长期一致性。 |
| ✅ To avoid disintegrated parts on faces and hands, we employ a textual inversion  $\tau^{\prime}$    ( **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In The Eleventh International Conference on Learning Representations.** ) on the negative side, which involves 10 token embeddings to be processed by the text encoder. | ✅ 为了避免脸部和手部部位破碎，我们在负片上采用了文本反转  $\tau^{\prime}$    ( **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In The Eleventh International Conference on Learning Representations.** )，这涉及文本编码器要处理的 10 个标记嵌入。 |
| ✅ By replacing  $\tau$  with  $\tau^{\prime}$  in ( 3 ) and ( 4 ), we can obtain the estimated noise on the negative side  $\overline{\boldsymbol{e}}_{t,-}(i)$ . | ✅ 通过在（3）和（4）中用 $\tau^{\prime}$ 替换 $\tau$ ，我们可以获得负侧 $\overline{\boldsymbol{e}}_{t,-}(i)$ 的估计噪声。 |
| ✅ Then, the guided estimated noise is | ✅ 然后，引导估计噪声为 |

**公式(6):** 
$$ \overline{\boldsymbol{e}}_{t}(i)=g\cdot\overline{\boldsymbol{e}}_{t,+}(i)+(1-g)\cdot\overline{\boldsymbol{e}}_{t,-}(i) $$

| 【第3.1节，第13段】原文 | 【第3.1节，第13段】翻译 |
| ---- | ---- |
| ✅ The classifier-free guidance scale  $g$  is set to 7 by default. | ✅ 无分类器指导量表 $g$ 默认设置为7。 |
| ✅ Based on empirical evidence, we skip the final attention layer of the text encoder, which can improve the visual quality slightly. | ✅ 根据经验证据，我们跳过文本编码器的最终注意层，这可以稍微提高视觉质量。 |
| ✅ The overall estimated noise of the whole video is | ✅ 整个视频的整体估计噪声为 |

**公式(7):** 
$$ \overline{\boldsymbol{e}}_{t}=\big{(}\overline{\boldsymbol{e}}(0),\overline{\boldsymbol{e}}(1),\cdots,\overline{\boldsymbol{e}}(n)\big{)}\in\mathbb{R}^{N\times H\times W\times C} $$

| 【第3.1节，第14段】原文 | 【第3.1节，第14段】翻译 |
| ---- | ---- |
| ✅ After that, we utilize a DDIM ( **Denoising Diffusion Implicit Models. In International Conference on Learning Representations.** ) scheduler to control the generation process. | ✅ 之后，我们利用 DDIM ( **Denoising Diffusion Implicit Models. In International Conference on Learning Representations.** ) 调度程序来控制生成过程。 |

**公式(8):** 
$$ \boldsymbol{x}_{t-1}=\sqrt{\alpha_{t-1}}\left(\frac{\boldsymbol{x}_{t}-\sqrt{1-\alpha_{t}}\overline{\boldsymbol{e}}_{t}}{\sqrt{\alpha_{t}}}\right)+\sqrt{1-\alpha_{t-1}}\overline{\boldsymbol{e}}_{t} $$

| 【第3.1节，第15段】原文 | 【第3.1节，第15段】翻译 |
| ---- | ---- |
| ✅ where  $\alpha_{t}$  is the hyper-parameter that determines how much noise it contains in step  $t$ . | ✅ 其中 $\alpha_{t}$ 是决定步骤 $t$ 中包含多少噪音的超参数。 |
| ✅ We follow the implementation of DDIM in AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** ). | ✅ 我们遵循AnimateDiff ( **Guo et al. (2023)  Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.   Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.   arXiv preprint arXiv:2307.04725 (2023).** )中DDIM的实现。 |
| ✅ Despite the findings from recent studies suggesting that alternative schedulers, such as DPM-Solver ( **Lu et al. (2022)  Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022.   Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps.   Advances in Neural Information Processing Systems 35 (2022), 5775–5787.** ) and OLSS ( **Optimal Linear Subspace Search: Learning to Construct Fast and High-Quality Schedulers for Diffusion Models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 463–472.** ) , can achieve superior visual quality within a specified number of steps, we decide to employ such a straightforward scheduler due to memory constraints. | ✅ 尽管最近的研究结果表明替代调度程序（例如 DPM-Solver ( **Lu et al. (2022)  Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022.   Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps.   Advances in Neural Information Processing Systems 35 (2022), 5775–5787.** ) 和 OLSS ( **Optimal Linear Subspace Search: Learning to Construct Fast and High-Quality Schedulers for Diffusion Models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 463–472.** )）可以在指定的步骤数内实现卓越的视觉质量，但由于内存限制，我们决定采用这种简单的调度程序。 |
| ✅ This decision is driven by the fact that these alternative schedulers need to store all latent tensors throughout the generation process, posing challenges for processing long videos. | ✅ 这一决定是基于这样一个事实：这些替代调度程序需要在整个生成过程中存储所有潜在张量，这对处理长视频提出了挑战。 |
| ✅ Additionally, we set the number of denoising steps  $T$  to only 10 for faster generation without compromising the resulting quality. | ✅ 此外，我们将去噪步骤数  $T$  设置为仅 10，以实现更快的生成速度，同时不影响最终质量。 |

### 3.2 Adding Editing Signals to Toon Shading

| 【第3.2节，第1段】原文 | 【第3.2节，第1段】翻译 |
| ---- | ---- |
| ✅ In the main toon shading pipeline, we decompose the information in the input video into outlines and colors. | ✅ 在主卡通着色管道中，我们将输入视频中的信息分解为轮廓和颜色。 |
| ✅ In practice, we can edit the content by modifying the outline video or color video. | ✅ 在实际操作中，我们可以通过修改轮廓视频或者彩色视频来编辑内容。 |
| ✅ Notably, due to the lack of reliable video editing methods for structural information, we mainly focus on editing the color information. | ✅ 值得注意的是，由于缺乏可靠的结构信息视频编辑方法，我们主要关注编辑颜色信息。 |
| ✅ We observe that the ControlNet model used for processing color videos can assist the UNet in generating high-quality videos, even if the color videos are blurry. | ✅ 我们观察到，用于处理彩色视频的 ControlNet 模型可以协助 UNet 生成高质量的视频，即使彩色视频很模糊。 |
| ✅ This noteworthy insight implies a robust fault tolerance within our approach to video editing methods. | ✅ 这一值得注意的见解意味着我们的视频编辑方法具有强大的容错能力。 |
| ✅ Consequently, we are motivated to design a dedicated branch to support video editing. | ✅ 因此，我们有动力设计一个专门的分支来支持视频编辑。 |

| 【第3.2节，第2段】原文 | 【第3.2节，第2段】翻译 |
| ---- | ---- |
| ✅ To achieve this, we add an editing branch to generate text-guided editing signals for the main toon shading pipeline, where the editing signal is passed in the format of a color video. | ✅ 为了实现这一点，我们添加了一个编辑分支，为主卡通着色管道生成文本引导的编辑信号，其中编辑信号以彩色视频的格式传递。 |
| ✅ The architecture of the editing branch is shown in the bottom part of Figure 1. | ✅ 编辑分支的架构如图1的底部所示。 |
| ✅ Similar to the main toon shading pipeline, we divide the synthesis of the editing signal into four subtasks: | ✅ 与主卡通着色管道类似，我们将编辑信号的合成分为四个子任务： |

| 【第3.2节，第3段】原文 | 【第3.2节，第3段】翻译 |
| ---- | ---- |
| ✅ Stylization : we leverage the same Stable Diffusion model as that in the main toon shading pipeline. | ✅ Stylization：我们利用与主卡通着色管道中相同的稳定扩散模型。 |

| 【第3.2节，第4段】原文 | 【第3.2节，第4段】翻译 |
| ---- | ---- |
| ✅ Consistency enhancement : we use cross-frame attention and FastBlend ( **Duan et al. (2023b)  Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, Jun Huang, and Mingyi Jin. 2023b.   FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier.   arXiv preprint arXiv:2311.09265 (2023).** ) to improve consistency. | ✅ Consistency enhancement：我们使用跨帧注意力和 FastBlend ( **Duan et al. (2023b)  Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, Jun Huang, and Mingyi Jin. 2023b.   FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier.   arXiv preprint arXiv:2311.09265 (2023).** ) 来提高一致性。 |
| ✅ While the motion modules based on AnimateDiff can make the video fluent, there are instances where they compromise visual quality. | ✅ 基于AnimateDiff的运动模块虽然可以使视频流畅，但在某些情况下它们会影响视觉质量。 |
| ✅ This pitfall is due to their reliance on a modified DDIM scheduler, which will be further discussed in the following experiments. | ✅ 这个缺陷是由于他们依赖于修改后的 DDIM 调度程序，这将在接下来的实验中进一步讨论。 |
| ✅ This is also the reason why a single editing branch cannot synthesize a high-quality video. | ✅ 这也是单一剪辑部门无法合成高质量视频的原因。 |
| ✅ To release the potential of the diffusion model itself, we use the DDIM scheduler consistent with its training process, omitting these motion modules. | ✅ 为了释放扩散模型本身的潜力，我们使用与其训练过程一致的 DDIM 调度程序，省略了这些运动模块。 |
| ✅ Instead, we leverage cross-frame attention and FastBlend to improve consistency, where cross-frame attention is a widely demonstrated effective technique ( **1. Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ｜ **2. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.** ) , and FastBlend is a model-free deflickering approach for post-processing. | ✅ 相反，我们利用跨帧注意力和 FastBlend 来提高一致性，其中跨帧注意力是一种被广泛证明的有效技术 ( **1. Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ｜ **2. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.** )，而 FastBlend 是一种用于后期处理的无模型去闪烁方法。 |

| 【第3.2节，第5段】原文 | 【第3.2节，第5段】翻译 |
| ---- | ---- |
| ✅ Structure guidance : we employ depth estimation ( **Ranftl et al. (2020)  René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. 2020.   Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer.   IEEE transactions on pattern analysis and machine intelligence 44, 3 (2020), 1623–1637.** ) and softedge ( **Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision. 1395–1403.** ) to represent the structural information and use two ControlNet models for precise structure guidance. | ✅ Structure guidance：我们采用深度估计( **Ranftl et al. (2020)  René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. 2020.   Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer.   IEEE transactions on pattern analysis and machine intelligence 44, 3 (2020), 1623–1637.** )和软边缘( **Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision. 1395–1403.** )来表示结构信息，并使用两个ControlNet模型进行精确的结构指导。 |
| ✅ Previous studies ( **1. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ｜ **2. Duan et al. (2023c)  Zhongjie Duan, Lizhou You, Chengyu Wang, Cen Chen, Ziheng Wu, Weining Qian, Jun Huang, Fei Chao, and Rongrong Ji. 2023c.   DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis.   arXiv preprint arXiv:2308.03463 (2023).** ) have empirically demonstrated the efficacy of these configurations in preserving structural information, particularly in instances of significant video editing. | ✅ 先前的研究 ( **1. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ｜ **2. Duan et al. (2023c)  Zhongjie Duan, Lizhou You, Chengyu Wang, Cen Chen, Ziheng Wu, Weining Qian, Jun Huang, Fei Chao, and Rongrong Ji. 2023c.   DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis.   arXiv preprint arXiv:2308.03463 (2023).** ) 已经通过实证研究证明了这些配置在保存结构信息方面的有效性，特别是在进行大量视频编辑的情况下。 |

| 【第3.2节，第6段】原文 | 【第3.2节，第6段】翻译 |
| ---- | ---- |
| ✅ Colorization : the color is determined by the given prompt. | ✅ Colorization：颜色由给定的提示决定。 |
| ✅ Note that sometimes the classifier-free guidance mechanism fails to generate the correct color in several frames. | ✅ 请注意，有时无分类器引导机制无法在几帧中生成正确的颜色。 |
| ✅ In such instances, FastBlend serves as a corrective measure, leveraging information from neighboring frames to rectify deficient color. | ✅ 在这种情况下，FastBlend 可作为一种纠正措施，利用相邻帧的信息来纠正缺陷的色彩。 |

| 【第3.2节，第7段】原文 | 【第3.2节，第7段】翻译 |
| ---- | ---- |
| ✅ The other components of the editing branch are similar to those of the main toon shading pipeline. | ✅ 编辑分支的其他组件与主卡通着色管道的组件类似。 |
| ✅ The same sliding window mechanism is applied on this branch. | ✅ 该分支上也应用了同样的滑动窗口机制。 |
| ✅ While the color video synthesized by this branch may exhibit blurring, it maintains a high level of visual coherence, suitable for guiding the main toon shading pipeline to synthesize a high-quality video. | ✅ 该分支合成的彩色视频虽然可能显得模糊，但保持了较高的视觉连贯性，适合引导主卡通着色管道合成高质量的视频。 |

### 3.3 Synthesizing High-Resolution Long Videos

| 【第3.3节，第1段】原文 | 【第3.3节，第1段】翻译 |
| ---- | ---- |
| ✅ We implement Diffutoon based on the DiffSynth framework ( **Duan et al. (2023c)  Zhongjie Duan, Lizhou You, Chengyu Wang, Cen Chen, Ziheng Wu, Weining Qian, Jun Huang, Fei Chao, and Rongrong Ji. 2023c.   DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis.   arXiv preprint arXiv:2308.03463 (2023).** ) , which can process the whole video in the latent space. | ✅ 我们基于 DiffSynth 框架 ( **Duan et al. (2023c)  Zhongjie Duan, Lizhou You, Chengyu Wang, Cen Chen, Ziheng Wu, Weining Qian, Jun Huang, Fei Chao, and Rongrong Ji. 2023c.   DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis.   arXiv preprint arXiv:2308.03463 (2023).** ) 实现了 Diffutoon，它可以在潜在空间中处理整个视频。 |
| ✅ To reduce the required GPU memory and improve computational efficiency, we adopt flash attention ( **Dao et al. (2022)  Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.   Flashattention: Fast and memory-efficient exact attention with io-awareness.   Advances in Neural Information Processing Systems 35 (2022), 16344–16359.** ) in all attention layers, including the text encoder, UNet, VAE, ControlNet models, and motion modules. | ✅ 为了减少所需的 GPU 内存并提高计算效率，我们在所有注意力层中采用闪存注意力 ( **Dao et al. (2022)  Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.   Flashattention: Fast and memory-efficient exact attention with io-awareness.   Advances in Neural Information Processing Systems 35 (2022), 16344–16359.** )，包括文本编码器、UNet、VAE、ControlNet 模型和运动模块。 |
| ✅ This memory-efficient attention implementation empowers the direct synthesis of videos in exceptionally high resolution. | ✅ 这种节省内存的注意力机制的实现使得直接合成极高分辨率的视频成为可能。 |
| ✅ The sliding window mechanism is capable of extending the length of videos. | ✅ 滑动窗口机制可以延长视频的长度。 |
| ✅ With the above settings, our pipeline succeeds in synthesizing remarkably detailed, high-resolution, and extended-duration videos. | ✅ 通过上述设置，我们的流程成功合成了细节丰富、分辨率高、时长较长的视频。 |

## 4 Experiments

| 【第4节，第1段】原文 | 【第4节，第1段】翻译 |
| ---- | ---- |
| ✅ Our primary focus centers on the rendering of high-resolution videos with rapid and substantial motion. | ✅ 我们的主要重点是渲染具有快速和大量动作的高分辨率视频。 |
| ✅ To evaluate the efficacy of our proposed approach, we create a dataset comprising 10 videos sourced from a video platform 222https://www.bilibili.com/. | ✅ 为了评估我们提出的方法的有效性，我们创建了一个包含来自视频平台 222https://www.bilibili.com/ 的 10 个视频的数据集。 |
| ✅ This dataset will be released publicly. | ✅ 该数据集将会公开发布。 |
| ✅ In our experiments, we achieve a video resolution of up to  $1536\times 1536$  , resulting in visually impressive frames. | ✅ 在我们的实验中，我们实现了高达  $1536\times 1536$  的视频分辨率，从而获得了视觉上令人印象深刻的帧。 |
| ✅ The detailed settings of models and parameters are presented in the appendix. | ✅ 模型和参数的详细设置见附录。 |

### 4.1 Comparison with Baseline Methods

| 【第4.1节，第1段】原文 | 【第4.1节，第1段】翻译 |
| ---- | ---- |
| ✅ The evaluation involves two distinct tasks: toon shading, where we exclusively employ the main toon shading pipeline to transform input videos into an anime style, and toon shading with editing signals, where manually crafted editing prompts are used to edit the content during the rendering process. | ✅ 评估涉及两个不同的任务：卡通着色，我们专门使用主卡通着色管道将输入视频转换为动漫风格；卡通着色与编辑信号，其中手工制作的编辑提示用于在渲染过程中编辑内容。 |
| ✅ In the two tasks, we conduct comparative evaluations with other state-of-the-art methods, including Rerender-a-video ( **Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ) , an open-source method that utilizes a special pipeline for video synthesis. | ✅ 在这两个任务中，我们与其他最先进的方法进行了比较评估，包括 Rerender-a-video ( **Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** )，这是一种利用特殊管道进行视频合成的开源方法。 |
| ✅ To ensure a fair comparison, we replace the default model of Rerender-a-video with the diffusion model from our approach. | ✅ 为了确保公平比较，我们用我们方法中的扩散模型替换了 Rerender-a-video 的默认模型。 |
| ✅ Additionally, we involve several popular closed-source models that have demonstrated competitiveness in comparison to existing methods. | ✅ 此外，我们还采用了几种流行的闭源模型，与现有方法相比，这些模型已表现出竞争力。 |
| ✅ These models include Gen-1 ( **Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ) and DomoAI ( **DOMO.AI (2024)  Group DOMO.AI. 2024.   DOMO.AI.       https://ai.domo.com/, Last accessed on 2024-01-18.** ). | ✅ 这些模型包括 Gen-1 ( **Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.** ) 和 DomoAI ( **DOMO.AI (2024)  Group DOMO.AI. 2024.   DOMO.AI.       https://ai.domo.com/, Last accessed on 2024-01-18.** )。 |
| ✅ Gen-1, while not specifically tailored for toon shading, is evaluated exclusively in the second task. | ✅ Gen-1 虽然不是专门为卡通着色而设计的，但在第二项任务中专门进行了评估。 |
| ✅ DomoAI offers several models for users on Discord 333https://discord.com/ , and in our experiments, we employ the “Anime v2 - Japanese anime style” version. | ✅ DomoAI 为 Discord 333https://discord.com/ 上的用户提供了多种模型，在我们的实验中，我们采用了“Anime v2 - 日本动漫风格”版本。 |
| ✅ Due to the length limitation of DomoAI, we only use 10 seconds from each video in our experiments. | ✅ 由于 DomoAI 的长度限制，我们在实验中仅使用每个视频的 10 秒。 |
| ✅ This comprehensive comparative analysis aims to evaluate the performance of our approach relative to both open-source and closed-source state-of-the-art methods across diverse tasks. | ✅ 这一全面的比较分析旨在评估我们的方法相对于开源和闭源的最先进的方法在不同任务上的表现。 |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S4.T1.3"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S4.T1.3.4.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T1.3.4.1.1" rowspan="2" style="padding-left:2.5pt;padding-right:2.5pt;">Task</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T1.3.4.1.2" rowspan="2" style="padding-left:2.5pt;padding-right:2.5pt;">Method</th><td class="ltx_td ltx_align_center ltx_border_t" colspan="3" id="S4.T1.3.4.1.3" style="padding-left:2.5pt;padding-right:2.5pt;">Metric</td></tr><tr class="ltx_tr" id="S4.T1.3.3"><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.1.1.1" style="padding-left:2.5pt;padding-right:2.5pt;"><table class="ltx_tabular ltx_align_middle" id="S4.T1.1.1.1.1"><tr class="ltx_tr" id="S4.T1.1.1.1.1.2"><td class="ltx_td ltx_nopad_r ltx_align_center" id="S4.T1.1.1.1.1.2.1" style="padding-left:2.5pt;padding-right:2.5pt;">Aesthetic</td></tr><tr class="ltx_tr" id="S4.T1.1.1.1.1.1"><td class="ltx_td ltx_nopad_r ltx_align_center" id="S4.T1.1.1.1.1.1.1" style="padding-left:2.5pt;padding-right:2.5pt;">score <math alttext="\uparrow" class="ltx_Math" display="inline" id="S4.T1.1.1.1.1.1.1.m1.1"><semantics id="S4.T1.1.1.1.1.1.1.m1.1a"><mo id="S4.T1.1.1.1.1.1.1.m1.1.1" stretchy="false" xref="S4.T1.1.1.1.1.1.1.m1.1.1.cmml">↑</mo><annotation-xml encoding="MathML-Content" id="S4.T1.1.1.1.1.1.1.m1.1b"><ci id="S4.T1.1.1.1.1.1.1.m1.1.1.cmml" xref="S4.T1.1.1.1.1.1.1.m1.1.1">↑</ci></annotation-xml><annotation encoding="application/x-tex" id="S4.T1.1.1.1.1.1.1.m1.1c">\uparrow</annotation></semantics></math></td></tr></table></td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.2.2.2" style="padding-left:2.5pt;padding-right:2.5pt;"><table class="ltx_tabular ltx_align_middle" id="S4.T1.2.2.2.1"><tr class="ltx_tr" id="S4.T1.2.2.2.1.2"><td class="ltx_td ltx_nopad_r ltx_align_center" id="S4.T1.2.2.2.1.2.1" style="padding-left:2.5pt;padding-right:2.5pt;">CLIP</td></tr><tr class="ltx_tr" id="S4.T1.2.2.2.1.1"><td class="ltx_td ltx_nopad_r ltx_align_center" id="S4.T1.2.2.2.1.1.1" style="padding-left:2.5pt;padding-right:2.5pt;">score <math alttext="\uparrow" class="ltx_Math" display="inline" id="S4.T1.2.2.2.1.1.1.m1.1"><semantics id="S4.T1.2.2.2.1.1.1.m1.1a"><mo id="S4.T1.2.2.2.1.1.1.m1.1.1" stretchy="false" xref="S4.T1.2.2.2.1.1.1.m1.1.1.cmml">↑</mo><annotation-xml encoding="MathML-Content" id="S4.T1.2.2.2.1.1.1.m1.1b"><ci id="S4.T1.2.2.2.1.1.1.m1.1.1.cmml" xref="S4.T1.2.2.2.1.1.1.m1.1.1">↑</ci></annotation-xml><annotation encoding="application/x-tex" id="S4.T1.2.2.2.1.1.1.m1.1c">\uparrow</annotation></semantics></math></td></tr></table></td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.3.3.3" style="padding-left:2.5pt;padding-right:2.5pt;"><table class="ltx_tabular ltx_align_middle" id="S4.T1.3.3.3.1"><tr class="ltx_tr" id="S4.T1.3.3.3.1.2"><td class="ltx_td ltx_nopad_r ltx_align_center" id="S4.T1.3.3.3.1.2.1" style="padding-left:2.5pt;padding-right:2.5pt;">Pixel</td></tr><tr class="ltx_tr" id="S4.T1.3.3.3.1.1"><td class="ltx_td ltx_nopad_r ltx_align_center" id="S4.T1.3.3.3.1.1.1" style="padding-left:2.5pt;padding-right:2.5pt;">MSE <math alttext="\downarrow" class="ltx_Math" display="inline" id="S4.T1.3.3.3.1.1.1.m1.1"><semantics id="S4.T1.3.3.3.1.1.1.m1.1a"><mo id="S4.T1.3.3.3.1.1.1.m1.1.1" stretchy="false" xref="S4.T1.3.3.3.1.1.1.m1.1.1.cmml">↓</mo><annotation-xml encoding="MathML-Content" id="S4.T1.3.3.3.1.1.1.m1.1b"><ci id="S4.T1.3.3.3.1.1.1.m1.1.1.cmml" xref="S4.T1.3.3.3.1.1.1.m1.1.1">↓</ci></annotation-xml><annotation encoding="application/x-tex" id="S4.T1.3.3.3.1.1.1.m1.1c">\downarrow</annotation></semantics></math></td></tr></table></td></tr><tr class="ltx_tr" id="S4.T1.3.5.2"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T1.3.5.2.1" rowspan="3" style="padding-left:2.5pt;padding-right:2.5pt;">Toon shading</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T1.3.5.2.2" style="padding-left:2.5pt;padding-right:2.5pt;">Rerender-a-video</th><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.3.5.2.3" style="padding-left:2.5pt;padding-right:2.5pt;">5.35</td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.3.5.2.4" style="padding-left:2.5pt;padding-right:2.5pt;">-</td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.3.5.2.5" style="padding-left:2.5pt;padding-right:2.5pt;">200.46</td></tr><tr class="ltx_tr" id="S4.T1.3.6.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S4.T1.3.6.3.1" style="padding-left:2.5pt;padding-right:2.5pt;">DomoAI</th><td class="ltx_td ltx_align_center" id="S4.T1.3.6.3.2" style="padding-left:2.5pt;padding-right:2.5pt;">6.26</td><td class="ltx_td ltx_align_center" id="S4.T1.3.6.3.3" style="padding-left:2.5pt;padding-right:2.5pt;">-</td><td class="ltx_td ltx_align_center" id="S4.T1.3.6.3.4" style="padding-left:2.5pt;padding-right:2.5pt;">-</td></tr><tr class="ltx_tr" id="S4.T1.3.7.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S4.T1.3.7.4.1" style="padding-left:2.5pt;padding-right:2.5pt;">Diffutoon</th><td class="ltx_td ltx_align_center" id="S4.T1.3.7.4.2" style="padding-left:2.5pt;padding-right:2.5pt;">6.47</td><td class="ltx_td ltx_align_center" id="S4.T1.3.7.4.3" style="padding-left:2.5pt;padding-right:2.5pt;">-</td><td class="ltx_td ltx_align_center" id="S4.T1.3.7.4.4" style="padding-left:2.5pt;padding-right:2.5pt;">188.87</td></tr><tr class="ltx_tr" id="S4.T1.3.8.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r ltx_border_t" id="S4.T1.3.8.5.1" rowspan="4" style="padding-left:2.5pt;padding-right:2.5pt;">Toon shadingwith editingsignals</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T1.3.8.5.2" style="padding-left:2.5pt;padding-right:2.5pt;">Rerender-a-video</th><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.3.8.5.3" style="padding-left:2.5pt;padding-right:2.5pt;">5.40</td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.3.8.5.4" style="padding-left:2.5pt;padding-right:2.5pt;">28.63</td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T1.3.8.5.5" style="padding-left:2.5pt;padding-right:2.5pt;">266.23</td></tr><tr class="ltx_tr" id="S4.T1.3.9.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S4.T1.3.9.6.1" style="padding-left:2.5pt;padding-right:2.5pt;">DomoAI</th><td class="ltx_td ltx_align_center" id="S4.T1.3.9.6.2" style="padding-left:2.5pt;padding-right:2.5pt;">6.25</td><td class="ltx_td ltx_align_center" id="S4.T1.3.9.6.3" style="padding-left:2.5pt;padding-right:2.5pt;">29.01</td><td class="ltx_td ltx_align_center" id="S4.T1.3.9.6.4" style="padding-left:2.5pt;padding-right:2.5pt;">-</td></tr><tr class="ltx_tr" id="S4.T1.3.10.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S4.T1.3.10.7.1" style="padding-left:2.5pt;padding-right:2.5pt;">Gen-1</th><td class="ltx_td ltx_align_center" id="S4.T1.3.10.7.2" style="padding-left:2.5pt;padding-right:2.5pt;">6.11</td><td class="ltx_td ltx_align_center" id="S4.T1.3.10.7.3" style="padding-left:2.5pt;padding-right:2.5pt;">28.91</td><td class="ltx_td ltx_align_center" id="S4.T1.3.10.7.4" style="padding-left:2.5pt;padding-right:2.5pt;">-</td></tr><tr class="ltx_tr" id="S4.T1.3.11.8"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r" id="S4.T1.3.11.8.1" style="padding-left:2.5pt;padding-right:2.5pt;">Diffutoon</th><td class="ltx_td ltx_align_center ltx_border_b" id="S4.T1.3.11.8.2" style="padding-left:2.5pt;padding-right:2.5pt;">6.37</td><td class="ltx_td ltx_align_center ltx_border_b" id="S4.T1.3.11.8.3" style="padding-left:2.5pt;padding-right:2.5pt;">30.69</td><td class="ltx_td ltx_align_center ltx_border_b" id="S4.T1.3.11.8.4" style="padding-left:2.5pt;padding-right:2.5pt;">143.51</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 1.  Quantitative results of each approach. | ✅ Table 1.  每种方法的定量结果。 |

![figure](https://ar5iv.labs.arxiv.org/html/2401.16224/assets/figure/case_study/input.jpg)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 2.  Visual comparison with other methods. | ✅ Figure 2.  与其他方法进行视觉比较。 |
| ✅ The prompt used for editing is “best quality, perfect anime illustration, a girl is dancing, smile, solo,  orange dress, black hair, white shoes, blue sky  ”. | ✅ 编辑使用的提示是“最佳品质，完美的动漫插图，一个女孩正在跳舞，微笑，独奏， orange dress, black hair, white shoes, blue sky ”。 |
| ✅ Since the resolution of our generated video is extremely high, we enlarge some areas to view details. | ✅ 由于我们生成的视频的分辨率极高，我们放大了一些区域以查看细节。 |
| ✅ We highly recommend readers to see the videos on our project page. | ✅ 我们强烈建议读者观看我们项目页面上的视频。 |

![figure](https://ar5iv.labs.arxiv.org/html/2401.16224/assets/figure/case_study/outline_video.jpg)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 3.  Intermediate results of Diffutoon. | ✅ Figure 3.  Diffutoon 的中间结果。 |
| ✅ In the main toon shading pipeline, the video is synthesized according to the outline video and the color video. | ✅ 在主卡通着色管道中，根据轮廓视频和颜色视频合成视频。 |
| ✅ When the editing branch is enabled, the generated color video contains the editing signals. | ✅ 当编辑分支启用时，生成的彩色视频包含编辑信号。 |

| 【第4.1节，第2段】原文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 【第4.1节，第2段】翻译                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✅ Currently, finding accurate evaluation metrics to measure video quality remains challenging, and there has been some controversy in recent years concerning evaluation metrics ( **1. Brooks et al. (2022)  Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei Efros, and Tero Karras. 2022.   Generating long videos of dynamic scenes.   Advances in Neural Information Processing Systems 35 (2022), 31769–31781.** ｜ **2. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.** ｜ **3. Ouyang et al. (2023)  Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2023.   Codef: Content deformation fields for temporally consistent video processing.   arXiv preprint arXiv:2308.07926 (2023).** ). | ✅ 目前，找到准确的评估指标来衡量视频质量仍然具有挑战性，近年来有关评估指标( **1. Brooks et al. (2022)  Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei Efros, and Tero Karras. 2022.   Generating long videos of dynamic scenes.   Advances in Neural Information Processing Systems 35 (2022), 31769–31781.** ｜ **2. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.** ｜ **3. Ouyang et al. (2023)  Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2023.   Codef: Content deformation fields for temporally consistent video processing.   arXiv preprint arXiv:2308.07926 (2023).** )存在一些争议。 |
| ✅ In our experiments, we evaluate the quality of videos generated by each method in three aspects.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | ✅ 在我们的实验中，我们从三个方面评估每种方法生成的视频的质量。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ✅ 1) Aesthetics : Visual appeal is quantified through an aesthetic score ( **Schuhmann et al. (2022)  Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022.   Laion-5b: An open large-scale dataset for training next generation image-text models.   Advances in Neural Information Processing Systems 35 (2022), 25278–25294.** ) , providing a measure of the overall visual quality of the generated videos.                                                                                                                                                                                                                                                                                                                                                                                                                                    | ✅ 1）Aesthetics：视觉吸引力通过美学分数( **Schuhmann et al. (2022)  Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022.   Laion-5b: An open large-scale dataset for training next generation image-text models.   Advances in Neural Information Processing Systems 35 (2022), 25278–25294.** )进行量化，衡量生成视频的整体视觉质量。                                                                                                                                                                                                                                                                                                                                                                                                        |
| ✅ 2) Text-video similarity : To evaluate the relevance of generated videos to the given text in the toon shading with editing signals task, we use the cosine similarity calculated by the CLIP model ( **Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.** ).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ✅ 2）Text-video similarity：为了评估生成的视频与带有编辑信号卡通着色任务中给定文本的相关性，我们使用 CLIP 模型 ( **Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.** ) 计算的余弦相似度。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ✅ 3) Video consistency : Evaluating video consistency is challenging.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | ✅ 3）Video consistency：评估视频一致性具有挑战性。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ✅ While earlier studies ( **1. Wang et al. (2023)  Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. 2023.   Zero-shot video editing using off-the-shelf image diffusion models.   arXiv preprint arXiv:2303.17599 (2023).** ｜ **2. Qi et al. (2023)  Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023.   Fatezero: Fusing attentions for zero-shot text-based video editing.   arXiv preprint arXiv:2303.09535 (2023).** ) commonly utilized feature similarity of adjacent frames, this approach is limited by embeddings computed by the CLIP model, which is specifically designed for images with a resolution of  $224\times 224$ .                                                                                                                                                                                                                                | ✅ 虽然早期研究 ( **1. Wang et al. (2023)  Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. 2023.   Zero-shot video editing using off-the-shelf image diffusion models.   arXiv preprint arXiv:2303.17599 (2023).** ｜ **2. Qi et al. (2023)  Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023.   Fatezero: Fusing attentions for zero-shot text-based video editing.   arXiv preprint arXiv:2303.09535 (2023).** ) 通常利用相邻帧的特征相似性，但这种方法受到 CLIP 模型计算的嵌入的限制，该模型是专门为具有  $224\times 224$  分辨率的图像设计的。                                                                                                                                                                                                                                           |
| ✅ Therefore, this metric is not suitable for our experiments.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | ✅ 因此，该指标不适合我们的实验。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ✅ Following Rerender-a-video ( **Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ) and Pix2Video ( **Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.** ) , we adopt pixel-MSE as a metric for video consistency.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | ✅ 继 Rerender-a-video ( **Yang et al. (2023)  Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.   Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.   arXiv preprint arXiv:2306.07954 (2023).** ) 和 Pix2Video ( **Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.** ) 之后，我们采用像素 MSE 作为视频一致性的度量。                                                                                                                                                                                                                                                                                                                                                                                                               |
| ✅ Pixel-MSE is the mean square error between the warped frame and its corresponding target frame, where the warped frame is computed according to the estimated optical flow ( **Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer, 402–419.** ).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ✅ Pixel-MSE 是扭曲帧与其对应目标帧之间的均方误差，其中扭曲帧是根据估计的光流 ( **Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer, 402–419.** ) 计算的。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ✅ Note that the services provided by DomoAI and Gen-1 can only support 24 fps, which is not aligned with the original video.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | ✅ 请注意，DomoAI和Gen-1提供的服务只能支持24 fps，这与原始视频不一致。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ✅ Consequently, the calculation of pixel-MSE for these two methods is not feasible.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | ✅ 因此，这两种方法计算像素MSE是不可行的。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ✅ The quantitative results are shown in Table 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | ✅ 定量结果如表1所示。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ✅ Our approach significantly surpasses other baseline models in both two tasks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | ✅ 我们的方法在这两项任务上都显著超越了其他基线模型。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ✅ The experimental results demonstrate the effectiveness of our method.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ✅ 实验结果证明了我们方法的有效性。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle" id="S4.T2.1"><tbody class="ltx_tbody"><tr class="ltx_tr" id="S4.T2.1.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T2.1.1.1.1" rowspan="2" style="padding-left:3.2pt;padding-right:3.2pt;">Task</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T2.1.1.1.2" rowspan="2" style="padding-left:3.2pt;padding-right:3.2pt;">Baseline</th><td class="ltx_td ltx_align_center ltx_border_t" colspan="2" id="S4.T2.1.1.1.3" style="padding-left:3.2pt;padding-right:3.2pt;">Preference</td></tr><tr class="ltx_tr" id="S4.T2.1.2.2"><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T2.1.2.2.1" style="padding-left:3.2pt;padding-right:3.2pt;">Diffutoon</td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T2.1.2.2.2" style="padding-left:3.2pt;padding-right:3.2pt;">Other</td></tr><tr class="ltx_tr" id="S4.T2.1.3.3"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T2.1.3.3.1" rowspan="2" style="padding-left:3.2pt;padding-right:3.2pt;">Toon shading</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T2.1.3.3.2" style="padding-left:3.2pt;padding-right:3.2pt;">Rerender-a-video</th><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T2.1.3.3.3" style="padding-left:3.2pt;padding-right:3.2pt;">98.21%</td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T2.1.3.3.4" style="padding-left:3.2pt;padding-right:3.2pt;">1.79%</td></tr><tr class="ltx_tr" id="S4.T2.1.4.4"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S4.T2.1.4.4.1" style="padding-left:3.2pt;padding-right:3.2pt;">DomoAI</th><td class="ltx_td ltx_align_center" id="S4.T2.1.4.4.2" style="padding-left:3.2pt;padding-right:3.2pt;">90.77%</td><td class="ltx_td ltx_align_center" id="S4.T2.1.4.4.3" style="padding-left:3.2pt;padding-right:3.2pt;">9.23%</td></tr><tr class="ltx_tr" id="S4.T2.1.5.5"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r ltx_border_t" id="S4.T2.1.5.5.1" rowspan="3" style="padding-left:3.2pt;padding-right:3.2pt;">Toon shadingwith editing signals</th><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" id="S4.T2.1.5.5.2" style="padding-left:3.2pt;padding-right:3.2pt;">Rerender-a-video</th><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T2.1.5.5.3" style="padding-left:3.2pt;padding-right:3.2pt;">97.44%</td><td class="ltx_td ltx_align_center ltx_border_t" id="S4.T2.1.5.5.4" style="padding-left:3.2pt;padding-right:3.2pt;">2.56%</td></tr><tr class="ltx_tr" id="S4.T2.1.6.6"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r" id="S4.T2.1.6.6.1" style="padding-left:3.2pt;padding-right:3.2pt;">DomoAI</th><td class="ltx_td ltx_align_center" id="S4.T2.1.6.6.2" style="padding-left:3.2pt;padding-right:3.2pt;">82.35%</td><td class="ltx_td ltx_align_center" id="S4.T2.1.6.6.3" style="padding-left:3.2pt;padding-right:3.2pt;">17.65%</td></tr><tr class="ltx_tr" id="S4.T2.1.7.7"><th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r" id="S4.T2.1.7.7.1" style="padding-left:3.2pt;padding-right:3.2pt;">Gen-1</th><td class="ltx_td ltx_align_center ltx_border_b" id="S4.T2.1.7.7.2" style="padding-left:3.2pt;padding-right:3.2pt;">95.74%</td><td class="ltx_td ltx_align_center ltx_border_b" id="S4.T2.1.7.7.3" style="padding-left:3.2pt;padding-right:3.2pt;">4.26%</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 2.  User preference in human evaluation. | ✅ Table 2.  人工评价中的用户偏好。 |

| 【第4.1节，第3段】原文 | 【第4.1节，第3段】翻译 |
| ---- | ---- |
| ✅ In addition to using the aforementioned metrics to evaluate each method, we also conducted a human evaluation involving 10 participants. | ✅ 除了使用上述指标评估每种方法之外，我们还进行了涉及 10 名参与者的人工评估。 |
| ✅ In each evaluation episode, each participant is presented with two videos, one generated by our method and the other generated by a randomly selected baseline method. | ✅ 在每次评估过程中，我们会向每位参与者展示两个视频，一个由我们的方法生成，另一个由随机选择的基线方法生成。 |
| ✅ Participants are asked to choose the video with the better visual effects. | ✅ 要求参与者选择视觉效果更好的视频。 |
| ✅ We recorded the proportion of participant choices in Table 2. | ✅ 我们在表2中记录了参与者选择的比例。 |
| ✅ Among these results, it is evident that users overwhelmingly believe that our method is capable of producing videos with superior visual effects. | ✅ 在这些结果中，很明显，绝大多数用户相信我们的方法能够制作具有卓越视觉效果的视频。 |
| ✅ This further demonstrates the superiority of our approach. | ✅ 这进一步证明了我们方法的优越性。 |

### 4.2 Case Study

![figure](https://ar5iv.labs.arxiv.org/html/2401.16224/assets/figure/case_study/toon_video_compare_without_outline.jpg)

| 【图标题】原文                                                  | 【图标题】翻译                   |
| -------------------------------------------------------- | ------------------------- |
| ✅ Figure 4.  Video rendered without outline information. | ✅ Figure 4.  渲染的视频没有轮廓信息。 |

![figure](https://ar5iv.labs.arxiv.org/html/2401.16224/assets/figure/case_study/toon_video_compare_without_color.jpg)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 5.  Video rendered without color information. | ✅ Figure 5.  呈现的视频没有颜色信息。 |

| 【第4.2节，第1段】原文 | 【第4.2节，第1段】翻译 |
| ---- | ---- |
| ✅ Figure 2 presents video samples generated by different methods. | ✅ 图2展示了通过不同方法生成的视频样本。 |
| ✅ In the original video (Figure 2 a), a girl is dancing with fast movements, posing a significant challenge for each video processing method. | ✅ 原始视频（图2 a）中，一个女孩正在快速跳舞，这对每种视频处理方法都提出了巨大的挑战。 |
| ✅ Gen-1 and Rerender-a-video struggle to effectively handle high-resolution videos, resulting in facial distortions of the character. | ✅ Gen-1 和 Rerender-a-video 难以有效处理高分辨率视频，导致角色面部扭曲。 |
| ✅ In the videos generated by DomoAI (Figure 2 e and Figure 2 f), there is missing content in the third frame, and the character’s movements in the fourth frame do not align with the original video. | ✅ DomoAI生成的视频（图2e和图2f）中，第三帧缺少内容，第四帧中角色的动作与原始视频不一致。 |
| ✅ This indicates that DomoAI cannot accurately capture motion features from the original video and reproduce the character’s pose. | ✅ 这表明DomoAI无法准确捕捉原始视频中的运动特征并重现角色的姿势。 |
| ✅ Contrastingly, videos generated by Diffutoon (Figure 2 g and Figure 2 h) showcase the preservation of details such as lighting, hair, and pose, while maintaining a visual style closely aligned with anime aesthetics. | ✅ 相比之下，Diffutoon 生成的视频（图 2 g 和图 2 h）展示了对灯光、头发和姿势等细节的保留，同时保持了与动漫美学紧密一致的视觉风格。 |
| ✅ Notably, in the toon shading with editing signals task, our method successfully achieves precise control based on the color information from the given text. | ✅ 值得注意的是，在使用编辑信号任务的卡通阴影中，我们的方法成功地实现了基于给定文本的颜色信息的精确控制。 |
| ✅ These results intuitively highlight the robustness and efficacy of our approach. | ✅ 这些结果直观地突出了我们方法的稳健性和有效性。 |

| 【第4.2节，第2段】原文 | 【第4.2节，第2段】翻译 |
| ---- | ---- |
| ✅ In Figure 3 , we present the intermediate results of Diffutoon, including the outline video and the color video generated by the editing branch. | ✅ 在图 3 中，我们展示了 Diffutoon 的中间结果，包括轮廓视频和编辑分支生成的彩色视频。 |
| ✅ The outline video precisely retains the structural information for rendering an anime-style frame, ensuring the visual quality. | ✅ 轮廓视频精确保留了动画风格帧渲染的结构信息，保证了视觉质量。 |
| ✅ The generated color video exhibits blurriness due to the rapid motion of the dancing girl. | ✅ 由于跳舞女孩的快速运动，生成的彩色视频显得模糊。 |
| ✅ It implies that the editing branch, when operating independently, fails to produce a video of high quality. | ✅ 这意味着编辑部门在独立运作时无法制作出高质量的视频。 |
| ✅ The outline video and the color video provide essential information for rendering a high-resolution video in Figure 2 h. | ✅ 轮廓视频和彩色视频为图 2 h 中的高分辨率视频渲染提供了必要的信息。 |
| ✅ For more video examples, please see the project page. | ✅ 更多视频示例请参阅项目页面。 |

### 4.3 Ablation Study

| 【第4.3节，第1段】原文 | 【第4.3节，第1段】翻译 |
| ---- | ---- |
| ✅ Since the effectiveness of the motion modules has been widely evaluated by prior work ( **Xing et al. (2023)  Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. 2023.   A survey on video diffusion models.   arXiv preprint arXiv:2310.10647 (2023).** ) , we further investigate the effectiveness of the two ControlNet models in the main toon shading pipeline. | ✅ 由于运动模块的有效性已通过先前的研究 ( **Xing et al. (2023)  Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. 2023.   A survey on video diffusion models.   arXiv preprint arXiv:2310.10647 (2023).** ) 得到广泛评估，我们进一步研究了主卡通着色管道中两个 ControlNet 模型的有效性。 |
| ✅ The rendered videos without each ControlNet model are shown in Figure 4 and Figure 5. | ✅ 不含各个 ControlNet 模型的渲染视频如图 4 和图 5 所示。 |
| ✅ The lack of outline information results in mutilation within the frame. | ✅ 缺少轮廓信息导致框架内的残缺。 |
| ✅ The lack of color guidance results in poor visual quality, with noticeable flickering on the head. | ✅ 缺乏色彩引导会导致视觉质量差，头部会出现明显的闪烁。 |
| ✅ It proves that the outline and color are both essential. | ✅ 事实证明，轮廓和色彩都很重要。 |

| 【第4.3节，第2段】原文 | 【第4.3节，第2段】翻译 |
| ---- | ---- |
| ✅ In the toon shading with editing signals task, we design an alternative approach that only contains a single pipeline. | ✅ 在带有编辑信号任务的卡通着色中，我们设计了一种仅包含单个管道的替代方法。 |
| ✅ This approach is constructed based on the editing branch, wherein we replace FastBlend with AnimateDiff. | ✅ 这种方法是基于编辑分支构建的，其中我们用 AnimateDiff 替换了 FastBlend。 |
| ✅ The video generated by this approach is presented in Figure 6. | ✅ 该方法生成的视频如图 6 所示。 |
| ✅ We observe that this video is dark and lacks aesthetic appeal. | ✅ 我们观察到该视频很暗并且缺乏美感。 |
| ✅ As we mentioned in Section 3.2 , the reason is that AnimateDiff relies on a modified DDIM scheduler. | ✅ 正如我们在 3.2 部分中提到的，原因是 AnimateDiff 依赖于修改后的 DDIM 调度程序。 |
| ✅ This scheduler is inconsistent with the Stable Diffusion backbone and the inconsistency is detrimental for synthesizing high-quality videos. | ✅ 该调度程序与稳定扩散主干不一致，这种不一致对于合成高质量视频是有害的。 |
| ✅ However, this pitfall has minimal influence on the main toon shading pipeline, because the color is fixed by the ControlNet. | ✅ 然而，这个缺陷对主卡通着色管道的影响很小，因为颜色是由 ControlNet 固定的。 |
| ✅ It proves the necessity of maintaining a separate pipeline architecture. | ✅ 这证明了维护单独管道架构的必要性。 |

![figure](https://ar5iv.labs.arxiv.org/html/2401.16224/assets/figure/case_study/single_pipeline_compare.jpg)

| 【图标题】原文 | 【图标题】翻译 |
| ---- | ---- |
| ✅ Figure 6.  Video rendered by the editing branch with AnimateDiff. | ✅ Figure 6.  由编辑分支使用 AnimateDiff 渲染的视频。 |

## 5 Conclusion and Future Work

| 【第5节，第1段】原文 | 【第5节，第1段】翻译 |
| ---- | ---- |
| ✅ In this paper, we investigate an innovative form of toon shading based on diffusion models, intending to directly transmute photorealistic videos into anime styles. | ✅ 在本文中，我们研究了一种基于扩散模型的卡通着色的创新形式，旨在将真实感视频直接转化为动漫风格。 |
| ✅ We introduce an advanced toon shading approach which consists of a main toon shading pipeline and an editing branch. | ✅ 我们介绍了一种先进的卡通着色方法，它由主卡通着色管道和编辑分支组成。 |
| ✅ Our approach is capable of processing high-resolution long videos, and can also edit the video via the editing branch. | ✅ 我们的方法能够处理高分辨率的长视频，并且还可以通过编辑分支对视频进行编辑。 |
| ✅ The comprehensive experimental results demonstrate the efficacy of our approach. | ✅ 全面的实验结果证明了我们方法的有效性。 |
| ✅ However, Diffutoon is a toon shading method, not a general video stylization method, as it cannot handle other styles (e.g., realistic, oil painting, and ink painting). | ✅ 然而，Diffutoon 是一种卡通着色方法，而不是通用的视频风格化方法，因为它不能处理其他风格（例如现实主义、油画和水墨画）。 |
| ✅ In the future, we will focus on exploring more applications within the domain of video processing. | ✅ 未来我们将致力于探索视频处理领域的更多应用。 |

## 6 References

- 1
  - 

- 2
  - Pascal Barla, Joëlle Thollot, and Lee Markosian. 2006.
  - **X-toon: An extended toon shader. In Proceedings of the 4th international symposium on Non-photorealistic animation and rendering. 127–132.**
  - 

- 3
  - Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023.
  - **Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22563–22575.**
  - 

- 4
  - Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei Efros, and Tero Karras. 2022.
  - **Generating long videos of dynamic scenes.**
  - Advances in Neural Information Processing Systems 35 (2022), 31769–31781.
  - 

- 5
  - Tingfeng Cao, Chengyu Wang, Bingyan Liu, Ziheng Wu, Jinhui Zhu, and Jun Huang. 2023.
  - **BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track. 1–11.**
  - 

- 6
  - Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. 2023.
  - **Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.**
  - 

- 7
  - Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.
  - **Flashattention: Fast and memory-efficient exact attention with io-awareness.**
  - Advances in Neural Information Processing Systems 35 (2022), 16344–16359.
  - 

- 8
  - Group DOMO.AI. 2024.
  - **DOMO.AI.**
  - 
  - 
  - https://ai.domo.com/, Last accessed on 2024-01-18.

- 9
  - Zhongjie Duan, Chengyu Wang, Cen Chen, Jun Huang, and Weining Qian. 2023a.
  - **Optimal Linear Subspace Search: Learning to Construct Fast and High-Quality Schedulers for Diffusion Models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 463–472.**
  - 

- 10
  - Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, Jun Huang, and Mingyi Jin. 2023b.
  - **FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier.**
  - arXiv preprint arXiv:2311.09265 (2023).
  - 

- 11
  - Zhongjie Duan, Lizhou You, Chengyu Wang, Cen Chen, Ziheng Wu, Weining Qian, Jun Huang, Fei Chao, and Rongrong Ji. 2023c.
  - **DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis.**
  - arXiv preprint arXiv:2308.03463 (2023).
  - 

- 12
  - Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. 2023.
  - **Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7346–7356.**
  - 

- 13
  - Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. 2022.
  - **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In The Eleventh International Conference on Learning Representations.**
  - 

- 14
  - Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014.
  - **Generative adversarial nets.**
  - Advances in neural information processing systems 27 (2014).
  - 

- 15
  - Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. 2023.
  - **Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.**
  - arXiv preprint arXiv:2307.04725 (2023).
  - 

- 16
  - Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. 2023.
  - **ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models.**
  - arXiv preprint arXiv:2310.07702 (2023).
  - 

- 17
  - Jonathan Ho and Tim Salimans. 2021.
  - **Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.**
  - 

- 18
  - Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021.
  - **LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.**
  - 

- 19
  - Matis Hudon, Rafael Pagés, Mairéad Grogan, Jan Ondřej, and Aljoša Smolić. 2018.
  - **2D shading for cel animation. In Proceedings of the Joint Symposium on Computational Aesthetics and Sketch-Based Interfaces and Modeling and Non-Photorealistic Animation and Rendering. 1–12.**
  - 

- 20
  - Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. 2023.
  - **Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis.**
  - arXiv preprint arXiv:2306.08645 (2023).
  - 

- 21
  - Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023.
  - **Text2video-zero: Text-to-image diffusion models are zero-shot video generators.**
  - arXiv preprint arXiv:2303.13439 (2023).
  - 

- 22
  - Diederik P Kingma and Max Welling. 2013.
  - **Auto-encoding variational bayes.**
  - arXiv preprint arXiv:1312.6114 (2013).
  - 

- 23
  - Chenyang Lei, Xuanchi Ren, Zhaoxiang Zhang, and Qifeng Chen. 2023.
  - **Blind Video Deflickering by Neural Filtering with a Flawed Atlas. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10439–10448.**
  - 

- 24
  - Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. 2022.
  - **Srdiff: Single image super-resolution with diffusion probabilistic models.**
  - Neurocomputing 479 (2022), 47–59.
  - 

- 25
  - Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022.
  - **Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps.**
  - Advances in Neural Information Processing Systems 35 (2022), 5775–5787.
  - 

- 26
  - Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023.
  - **T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models.**
  - arXiv preprint arXiv:2302.08453 (2023).
  - 

- 27
  - Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2023.
  - **Codef: Content deformation fields for temporally consistent video processing.**
  - arXiv preprint arXiv:2308.07926 (2023).
  - 

- 28
  - Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023.
  - **Fatezero: Fusing attentions for zero-shot text-based video editing.**
  - arXiv preprint arXiv:2303.09535 (2023).
  - 

- 29
  - Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021.
  - **Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.**
  - 

- 30
  - René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. 2020.
  - **Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer.**
  - IEEE transactions on pattern analysis and machine intelligence 44, 3 (2020), 1623–1637.
  - 

- 31
  - Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022.
  - **High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.**
  - 

- 32
  - Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015.
  - **U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.**
  - 

- 33
  - Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023.
  - **Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22500–22510.**
  - 

- 34
  - Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022.
  - **Laion-5b: An open large-scale dataset for training next generation image-text models.**
  - Advances in Neural Information Processing Systems 35 (2022), 25278–25294.
  - 

- 35
  - Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015.
  - **Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.**
  - 

- 36
  - Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020.
  - **Denoising Diffusion Implicit Models. In International Conference on Learning Representations.**
  - 

- 37
  - Zachary Teed and Jia Deng. 2020.
  - **Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer, 402–419.**
  - 

- 38
  - Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. 2023.
  - **Zero-shot video editing using off-the-shelf image diffusion models.**
  - arXiv preprint arXiv:2303.17599 (2023).
  - 

- 39
  - Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. 2021.
  - **Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision. 1905–1914.**
  - 

- 40
  - Saining Xie and Zhuowen Tu. 2015.
  - **Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision. 1395–1403.**
  - 

- 41
  - Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. 2023.
  - **A survey on video diffusion models.**
  - arXiv preprint arXiv:2310.10647 (2023).
  - 

- 42
  - Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023.
  - **Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation.**
  - arXiv preprint arXiv:2306.07954 (2023).
  - 

- 43
  - Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023.
  - **Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.**
  - 

## 7 Appendix A Model Components

| 【第7节，第1段】原文 | 【第7节，第1段】翻译 |
| ---- | ---- |
| ✅ After experimental testing, we decided to utilize several open-source models obtained from the open-source community. | ✅ 经过实验测试，我们决定利用从开源社区获得的几个开源模型。 |
| ✅ These models are listed in Table 3. | ✅ 这些模型列在表 3 中。 |
| ✅ Benefiting from the abundant open-source models, we succeed in designing such a fantastic toon shading pipeline. | ✅ 得益于丰富的开源模型，我们成功设计出如此出色的卡通着色管道。 |

<table class="ltx_tabular ltx_guessed_headers ltx_align_middle" id="A1.T3.1"><thead class="ltx_thead"><tr class="ltx_tr" id="A1.T3.1.1.1"><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_t" id="A1.T3.1.1.1.1">Model type</th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="A1.T3.1.1.1.2"><table class="ltx_tabular ltx_align_middle" id="A1.T3.1.1.1.2.1"><tr class="ltx_tr" id="A1.T3.1.1.1.2.1.1"><td class="ltx_td ltx_nopad_r ltx_align_center" id="A1.T3.1.1.1.2.1.1.1">Main</td></tr><tr class="ltx_tr" id="A1.T3.1.1.1.2.1.2"><td class="ltx_td ltx_nopad_r ltx_align_center" id="A1.T3.1.1.1.2.1.2.1">toon</td></tr><tr class="ltx_tr" id="A1.T3.1.1.1.2.1.3"><td class="ltx_td ltx_nopad_r ltx_align_center" id="A1.T3.1.1.1.2.1.3.1">shading</td></tr><tr class="ltx_tr" id="A1.T3.1.1.1.2.1.4"><td class="ltx_td ltx_nopad_r ltx_align_center" id="A1.T3.1.1.1.2.1.4.1">pipeline</td></tr></table></th><th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" id="A1.T3.1.1.1.3"><table class="ltx_tabular ltx_align_middle" id="A1.T3.1.1.1.3.1"><tr class="ltx_tr" id="A1.T3.1.1.1.3.1.1"><td class="ltx_td ltx_nopad_r ltx_align_center" id="A1.T3.1.1.1.3.1.1.1">Video</td></tr><tr class="ltx_tr" id="A1.T3.1.1.1.3.1.2"><td class="ltx_td ltx_nopad_r ltx_align_center" id="A1.T3.1.1.1.3.1.2.1">editing</td></tr><tr class="ltx_tr" id="A1.T3.1.1.1.3.1.3"><td class="ltx_td ltx_nopad_r ltx_align_center" id="A1.T3.1.1.1.3.1.3.1">branch</td></tr></table></th><th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t" id="A1.T3.1.1.1.4">URL</th></tr></thead><tbody class="ltx_tbody"><tr class="ltx_tr" id="A1.T3.1.2.1"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A1.T3.1.2.1.1">Stable Diffusion</td><td class="ltx_td ltx_align_center ltx_border_t" id="A1.T3.1.2.1.2">✓</td><td class="ltx_td ltx_align_center ltx_border_t" id="A1.T3.1.2.1.3">✓</td><td class="ltx_td ltx_align_left ltx_border_t" id="A1.T3.1.2.1.4"><a class="ltx_ref ltx_url ltx_font_typewriter" href="https://civitai.com/models/34553/aingdiffusion" target="_blank" title="">https://civitai.com/models/34553/aingdiffusion</a></td></tr><tr class="ltx_tr" id="A1.T3.1.3.2"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T3.1.3.2.1">ControlNet (Outline)</td><td class="ltx_td ltx_align_center" id="A1.T3.1.3.2.2">✓</td><td class="ltx_td" id="A1.T3.1.3.2.3"></td><td class="ltx_td ltx_align_left" id="A1.T3.1.3.2.4"><a class="ltx_ref ltx_url ltx_font_typewriter" href="https://huggingface.co/lllyasviel/control_v11p_sd15_lineart" target="_blank" title="">https://huggingface.co/lllyasviel/control_v11p_sd15_lineart</a></td></tr><tr class="ltx_tr" id="A1.T3.1.4.3"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T3.1.4.3.1">ControlNet (Color)</td><td class="ltx_td ltx_align_center" id="A1.T3.1.4.3.2">✓</td><td class="ltx_td" id="A1.T3.1.4.3.3"></td><td class="ltx_td ltx_align_left" id="A1.T3.1.4.3.4"><a class="ltx_ref ltx_url ltx_font_typewriter" href="https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile" target="_blank" title="">https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile</a></td></tr><tr class="ltx_tr" id="A1.T3.1.5.4"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T3.1.5.4.1">ControlNet (Softedge)</td><td class="ltx_td" id="A1.T3.1.5.4.2"></td><td class="ltx_td ltx_align_center" id="A1.T3.1.5.4.3">✓</td><td class="ltx_td ltx_align_left" id="A1.T3.1.5.4.4"><a class="ltx_ref ltx_url ltx_font_typewriter" href="https://huggingface.co/lllyasviel/control_v11p_sd15_softedge" target="_blank" title="">https://huggingface.co/lllyasviel/control_v11p_sd15_softedge</a></td></tr><tr class="ltx_tr" id="A1.T3.1.6.5"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T3.1.6.5.1">ControlNet (Depth)</td><td class="ltx_td" id="A1.T3.1.6.5.2"></td><td class="ltx_td ltx_align_center" id="A1.T3.1.6.5.3">✓</td><td class="ltx_td ltx_align_left" id="A1.T3.1.6.5.4"><a class="ltx_ref ltx_url ltx_font_typewriter" href="https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth" target="_blank" title="">https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth</a></td></tr><tr class="ltx_tr" id="A1.T3.1.7.6"><td class="ltx_td ltx_align_left ltx_border_r" id="A1.T3.1.7.6.1">Motion modules</td><td class="ltx_td ltx_align_center" id="A1.T3.1.7.6.2">✓</td><td class="ltx_td" id="A1.T3.1.7.6.3"></td><td class="ltx_td ltx_align_left" id="A1.T3.1.7.6.4"><a class="ltx_ref ltx_url ltx_font_typewriter" href="https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt" target="_blank" title="">https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt</a></td></tr><tr class="ltx_tr" id="A1.T3.1.8.7"><td class="ltx_td ltx_align_left ltx_border_b ltx_border_r" id="A1.T3.1.8.7.1">Textual inversion</td><td class="ltx_td ltx_align_center ltx_border_b" id="A1.T3.1.8.7.2">✓</td><td class="ltx_td ltx_align_center ltx_border_b" id="A1.T3.1.8.7.3">✓</td><td class="ltx_td ltx_align_left ltx_border_b" id="A1.T3.1.8.7.4"><a class="ltx_ref ltx_url ltx_font_typewriter" href="https://civitai.com/models/11772" target="_blank" title="">https://civitai.com/models/11772</a></td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 3.  List of models utilized in Diffutoon. | ✅ Table 3.  Diffutoon 中使用的模型列表。 |

## 8 Appendix B Parameter Settings

<table class="ltx_tabular ltx_centering ltx_align_middle" id="A2.T4.1"><tbody class="ltx_tbody"><tr class="ltx_tr" id="A2.T4.1.2.1"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A2.T4.1.2.1.1">Components</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A2.T4.1.2.1.2">Parameter</td><td class="ltx_td ltx_align_center ltx_border_t" id="A2.T4.1.2.1.3">Value</td></tr><tr class="ltx_tr" id="A2.T4.1.3.2"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A2.T4.1.3.2.1" rowspan="9">Maintoonshadingpipeline</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A2.T4.1.3.2.2">frame height</td><td class="ltx_td ltx_align_center ltx_border_t" id="A2.T4.1.3.2.3">1536</td></tr><tr class="ltx_tr" id="A2.T4.1.4.3"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.4.3.1">frame width</td><td class="ltx_td ltx_align_center" id="A2.T4.1.4.3.2">1536</td></tr><tr class="ltx_tr" id="A2.T4.1.5.4"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.5.4.1">classifier-free guidance scale</td><td class="ltx_td ltx_align_center" id="A2.T4.1.5.4.2">7</td></tr><tr class="ltx_tr" id="A2.T4.1.6.5"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.6.5.1">denoising strength</td><td class="ltx_td ltx_align_center" id="A2.T4.1.6.5.2">1</td></tr><tr class="ltx_tr" id="A2.T4.1.7.6"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.7.6.1">inference steps</td><td class="ltx_td ltx_align_center" id="A2.T4.1.7.6.2">10</td></tr><tr class="ltx_tr" id="A2.T4.1.8.7"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.8.7.1">sliding window size</td><td class="ltx_td ltx_align_center" id="A2.T4.1.8.7.2">16</td></tr><tr class="ltx_tr" id="A2.T4.1.9.8"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.9.8.1">sliding window stride</td><td class="ltx_td ltx_align_center" id="A2.T4.1.9.8.2">8</td></tr><tr class="ltx_tr" id="A2.T4.1.10.9"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.10.9.1">conditioning scale (outline)</td><td class="ltx_td ltx_align_center" id="A2.T4.1.10.9.2">0.5</td></tr><tr class="ltx_tr" id="A2.T4.1.11.10"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.11.10.1">conditioning scale (color)</td><td class="ltx_td ltx_align_center" id="A2.T4.1.11.10.2">0.5</td></tr><tr class="ltx_tr" id="A2.T4.1.12.11"><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A2.T4.1.12.11.1" rowspan="9">Videoeditingbranch</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A2.T4.1.12.11.2">frame height</td><td class="ltx_td ltx_align_center ltx_border_t" id="A2.T4.1.12.11.3">512</td></tr><tr class="ltx_tr" id="A2.T4.1.13.12"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.13.12.1">frame width</td><td class="ltx_td ltx_align_center" id="A2.T4.1.13.12.2">512</td></tr><tr class="ltx_tr" id="A2.T4.1.14.13"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.14.13.1">classifier-free guidance scale</td><td class="ltx_td ltx_align_center" id="A2.T4.1.14.13.2">7</td></tr><tr class="ltx_tr" id="A2.T4.1.15.14"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.15.14.1">denoising strength</td><td class="ltx_td ltx_align_center" id="A2.T4.1.15.14.2">0.9</td></tr><tr class="ltx_tr" id="A2.T4.1.16.15"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.16.15.1">inference steps</td><td class="ltx_td ltx_align_center" id="A2.T4.1.16.15.2">20</td></tr><tr class="ltx_tr" id="A2.T4.1.17.16"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.17.16.1">sliding window size</td><td class="ltx_td ltx_align_center" id="A2.T4.1.17.16.2">8</td></tr><tr class="ltx_tr" id="A2.T4.1.18.17"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.18.17.1">sliding window stride</td><td class="ltx_td ltx_align_center" id="A2.T4.1.18.17.2">4</td></tr><tr class="ltx_tr" id="A2.T4.1.19.18"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.19.18.1">conditioning scale (depth)</td><td class="ltx_td ltx_align_center" id="A2.T4.1.19.18.2">0.5</td></tr><tr class="ltx_tr" id="A2.T4.1.20.19"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.20.19.1">conditioning scale (softedge)</td><td class="ltx_td ltx_align_center" id="A2.T4.1.20.19.2">0.5</td></tr><tr class="ltx_tr" id="A2.T4.1.21.20"><td class="ltx_td ltx_align_left ltx_border_b ltx_border_r ltx_border_t" id="A2.T4.1.21.20.1" rowspan="7">FastBlend</td><td class="ltx_td ltx_align_left ltx_border_r ltx_border_t" id="A2.T4.1.21.20.2">inference mode</td><td class="ltx_td ltx_align_center ltx_border_t" id="A2.T4.1.21.20.3">accurate</td></tr><tr class="ltx_tr" id="A2.T4.1.22.21"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.22.21.1">sliding window size</td><td class="ltx_td ltx_align_center" id="A2.T4.1.22.21.2">30</td></tr><tr class="ltx_tr" id="A2.T4.1.23.22"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.23.22.1">batch size</td><td class="ltx_td ltx_align_center" id="A2.T4.1.23.22.2">64</td></tr><tr class="ltx_tr" id="A2.T4.1.24.23"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.24.23.1">tracking mechanism</td><td class="ltx_td ltx_align_center" id="A2.T4.1.24.23.2">enabled</td></tr><tr class="ltx_tr" id="A2.T4.1.25.24"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.25.24.1">patch size</td><td class="ltx_td ltx_align_center" id="A2.T4.1.25.24.2">5</td></tr><tr class="ltx_tr" id="A2.T4.1.26.25"><td class="ltx_td ltx_align_left ltx_border_r" id="A2.T4.1.26.25.1">number of iterations</td><td class="ltx_td ltx_align_center" id="A2.T4.1.26.25.2">5</td></tr><tr class="ltx_tr" id="A2.T4.1.1"><td class="ltx_td ltx_align_left ltx_border_b ltx_border_r" id="A2.T4.1.1.1">guide weight <math alttext="\alpha" class="ltx_Math" display="inline" id="A2.T4.1.1.1.m1.1"><semantics id="A2.T4.1.1.1.m1.1a"><mi id="A2.T4.1.1.1.m1.1.1" xref="A2.T4.1.1.1.m1.1.1.cmml">α</mi><annotation-xml encoding="MathML-Content" id="A2.T4.1.1.1.m1.1b"><ci id="A2.T4.1.1.1.m1.1.1.cmml" xref="A2.T4.1.1.1.m1.1.1">𝛼</ci></annotation-xml><annotation encoding="application/x-tex" id="A2.T4.1.1.1.m1.1c">\alpha</annotation></semantics></math></td><td class="ltx_td ltx_align_center ltx_border_b" id="A2.T4.1.1.2">10</td></tr></tbody></table>

| 【表标题】原文 | 【表标题】翻译 |
| ---- | ---- |
| ✅ Table 4.  Parameter settings in the experiments. | ✅ Table 4.  实验中的参数设置。 |

| 【第8节，第1段】原文 | 【第8节，第1段】翻译 |
| ---- | ---- |
| ✅ The parameter settings of our approach are detailed in Table 4. | ✅ 我们的方法的参数设置详见表4。 |
| ✅ Since our approach has a robust tolerance to color video, we use a lower resolution and sliding window size in the editing branch for faster generation. | ✅ 由于我们的方法对彩色视频具有强大的容忍度，因此我们在编辑分支中使用较低的分辨率和滑动窗口大小来加快生成速度。 |
| ✅ The denoising strength quantifies the extent of noise introduced into the video, with a value of 1 indicating complete frame replacement and rerendering, while 0 implies no modifications to the video. | ✅ 去噪强度量化了引入视频的噪声程度，值为 1 表示完全替换帧并重新渲染，而 0 表示不对视频进行修改。 |
| ✅ In the editing branch, we set the denoising strength to 0.9, retaining a little information from the input video. | ✅ 在编辑分支中，我们将去噪强度设置为 0.9，保留了输入视频中的一些信息。 |
| ✅ The number of inference steps is 20 in the editing branch, which is larger than that of the main toon shading pipeline. | ✅ 编辑分支中的推理步骤数为 20，大于主卡通着色管道的推理步骤数。 |
| ✅ This adjustment is based on empirical findings that fewer steps may lead to frames that are misaligned with the desired editing prompt. | ✅ 此调整基于经验发现，即步骤较少可能会导致帧与所需的编辑提示不对齐。 |
| ✅ These parameters are manually tuned to optimize speed without compromising the resulting quality. | ✅ 这些参数是手动调整的，以优化速度，而不会影响最终的质量。 |
| ✅ For the parameters associated with FastBlend, the accurate inference mode is utilized, and readers can refer to the original paper ( **Duan et al. (2023b)  Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, Jun Huang, and Mingyi Jin. 2023b.   FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier.   arXiv preprint arXiv:2311.09265 (2023).** ) for more comprehensive details on these configurations. | ✅ 对于与 FastBlend 相关的参数，采用了准确推理模式，读者可以参考原始论文 ( **Duan et al. (2023b)  Zhongjie Duan, Chengyu Wang, Cen Chen, Weining Qian, Jun Huang, and Mingyi Jin. 2023b.   FastBlend: a Powerful Model-Free Toolkit Making Video Stylization Easier.   arXiv preprint arXiv:2311.09265 (2023).** ) 了解有关这些配置的更全面的详细信息。 |