# Survey of Video and Multimodal Generation

## TL;DR
- Video generation has transitioned from unstable GAN/RNN models to powerful latent diffusion models and diffusion transformers [1][2][3][4].
- Modern evaluation frameworks have shifted from simple frame-level metrics to hierarchical benchmarks like VBench and human-preference alignments [5][6].
- Recent innovations focus on long-form generation, physics-based control, and multimodal synchronization between audio and video [7][8][9][10].
- Current open challenges include maintaining temporal consistency and achieving reliable physical plausibility in complex scenes [11].

## Background
Video generation is the task of synthesizing dynamic visual content from textual or multimodal inputs. Historically constrained by the computational intensity of modeling temporal dependencies, the field has seen rapid development over the last few years, evolving from early GAN-based architectures to modern diffusion-based systems [1][12]. This shift is critical for high-fidelity content creation and simulation, enabling advancements in creative arts, world modeling, and interactive media.

## Foundations of Video Generation
Early milestones were dominated by Generative Adversarial Networks (GANs), which framed video synthesis as a minimax game [1][12]. Architectures like VGAN employed 3D convolutions to model spatio-temporal features, while models such as MoCoGAN decoupled content from motion to improve synthesis control [1]. Despite their impact, GANs struggled with training stability [12]. By 2022, research shifted toward diffusion models, which extended image synthesis techniques to video by incorporating temporal attention and joint training on image-video data [2][13]. These approaches provided superior distribution modeling, offering greater stability and quality compared to their predecessors [1][13].

## Modern Architectural Paradigms
Current state-of-the-art video generation relies on Latent Diffusion Models (LDMs) and Diffusion Transformers (DiT) [4]. LDMs reduce computational demands by processing video in lower-dimensional latent spaces [4]. Latte, a significant architecture in this domain, utilizes transformer blocks to model video distributions in latent space, achieving highly competitive results [3]. Simultaneously, DiT-based models improve temporal consistency and architectural stability by replacing U-Net foundations with transformer-based components [4]. Further advancements include hybrid architectures, such as the dual-stream CrossDiT employed in Kandinsky 6.0, which enables high-quality synchronized audio and video generation [7].

## Benchmarking and Evaluation
Evaluating video generation is complex, as metrics must account for both visual fidelity and temporal dynamics [5]. Traditional metrics often fail to capture the spatiotemporal consistency and semantic nuances required for high-quality video generation [5]. VBench provides a hierarchical framework to disentangle attributes such as motion smoothness, temporal flickering, and text alignment [5][6]. Other specialized benchmarks, like VTR-Bench, target specific capabilities such as visual text rendering accuracy [14], while frameworks like WorldScore assess world generation models by focusing on controllability and scene dynamics [15]. Human-preference datasets, including AIGVE-60K, have become central to aligning generative outputs with human expectations [5].

## Recent Trends and Challenges
Recent research prioritizes structural and multimodal controllability [11]. The LongVie framework addresses temporal degradation in ultra-long video generation through an autoregressive approach [8]. For interaction, "Force Prompting" has introduced methods for utilizing physical forces (e.g., wind) as control signals to enable physically realistic responses without extensive simulation [9]. Programmatic approaches like MaLiang-Harness seek to bridge the "Program-to-Visual" gap, using MLLM-driven inspection to improve composition [10]. Despite these strides, open problems remain: maintaining physical plausibility over extended time horizons, consistent depth estimation, and ensuring reliability across diverse, non-human-centric domains are primary hurdles [11].

## References
[1] Evolution of Video Generative Foundations. web. https://arxiv.org/html/2604.06339v1 (n.d.)
[2] Video Diffusion Models. hf-search. https://huggingface.co/papers/2204.03458 (2022-04-07)
[3] Latte: Latent Diffusion Transformer for Video Generation. hf-search. https://huggingface.co/papers/2401.03048 (2024-01-05)
[4] A Survey on Video Diffusion Models. web. https://dl.acm.org/doi/full/10.1145/3696415 (2024-11-07)
[5] Generative AI Video Evaluation: Survey of Metrics, Benchmarks, and Trustworthiness. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/papers/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.pdf (n.d.)
[6] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://arxiv.org/abs/2311.17982 (2023-11-29)
[7] Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation. hf-daily. https://huggingface.co/papers/2610.05608 (2026-10-04)
[8] LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation. hf-search. https://huggingface.co/papers/2508.03694 (2025-08-05)
[9] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/953dcf20cbd275465066ad63ea111f13-Paper-Conference.pdf (n.d.)
[10] MaLiang-Harness: A Programmable Path to Image and Video Generation. hf-daily. https://huggingface.co/papers/2609.34309 (2026-09-28)
[11] Controllable Video Generation: A Survey. web. https://arxiv.org/html/2507.16869v2 (2026-01-16)
[12] Generative Adversarial Networks for Image and Video Synthesis: Algorithms and Applications. web. https://ar5iv.labs.arxiv.org/html/2008.02793 (n.d.)
[13] Diffusion Models for Video Prediction and Infilling. hf-search. https://huggingface.co/papers/2206.07696 (2022-06-15)
[14] VTR-Bench: A Systematic Benchmark for Evaluating Visual Text Rendering in Video Generation. hf-search. https://huggingface.co/papers/2610.01499 (2026-10-01)
[15] WorldScore: A Unified Evaluation Benchmark for World Generation. hf-search. https://huggingface.co/papers/2504.00983 (2025-04-01)
