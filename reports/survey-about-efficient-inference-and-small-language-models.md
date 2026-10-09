# Survey on Efficient Inference and Small Language Models

## TL;DR
- Small Language Models (SLMs) balance performance and efficiency, enabling deployment on resource-constrained devices [1][2].
- Model compression techniques like quantization, pruning, and knowledge distillation are essential for reducing memory and computational overhead [3][4].
- Hardware-aware optimization, including specialized kernels and runtime engines, maximizes throughput and energy efficiency [5][6][7].
- Emerging trends focus on disaggregated quantization [8], energy-aware benchmarking [9], and longitudinal agent evaluation [10].

## Background
The demand for deploying Large Language Models (LLMs) on edge and consumer hardware has driven interest in Small Language Models (SLMs) and efficient inference [1]. SLMs aim to maintain high accuracy and adaptability while adhering to strict constraints on hardware, latency, and privacy [1][2]. As compute demands grow, bridging the performance gap between compact models and their larger counterparts through architectural innovation and optimization has become a central challenge in AI [2].

## Architectural Motivations for SLMs
SLMs are designed to address the scalability and efficiency limitations of massive Transformer models [1]. Beyond standard architectures, new designs like Mamba and xLSTM offer linear-scaling inference, avoiding the quadratic complexity associated with traditional attention mechanisms [2]. These models are not only useful for standalone deployment on mobile or edge devices but also function effectively as specialized reward or proxy models to support larger LLM ecosystems [2]. Furthermore, techniques such as test-time compute scaling are increasingly used to elevate SLM performance [2], while fine-tuning on domain-specific behavioral data allows SLMs to match large model baselines in targeted applications [11].

## Model Compression Strategies
To fit large models into memory or improve runtime, various compression techniques are employed [3][4]. Quantization reduces parameter bit-precision, with Post-Training Quantization (PTQ) offering immediate benefits and Quantization-Aware Training (QAT) providing better performance recovery through retraining [4]. Pruning removes redundant layers or neurons—either through structured or unstructured methods—to streamline execution [4]. Knowledge distillation transfers capabilities from a large teacher model to a smaller student, using both black-box and white-box approaches [4]. Recent advancements specifically address the memory and accuracy bottlenecks of recurrent states in linear attention models by modeling the spatial and temporal impact of quantization errors [12], or by disaggregating quantization strategies across different prefill and decoding phases [8].

## Hardware-Aware Optimization
Efficiency relies heavily on the synergy between model algorithms and the underlying hardware [7]. Libraries like FlashInfer [6] and TensorRT-LLM [5] provide optimized kernels for attention and Mixture-of-Experts (MoE) operations, while frameworks like Llaminar [13] support heterogeneous deployments by treating CPU and GPU sockets as first-class resources. Innovations also extend to FPGA accelerators, which can shift computationally expensive operations to memory-based look-up tables to improve energy efficiency [14]. Parallel prompt decoding is another key technique to reduce parameter overhead, directly accelerating inference throughput [15].

## Trends and Open Problems
Efficient deployment is increasingly focused on managing the KV cache, which remains a dynamic memory bottleneck [7]. Strategies such as paged attention and the separation of prefill and decoding stages are common in modern serving systems [7][16]. As deployment grows, the need for specialized benchmarks becomes critical; current initiatives include TraceDance for agent behavior [17], energy-efficiency monitoring [9], and evaluating longitudinal memory in AI companions [10]. These efforts highlight an ongoing shift from generic model metrics toward deployment-specific objectives, balancing system-level reliability with individual user experience.

## References
[1] A Survey on Small Language Models. web. https://aclanthology.org/2025.ranlp-1.93.pdf (n.d.)
[2] A Survey on Small Language Models in the Era of Large Language Models: Architecture, Capabilities, and Trustworthiness. web. https://dl.acm.org/doi/10.1145/3711896.3736563 (2025-08-03)
[3] A Survey on Model Compression for Large Language Models. hf-search. https://huggingface.co/papers/2308.07633 (2023-08-15)
[4] A Survey on Model Compression for Large Language Models. web. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482/A-Survey-on-Model-Compression-for-Large-Language (2024-11-27)
[5] TensorRT-LLM. web. https://github.com/NVIDIA/TensorRT-LLM/ (n.d.)
[6] FlashInfer. web. https://github.com/arpera/flashinfer (n.d.)
[7] Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems. web. https://dl.acm.org/doi/full/10.1145/3754448 (2025-09-04)
[8] Disaggregated Quantization: Specializing LLM Prefill and Decode. hf-daily. https://huggingface.co/papers/2609.26333 (2026-09-22)
[9] Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures. hf-search. https://huggingface.co/papers/2604.09048 (2026-04-10)
[10] RealCompanion: Benchmarking Human Understanding from Reasoning over Longitudinal Real-World Conversations. hf-daily. https://huggingface.co/papers/2610.01780 (2026-10-01)
[11] Small Foundation Models of Human Cognition and Behaviour. hf-search. https://huggingface.co/papers/2608.05224 (2026-08-09)
[12] STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization. hf-daily. https://huggingface.co/papers/2609.38169 (2026-09-29)
[13] Llaminar. web. https://github.com/Llaminar/llaminar (n.d.)
[14] LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs. hf-search. https://huggingface.co/papers/2511.06174 (2025-11-09)
[15] Hardware-Aware Parallel Prompt Decoding for Memory-Efficient Acceleration of LLM Inference. hf-search. https://huggingface.co/papers/2405.18628 (2024-05-28)
[16] Taming the Titans: A Survey of Efficient LLM Inference Serving. web. https://aclanthology.org/2025.inlg-main.32.pdf (n.d.)
[17] TraceDance: An Automated System for Building Agent Behavior Benchmarks from Real-World Agent Deployment Traces. hf-search. https://huggingface.co/papers/2609.33295 (2026-09-27)
