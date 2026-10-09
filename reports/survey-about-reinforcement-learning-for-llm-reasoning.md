# Reinforcement Learning for LLM Reasoning: A Survey

## TL;DR
- Reinforcement learning enhances LLM reasoning by optimizing holistic criteria [1][2] and enabling stepwise verifiable rewards [2].
- PPO and GRPO dominate current approaches, but newer value-based methods offer improved sample efficiency [3][4].
- Benchmarking reasoning capabilities remains challenging, with pass@k metrics often masking limited reasoning coverage [5][6].
- Emerging trends focus on test-time compute, adaptive budget allocation, and the use of reinforcement-learned teachers [7][8][9].

## Background
Reinforcement learning (RL) has become central to aligning Large Language Models (LLMs) with complex reasoning goals. Unlike supervised fine-tuning, which focuses on token-level imitation, RL allows for the optimization of sequence-level objectives, such as logical coherence, factual accuracy, and multi-step problem-solving [1][2]. As LLMs are increasingly deployed for reasoning-intensive tasks, RL provides the necessary framework to incentivize correct outcome generation and stepwise reasoning in scenarios where human-labeled chains of thought are sparse or expensive to obtain [1][10].

## Paradigms of RL-based Reasoning
Training LLMs for reasoning with RL typically involves three core components: reward model training, preference-based fine-tuning, and policy optimization [1]. Paradigms range from traditional Reinforcement Learning from Human Feedback (RLHF), which utilizes human preferences [1], to Reinforcement Learning from AI Feedback (RLAIF), which scales feedback via model-based evaluations [1]. Outcome-based approaches, such as Reinforcement Learning with Verifiable Rewards (RLVR), are critical for mathematical and logical tasks, where the correctness of an answer can be programmatically verified [2]. Additionally, Simplified Preference Optimization (DPO) has emerged as an alternative that bypasses explicit reward modeling [1]. On-policy distillation (OPD) is another prominent technique, where agents learn by distilling knowledge from a policy that is continuously updated during training [11].

## Methodological Evolution
Current methodological development is defined by a shift from strictly policy-gradient-based methods to exploring value-based alternatives [4]. While proximal policy optimization (PPO) and group relative policy optimization (GRPO) are industry standards, they are fundamentally on-policy and sample-inefficient, requiring fresh samples for every update [4]. Research into the "value-gradient hypothesis" suggests that even critic-free methods implicitly leverage value-like signals via the actor's hidden states during backpropagation [3]. Emerging off-policy value-based methods, such as ReVal, address sample inefficiency by integrating replay buffers, enabling models to learn from historical reasoning trajectories more effectively [4]. Other developments, like S-GRPO and T-SPMO, specifically optimize these techniques for reasoning tasks under memory and compute constraints [12].

## Benchmarks and Evaluation
Evaluation of reasoning models frequently relies on benchmarks such as GSM8K, MATH500, and AMC23, measured primarily using pass@k metrics [5]. However, these metrics can be deceptive; while RL-trained models often show superior performance at low-k values, they may fail to scale their reasoning coverage for higher k, suggesting they might overfit to common solution paths [5]. To audit these limitations, new frameworks like the Oracle Performance Gap (OPG) have been introduced to test if models generalize beyond their training splits [6]. Furthermore, non-standard assessment methods, such as the TRACE framework, utilize metacognitive theories to evaluate the structural validity of chains of thought, providing a more granular reward signal than outcome-based verification alone [13].

## Trends and Open Problems
The field is moving toward scaling reasoning through test-time compute [7]. Rather than solely increasing model size, recent approaches focus on adaptive budget allocation and search-based inference procedures that allow the model to spend more compute on difficult problems [7][8]. Another critical trend is the use of Reinforcement-Learned Teachers (RLTs), which distill reasoning strategies into student models to improve exploration efficiency and reduce reliance on sparse outcome rewards [9]. Despite these advancements, significant open problems remain, including the need to extend RL theory from simple proxy rewards to binary outcomes and the challenge of fostering robust, multi-step reasoning without human supervision [14][10].

## References
[1] Reinforcement Learning Enhanced LLMs: A Survey. web. https://arxiv.org/html/2412.10400 (2024-12-10)
[2] A Technical Survey of Reinforcement Learning Techniques for Large Language Models. web. https://dl.acm.org/doi/full/10.1145/3834858 (2026-09-29)
[3] Value-Gradient Hypothesis of RL for LLMs. web. https://arxiv.org/html/2605.21654 (2026-05-20)
[4] Off-Policy Value-Based Reinforcement Learning for Large Language Models. web. https://ar5iv.labs.arxiv.org/html/2603.23355 (2026-03-20)
[5] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf (2025-12-01)
[6] Rethinking RL Evaluation: Can Benchmarks Truly Reveal Failures of RL Methods?. web. https://aclanthology.org/2026.findings-acl.769.pdf (2026-01-01)
[7] A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks. hf-search. https://huggingface.co/papers/2501.10069 (2025-01-17)
[8] Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs. hf-search. https://huggingface.co/papers/2507.02076 (2025-07-02)
[9] Reinforcement Learning Teachers of Test Time Scaling. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/9a6b278218966499194491f55ccf8b75-Paper-Conference.pdf (2025-12-01)
[10] Theoretical Analysis of Reinforcement Learning for LLM Reasoning. web. https://arxiv.org/pdf/2602.01523 (2026-02-01)
[11] Scaling Properties of Same-Family On-Policy Distillation. hf-daily. https://huggingface.co/papers/2609.32722 (2026-09-26)
[12] Reinforcement Learning for LLM Reasoning Under Memory Constraints. hf-search. https://huggingface.co/papers/2504.20834 (2025-04-29)
[13] TRACE: Toulmin-based Reasoning Assessment through Constructive Elements for LLM CoT Evaluation. hf-search. https://huggingface.co/papers/2605.29656 (2026-05-28)
[14] Reasoning Beyond Limits: Advances and Open Problems for LLM Reasoning. web. https://arxiv.org/abs/2503.22732 (2025-03-26)
