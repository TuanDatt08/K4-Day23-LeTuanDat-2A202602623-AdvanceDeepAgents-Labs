# World Models: A Survey of Foundations, Architectures, and Embodied Applications

## TL;DR
- World models function as internal simulators that allow agents to predict dynamics and plan actions by approximating physical processes [1][2].
- Modern architectures like Joint Embedding Predictive Architectures (JEPA) move beyond pixel-level reconstruction toward efficient latent-space planning [3][4].
- Evaluation is shifting from visual fidelity to closed-loop task success in robotics, using benchmarks that measure physical plausibility and simulative reasoning [5][6].
- Research trends increasingly focus on interactive, decision-coupled simulations and the integration of causal reasoning for embodied agents [7][8].

## Background
World models are internal representations that allow agents to simulate future scenarios, infer states, and evaluate potential interventions without requiring real-world interactions [2]. The foundational concept stems from the idea that biological organisms thrive by precomputing strategies within "working models" of their physical environment [2]. Early research in Predictive State Representations (PSRs) established a framework for using future-looking statistics rather than past history to represent current states [9]. In recent years, the maturation of large-scale generative models has re-invigorated this field, shifting the focus toward models that can serve as simulators for autonomous systems and embodied AI [1][7].

## Foundations and Functional Roles
At its core, a world model is a finite-resource approximation of the environment's state-transition processes [2]. These models are typically categorized by their roles in internal simulation: renderers that generate future observations, simulators that propagate dynamics, and planners that select actions [2]. PSR-based frameworks have historically provided a mathematical foundation for these models by representing the state as a vector of outcomes for future "tests" [9]. Modern interpretations differentiate between explicit models—which denoise future frames for visual prediction—and latent models that bypass pixel reconstruction to prioritize computational efficiency, particularly in real-time control scenarios [10].

## Modern Architectures: From Generative to Latent Prediction
Current research trends prioritize efficiency and task-relevance, leading to the development of Joint Embedding Predictive Architectures (JEPA) [3]. Unlike traditional generative models, JEPAs perform prediction in a shared latent space, which reduces computational overhead while maintaining high accuracy for downstream planning [4]. For instance, Auto-JEPA demonstrates the effectiveness of predicting future intent in autonomous driving without dense scene reconstruction [11]. Variational approaches, such as VJEPA, have further extended this by introducing probabilistic frameworks that enable uncertainty-aware belief propagation and planning [3]. Conversely, while explicit generative models provide high-fidelity visual outputs, they often encounter challenges with generalization compared to their latent counterparts [10].

## Evaluation Benchmarks for Embodied Systems
The evaluation of world models is undergoing a significant transition from open-loop metrics, such as visual fidelity, toward closed-loop benchmarks [6]. New protocols like RoboWM-Bench evaluate the "bridge" between predicted behaviors and executable actions, testing spatial reasoning and contact stability in robotic manipulation [5]. Other frameworks, such as the World Reasoning Arena, focus on simulative reasoning and long-horizon forecasting rather than simple frame prediction [12]. Additionally, environments like WorldAuditBench emphasize multimodal perception, requiring agents to identify physical anomalies in 3D interactive spaces, which highlights the need for robust action-perception coupling in embodied AI [13].

## Trends and Open Problems
The field is currently moving away from passive video generation toward interactive simulations that support embodied training [7]. A critical challenge remains the accumulation of errors in long-horizon rollouts, which impacts temporal consistency [14]. Furthermore, while integrating large language models with causal representation learning has shown promise for enhancing reasoning [8], world models still struggle with strict adherence to physical laws in complex scenarios [7]. Achieving a balance between computational efficiency for real-time control and the high-fidelity representation needed for safe physical interaction remains a central research goal [14].

## References
[1] Understanding World or Predicting Future? A Comprehensive Survey of World Models. hf-search. https://huggingface.co/papers/2411.14499 (2024-11-21)
[2] A Definition and Roadmap for World Models. web. https://arxiv.org/html/2607.06401 (2026-07-01)
[3] VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models. web. https://arxiv.org/html/2601.14354 (2026-01-26)
[4] UniJEPA: A Unified Joint-Embedding Predictive Architecture for Task-Agnostic Visual World Modeling. web. https://arxiv.org/abs/2608.07409 (2026-08-07)
[5] RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation. web. https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/papers/Jiang_RoboWM-Bench_A_Benchmark_for_Evaluating_World_Models_in_Robotic_Manipulation_CVPRW_2026_paper.pdf (2026-06-01)
[6] Survey: Do World Models Make Better Robots?. web. https://arxiv.org/abs/2609.29669 (2026-09-29)
[7] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[8] Language Agents Meet Causality -- Bridging LLMs and Causal World Models. hf-search. https://huggingface.co/papers/2410.19923 (2024-10-25)
[9] Predictive Representations of State. web. https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf (2001-01-01)
[10] What Makes World Action Models Generalize? An Empirical Study of Test-Time Future Modeling. hf-daily. https://huggingface.co/papers/2609.34981 (2026-09-29)
[11] Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving. hf-search. https://huggingface.co/papers/2607.29031 (2026-07-31)
[12] World Reasoning Arena. hf-search. https://huggingface.co/papers/2603.25887 (2026-03-26)
[13] WorldAuditBench: Interactive 3D World Auditing with Multimodal Agents. hf-daily. https://huggingface.co/papers/2609.40325 (2026-09-30)
[14] A Comprehensive Survey on World Models for Embodied AI. web. https://arxiv.org/html/2510.16732v1 (2025-10-19)
