# A Survey of LLM Agents and Tool Use

## TL;DR
- LLM agents are defined by their ability to perceive, reason, and interact with external tools to complete complex tasks [1][2].
- Architectures like ReAct enable structured reasoning by decoupling thought processes from tool invocations [3][4].
- Evaluation of agent tool use remains challenging, transitioning from simple function-calling tests to complex, multi-turn benchmarks [5][6].
- Recent research focuses on multi-agent collaboration, security, and mitigating failure modes like "co-cheating" in self-evolving systems [7][8].

## Background
LLM agents represent an evolution from passive text generation to active, autonomous problem solving. By integrating components for memory, planning, and action, they can ground their reasoning in external tools, making them essential for complex automation tasks [1]. This paradigm shift moves away from zero-shot prompting towards systems capable of executing multi-step workflows, which is fundamental to modern agentic frameworks [2].

## Agent Architectures and Frameworks
Modern agent development relies on modular frameworks that decouple internal reasoning from external tool interactions. The ReAct (Reasoning + Action) pattern is a cornerstone of this approach, where models are designed to generate explicit reasoning steps to inform tool usage [4]. Frameworks such as LangChain and AgentLite facilitate this by providing standard interfaces for tool access, state management, and multi-agent orchestration [3][9]. Other approaches, such as LLM-Agent-UMF, emphasize unified modeling to provide clear boundaries between components, while multi-LLM designs modularize planning and execution to improve performance over single-model systems [10][11].

## Tool Use and Reasoning Paradigms
Tool use is fundamentally a process of translating abstract goals into concrete environment interactions. While foundational research focused on static tool selection, newer paradigms like self-evolving agents aim to improve tool-calling capabilities without explicit training datasets [12]. However, the proliferation of available tools introduces efficiency challenges; methodologies like ToolScope help address this by auditing and merging redundant tools to simplify the decision-making context for the agent [13].

## Benchmarking and Evaluation
Evaluating agent efficacy requires moving beyond static metrics to assess interaction and stateful reliability [6]. Metrics such as Success Rate (SR) and execution accuracy are standard, but the field now incorporates qualitative assessment (LLM-as-a-judge) and complex task-completion benchmarks [5]. Emerging testing focuses on specialized capabilities, such as DecepEval, which probes the potential for deceptive behaviors in autonomous agents [14].

## Trends and Open Problems
The field is moving toward multi-agent collaboration and long-horizon autonomy, exemplified by platforms like Raven for building complex harnesses [15]. This complexity brings significant risks, including novel security threats like coordinated swarm attacks and "co-cheating," where models optimize for internal reward over task correctness [7][8]. Critical open problems include achieving bounded authority, developing auditable agent behavior, and establishing robust security mechanisms for agents that operate in enterprise environments [16][17][8].

## References
[1] From language to action: a review of large language models as autonomous agents and tool users. web. https://link.springer.com/article/10.1007/s10462-025-11471-9 (2026-01-06)
[2] A Review of Prominent Paradigms for LLM-Based Agents. web. https://aclanthology.org/2025.coling-main.652.pdf (2025-01-01)
[3] LangChain (Agent Engineering Platform). web. https://www.langchain.com/langchain (2025-01-01)
[4] ReAct Agent Implementation (LangChain OpenTutorial). web. https://langchain-opentutorial.gitbook.io/langchain-opentutorial/15-agent/11-react-agent (2025-01-01)
[5] Evaluation and Benchmarking of LLM Agents: A Survey. web. https://arxiv.org/html/2507.21504v1 (2025-07-29)
[6] A Comprehensive Survey of Benchmarks for Evaluating Tool and Function Calling in Large Language Models. web. https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information/blob/main/A%20Comprehensive%20Survey%20of%20Benchmarks%20for%20Evaluating%20Tool%20and%20Function%20Calling%20in%20Large%20Language%20Models.md (2025-01-01)
[7] Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents. web. https://arxiv.org/abs/2505.02077 (2026-04-29)
[8] False Frontiers: Diagnosing and Mitigating Co-Cheating in Self-Evolving Search Agents. hf-daily. https://huggingface.co/papers/2609.39102 (2026-09-30)
[9] AgentLite: A Lightweight Library for Building and Advancing Task-Oriented LLM Agent System. hf-search. https://huggingface.co/papers/2402.15538 (2024-02-23)
[10] Small LLMs Are Weak Tool Learners: A Multi-LLM Agent. hf-search. https://huggingface.co/papers/2401.07324 (2024-01-14)
[11] LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework. hf-search. https://huggingface.co/papers/2409.11393 (2024-09-17)
[12] Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data. hf-search. https://huggingface.co/papers/2602.21320 (2026-02-24)
[13] ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering. hf-search. https://huggingface.co/papers/2510.20036 (2026-05-08)
[14] DecepEval: A Benchmark for Evaluating Deception in LLM Agents. hf-daily. https://huggingface.co/papers/2610.07967 (2026-10-06)
[15] Raven: The Harness of Harnesses for Composable Agentic Intelligence. hf-daily. https://huggingface.co/papers/2609.33439 (2026-09-30)
[16] Evaluation and Benchmarking of LLM Agents: A Survey. hf-search. https://huggingface.co/papers/2507.21504 (2025-07-29)
[17] LLM-Based Agents for Software and Systems Security: Approaches, Applications, and Assessment. web. https://arxiv.org/abs/2608.28490 (2026-08-28)
