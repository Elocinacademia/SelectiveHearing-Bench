# SelectiveHearing-Bench
Large audio language models (LALMs) are increasingly deployed in real-world settings where they inevitably capture speech from unintended nearby bystanders, raising privacy risks that existing benchmarks and defenses largely overlook. We introduce SH-Bench, the first benchmark designed to evaluate selective hearing: a model’s ability to attend to an intended main speaker while refusing to process or reveal information about incidental bystander speech. SH-Bench contains 3,968 multi-speaker audio mixtures spanning both real-world and synthetic scenarios, paired with 77k multiple-choice questions that probe models under general and selective operating modes. We propose Selective Efficacy (SE), a unified metric capturing both multi-speaker comprehension and bystander-privacy protection. Our evaluation of state-of-the-art open-source and proprietary LALMs reveals substantial privacy leakage, with strong audio understanding failing to translate into selective protection of bystander privacy. To mitigate this gap, we introduce Bystander Privacy Fine-Tuning (BPFT), a training pipeline that teaches models to refuse bystander-related queries without degrading main-speaker comprehension. BPFT yields substantial gains which improve SE by up to 15.9\% over Gemini 2.5 Pro, demonstrating that selective hearing is learnable but far from achieved in current LALMs. SH-Bench and BPFT provide the first systematic framework for measuring and improving bystander privacy in audio foundation models.

<div style='display:flex; gap: 0.25rem; '>
<a href='paper'><img src='https://img.shields.io/badge/arXiv-PDF-red'></a>
<a href='https://huggingface.co/datasets/BrianatCambridge/SelectiveHearingBench'><img src='https://img.shields.io/badge/dataset-SHBench-blue'></a> 
<a href='https://huggingface.co/BrianatCambridge/Qwen2.5_Omni_BPFT'><img src='https://img.shields.io/badge/checkpoint-Qwen2.5_Omni_BPFT-yellow'></a>
<a href='https://huggingface.co/BrianatCambridge/StepAudio2mini_BPFT'><img src='https://img.shields.io/badge/checkpoint-Step_Audio_2_mini_BPFT-yellow'></a>
</div>

## Preparation
Download data from huggingface. Put data into local dir, e.g. `dataset`\

## Inference with Qwen2.5-Omni
Run in general mode:\
`python inference_qwen25_omni.py dataset/all_questions_descfilter_bg.json basic`
You need to change to your actual data directory.

Run in selective mode:\
`python inference_qwen25_omni.py dataset/all_questions_descfilter_bg.json selective`

## Evaluation
Evaluate Qwen 2.5 Omni tested under general mode
`evaluate.py results/qwen25omni_basic_qa.json`

## Train with BPFT
Qwen2.5-Omni as an example (8 GPU)\
`bash run.sh`

## Inference with Qwen2.5-Omni + BPFT
`bash test.sh`

