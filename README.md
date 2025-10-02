# Coding AI model distillation

## Introduction
A code generation AI model using knowledge distillation. 
- The teacher model will be a large pre-trained model (e.g., StarCoder or a similar open-source code model), which has been trained on massive datasets like StarCoderData. 
- The student model will be a smaller model built from scratch and trained via distillation to mimic the teacher to make it more efficient in terms of size, inference speed and compute requirements while maintaining reasonable performance.

The distillation process related to transfers knowledge from the teacher (soft probabilities) to the student and often leading to better generalization than training the student on raw data alone. 

## My approach
- Teacher model use a pre-trained model from Hugging Face (bigcode/starcoder2-3b, ~3B parameters)
- Student model use a small transformer (~125M parameters, like a mini-GPT) built from scratch using Python
- Efficiency proof by comparing model size, inference latency, and benchmark scores (pass@1 on HumanEval)
- Tools:
    - Hugging Face Transformers/Datasets for model handling and data loading
    - PyTorch or custom training
    - Weights & Biases (W&B) for logging (experiement tracking)
 
![ads](https://wandb.ai/nguyencongtuan/code-distillation?nw=nwusernguyencongtuan0810&panelDisplayName=train_loss&panelSectionName=Charts)
