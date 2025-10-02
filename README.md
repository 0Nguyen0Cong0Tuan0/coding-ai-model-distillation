# Coding AI model distillation

## Introduction
A code generation AI model using knowledge distillation. 
- The teacher model will be a large pre-trained model (e.g., StarCoder or a similar open-source code model), which has been trained on massive datasets like StarCoderData. 
- The student model will be a smaller model built from scratch and trained via distillation to mimic the teacher to make it more efficient in terms of size, inference speed and compute requirements while maintaining reasonable performance.

The distillation process related to transfers knowledge from the teacher (soft probabilities) to the student and often leading to better generalization than training the student on raw data alone. 

## My approach
- Teacher model use a pre-trained model from Hugging Face (bigcode/starcoder2-3b, ~3B parameters).
- Student model use a small transformer (~44M parameters, like a mini-GPT) built from scratch using Python.
- Efficiency proof by comparing model size, inference latency, and benchmark scores (pass@1 on HumanEval).
- Tools:
    - Hugging Face Transformers/Datasets for model handling and data loading.
    - PyTorch or custom training.
    - Weights & Biases (W&B) for logging (experiement tracking).
 
## Dataset
Train dataset information
- Name: google-research-datasets/mbpp (full).
- Features: ['task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list'].
- Number of rows: 374.

Evaluation dataset information
- Name: openai/openai_humanevel.
- Features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'].
- Number of rows: 164.

## The teacher model
- StarCoder2-3B (pre-trained on code).
- Use bfloat16 for efficiency and auto device mapping.
- Set to eval mode for inference (soft labels in distillation).
```markdown
Starcoder2ForCausalLM(
  (model): Starcoder2Model(
    (embed_tokens): Embedding(49152, 3072)
    (layers): ModuleList(
      (0-29): 30 x Starcoder2DecoderLayer(
        (self_attn): Starcoder2Attention(
          (q_proj): Linear(in_features=3072, out_features=3072, bias=True)
          (k_proj): Linear(in_features=3072, out_features=256, bias=True)
          (v_proj): Linear(in_features=3072, out_features=256, bias=True)
          (o_proj): Linear(in_features=3072, out_features=3072, bias=True)
        )
        (mlp): Starcoder2MLP(
          (c_fc): Linear(in_features=3072, out_features=12288, bias=True)
          (c_proj): Linear(in_features=12288, out_features=3072, bias=True)
          (act): PytorchGELUTanh()
        )
        (input_layernorm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)
      )
    )
    (norm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)
    (rotary_emb): Starcoder2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=49152, bias=False)
)
```

## The student models

Student models:
| Name | Architecture | #Params | #Layers | #Embeddings | #Heads | #Positions | 
| --- | --- | --- | --- | --- | --- | --- |
| model_44m_t1.5_a0.3 | GPT2-like | 6 | 512 | 8 | 512 | 44M | 
| model_44m_t1.7_a0.6 | GPT2-like | 6 | 512 | 8 | 512 | 44M | 
| model_44m_t2.0_a0.5 | GPT2-like | 6 | 512 | 8 | 512 | 44M |   
| model_202m_t2.0_a0.5 | GPT2-like | 202M | 12 | 1024 | 16 | 512 |
| model_202m_t2.0_a0.7 | GPT2-like | 202M | 12 | 1024 | 16 | 512 | 

Hyperparameters:
| Temperature | Alpha | Epochs |  
| 1.5 | 0.3 | 7 | 
| 1.7 | 0.6 | 7 | 
| 2.0 | 0.5 | 7 |
| 2.0 | 0.5 | 7 |  
| 2.0 | 0.7 | 7 |

## Training and evaluation setup
*Training*
- The DataLoaders shuffle the training dataset but not evaluation dataset.
- Temperature (softens logits) and alpha (balances KL div (Kullback–Leibler divergence) and cross-entropy (CE) loss.
- Loop over hyperparameters and students by training with KL div and CE loss. Log to W&B and then save models.

*Evaluation*
- Init Weights & Biases (W&B) for eval logging/tracking.
- On the metrics of code_eval for Pass@k (code correctness) and BLEU for similarity.
- Trying on evaluation on just 20 problems (Because my device does not have enough resources for efficient distillation process).

## Results
| Model | Params (M) | Pass@1 | Pass@10 | BLEU | Avg Latency (s) |
| Teacher model (StarCode2-3B) | 3030.371 | 0.05 (avg) | 0.52 (avg) | 0.007923 (avg) | 52.448 |
| model_t2.0_a0.5 | 202.013 | 0.8 | 1 | 0.01424 | 4.725 |
| model_t2.0_a0.7 | 202.013 | 1 | 0.95 0.008332 | 4.926 |
| model_t1.7_a0.6 | 44.343 | 0.5 | 0.95 | 0.01512 | 2.489 | 
| model_t1.5_a0.3 | 44.343 | 0.95 | 0.95 | 0.01677 | 2.478 |
| model_t2.0_a0.5 | 44.343 | 1 | 0.9 | 0.02144 | 2.491 |




