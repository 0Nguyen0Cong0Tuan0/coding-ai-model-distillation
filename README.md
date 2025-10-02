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
 
## Dataset
Train dataset information
- Name: google-research-datasets/mbpp (full)
- Features: ['task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list']
- Number of rows: 374

Evaluation dataset information
- Name: openai/openai_humanevel
- Features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point']
- Number of rows: 164

## The teacher model
- StarCoder2-3B (pre-trained on code)
- Use bfloat16 for efficiency and auto device mapping
- Set to eval mode for inference (soft labels in distillation)
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


