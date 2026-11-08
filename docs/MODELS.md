# Model Provenance and Quantization Manifest (`docs/MODELS.md`)

## Runtime Environment

- **Conda Environment**: `rag` (`Python 3.12`)
- **`mlx` version**: `0.32.4`
- **`mlx-lm` version**: `0.32.2`
- **Quantization**: 4-bit affine quantization (`bits: 4`, `group_size: 64`) converted locally with `mlx_lm.convert`.
- **Offline Execution**: All validation and inference runs operate strictly offline (`HF_HUB_OFFLINE=1`) against local MLX directories.

---

## Model Provenance Table

| Study Model ID | Hugging Face Repo ID | Commit SHA | Local HF Path | Local MLX Path | Quantization (`bits` / `group_size`) | Effective Bits / Weight | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen3-4B` | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `models/hf/Qwen__Qwen3-4B` | `models/mlx/qwen3-4b-4bit` | `4` / `64` (affine) | `4.500` | Apache-2.0 |
| `Qwen3-8B` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `models/hf/Qwen__Qwen3-8B` | `models/mlx/qwen3-8b-4bit` | `4` / `64` (affine) | `4.501` | Apache-2.0 |
| `Llama-3.1-8B-Instruct` | `unsloth/Meta-Llama-3.1-8B-Instruct` | `a2856192dd7c25b842431f39c179a6c2c2f627d1` | `models/hf/unsloth__Meta-Llama-3.1-8B-Instruct` | `models/mlx/llama31-8b-4bit` | `4` / `64` (affine) | `4.500` | Llama 3.1 Community License |

### Note on `Llama-3.1-8B-Instruct` Mirror & Licensing

The `Llama-3.1-8B-Instruct` weights come from the community mirror `unsloth/Meta-Llama-3.1-8B-Instruct` because Meta's gated access request on Hugging Face was declined. The **Llama 3.1 Community License** applies to these weights and all derivative quantized artifacts.

---

## File Checksums (SHA-256)

### 1. `Qwen3-4B` (`Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`)

| Copy | File | SHA-256 |
| :--- | :--- | :--- |
| HF (`models/hf/Qwen__Qwen3-4B`) | `model-00001-of-00003.safetensors` | `328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223` |
| HF (`models/hf/Qwen__Qwen3-4B`) | `model-00002-of-00003.safetensors` | `6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5` |
| HF (`models/hf/Qwen__Qwen3-4B`) | `model-00003-of-00003.safetensors` | `e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1` |
| HF (`models/hf/Qwen__Qwen3-4B`) | `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| MLX (`models/mlx/qwen3-4b-4bit`) | `model.safetensors` | `365a6fff57490ca4e89e354c3295b7f2f60ac3fcb0a8a3fa06e21b80bb4d19ca` |
| MLX (`models/mlx/qwen3-4b-4bit`) | `tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |

### 2. `Qwen3-8B` (`Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`)

| Copy | File | SHA-256 |
| :--- | :--- | :--- |
| HF (`models/hf/Qwen__Qwen3-8B`) | `model-00001-of-00005.safetensors` | `31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f` |
| HF (`models/hf/Qwen__Qwen3-8B`) | `model-00002-of-00005.safetensors` | `5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282` |
| HF (`models/hf/Qwen__Qwen3-8B`) | `model-00003-of-00005.safetensors` | `c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836` |
| HF (`models/hf/Qwen__Qwen3-8B`) | `model-00004-of-00005.safetensors` | `b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a` |
| HF (`models/hf/Qwen__Qwen3-8B`) | `model-00005-of-00005.safetensors` | `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff` |
| HF (`models/hf/Qwen__Qwen3-8B`) | `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| MLX (`models/mlx/qwen3-8b-4bit`) | `model.safetensors` | `a1411b70b4cfb45016fc69228d58716de58413f5e694561eee8a82b20cfbebf5` |
| MLX (`models/mlx/qwen3-8b-4bit`) | `tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |

### 3. `Llama-3.1-8B-Instruct` (`unsloth/Meta-Llama-3.1-8B-Instruct` @ `a2856192dd7c25b842431f39c179a6c2c2f627d1`)

| Copy | File | SHA-256 |
| :--- | :--- | :--- |
| HF (`models/hf/unsloth__Meta-Llama-3.1-8B-Instruct`) | `model-00001-of-00004.safetensors` | `2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668` |
| HF (`models/hf/unsloth__Meta-Llama-3.1-8B-Instruct`) | `model-00002-of-00004.safetensors` | `09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15` |
| HF (`models/hf/unsloth__Meta-Llama-3.1-8B-Instruct`) | `model-00003-of-00004.safetensors` | `fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa` |
| HF (`models/hf/unsloth__Meta-Llama-3.1-8B-Instruct`) | `model-00004-of-00004.safetensors` | `92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b` |
| HF (`models/hf/unsloth__Meta-Llama-3.1-8B-Instruct`) | `tokenizer.json` | `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b` |
| MLX (`models/mlx/llama31-8b-4bit`) | `model.safetensors` | `fe36d962309a5c379bc203996f56a6dde228652d0bb3ed78d4bf60dc9ea7da6c` |
| MLX (`models/mlx/llama31-8b-4bit`) | `tokenizer.json` | `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b` |
