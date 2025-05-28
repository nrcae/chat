# AI Chat
LLM Chat with efficient, local quantized 7B Model.

## Model Preparation

1.  Download a GGUF model file (e.g., `model.gguf`).
2.  Place the `model.gguf` file in a directory that will be used for the Kubernetes PersistentVolume. For K3d, the `pv-pvc-k3d.yaml` assumes the model is at `/tmp/model.gguf` on your K3d host machine (or the machine where K3d nodes run). Modify this path in `pv-pvc-k3d.yaml` if your model is elsewhere.

## Configuration

The chatbot can be configured via command-line arguments to `chat.py` or environment variables in the `Dockerfile`.

### `chat.py` Command-Line Arguments

| Argument              | Default Value        | Description                                                                   |
| --------------------- | -------------------- | ----------------------------------------------------------------------------- |
| `--model-path`        | `app/model/model.gguf` | Path to the GGUF model file.                                                  |
| `--n-gpu-layers`      | `-1`                 | Number of layers to offload to GPU (-1 for model default, 0 for CPU only).    |
| `--n-threads`         | `None` (all cores)   | Number of CPU threads to use.                                                 |
| `--n-ctx`             | `2048`               | Context window size for the model.                                            |
| `--max-tokens-response`| `350`                | Maximum number of tokens the AI will generate in a single response.          |
| `--chat-format`       | `mistral-instruct`   | Chat format string to use with `llama-cpp-python` (e.g., `llama-2`, `chatml`). |

## Kubernetes Deployment
k3d cluster create chatbot-k3s-cluster \
  --volume ~/VSCode/ai_chat/model/:/mnt/k3d_model_volume@all \
  --agents


Delete with: k3d cluster delete chatbot-k3s-cluster
Import: k3d image import chatbot-k3s-app:latest -c chatbot-k3s-cluster
