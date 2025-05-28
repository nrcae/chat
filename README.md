# AI Chat
LLM Chat with efficient, local quantized 7B Model.

# Kubernetes Deployment
k3d cluster create chatbot-k3s-cluster \
  --volume ~/VSCode/ai_chat/model/:/mnt/k3d_model_volume@all \
  --agents


Delete with: k3d cluster delete chatbot-k3s-cluster
Import: k3d image import chatbot-k3s-app:latest -c chatbot-k3s-cluster
