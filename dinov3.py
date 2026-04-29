from transformers import DINOv3ViTModel


model = DINOv3ViTModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m"
)
print("Model loaded successfully, hidden_size =", model.config.hidden_size)
