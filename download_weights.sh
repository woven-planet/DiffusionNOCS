# DINOv2 Small
wget "https://huggingface.co/facebook/dinov2-small/raw/main/config.json" -P ./diffusion_nocs/resources/dinov2-small/
wget "https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors" -P ./diffusion_nocs/resources/dinov2-small/

# DiffusionNOCS
wget "https://huggingface.co/TRI-ML/DiffusionNOCS/resolve/main/category6.pt" -P ./diffusion_nocs/resources/
wget "https://huggingface.co/TRI-ML/DiffusionNOCS/resolve/main/pca6.pkl" -P ./diffusion_nocs/resources/
