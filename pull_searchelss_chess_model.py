from huggingface_hub import snapshot_download
model_path = snapshot_download(
    repo_id="dbest-isi/searchless-chess-9M-selfplay",
    local_dir="./searchless_chess_model"
)
print(f"Model downloaded to: {model_path}")
