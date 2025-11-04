# The File ID is the long string in the middle of your URL
FILE_ID="1GKssE5xh4VbVKZMFlrm5XwRP_Q0nC75t"

# You can also just give gdown the full URL
gdown "https://drive.google.com/file/d/1GKssE5xh4VbVKZMFlrm5XwRP_Q0nC75t/view?usp=drive_link"

# Or, to specify the output name:
gdown --output chessformer_epoch_13.pth "https://drive.google.com/file/d/1GKssE5xh4VbVKZMFlrm5XwRP_Q0nC75t/view?usp=drive_link"