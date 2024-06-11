export SRC=https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main

# Download YCB-V
echo 'Downloading YCB-V'
wget $SRC/ycbv/ycbv_base.zip -P datasets          # Base archive with dataset info, camera parameters, etc.
wget $SRC/ycbv/ycbv_models.zip -P datasets        # 3D object models.
wget $SRC/ycbv/ycbv_test_bop19.zip -P datasets    # Test images.

# Download HOPE
echo 'Downloading HOPE'
wget $SRC/hope/hope_base.zip -P datasets            # Base archive with dataset info, camera parameters, etc.
wget $SRC/hope/hope_models.zip -P datasets          # 3D object models.
wget $SRC/hope/hope_val_realsense.zip -P datasets   # Test images.

# Download TYOL
echo 'Downloading TYOL'
wget $SRC/tyol/tyol_base.zip -P datasets         # Base archive with dataset info, camera parameters, etc.
wget $SRC/tyol/tyol_models.zip -P datasets       # 3D object models.
wget $SRC/tyol/tyol_test_all.zip -P datasets     # Test images.

# Unzip all files
echo 'Unzipping all files'
unzip datasets/hope_base.zip -d datasets                # Contains folder "hope".
unzip datasets/hope_models.zip -d datasets/hope         # Unpacks to "hope".
unzip datasets/hope_val_realsense.zip -d datasets/hope  # Unpacks to "hope".
unzip datasets/tyol_base.zip -d datasets                # Contains folder "tyol".
unzip datasets/tyol_models.zip -d datasets/tyol         # Unpacks to "tyol".
unzip datasets/tyol_test_all.zip -d datasets/tyol       # Unpacks to "tyol".
unzip datasets/ycbv_base.zip -d datasets                # Contains folder "ycbv".
unzip datasets/ycbv_models.zip -d datasets/ycbv         # Unpacks to "ycbv".
unzip datasets/ycbv_test_bop19.zip -d datasets/ycbv     # Unpacks to "ycbv".

# Remove all ZIP files
echo 'Removing all ZIP files'
rm datasets/*.zip
