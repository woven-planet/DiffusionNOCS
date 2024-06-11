# Check if --save_models argument is provided to the shell script
save_models_flag=""
if [ "$1" == "--save_models" ]; then
    save_models_flag="--save_models"
fi

# Convert YCB-V
echo 'Converting YCB-V'
python3 scripts/convert2nocs.py --data_dir datasets/ycbv --output_dir datasets/ycbv_nocs --data_split test $save_models_flag

# Convert HOPE
echo 'Converting HOPE'
python3 scripts/convert2nocs.py --data_dir datasets/hope --output_dir datasets/hope_nocs --data_split val $save_models_flag

# Convert TYOL
echo 'Converting TYOL'
python3 scripts/convert2nocs.py --data_dir datasets/tyol --output_dir datasets/tyol_nocs --data_split test $save_models_flag
