
DATA_ROOT_DIR="/media/vincent/HDD-02/gc4dgs/data"
MVS_ROOT_PATH="/media/vincent/HDD-02/gc4dgs/mvs_modules/configs/config_mvsformer.json"
DV2_METRIC_ROOT_PATH="/media/vincent/HDD-02/gc4dgs/utils/DV2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"

DATASETS=(
    dynerf
    )

SCENES=(
    coffee_martini
    cook_spinach
    cut_roasted_beef
    flame_salmon
    flame_steak
    sear_steak
    )

TRAIN_VIEWS=(
            # 2
            3
            # 4
            # 5
            )

RESOLUTION=(
            # 1
            2
            # 4
            # 8
            )

RESOLUTION2=(
            1352,1014
            )

TRAIN_CONFIGS=(
    MVS
    # COLMAP
    # dust3r
    # mast3r
    ) 

for SCENE in "${SCENES[@]}"; do

# ----- (1) MVSFormer_Geometric_Initialization -----
    SCENE_PATH=${DATA_ROOT_DIR}/${DATASETS}/${SCENE}

    CMD_1="python ./scripts/pre_dynerf/pose2MVS.py \
    --path ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --train_views ${TRAIN_VIEWS} \
    --resolution ${RESOLUTION} \
    --scene ${SCENE} \
    --colmap True \
    "

    CMD_2="python ./mvs2points.py \
    --path ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --mvs_config ${MVS_ROOT_PATH} \
    --dataset ${DATASETS} \
    --resolution ${RESOLUTION2} \
    "

    # Downsampling of Dense Points from COLMAP and MVSFormer
    CMD_3="python ./utils/o3dPre.py \
    --mvs_input ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --output_path ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --colmap True \
    --mvsformer True\
    "

    # ----- (2) DepthAnythingV2_For_Depth_Estimation -----
    CMD_4="python ./utils/DV2/metric_depth/MeDV2run.py \
    --encoder vitl \
    --load-from ${DV2_METRIC_ROOT_PATH} \
    --max-depth 26.5 \
    --img-path ${SCENE_PATH}/${TRAIN_VIEWS}_views/images \
    --outdir ${SCENE_PATH}/${TRAIN_VIEWS}_views/depths \
    --input-size 1000 \
    --pred-only \
    --save-numpy \
    --grayscale\
    "

    # ----- (3) Train, Render and Metrics  -----
    CMD_5="python ./train.py \
    --config configs/${DATASETS}/${SCENE}_${TRAIN_VIEWS}_views_${TRAIN_CONFIGS}.yaml \
    --pcd_init ${TRAIN_CONFIGS} \
    "

    CMD_6="python ./render.py \
    --model_path output/${DATASETS}/${SCENE}/${TRAIN_VIEWS}_views/${TRAIN_CONFIGS} \
    --loaded_pth output/${DATASETS}/${SCENE}/${TRAIN_VIEWS}_views/${TRAIN_CONFIGS}/chkpnt_best.pth \
    --pcd_init ${TRAIN_CONFIGS} \
    "

    CMD_7="python ./metrics.py \
    --model_path output/${DATASETS}/${SCENE}/${TRAIN_VIEWS}_views/${TRAIN_CONFIGS} \
    "


    echo "========= ${SCENE}: MVSFormer_Geometric_Initialization   ========="
    # eval $CMD_1
    # eval $CMD_2
    # echo "========= ${SCENE}: Downsampling of Dense Points         ========="
    # eval $CMD_3
    # echo "========= ${SCENE}: DepthAnythingV2_For_Depth_Estimation ========="
    # eval $CMD_4
    echo "========= ${SCENE}: Train_Render_and_Metrics             ========="
    eval $CMD_5
    eval $CMD_6
    eval $CMD_7
done