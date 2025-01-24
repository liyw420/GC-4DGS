
DATA_ROOT_DIR="/media/vincent/HDD-02/fs4dgs/data"
MVS_ROOT_PATH="/media/vincent/HDD-02/fs4dgs/mvs_modules/configs/config_mvsformer.json"
DV2_METRIC_ROOT_PATH="/media/vincent/HDD-02/fs4dgs/utils/DV2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"

DATASETS=(
    technicolor
    )

SCENES=(
    Painter
    Theater
    Train
    Remy
    Birthday
    )

TRAIN_VIEWS=(
            3
            # 4
            # 5
            # 15
            )

RESOLUTION=(
            # 1
            2
            # 4
            # 8
            )

RESOLUTION2=(
            1024,544
            )

TRAIN_CONFIGS=(
    # MVS
    COLMAP
    )

for SCENE in "${SCENES[@]}"; do

    # ----- (1) MVSFormer_Geometric_Initialization -----
    SCENE_PATH=${DATA_ROOT_DIR}/${DATASETS}/${SCENE}

    CMD_1="python ./scripts/pre_technicolor/pre_technicolor.py \
    --videopath ${SCENE_PATH} \
    --train_views ${TRAIN_VIEWS} \
    --resolution ${RESOLUTION} \
    "

    CMD_2="export PYTHONPATH=\$PYTHONPATH:$(dirname "$DATA_ROOT_DIR")"

    CMD_3="python ./scripts/pre_technicolor/imgs2poses.py \
    --match_type exhaustive_matcher \
    --scenedir ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    "

    CMD_4="python ./scripts/pre_technicolor/pose2MVS.py \
    --path ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --train_views ${TRAIN_VIEWS} \
    --resolution ${RESOLUTION} \
    --colmap True \
    "

    CMD_5="python ./mvs2points.py \
    --path ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --mvs_config ${MVS_ROOT_PATH} \
    --dataset ${DATASETS} \
    --resolution ${RESOLUTION2} \
    "
    # Downsampling of Dense Points from COLMAP and MVSFormer
    CMD_6="python ./utils/o3dPre.py \
    --mvs_input ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --output_path ${SCENE_PATH}/${TRAIN_VIEWS}_views \
    --colmap True \
    --mvsformer True\
    "

    # ----- (2) DepthAnythingV2_For_Depth_Estimation -----
    CMD_7="python ./utils/DV2/metric_depth/MeDV2run.py \
    --encoder vitl \
    --load-from ${DV2_METRIC_ROOT_PATH} \
    --max-depth 3.6 \
    --img-path ${SCENE_PATH}/${TRAIN_VIEWS}_views/images \
    --outdir ${SCENE_PATH}/${TRAIN_VIEWS}_views/depths \
    --input-size 1000 \
    --pred-only \
    --save-numpy \
    "

    # ----- (3) Train, Render and Metrics  -----
    CMD_8="python ./train.py \
    --config configs/${DATASETS}/${SCENE}_MVS_${TRAIN_VIEWS}_views.yaml \
    --pcd_init ${TRAIN_CONFIGS} \
    "

    CMD_9="python ./render.py \
    --model_path output/${DATASETS}/${SCENE}/${TRAIN_VIEWS}_views/MVS \
    --loaded_pth output/${DATASETS}/${SCENE}/${TRAIN_VIEWS}_views/MVS/chkpnt_best.pth \
    --pcd_init ${TRAIN_CONFIGS} \
    "

    CMD_10="python ./metrics.py \
    --model_path output/${DATASETS}/${SCENE}/${TRAIN_VIEWS}_views/MVS \
    "


    echo "========= ${SCENE}: MVSFormer_Geometric_Initialization   ========="
    # eval $CMD_1
    # eval $CMD_2
    # eval $CMD_3
    # eval $CMD_4
    # eval $CMD_5
    # echo "========= ${SCENE}: Downsampling of Dense Points         ========="
    # eval $CMD_6
    # echo "========= ${SCENE}: DepthAnythingV2_For_Depth_Estimation ========="
    # eval $CMD_7
    # echo "========= ${SCENE}: Train_Render_and_Metrics             ========="
    eval $CMD_8
    eval $CMD_9
    eval $CMD_10

done