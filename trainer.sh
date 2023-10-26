EXPNAME='extrasensory_lr1e-3_lambda_0.3_0.1_commonD_3072'

CUDA_VISIBLE_DEVICES=0 nohup python3 -u code/trainer.py \
    --seed 2 \
    --expName $EXPNAME \
    --batch_size 1024 \
    --num_workers 2 \
    --lr 1e-3 \
    --epoch 300 \
    --clip_grad 1000 \
    --gamma 0.998 \
    --mask \
    --hetero \
    --use_cuda \
    --reduction 'mean' \
    --users 3 \
    --phonePlacements 5 \
    --activities 12 \
    --dataPath 'data/extrasensoryRuled_sampled3user/' \
    --outputPath 'output/'$EXPNAME \
    --xPath 'feature_ruled' \
    --yPath 'y_ruled' \
    --weightPath 'mask_expanded_ruled' \
    --nodePath 'nodeInit_ruled' \
    --hyperIndexPath 'adj_ruled' \
    --hyperWeightPath 'count_ruled' \
    --hyperAttrPath 'edgeInit_ruled' \
    --modelPath 'model.pkl' \
    --resultPath 'result.npy' \
    --lossPath $EXPNAME'.jpg' \
    --model_dropout1 0.05 \
    --model_dropout2 0.05 \
    --model_commonDim 3072 \
    --model_leakySlope_g 0.2 \
    --model_leakySlope_x 0.2 \
    --hgcn_l1_before_leakySlope 0.2 \
    --hgcn_l1_in_channels 152 \
    --hgcn_l1_out_channels 256 \
    --hgcn_l1_use_attention \
    --hgcn_l1_heads 2 \
    --hgcn_l1_negative_slope 0.2 \
    --hgcn_l1_dropout 0.05 \
    --hgcn_l1_bias \
    --hgcn_l1_after_leakySlope 0.2 \
    --hgcn_l2_before_leakySlope 0.2 \
    --hgcn_l2_in_channels -1 \
    --hgcn_l2_out_channels 256 \
    --hgcn_l2_heads 2 \
    --hgcn_l2_negative_slope 0.2 \
    --hgcn_l2_dropout 0.05 \
    --hgcn_l2_bias \
    --hgcn_l2_after_leakySlope 0.2 \
    --lambda1 0.3 \
    --lambda2 0.1 \
    1> "log/"$EXPNAME".log" \
    2> "log/"$EXPNAME".err" &

