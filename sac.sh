#!/bin/bash
for (( i=0; i<1; i++ ))
do
    # mlagents-learn config/sac_transfer/3DBall.yaml \
    #     --run-id newball_sac_f16_final \
    #     --env=builds/linux/newball --no-graphics

    mlagents-learn config/sac_transfer/3DBallHard_transfer_variant.yaml \
        --run-id hardball_sac_f8_final_transfer_rewonly_${i} \
        --env=builds/linux/ball_hard --no-graphics

    # mlagents-learn config/sac_transfer/3DBallHard.yaml \
    #     --run-id hardball_sac_f8_final_aux_${i} \
    #     --env=builds/linux/ball_hard --no-graphics
done

# NAME="f16_enc-ac-cri_nolinear"
# mlagents-learn config/sac_transfer/3DBall.yaml --run-id ball_sac_${NAME} --env=builds/linux/ball --no-graphics
# mlagents-learn config/sac_transfer/3DBallHard.yaml --run-id ballhard_sac_${NAME} --env=builds/linux/ball_hard --no-graphics
# mlagents-learn config/sac_transfer/3DBallHard_transfer.yaml --run-id ballhard_sac_${NAME}_transfer --env=builds/linux/ball_hard --no-graphics

# NAME="f16_enc-act_nolinear"
# for (( i=0; i<5; i++ ))
# do  
    # mlagents-learn config/sac_transfer/3DBall.yaml --run-id ball_sac_${NAME}_${i} --env=builds/linux/ball --no-graphics
    # mlagents-learn config/sac_transfer/3DBallHard.yaml --run-id ballhard_sac_${NAME}_${i} --env=builds/linux/ball_hard --no-graphics
    # mlagents-learn config/sac_transfer/transfer/3DBallHard_transfer_${i}.yaml --run-id ballhard_sac_${NAME}_transfer_${i} --env=builds/linux/ball_hard --no-graphics
# done

# mlagents-learn config/sac_transfer/3DBallHard.yaml --run-id ballhard_sac_f8_enc-actor_lr1e4 --env=builds/server/3dball_hard --no-graphics
# mlagents-learn config/sac_transfer/3DBallHard_transfer.yaml --run-id ballhard_sac_f8_enc-actor_lr1e4_transfer --env=builds/server/3dball_hard --no-graphics

# mlagents-learn config/sac_transfer/PushBlock_transfer.yaml --run-id pushblock30_sac_f256_enc-actor_transfer --env builds/server/pushblock_ray30 --no-graphics
# mlagents-learn config/sac_transfer/PushBlock.yaml --run-id pushblock30_sac_f256_enc-actor --env builds/server/pushblock_ray30 --no-graphics

# mlagents-learn config/sac_transfer/PushBlock_transfer.yaml --run-id pushblocklen4_sac_f256_enc-actor_transfer --env builds/server/pushblock_ray3_len4 --no-graphics
# mlagents-learn config/sac_transfer/PushBlock.yaml --run-id pushblocklen4_sac_f256_enc-actor --env builds/server/pushblock_ray3_len4 --no-graphics