
#!/bin/bash
NAME="nolinear"
mlagents-learn config/sac_transfer/3DBall.yaml --run-id ball_sac_f16_enc-actor_${NAME} --env=builds/server/3dball --no-graphics
# mlagents-learn config/sac_transfer/3DBallVisual.yaml --run-id ballvisual_sac_f16_enc-actor_${NAME} --env=builds/server/ball_visual 
mlagents-learn config/sac_transfer/3DBallVisual_transfer.yaml --run-id ballvisual_sac_f16_enc-actor_${NAME}_transfer --env=builds/server/visualball_new 


# mlagents-learn config/sac_transfer/3DBallHard.yaml --run-id ballhard_sac_f8_enc-actor_lr1e4 --env=builds/server/3dball_hard --no-graphics
# mlagents-learn config/sac_transfer/3DBallHard_transfer.yaml --run-id ballhard_sac_f8_enc-actor_lr1e4_transfer --env=builds/server/3dball_hard --no-graphics

# mlagents-learn config/sac_transfer/PushBlock_transfer.yaml --run-id pushblock30_sac_f256_enc-actor_transfer --env builds/server/pushblock_ray30 --no-graphics
# mlagents-learn config/sac_transfer/PushBlock.yaml --run-id pushblock30_sac_f256_enc-actor --env builds/server/pushblock_ray30 --no-graphics

# mlagents-learn config/sac_transfer/PushBlock_transfer.yaml --run-id pushblocklen4_sac_f256_enc-actor_transfer --env builds/server/pushblock_ray3_len4 --no-graphics
# mlagents-learn config/sac_transfer/PushBlock.yaml --run-id pushblocklen4_sac_f256_enc-actor --env builds/server/pushblock_ray3_len4 --no-graphics