
mlagents-learn config/ppo_transfer/3DBall.yaml --run-id ball_ppo_f8_auxiliary \
    --env=builds/linux/ball --no-graphics
mlagents-learn config/ppo_transfer/3DBallHard.yaml --run-id hardball_ppo_f8_auxiliary \
    --env=builds/linux/ball_hard --no-graphics
mlagents-learn config/ppo_transfer/3DBallHard_transfer.yaml --run-id hardball_ppo_f8_auxiliary_transfer \
    --env=builds/linux/ball_hard --no-graphics

# mlagents-learn config/ppo_transfer/PushBlock.yaml --run-id pushblock_ppo_f1000_actionlinear --env=builds/server/pushblock --no-graphics
# mlagents-learn config/ppo_transfer/PushBlock_transfer.yaml --run-id pushblock_ppo_f1000_actionlinear_transfer_c10 --env=builds/server/pushblock --no-graphics