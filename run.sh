
#mlagents-learn config/dqn/PushBlock.yaml --run-id pushblock_dqn_f128_transonly --env builds/linux/pushblock
mlagents-learn config/dqn/PushBlock_transfer.yaml --run-id pushblock_dqn_f64_transonly_transfer --env  builds/linux/pushblock
