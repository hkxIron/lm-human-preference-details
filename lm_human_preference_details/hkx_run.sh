# accelerate launch  \
# --num_processes 1  \
# lm_human_preference_details/train_reward_accelerate.py --track


echo "本地训练：train reward model"
#accelerate launch --num_processes 2 train_reward_accelerate.py --base_model /home/hkx/data/work/hf_data_and_model/models/Qwen/Qwen1.5-0.5B/ --save_path ../models/reward --deepspeed --no-cuda --no-track

echo "本地训练：train policy model"
accelerate launch --num_processes 2 train_policy_accelerate.py --base_model /home/hkx/data/work/hf_data_and_model/models/Qwen/Qwen1.5-0.5B/ --save_path ../models/policy --rewards.trained_model ../models/reward/pytorch.bin --no-cuda --no-track --task.temperature 0.65 