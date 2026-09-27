import numpy as np
import simpler_env
from matplotlib import pyplot as plt
from rt1_pytorch import RT1Inference
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def main(model_path):
    env = simpler_env.make("google_robot_pick_coke_can", render_mode="human")
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print("Reset info:", reset_info)
    print("Instruction:", instruction)

    model = RT1Inference(pt_model_path=model_path, policy_setup="google_robot")
    model.reset(instruction)
    print("RT-1 loaded successfully")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    done, truncated, step = False, False, 0
    while not (done or truncated):
        ax.clear()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Step {step}\n{instruction}")
        plt.pause(0.01)

        raw_action, action = model.step(image, instruction)
        env_action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        obs, reward, done, truncated, info = env.step(env_action)

        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction
            print("New Instruction:", instruction)
        step += 1

    episode_stats = info.get("episode_stats", {})
    print("Episode stats:", episode_stats)
    env.close()


if __name__ == "__main__":
    # model_path = "models/rt1x_logits_pt_model.pt"       # RT-1-X model
    model_path = "models/rt1_logits_pt_model.pt"        # RT-1 Converged
    # model_path = "models/rt1_15_logits_pt_model.pt"     # RT-1 15%
    # model_path = "models/rt1_begin_logits_pt_model.pt"  # RT-1 begin
    main(model_path)
