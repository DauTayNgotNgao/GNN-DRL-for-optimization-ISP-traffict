import numpy as np
from gnn_ppo_isp import load_germany50_from_tgz, ISPEnv

TGZ_PATH = "directed-germany50-DFN-aggregated-5min-over-1day-native (1).tgz"

def run_baseline(mode="ISP-only"):
    edge_index, capacities, traffic_series, _ = load_germany50_from_tgz(TGZ_PATH)
    env = ISPEnv(edge_index, capacities, traffic_series)

    state, _ = env.reset()
    max_utils = []

    for _ in range(200):
        if mode == "ISP-only":
            action = np.zeros(env.num_edges, dtype=np.float32)
        elif mode == "random":
            action = np.random.uniform(-1, 1, env.num_edges).astype(np.float32)
        else:
            raise ValueError("Unknown mode")

        state, reward, done, _, info = env.step(action)
        max_utils.append(info["max_util"])

        if done:
            break

    return np.mean(max_utils)

if __name__ == "__main__":
    print("ISP-only Avg MLU:", run_baseline("ISP-only"))
    print("Random Avg MLU:", run_baseline("random"))
