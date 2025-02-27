### RLs: Reinforcement Learning Algorithms in Gaming-Like Style

This repository demonstrates the application of Reinforcement Learning (RL) algorithms in gaming-like scenarios, making learning RL intuitive, practical, and engaging.

<img src="utils/images/RLS-DECO.jpg"/>

---

<img src="utils/images/napkin-selection (10).png"/>
<img src="utils/images/napkin-selection (8).png"/>


<!--This repository demonstrates the application of Reinforcement Learning (RL) algorithms in gaming-like scenarios, making learning RL intuitive, practical, and engaging. Whether you’re a beginner exploring RL concepts or an experienced researcher, this project serves as an interactive playground to understand, implement, and visualize RL techniques in a gaming framework.-->

<!-- ### Features
-	Interactive Gaming Environments: Test RL algorithms in dynamic, gaming-style simulations.
-	Pre-implemented RL Algorithms:
    -	Q-Learning
    -	Deep Q-Networks (DQN)
    -	Double DQN
    -	Policy Gradient Methods
    -	Proximal Policy Optimization (PPO)
    -	Customizable Environments: Easily modify or create new gaming scenarios for experimentation.
    -	Visualization: Track agent learning progress with rich visualizations and performance metrics.
    -	Modular Design: Well-organized and modular code for ease of understanding and contribution. -->

### Project Structure
```
RLs/
├── algorithms/          # Core RL algorithms (Q-Learning, DQL, etc.)
├── environments/        # Prebuilt gaming-style RL environments
├── config/              # Configuration files for training and environments
├── prebuilt/            # Store trained model utils
├── tests/               # Test scripts for validation
├── utils/               # Helper functions for metrics, visualization, etc.
├── x-samples/           # Examples
└── main.py              # Main entry point for running experiments
```

### Installation
```bash
# Clone
git clone https://github.com/Genereux-akotenou/RLs.git
cd RLs

# Create env
python -m venv .rl_env
source .rl_env/bin/activate
pip install -r requirements.txt

# Ready?
# NB: Before you start a notebook in this project make sur to select as a kernel '.rl_env/bin/python'
```

### Examples
1. Run a Predefined Scenario
Execute an RL algorithm in a prebuilt gaming environment:
```bash
python main.py --env "FrozenLake" --algo "DQN" --mode "test" --test_episodes 3 --verbose "1"
    --model_path "prebuilt/frozenlake-v1/weights_0150.weights.h5" \

```

2. Training a DQN Agent in a FrozenLake Environment
```bash
python main.py --env "FrozenLake" --algo "DQN" --mode "train" \
    --output_dir "prebuilt/frozenlake-v1" --map "SFFF" "FHFH" "FFFH" "HFFG" \
    --batch_size 32 --n_episodes 1000 --max_steps 300 --verbose "0"
```

3. Comparing Algorithms (comming)
<!-- Run multiple agents for benchmarking: -->
```bash
python benchmark.py --env "CustomGame" --algos "DQN PPO"
```

4. Visualizing Results (comming)
<!-- Generate performance graphs: -->
```bash
python visualize_results.py --log_dir "./logs/"
```

### Contributing
Contributions are welcome! Please follow these steps:
1.	Fork the repository.
2.	Create a new branch for your feature or bug fix.
3.	Submit a pull request with a detailed description.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

<!-- ### Contact
For questions or suggestions, feel free to reach out:
-	Email: Mahouzonssou.AKOTENOU@um6p.ma -->