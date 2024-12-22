### RLs: Reinforcement Learning Algorithms in Gaming-Like Style

This repository demonstrates the application of Reinforcement Learning (RL) algorithms in gaming-like scenarios, making learning RL intuitive, practical, and engaging.

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
├── environments/        # Prebuilt gaming-style RL environments
├── algorithms/          # Core RL algorithms (Q-Learning, PPO, etc.)
├── config/              # Configuration files for training and environments
├── utils/               # Helper functions for metrics, visualization, etc.
├── tests/               # Test scripts for validation
└── main.py              # Main entry point for running experiments
```

### Installation
```bash
git clone https://github.com/Genereux-akotenou/RLs.git
cd RLs
pip install -r requirements.txt
```

### Examples
1. Run a Predefined Scenario
Execute an RL algorithm in a prebuilt gaming environment:
```bash
python main.py --env "Maze" --algo "DQN"
```

2. Visualize Training
Enable real-time training visualizations:
```bash
python main.py --env "Maze" --algo "DQN" --visualize True
```

3. Training a DQN Agent in a Maze Environment
```bash
python main.py --env "Maze" --algo "DQN"
```

4. Comparing Algorithms
Run multiple agents for benchmarking:
```bash
python benchmark.py --env "CustomGame" --algos "DQN PPO"
```

5. Visualizing Results
Generate performance graphs:
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

### Contact
For questions or suggestions, feel free to reach out:
-	Email: Mahouzonssou.AKOTENOU@um6p.ma