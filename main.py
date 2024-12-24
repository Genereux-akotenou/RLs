import argparse
from environments import FrozenLake, CartPole

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test reinforcement learning models.")
    parser.add_argument("--env", type=str, required=True, help="Environment to use, e.g., 'FrozenLake'")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm to use, e.g., 'DQN'")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Mode: 'train' or 'test'")
    parser.add_argument("--output_dir", type=str, default="model_output/frozenlake-v1/", help="Directory to store the model")
    parser.add_argument("--model_path", type=str, help="Path to load the model (required for testing)")
    parser.add_argument("--map", type=str, nargs="+", help="Custom map for FrozenLake (as a list of strings)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--n_episodes", type=int, default=200, help="Number of episodes for training")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--test_episodes", type=int, default=3, help="Number of episodes for testing")
    parser.add_argument("--is_slippery", action="store_true", help="Enable slippery mode")
    parser.add_argument("--render_mode", type=str, default="human", help="Rendering mode, e.g., 'human', 'rgb_array'")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1], help="Display logs")
    parser.add_argument("--hole_position", type=int, nargs='+', help="List of integers representing the hole positions")
    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    if args.env == "FrozenLake":
        custom_map = args.map or [
            'SFFF',
            'FHFH',
            'FFFH',
            'HFFG'
        ]
        if args.mode == "train":
            game = FrozenLake(args.algo, custom_map, is_slippery=args.is_slippery, render_mode=args.render_mode, verbose=args.verbose, hole_position=args.hole_position)
            game.batch_size = args.batch_size
            game.n_episodes = args.n_episodes
            game.max_steps = args.max_steps
            game.train(args.output_dir)
        elif args.mode == "test":
            if not args.model_path:
                raise ValueError("For testing, you must specify a --model_path")
            game = FrozenLake(args.algo, custom_map, is_slippery=args.is_slippery, render_mode=args.render_mode, verbose=args.verbose, hole_position=args.hole_position)
            game.max_steps = args.max_steps
            game.test(args.model_path, test_episodes=args.test_episodes)
    # --------------------------------------------------------------------------------
    elif args.env == "CartPole":
        if args.mode == "train":
            game = CartPole(args.algo, render_mode=args.render_mode, verbose=args.verbose)
            game.batch_size = args.batch_size
            game.n_episodes = args.n_episodes
            game.max_steps = args.max_steps
            game.train(args.output_dir)
        elif args.mode == "test":
            if not args.model_path:
                raise ValueError("For testing, you must specify a --model_path")
            game = CartPole(args.algo, render_mode=args.render_mode, verbose=args.verbose)
            game.max_steps = args.max_steps
            game.test(args.model_path, test_episodes=args.test_episodes)
    else:
        print(f"Environment '{args.env}' not supported.")