import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List
from langgraph.graph import StateGraph
from llm import get_memory_graph, invoke_graph
from loguru import logger
import fire

# Configure logging
logger.add("experiment.log", 
           format="{time} {level} {message}", 
           level="INFO", 
           backtrace=True, 
           diagnose=True, 
           mode="w")

@dataclass
class ExperimentSettings:
    """Settings for running the experiment."""
    instructions: str = (
        """
        You are a Technion student participating in a paid experiment.
        Instructions:
        - Play a series of games involving a money machine.
        - Each button press results in either winning or losing points.
        - Aim to accumulate as many points as possible.
        - The final bonus is determined based on total points (100 points = 1 Agora).
        - Machines may vary between participants.
        Good luck!
        """
    )
    new_game: str = "New Game (may differ from previous ones)"
    trial_instruction: str = "Please reply with 'LEFT' or 'RIGHT'."
    trial_payoff_template: str = "You won {points} points (you chose {chosen_button})."
    end_game_template: str = "End of game. You won {points} points."
    do_not_respond: str = "Please respond with an empty string."
    summary_prompt: str = "Summarize game rewards concisely to aid decision-making."
    n_trials: int = 100
    model: str = "gpt-4o"
    data_file: str = "artifacts/2008/estimation_data.csv"
    output_path: str = "results/2008"
    seed: int = 42
    llm_per_problem: bool = True
    checkpoint_per_problem: bool = True  # NEW FLAG: True = per problem, False = per subject
    n_jobs: int = 4

def load_data(settings: ExperimentSettings) -> pd.DataFrame:
    """Load and preprocess experiment data."""
    logger.info(f"Loading data from {settings.data_file}")
    df = pd.read_csv(settings.data_file, index_col=0)
    return df.drop(['Choice', 'Payoff'], axis=1).groupby(["Id", "Order"]).first().reset_index()

def parse_decision(content: str) -> Optional[str]:
    """Parse the decision from the content string."""
    content_lower = content.lower()
    if "left" in content_lower:
        return "left"
    elif "right" in content_lower:
        return "right"
    return None

def get_choice(app: StateGraph, settings: ExperimentSettings, initial_response: Optional[str] = None, max_retry: int = 3) -> str:
    """Obtain the choice from the user with retries, optionally using an initial response."""
    if initial_response:
        choice = parse_decision(initial_response.strip().lower())
        if choice:
            return choice  # If valid, return immediately

    for attempt in range(max_retry):
        content = invoke_graph(app, settings.trial_instruction).strip().lower()
        choice = parse_decision(content)
        if choice:
            return choice
        logger.warning(f"Invalid response received: {content}. Retrying {attempt + 1}/{max_retry}...")

    logger.error(f"Max retries reached. Last response: {content}")
    raise ValueError(f"Invalid response after {max_retry} attempts: {content}")

def run_problem(app: Optional[StateGraph], settings: ExperimentSettings, problem: pd.Series, subject_id: int, problem_number: int) -> List[dict]:
    """Run all trials for a given problem while minimizing LLM calls."""
    results = []
    total_points = 0
    risky_side = np.random.choice(["left", "right"])

    if (app is None) or settings.llm_per_problem:
        app = get_memory_graph(
            model=settings.model,
            summary_prompt=settings.summary_prompt,
            system_prompt=settings.instructions
        )

    # Start the game and ask for the first decision
    first_response = invoke_graph(app, f"{settings.new_game}\n{settings.trial_instruction}").strip().lower()
    choice = get_choice(app, settings, initial_response=first_response)

    for trial_num in tqdm(range(settings.n_trials), desc=f"Subject {subject_id}, Problem {problem_number}"):
        points = (
            np.random.choice([problem.Low, problem.High], p=[1 - problem.Phigh, problem.Phigh])
            if choice == risky_side else problem.Medium
        )

        trial_payoff_message = settings.trial_payoff_template.format(points=points, chosen_button=choice)
        total_points += points

        trial_result = {
            **problem.to_dict(),
            "risky_side": risky_side,
            "side": choice,
            "Choice": int(choice == risky_side),  # 1 if risky side, else 0
            "Payoff": points,
            "total_points": total_points,
            "Trial": trial_num + 1
        }
        results.append(trial_result)

        # No need to ask for a choice after the last trial
        if trial_num < settings.n_trials - 1:
            response = invoke_graph(app, f"{trial_payoff_message}\n{settings.trial_instruction}").strip().lower()
            choice = get_choice(app, settings, initial_response=response)

        else:
            invoke_graph(app, settings.end_game_template.format(points=total_points))

    return results

def run_subject(settings: ExperimentSettings, df_id: pd.DataFrame, subject_id: int):
    """Run the experiment for a single subject with optional per-problem checkpointing."""
    if not settings.checkpoint_per_problem:
        # Check if the full subject file exists
        output_file = os.path.join(settings.output_path, f"{subject_id}.csv")
        if os.path.exists(output_file):
            logger.info(f"Skipping completed subject {subject_id}")
            return
    
    # Initialize LLM app at subject level if needed
    app = None if settings.llm_per_problem else get_memory_graph(
        model=settings.model,
        summary_prompt=settings.summary_prompt,
        system_prompt=settings.instructions
    )

    all_results = []

    for problem_number, (_, problem) in tqdm(enumerate(df_id.iterrows()), total=len(df_id), desc=f"Subject {subject_id} - Problems"):
        
        # Check if the problem is already processed if checkpointing per problem
        problem_file = os.path.join(settings.output_path, f"{subject_id}_{problem_number}.csv")
        if settings.checkpoint_per_problem and os.path.exists(problem_file):
            logger.info(f"Skipping completed problem {problem_number} for subject {subject_id}")
            continue

        # Run the problem
        problem_results = run_problem(app, settings, problem, subject_id, problem_number)
        all_results.extend(problem_results)

        # Save results immediately if checkpointing per problem
        if settings.checkpoint_per_problem:
            pd.DataFrame(problem_results).to_csv(problem_file, index=False)
            logger.info(f"Checkpoint saved: {problem_file}")

    # Save results after all problems if checkpointing per subject
    if not settings.checkpoint_per_problem and all_results:
        subject_file = os.path.join(settings.output_path, f"{subject_id}.csv")
        pd.DataFrame(all_results).to_csv(subject_file, index=False)
        logger.info(f"Checkpoint saved: {subject_file}")

def run_experiment(settings: ExperimentSettings = ExperimentSettings()):
    """Execute the full experiment with flexible checkpointing."""
    logger.info("Starting Experiment 2008")
    np.random.seed(settings.seed)

    df = load_data(settings)
    unique_subjects = df["Id"].unique()

    if settings.n_jobs == 1:
        for subject_id in tqdm(unique_subjects, desc="Subjects"):
            run_subject(settings, df[df["Id"] == subject_id], subject_id)
    else:
        Parallel(n_jobs=settings.n_jobs, backend="threading", verbose=True)(
            delayed(run_subject)(settings, df[df["Id"] == subject_id], subject_id) for subject_id in tqdm(unique_subjects, desc="Subjects")
        )

if __name__ == "__main__":
    fire.Fire(run_experiment)
