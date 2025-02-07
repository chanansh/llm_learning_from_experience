import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from langgraph.graph import StateGraph
from llm import get_memory_graph, invoke_graph
from loguru import logger
import fire
# Configure logging

@dataclass
class ExperimentSettings:
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
    trial_instruction: str = "Choose now by replying 'LEFT' or 'RIGHT' only. No other output is allowed!"
    trial_payoff_template: str = "You won {points} points (you chose {chosen_button})."
    end_game_template: str = "End of game. You won {points} points."
    do_not_respond: str = "No response expected. Reply with an empty string."
    summary_prompt: str = "Summarize game rewards concisely to aid decision-making."
    n_trials: int = 100
    model: str = "gpt-4o-mini" # gpt-4o, gpt-4o-mini
    data_file: str = "artifacts/2008/estimation_data.csv"
    output_path: str = "results/2008"
    seed:int = 42
    llm_per_problem: bool = True
    n_jobs = -1

def load_data(settings: ExperimentSettings) -> pd.DataFrame:
    """Load and preprocess experiment data."""
    logger.info(f"Loading data from {settings.data_file}")
    df = pd.read_csv(settings.data_file, index_col=0)
    return df.drop(['Choice', 'Payoff'], axis=1).groupby(["Id", "Order"]).first().reset_index()

def parse_decision(content: str) -> str:
    """Parse the decision from the content."""
    return "left" if "left" in content.lower() else "right" if "right" in content else None

def run_trial(app: StateGraph, settings: ExperimentSettings, problem: pd.Series, risky_side: str) -> dict:
    """Execute a single trial."""
    choice = get_choice(app, settings)
    points = (
        np.random.choice([problem.Low, problem.High], p=[1 - problem.Phigh, problem.Phigh])
        if choice == risky_side else problem.Medium
    )
    
    trial_payoff_message = settings.trial_payoff_template.format(points=points, chosen_button=choice)
    invoke_graph(app, trial_payoff_message + "\n" + settings.do_not_respond)
    
    return {
        **problem.to_dict(),
        "risky_side": risky_side,
        "side": choice,
        "Choice": int(choice == risky_side),  # 1 if risky side, else 0
        "Payoff": points,
    }

def get_choice(app, settings, max_retry:int=3):
    """Get the choice from the user.
    Args:
        app: The memory graph application.
        settings: The experiment settings.
        max_retry: The maximum number of retries.
    Returns:
        Tuple[str, Optional[str]]: The content and choice.
    Raises:
        ValueError: If an invalid response is received after max_retry attempts.
    """
    content = invoke_graph(app, settings.trial_instruction).strip().lower()
    choice = parse_decision(content)
    # retry if invalid response
    for i_try in range(max_retry):
        if choice is not None:
            break
        logger.warning(f"Invalid response received: {content}. retrying {i_try + 1}/{max_retry} ...")
        content = invoke_graph(app, settings.trial_instruction).strip().lower()
        choice = parse_decision(content)
    if choice is None:
        logger.error(f"Invalid response received: {content}")
        raise ValueError(f"Invalid response: {content}")
    return choice


def run_problem(app: StateGraph, 
                settings: ExperimentSettings, 
                problem: pd.Series,
                message: str | None = None) -> list:
    """Execute all trials for a given problem."""
    if settings.llm_per_problem:
        if app is not None:
            raise ValueError("LLM per problem is enabled. App should be None.")
        logger.info("LLM per problem is enabled so initializing the app at problem level.")
        app = get_memory_graph(
            model=settings.model, 
            summary_prompt=settings.summary_prompt,
            system_prompt=settings.instructions + "\n" + settings.do_not_respond)
    risky_side = np.random.choice(["left", "right"])
    logger.info(f"Risky side: {risky_side}")
    invoke_graph(app, settings.new_game + "\n" + settings.do_not_respond)

    total_points = 0
    results = []
    
    for trial_num in tqdm(range(settings.n_trials), desc="Trials"):
        logger.info(f"Trial {trial_num + 1}/{settings.n_trials} [{message}]")
        trial_result = run_trial(app, settings, problem, risky_side)
        total_points += trial_result["Payoff"]
        trial_result.update({"total_points": total_points, "Trial": trial_num + 1})
        results.append(trial_result)
    
    return results


def run_subject(settings: ExperimentSettings, 
                df_id: pd.DataFrame, 
                subject_id: int):
    """Execute all problems for a given subject."""
    if settings.llm_per_problem:
        logger.info("LLM per problem is enabled so no need to initialize the app at subject level.")
        app = None
    else:
        logger.info("LLM per problem is disabled so initializing the app at subject level.")
        app = get_memory_graph(
            model=settings.model, 
            summary_prompt=settings.summary_prompt,
            system_prompt=settings.instructions + "\n" + settings.do_not_respond)
    subject_results = []
    #invoke_graph(app, settings.instructions + "\n" + settings.do_not_respond)
    
    for iproblem, (_, problem) in tqdm(enumerate(df_id.iterrows()), desc=f"Subject {subject_id} - Problems"):
        logger.info(f"Running problem {problem.Problem} for subject {subject_id} ({iproblem + 1}/{len(df_id)})")
        message = f"Running subject {subject_id} - Problem {problem.Problem} ({iproblem + 1}/{len(df_id)})"
        problem_results = run_problem(app, settings, problem, message)
        subject_results.extend(problem_results)
    
    return subject_results
       
def get_subject_to_run_on(settings, unique_subjects):
    """Determine which subjects have not been processed yet."""
    existing_files = {f.split(".")[0] for f in os.listdir(settings.output_path) if f.endswith(".csv")}
    return [subj for subj in unique_subjects if str(subj) not in existing_files]

def process_subject(settings, df_id, subject_id):
    """Process a single subject and save results to a unique file."""
    logger.info(f"Starting subject {subject_id}")
    subject_results = run_subject(settings, df_id, subject_id)
    output_file = os.path.join(settings.output_path, f"{subject_id}.csv")
    
    pd.DataFrame(subject_results).to_csv(output_file, index=False)
    
    logger.info(f"Completed subject {subject_id}")

def run_experiment(settings = ExperimentSettings()):
    logger.info("Running Experiment 2008")
    logger_filename = ExperimentSettings.output_path + "/log.log"
    logger.add(logger_filename, format="{time} {level} {message}", level="INFO", backtrace=True, diagnose=True, mode="w")
    """Execute the full experiment with optional parallel execution."""
    logger.info(f"Starting experiment with settings: {settings}")
    np.random.seed(settings.seed)
    df = load_data(settings)
    unique_subjects = df["Id"].unique()
    logger.info(f"Unique subjects: {len(unique_subjects)}")
    
    subject_to_run_on = get_subject_to_run_on(settings, unique_subjects)
    
    if settings.n_jobs == 1:
        for subject_id in tqdm(subject_to_run_on, desc="Subjects"):
            process_subject(settings, df, subject_id)
    else:
        Parallel(n_jobs=settings.n_jobs, backend="threading", verbose=True)(
            delayed(process_subject)(settings, df[df["Id"] == subject_id].copy(), subject_id) for subject_id in tqdm(subject_to_run_on, desc="Subjects")
        )

if __name__ == "__main__":
    fire.Fire(run_experiment)
