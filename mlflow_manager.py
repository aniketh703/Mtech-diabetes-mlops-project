"""
MLflow Experiment Management Script for Diabetes Prediction Project

This script provides utilities to manage MLflow experiments, runs, and models.
"""

import argparse
import mlflow
import mlflow.sklearn
from src.mlflow_utils import get_mlflow_manager
import pandas as pd
import sys


def start_mlflow_ui():
    """Start MLflow UI server"""
    print("Starting MLflow UI...")
    print("MLflow UI will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    import subprocess
    subprocess.run(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])


def list_experiments():
    """List all MLflow experiments"""
    mlflow_manager = get_mlflow_manager()
    
    print("\n=== MLflow Experiments ===")
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        print(f"ID: {exp.experiment_id}")
        print(f"Name: {exp.name}")
        print(f"Lifecycle Stage: {exp.lifecycle_stage}")
        print(f"Artifact Location: {exp.artifact_location}")
        print("-" * 50)


def list_runs(experiment_name=None, max_results=10):
    """List runs in an experiment"""
    mlflow_manager = get_mlflow_manager()
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    runs = mlflow_manager.list_runs(max_results=max_results)
    
    print(f"\n=== Recent {max_results} Runs ===")
    if not runs.empty:
        # Select relevant columns for display
        display_columns = ['run_id', 'status', 'start_time', 'metrics.test_accuracy', 
                          'metrics.test_f1_score', 'tags.mlflow.runName']
        
        # Filter columns that exist
        available_columns = [col for col in display_columns if col in runs.columns]
        
        print(runs[available_columns].to_string(index=False))
    else:
        print("No runs found in the experiment.")


def compare_runs(run_ids):
    """Compare multiple runs"""
    if len(run_ids) < 2:
        print("Please provide at least 2 run IDs for comparison")
        return
    
    print(f"\n=== Comparing Runs: {', '.join(run_ids)} ===")
    
    runs_data = []
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        run_data = {
            'run_id': run_id[:8],  # Shortened for display
            'status': run.info.status,
            'start_time': run.info.start_time
        }
        
        # Add metrics
        for metric_name, metric_value in run.data.metrics.items():
            run_data[f"metrics.{metric_name}"] = metric_value
        
        # Add parameters
        for param_name, param_value in run.data.params.items():
            run_data[f"params.{param_name}"] = param_value
        
        runs_data.append(run_data)
    
    comparison_df = pd.DataFrame(runs_data)
    print(comparison_df.to_string(index=False))


def get_best_model(metric="test_accuracy"):
    """Get the best performing model"""
    mlflow_manager = get_mlflow_manager()
    
    best_run = mlflow_manager.get_best_run(metric_name=metric)
    
    if best_run is not None:
        print(f"\n=== Best Model (by {metric}) ===")
        print(f"Run ID: {best_run['run_id']}")
        print(f"Run Name: {best_run.get('tags.mlflow.runName', 'N/A')}")
        print(f"{metric}: {best_run.get(metric, 'N/A')}")
        print(f"Start Time: {best_run['start_time']}")
        
        # Load and return model
        model = mlflow_manager.load_model_from_run(best_run['run_id'])
        print("Model loaded successfully!")
        return model, best_run['run_id']
    else:
        print(f"No runs found with metric: {metric}")
        return None, None


def promote_model(run_id, stage="Production"):
    """Promote a model to production stage"""
    mlflow_manager = get_mlflow_manager()
    
    result = mlflow_manager.promote_model(run_id, stage)
    print(f"\n=== Model Promotion ===")
    print(result)


def cleanup_experiments(experiment_name=None, keep_last_n=10):
    """Clean up old experiment runs"""
    mlflow_manager = get_mlflow_manager()
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    runs = mlflow_manager.list_runs(max_results=1000)
    
    if len(runs) <= keep_last_n:
        print(f"Only {len(runs)} runs found, nothing to clean up.")
        return
    
    runs_to_delete = runs.iloc[keep_last_n:]
    
    print(f"\n=== Cleanup ===")
    print(f"Will delete {len(runs_to_delete)} old runs, keeping the latest {keep_last_n}")
    
    confirmation = input("Are you sure you want to delete these runs? (y/N): ")
    if confirmation.lower() == 'y':
        client = mlflow.tracking.MlflowClient()
        for _, run in runs_to_delete.iterrows():
            try:
                client.delete_run(run['run_id'])
                print(f"Deleted run: {run['run_id'][:8]}")
            except Exception as e:
                print(f"Failed to delete run {run['run_id'][:8]}: {e}")
    else:
        print("Cleanup cancelled.")


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="MLflow Experiment Management")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # UI command
    subparsers.add_parser('ui', help='Start MLflow UI')
    
    # List experiments
    subparsers.add_parser('list-experiments', help='List all experiments')
    
    # List runs
    runs_parser = subparsers.add_parser('list-runs', help='List runs in experiment')
    runs_parser.add_argument('--experiment', type=str, help='Experiment name')
    runs_parser.add_argument('--max-results', type=int, default=10, help='Max number of runs to show')
    
    # Compare runs
    compare_parser = subparsers.add_parser('compare', help='Compare multiple runs')
    compare_parser.add_argument('run_ids', nargs='+', help='Run IDs to compare')
    
    # Best model
    best_parser = subparsers.add_parser('best-model', help='Get best performing model')
    best_parser.add_argument('--metric', type=str, default='test_accuracy', help='Metric to optimize')
    
    # Promote model
    promote_parser = subparsers.add_parser('promote', help='Promote model to production')
    promote_parser.add_argument('run_id', help='Run ID of model to promote')
    promote_parser.add_argument('--stage', type=str, default='Production', help='Stage to promote to')
    
    # Cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old runs')
    cleanup_parser.add_argument('--experiment', type=str, help='Experiment name')
    cleanup_parser.add_argument('--keep', type=int, default=10, help='Number of runs to keep')
    
    args = parser.parse_args()
    
    if args.command == 'ui':
        start_mlflow_ui()
    elif args.command == 'list-experiments':
        list_experiments()
    elif args.command == 'list-runs':
        list_runs(args.experiment, args.max_results)
    elif args.command == 'compare':
        compare_runs(args.run_ids)
    elif args.command == 'best-model':
        get_best_model(args.metric)
    elif args.command == 'promote':
        promote_model(args.run_id, args.stage)
    elif args.command == 'cleanup':
        cleanup_experiments(args.experiment, args.keep)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()