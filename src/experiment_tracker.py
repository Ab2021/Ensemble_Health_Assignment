"""
Lightweight Experiment Tracker -- MLflow-compatible structure for model versioning.

Key features:
  - Self-contained (no external service, HIPAA-safe -- no telemetry/cloud)
  - MLflow-compatible artifact layout for future migration
  - Tracks: parameters, metrics, tags, model artifacts, predictions
  - Each run gets a unique subfolder in data/output/experiments/

Directory structure per run:
  experiments/
    <experiment_name>/
      <run_id>/
        params.json        -- hyperparameters
        metrics.json       -- evaluation metrics
        tags.json          -- run metadata
        model.pkl          -- serialized sklearn model
        preprocessor.pkl   -- ColumnTransformer/StandardScaler
        predictions.csv    -- scored current claims
        feature_names.json -- ordered feature list
"""
import os
import json
import uuid
import pickle
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentRun:
    """Metadata for a single experiment run."""
    run_id: str
    experiment_name: str
    timestamp: str
    status: str = 'RUNNING'  # RUNNING, FINISHED, FAILED

    # Hyperparameters
    params: Dict[str, Any] = field(default_factory=dict)

    # Evaluation metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Tags for organization
    tags: Dict[str, str] = field(default_factory=dict)

    # Artifact paths (set after saving)
    artifact_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """Manages experiment runs and artifact storage.

    Usage:
        tracker = ExperimentTracker('data/output/experiments')
        run = tracker.create_run('lr_hyperparameter_sweep')
        # ... train model, evaluate ...
        tracker.save_model(run.run_id, model)
        tracker.save_preprocessor(run.run_id, preprocessor)
        tracker.save_predictions(run.run_id, df_predictions)
        tracker.log_metrics(run.run_id, {'roc_auc': 0.69, 'capture_at_25': 0.46})
        tracker.log_params(run.run_id, {'C': 0.1, 'solver': 'lbfgs'})
        tracker.finish_run(run.run_id)
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._runs: Dict[str, ExperimentRun] = {}

    # -- Run lifecycle --------------------------------

    def create_run(
        self,
        experiment_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """Create a new experiment run with a unique ID."""
        run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tags=tags or {},
        )
        self._runs[run_id] = run
        return run

    def _run_dir(self, run_id: str) -> str:
        run = self._runs[run_id]
        path = os.path.join(self.base_dir, run.experiment_name, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def log_params(self, run_id: str, params: Dict[str, Any]):
        self._runs[run_id].params.update(params)
        self._save_json(run_id, 'params.json', self._runs[run_id].params)

    def log_metrics(self, run_id: str, metrics: Dict[str, float]):
        self._runs[run_id].metrics.update(metrics)
        self._save_json(run_id, 'metrics.json', self._runs[run_id].metrics)

    def log_tags(self, run_id: str, tags: Dict[str, str]):
        self._runs[run_id].tags.update(tags)
        self._save_json(run_id, 'tags.json', self._runs[run_id].tags)

    def save_model(self, run_id: str, model, filename: str = 'model.pkl'):
        path = os.path.join(self._run_dir(run_id), filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_preprocessor(self, run_id: str, preprocessor, filename: str = 'preprocessor.pkl'):
        path = os.path.join(self._run_dir(run_id), filename)
        with open(path, 'wb') as f:
            pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_predictions(self, run_id: str, df_predictions, filename: str = 'predictions.csv'):
        path = os.path.join(self._run_dir(run_id), filename)
        df_predictions.to_csv(path, index=False)

    def save_feature_names(self, run_id: str, feature_names: List[str]):
        self._save_json(run_id, 'feature_names.json', feature_names)

    def log_artifact(self, run_id: str, local_path: str, artifact_name: Optional[str] = None):
        """Copy a local file into the run's artifact directory."""
        import shutil
        name = artifact_name or os.path.basename(local_path)
        dest = os.path.join(self._run_dir(run_id), name)
        shutil.copy2(local_path, dest)

    def finish_run(self, run_id: str, status: str = 'FINISHED'):
        self._runs[run_id].status = status
        self._runs[run_id].artifact_uri = self._run_dir(run_id)
        run_data = self._runs[run_id].to_dict()
        self._save_json(run_id, 'run_metadata.json', run_data)

    def fail_run(self, run_id: str, error: str):
        self._runs[run_id].status = 'FAILED'
        self._runs[run_id].tags['error'] = error

    # -- Query ----------------------------------------

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self._runs.get(run_id)

    def list_experiments(self) -> List[str]:
        """List all experiment names with at least one run."""
        if not os.path.exists(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def list_runs(self, experiment_name: str) -> List[Dict]:
        """List all runs for an experiment with summary metrics."""
        exp_dir = os.path.join(self.base_dir, experiment_name)
        if not os.path.exists(exp_dir):
            return []
        runs = []
        for run_id in sorted(os.listdir(exp_dir)):
            run_path = os.path.join(exp_dir, run_id)
            if not os.path.isdir(run_path):
                continue
            meta_path = os.path.join(run_path, 'run_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    runs.append(json.load(f))
        return runs

    def compare_runs(self, experiment_name: str) -> Dict[str, Any]:
        """Generate a comparison table of all runs in an experiment."""
        runs_data = self.list_runs(experiment_name)
        if not runs_data:
            return {'experiment': experiment_name, 'runs': 0, 'comparison': {}}

        # Collect all unique metric keys
        all_metrics = set()
        for r in runs_data:
            all_metrics.update(r.get('metrics', {}).keys())

        comparison = {}
        for metric in sorted(all_metrics):
            values = {}
            for r in runs_data:
                if metric in r.get('metrics', {}):
                    values[r['run_id']] = r['metrics'][metric]
            comparison[metric] = values

        # Find best run per metric
        best = {}
        for metric, values in comparison.items():
            if values:
                best_run = max(values, key=values.get)
                best[metric] = {'run_id': best_run, 'value': values[best_run]}

        return {
            'experiment': experiment_name,
            'runs': len(runs_data),
            'comparison': comparison,
            'best_per_metric': best,
            'run_summaries': [
                {
                    'run_id': r['run_id'],
                    'params': r.get('params', {}),
                    'metrics': r.get('metrics', {}),
                    'status': r.get('status', 'UNKNOWN'),
                }
                for r in runs_data
            ],
        }

    # -- Internal helpers -----------------------------

    def _save_json(self, run_id: str, filename: str, data: Any):
        path = os.path.join(self._run_dir(run_id), filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_metadata(self, run_id: str):
        """Save current run metadata snapshot."""
        run = self._runs[run_id]
        self._save_json(run_id, 'run_metadata.json', run.to_dict())
