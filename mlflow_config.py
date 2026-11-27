"""Shared, portable MLflow configuration for Student 1."""
from pathlib import Path
import os
import sqlite3
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mlflow.db"
ARTIFACT_DIR = BASE_DIR / "mlruns"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"

def configure_mlflow(experiment_name: str = "Iris-Classification") -> None:
    import mlflow
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    # Repair artifact URIs stored by the original Colab run so this package
    # remains usable after being moved to another computer or repository.
    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "experiments" in tables:
                    cols = {r[1] for r in conn.execute("PRAGMA table_info(experiments)")}
                    if "artifact_location" in cols:
                        rows = conn.execute("SELECT experiment_id, artifact_location FROM experiments").fetchall()
                        for exp_id, old_uri in rows:
                            if old_uri and (old_uri.startswith("/content/mlruns") or old_uri.startswith("file:///content/mlruns")):
                                suffix = old_uri.split("mlruns", 1)[1].lstrip("/")
                                new_path = ARTIFACT_DIR / suffix
                                new_path.mkdir(parents=True, exist_ok=True)
                                conn.execute("UPDATE experiments SET artifact_location=? WHERE experiment_id=?", (new_path.as_uri(), exp_id))
                if "runs" in tables:
                    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
                    if "artifact_uri" in cols:
                        rows = conn.execute("SELECT run_uuid, artifact_uri FROM runs").fetchall()
                        for run_id, old_uri in rows:
                            if old_uri and (old_uri.startswith("/content/mlruns") or old_uri.startswith("file:///content/mlruns")):
                                suffix = old_uri.split("mlruns", 1)[1].lstrip("/")
                                new_path = ARTIFACT_DIR / suffix
                                new_path.mkdir(parents=True, exist_ok=True)
                                conn.execute("UPDATE runs SET artifact_uri=? WHERE run_uuid=?", (new_path.as_uri(), run_id))
                conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not repair MLflow database paths: {exc}") from exc

    # Create the experiment only if it does not already exist.
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(experiment_name)
