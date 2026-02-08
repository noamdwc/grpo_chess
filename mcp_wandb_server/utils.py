"""Utility functions for formatting wandb data for agent consumption."""
from typing import Any, Dict, List, Optional, Tuple
import json
import base64
import io
import os
import tempfile
import shutil


def format_metric_history(history: Any) -> Dict[str, List[Dict[str, float]]]:
    """
    Format wandb history into metric name -> list of {step, value} pairs.
    
    Args:
        history: List of dictionaries or pandas DataFrame from run.history()
        
    Returns:
        Dictionary mapping metric names to lists of {step, value} pairs
    """
    # Handle pandas DataFrame
    if hasattr(history, 'empty'):
        if history.empty:
            return {}
        # Convert DataFrame to list of dicts
        history = history.to_dict('records')
    
    if not history:
        return {}
    
    # Get all metric keys (exclude _step, _runtime, _timestamp, etc.)
    metric_keys = [k for k in history[0].keys() if not k.startswith("_")]
    
    result = {}
    for key in metric_keys:
        values = []
        for row in history:
            if key in row and row[key] is not None:
                step = row.get("_step", len(values))
                value = row[key]
                if isinstance(value, (int, float)):
                    values.append({"step": step, "value": float(value)})
        result[key] = values
    
    return result


def format_run_summary(run: Any) -> Dict[str, Any]:
    """
    Format a wandb run into a summary dictionary.
    
    Args:
        run: wandb.Api().run() object
        
    Returns:
        Dictionary with run summary information
    """
    # Handle created_at - it might be a datetime object or a string
    created_at = None
    if run.created_at:
        if hasattr(run.created_at, 'isoformat'):
            created_at = run.created_at.isoformat()
        else:
            created_at = str(run.created_at)
    
    summary = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": created_at,
        "tags": run.tags or [],
        "hyperparameters": dict(run.config) if run.config else {},
        "summary_metrics": dict(run.summary) if run.summary else {},
    }
    
    return summary


def format_runs_list(runs: List[Any]) -> List[Dict[str, Any]]:
    """
    Format a list of wandb runs into a list of dictionaries.
    
    Args:
        runs: List of wandb.Api().run() objects
        
    Returns:
        List of run metadata dictionaries
    """
    return [format_run_summary(run) for run in runs]


def downscale_image_bytes(image_bytes: bytes, max_dimension: int = 800) -> bytes:
    """Downscale an image so its largest dimension is at most max_dimension pixels.

    Returns PNG bytes. Returns original bytes if already small enough.
    """
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    if max(img.size) <= max_dimension:
        return image_bytes
    img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def format_plot_info(
    file: Any, include_image_data: bool = True
) -> Tuple[Dict[str, Any], Optional[bytes], Optional[str]]:
    """
    Format wandb file information for plots, optionally downloading raw image bytes.

    Args:
        file: wandb.Api().file() object
        include_image_data: If True, download raw image bytes

    Returns:
        Tuple of (metadata_dict, raw_image_bytes_or_None, mime_type_or_None)
    """
    file_type = "image" if file.name.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")) else "other"
    metadata = {
        "name": file.name,
        "size": file.size,
        "url": file.url if hasattr(file, "url") else None,
        "type": file_type,
    }

    raw_bytes = None
    mime_type = None

    # Download image data if requested and it's an image file (skip SVGs)
    if include_image_data and file_type == "image" and not file.name.endswith(".svg"):
        try:
            temp_dir = tempfile.mkdtemp()
            downloaded_file_path = None
            try:
                try:
                    downloaded_path = file.download(root=temp_dir, replace=True)
                except (TypeError, AttributeError):
                    downloaded_path = file.download(replace=True)

                if isinstance(downloaded_path, str):
                    if os.path.isabs(downloaded_path):
                        downloaded_file_path = downloaded_path
                    else:
                        for base_dir in [temp_dir, os.getcwd()]:
                            candidate = os.path.join(base_dir, downloaded_path)
                            if os.path.exists(candidate):
                                downloaded_file_path = candidate
                                break
                        if not downloaded_file_path:
                            for base_dir in [temp_dir, os.getcwd()]:
                                candidate = os.path.join(base_dir, os.path.basename(file.name))
                                if os.path.exists(candidate):
                                    downloaded_file_path = candidate
                                    break
                        if not downloaded_file_path:
                            raise FileNotFoundError(f"Could not locate downloaded file: {downloaded_path}")
                elif hasattr(downloaded_path, "read"):
                    raw_bytes = downloaded_path.read()
                    if hasattr(downloaded_path, "close"):
                        downloaded_path.close()
                else:
                    raise ValueError(f"Unexpected download return type: {type(downloaded_path)}")

                if downloaded_file_path:
                    with open(downloaded_file_path, "rb") as f:
                        raw_bytes = f.read()

                ext = file.name.lower().split(".")[-1] if "." in file.name else ""
                mime_types = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                }
                mime_type = mime_types.get(ext, "image/png")
            finally:
                try:
                    if downloaded_file_path and downloaded_file_path.startswith(temp_dir):
                        try:
                            os.remove(downloaded_file_path)
                        except Exception:
                            pass
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
        except Exception as e:
            metadata["image_data_error"] = str(e)

    return metadata, raw_bytes, mime_type


def format_comparison_table(runs: List[Any], metric_keys: List[str]) -> Dict[str, Any]:
    """
    Format a comparison table of metrics across runs.
    
    Args:
        runs: List of wandb.Api().run() objects
        metric_keys: List of metric keys to compare
        
    Returns:
        Dictionary with comparison data
    """
    comparison = {
        "runs": [],
        "metrics": metric_keys
    }
    
    for run in runs:
        run_data = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "metrics": {}
        }
        
        summary = run.summary or {}
        for metric_key in metric_keys:
            if metric_key in summary:
                run_data["metrics"][metric_key] = summary[metric_key]
            else:
                run_data["metrics"][metric_key] = None
        
        comparison["runs"].append(run_data)
    
    return comparison


def format_json_response(data: Any) -> str:
    """
    Format data as JSON string for MCP response.
    
    Args:
        data: Data to format
        
    Returns:
        JSON string
    """
    return json.dumps(data, indent=2, default=str)

