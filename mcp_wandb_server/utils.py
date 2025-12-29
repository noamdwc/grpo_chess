"""Utility functions for formatting wandb data for agent consumption."""
from typing import Any, Dict, List, Optional
import json
import base64
import io
import os
import tempfile
import shutil


def format_metric_history(history: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, float]]]:
    """
    Format wandb history into metric name -> list of {step, value} pairs.
    
    Args:
        history: List of dictionaries from run.history()
        
    Returns:
        Dictionary mapping metric names to lists of {step, value} pairs
    """
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
    summary = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at.isoformat() if run.created_at else None,
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


def format_plot_info(file: Any, include_image_data: bool = True) -> Dict[str, Any]:
    """
    Format wandb file information for plots, optionally including base64-encoded image data.
    
    Args:
        file: wandb.Api().file() object
        include_image_data: If True, download and encode image as base64
        
    Returns:
        Dictionary with plot metadata and optionally base64-encoded image data
    """
    result = {
        "name": file.name,
        "size": file.size,
        "url": file.url if hasattr(file, "url") else None,
        "type": "image" if file.name.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")) else "other"
    }
    
    # Download and encode image data if requested and it's an image file
    if include_image_data and result["type"] == "image":
        try:
            # Create a temporary directory for downloads
            temp_dir = tempfile.mkdtemp()
            downloaded_file_path = None
            try:
                # Download the file - wandb's download() returns a file path
                # Try with root parameter first, then fallback to default behavior
                try:
                    downloaded_path = file.download(root=temp_dir, replace=True)
                except (TypeError, AttributeError):
                    # If root parameter not supported, download to temp dir using default
                    downloaded_path = file.download(replace=True)
                
                # Determine the actual file path
                if isinstance(downloaded_path, str):
                    # Check if it's an absolute path
                    if os.path.isabs(downloaded_path):
                        downloaded_file_path = downloaded_path
                    else:
                        # Try relative to temp_dir, then current directory
                        for base_dir in [temp_dir, os.getcwd()]:
                            candidate = os.path.join(base_dir, downloaded_path)
                            if os.path.exists(candidate):
                                downloaded_file_path = candidate
                                break
                        
                        # If still not found, try with just the filename
                        if not downloaded_file_path:
                            for base_dir in [temp_dir, os.getcwd()]:
                                candidate = os.path.join(base_dir, os.path.basename(file.name))
                                if os.path.exists(candidate):
                                    downloaded_file_path = candidate
                                    break
                        
                        if not downloaded_file_path:
                            raise FileNotFoundError(f"Could not locate downloaded file: {downloaded_path}")
                elif hasattr(downloaded_path, "read"):
                    # It's a file-like object, read directly
                    image_data = downloaded_path.read()
                    if hasattr(downloaded_path, "close"):
                        downloaded_path.close()
                    downloaded_file_path = None  # No file to clean up
                else:
                    raise ValueError(f"Unexpected download return type: {type(downloaded_path)}")
                
                # Read file if we have a path
                if downloaded_file_path:
                    with open(downloaded_file_path, "rb") as f:
                        image_data = f.read()
                
                # Encode as base64
                base64_data = base64.b64encode(image_data).decode("utf-8")
                
                # Determine MIME type based on file extension
                ext = file.name.lower().split(".")[-1] if "." in file.name else ""
                mime_types = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                    "svg": "image/svg+xml"
                }
                mime_type = mime_types.get(ext, "image/png")
                
                result["image_data"] = base64_data
                result["mime_type"] = mime_type
                result["data_uri"] = f"data:{mime_type};base64,{base64_data}"
            finally:
                # Clean up temporary directory and downloaded files
                try:
                    # Remove downloaded file if it's in temp_dir
                    if downloaded_file_path and downloaded_file_path.startswith(temp_dir):
                        try:
                            os.remove(downloaded_file_path)
                        except Exception:
                            pass
                    # Remove temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass  # Ignore cleanup errors
        except Exception as e:
            # If download fails, still return metadata but note the error
            result["image_data_error"] = str(e)
            result["image_data"] = None
            result["mime_type"] = None
            result["data_uri"] = None
    
    return result


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

