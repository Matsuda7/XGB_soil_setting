"""Handle previous prediction chunks without mixing models or deleting data."""
from pathlib import Path
import tempfile


def prepare_chunk_output(output_dir: Path, resume: bool, policy: str = "error"):
    if policy not in {"archive", "error"}:
        raise ValueError("existing_chunks must be archive or error")
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume or not any(output_dir.glob("N_value_10m_*.csv.gz")):
        return None
    if policy == "error":
        raise FileExistsError(
            f"Prediction chunks already exist under {output_dir}. "
            "Set existing_chunks to archive for fresh predictions. "
            "Use --resume only with the same model, inputs and settings."
        )
    archive = Path(tempfile.mkdtemp(prefix=output_dir.name + "_previous_", dir=output_dir.parent))
    destination = archive / output_dir.name
    output_dir.rename(destination)
    output_dir.mkdir()
    return destination
