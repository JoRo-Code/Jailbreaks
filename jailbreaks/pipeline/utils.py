import logging
import pathlib
import functools
import concurrent.futures

import wandb

logger = logging.getLogger(__name__)

def fetch_artifacts(run, output_dir:pathlib.Path, art_type:str):
    for art in run.logged_artifacts():
        if art.type == art_type:
            run_dir = output_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            art.download(root=str(run_dir))
    return run.name

def fetch_all_artifacts(
    project:str, 
    output_dir:pathlib.Path, 
    art_type:str, 
    threads:int=8
    ):
    runs: wandb.Api.runs = wandb.Api().runs(f"{project}")
    fetch_arts = functools.partial(fetch_artifacts, output_dir=output_dir, art_type=art_type)
    logger.info("Fetching artifacts from %s", project)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        for finished in ex.map(fetch_arts, runs):
            logger.info("✓ %s", finished)

if __name__ == "__main__":
    fetch_all_artifacts(
        project="test", 
        output_dir=pathlib.Path("download"), 
        art_type="responses", 
        threads=8
    )
