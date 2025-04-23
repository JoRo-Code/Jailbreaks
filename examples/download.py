import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from jailbreaks.pipeline.utils import fetch_all_artifacts, FetchFilter

fetch_filter = FetchFilter(
    art_type="responses",
    #model_names=["Qwen/Qwen2-0.5B-Instruct"],
    n_runs=3,
    #benchmark_names=["harmless-bench"],
    #method_names=["diff-in-means"]
)

fetch_all_artifacts(
    project="jailbreaks_qwen_v2",
    output_dir=Path("data_test"),
    fetch_filter=fetch_filter
)