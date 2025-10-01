import time
import click
from loguru import logger
from molecule_generation import load_model_from_directory
from pathlib import Path
from tqdm.auto import trange


@click.command()
@click.option("--model_dir", type=str, default="outputs/checkpoints")
@click.option("--output_path", type=str, default="samples.csv")
def sample(model_dir, output_path):
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    total_iters = 100
    batch_size = 300
    reload_interval = 10

    file_exists = output_path.exists()
    mode = "a" if file_exists else "w"

    with output_path.open(mode) as f:
        if not file_exists:
            f.write("smiles\n")  # 只有首次创建文件时写入表头

        for i in trange(0, total_iters, reload_interval, desc="Sampling"):
            dynamic_seed = int(time.time_ns() % (2**32))
            with load_model_from_directory(model_dir, num_workers=4, seed=dynamic_seed) as model:
                for j in range(reload_interval):
                    if i + j >= total_iters:
                        break
                    sampled_smiles = model.sample(batch_size)
                    f.write("\n".join(sampled_smiles) + "\n")
                    f.flush()
                    logger.info(f"Sampled batch {i + j + 1}/{total_iters} written.")


if __name__ == "__main__":
    sample()