from pathlib import Path
import runpy

DATASET_MERGE = Path("/Users/yuki/Downloads/yuna-ai-dataset/merge_yuna_avl_torch.py")


def main():
	assert DATASET_MERGE.is_file(), DATASET_MERGE
	runpy.run_path(str(DATASET_MERGE), run_name="__main__")


if __name__ == "__main__":
	main()
