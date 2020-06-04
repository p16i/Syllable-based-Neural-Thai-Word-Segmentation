import os
import yaml
import glob

dry_run = True

if __name__ == "__main__":
    with open("./hyperopt-results.yml", "r") as fh:
        data = yaml.safe_load(fh)
        stat_files = list(map(lambda x: x["path"], data))

        print(f"we use {len(stat_files)} files")

        all_stat_files = glob.glob("./stats/*.csv")
        print(f"but we have {len(all_stat_files)} files")

        for f in all_stat_files:
            if f not in stat_files:
                print(f"Removing {f}")
                os.remove(f)