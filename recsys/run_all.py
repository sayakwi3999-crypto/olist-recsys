# -*- coding: utf-8 -*-
"""Run the whole pipeline: export data -> features -> train models -> offline evaluation."""
import sys
import time

import export_data
import features
import models as models_mod
import evaluate


def main():
    t0 = time.time()
    export_data.main()
    # Scenario A: time-based split
    data = features.build(force=True)
    model_dict = models_mod.build_models(data)
    df, sdf = evaluate.run(data, model_dict, tag="time_split",
                           title="Scenario A: time-based split (last 20% of interactions as test set)")
    # Scenario B: leave-one-out for repeat buyers
    data2 = features.build_repeat(force=True)
    model_dict2 = models_mod.build_models(data2)
    df2, sdf2 = evaluate.run(data2, model_dict2, tag="repeat",
                             title="Scenario B: repeat-buyer leave-one-out (last order as test set)")
    print(f"\n[run_all] finished in {time.time()-t0:.1f}s")
    print("[run_all] results written to recsys/output/")


if __name__ == "__main__":
    sys.exit(main())
