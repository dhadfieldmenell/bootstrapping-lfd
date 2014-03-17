import os
import os.path as osp


bootstrapping_dir = os.getenv("BOOTSTRAPPING_DIR")
models_dir        = osp.join(bootstrapping_dir, "models")
