# Review risk questions (ecs_hcm_model)

Yes/no questions asked of each builder change, answered from the diff alone.
Any "yes" marks the change risky and it gets a full review; all "no" gets a simple review.

1. **Does it change what is written to, overwritten in, or deleted from MinIO (model/checkpoint buckets, `model_upload_download.py`, checkpoint save/resume/`_best` selection)?**
   Why: trained weights are the expensive artifact; a bad key, prefix or delete loses days of GPU time.

2. **Does it change the model architecture or `state_dict` keys/shapes (`hcm_nn.py`, heads, backbone, `out_stride`)?**
   Why: existing checkpoints and the Jetson demo load with `strict=True`; a key change breaks resume and inference.

3. **Does it change how training targets or inputs are built (COCO/COCOA/ImageNet transforms, masks, normalization, input size, collate)?**
   Why: a silent label/target bug (e.g. the K-sampled vs union mask mismatch) trains a wrong model with no crash.

4. **Does it change the loss, optimizer, lr schedule, warmup/unfreeze, early stopping, or metric/validation computation?**
   Why: core training logic; errors show up only after a full cluster run, and metric bugs mislead model selection.

5. **Does it change the trainer CLI (flags, defaults, entrypoint) or the trainer image contract in `main.py` / `services/trainer`?**
   Why: the sys_dags DAG `a-ecs_hcm-trainer` calls this interface; a renamed flag or changed default breaks or silently alters runs.

6. **Does it touch `dockerfiles/`, `build.sh`, requirements, or the torch/CUDA version or library paths?**
   Why: the build/deploy path; a broken image or CUDA lib resolution fails every cluster run.

7. **Does it add or change credentials, endpoints, secrets handling, or anything read from the environment (MinIO/MLflow/Harbor keys, `ECS_HCM_*` overrides)?**
   Why: leaked keys are the one unrecoverable mistake; env overrides also gate safety thresholds.

8. **Does it change the training-safety guards (OOM skip, fp16/overflow handling, GPU preflight, kill/abort paths)?**
   Why: these guards keep a run alive or stop a bad one; weakening them wastes cluster time or corrupts weights.

9. **Does it change MLflow run creation, tagging, or logged metadata (including the uploaded model's metadata the Jetson reads)?**
   Why: stray or missing runs lose experiment history; metadata drift breaks the Jetson loader.

10. **Does it change code under `jetson/` or anything that runs on the Jetson (camera, device, served ports)?**
    Why: it runs on real hardware with a fragile installed environment and a live demo stream.

11. **Does it delete, skip, xfail, or loosen assertions in existing tests, or change CI tooling?**
    Why: weakened tests hide regressions in every area above.
