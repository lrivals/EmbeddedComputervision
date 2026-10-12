# Training on a Colab kernel from VS Code

The `_train` and `_sweep` notebooks (M14) open in VS Code and run on a Colab GPU runtime
through the Google Colab extension (`google.colab`). The notebook stays the repository
file; only the kernel is remote. The outputs go to Google Drive (`colab.sync_outputs`) and
are fetched on the PC through rclone ([donnees-drive.md](donnees-drive.md)).

## Who does what

| Step | Who | How |
|---|---|---|
| connect the Colab kernel, authorise Google and Drive | user | VS Code (extension UI) |
| set the parameters of a notebook | user or Claude | "Paramètres" cell (the file is generated: see below) |
| run the cells ("Run All") | **user** | VS Code |
| follow the loss, see the finished runs | user or Claude | rclone on `gdrive:EmbeddedCV/runs` |
| fetch the runs, evaluate, compare | user or Claude | `rclone copy`, inference notebooks locally |

Claude (Claude Code) cannot run cells on a kernel attached to VS Code. The Colab extension
only exposes UI commands (`colab.mountServer`, `colab.mountDrive`, `colab.openTerminal`…)
and keeps its tokens in the VS Code secret storage. The PC has no GPU, so no local run
either (tier N: hours on CPU). Claude prepares the parameters and follows the runs through
rclone, and the user is the one who clicks "Run All".

## Setup (once per Colab session)

1. Open the notebook in VS Code, for example
   [notebooks/kitti/tiny-yolov3-kitti_train.ipynb](../../notebooks/kitti/tiny-yolov3-kitti_train.ipynb).
2. "Select Kernel" → "Colab" → **new** server, type **GPU** (T4; L4 or A100 with Colab
   Pro), then Google sign-in. A server created as CPU does not change type: create another
   one. Check: `!nvidia-smi` in a cell shows the GPU. On a CPU runtime, the "Environnement"
   cell stops with "runtime Colab sans GPU" (before this check, CuPy said
   `cudaErrorInsufficientDriver`). The CuPy wheel follows the CUDA version of the driver
   (`cupy-cuda13x` from 13 on, otherwise `cupy-cuda12x`).
3. Kaggle token if the dataset is downloaded from Kaggle (crowdhuman, visdrone, flir
   without an archive on Drive): the `KGAT_…` token (kaggle.com, Settings, API Tokens) in
   the file `MyDrive/EmbeddedCV/access_token`, for example
   `printf %s "$KAGGLE_API_TOKEN" | rclone rcat gdrive:EmbeddedCV/access_token` (the old
   `kaggle.json` is also read there). The Colab secrets do not work here: under VS Code,
   `userdata.get` has no page to ask for access and ends in a `TimeoutException`.

The "Environnement" cell detects Colab (`"google.colab" in sys.modules`). The kernel runs
in `/content`, outside the repository: it therefore clones `REPO_URL` into
`/content/EmbeddedComputervision`, resets it to `REV` (default `main`) at each run, and
installs `python[data,plots]` and CuPy (`DEVICE = 'gpu'`). **The code that runs is that of
`REV` on GitHub, not that of the PC**: push first whatever must run.

The data goes through `colab.prepare_data(DATASET, DRIVE_DIR)` (order: marker file, Drive
archive, rclone, direct download; [M14](M14-notebooks.md#data-on-colab)). What is on the
Drive today:

| Dataset | Source on Colab |
|---|---|
| voc | `MyDrive/EmbeddedCV/data/voc.tar` |
| visdrone | `MyDrive/EmbeddedCV/data/visdrone.tar` |
| kitti | direct download (`get_datasets.sh kitti`) |
| flir | Kaggle API (`get_datasets.sh kaggle-download flir`, mirror `samdazel/teledyne-flir-adas-thermal-dataset-v2`; token in `MyDrive/EmbeddedCV/access_token`), then `push flir` for the following sessions |

To check on the first try: `drive.mount()` (`colab.mount_drive`) asks for an
authorisation in the notebook. If it does not show under VS Code, mount Drive with the
extension command "Colab: Mount Google Drive to Server…", then rerun the cell.

## Running

`DEVICE` is `'gpu'` by default in `_train` and `_sweep` (`'cpu'` to run them on a PC
without a GPU). Parameters to set in the "Paramètres" cell, before "Run All":

| Notebook | Parameters |
|---|---|
| `_train` | `ITERS` (4000), `BATCH` (16), `SUBSET` (50; 0: full split) |
| `_sweep` | `BATCHES` (`[8, 16, 32]`), `TRAIN_SUBSETS` (`[500, 0]`), `ITERS` (600 per run), `SKIP_DONE` |

These changes are not committed: the notebooks are generated from
`tools/notebooks/gabarits.py` (`make notebooks`) and `--check` refuses an edited cell. To
change a default value, change the template.

Suggested order, one notebook at a time per runtime:

1. `voc/tiny-yolov3-voc_train` (archive on Drive, reference of M2);
2. `kitti/tiny-yolov3-kitti_train`, then `_sweep`;
3. `visdrone/tiny-yolov3-visdrone_train`, then `_sweep`;
4. `flir/tiny-yolov3-flir_train` (first run: Kaggle download).

## Session disconnect

What goes to the Drive (`MyDrive/EmbeddedCV/runs/build/notebooks/<dataset>/<model>/`), and
when:

| What | When | By |
|---|---|---|
| the whole of `OUT` (checkpoints, `loss.csv`, `log.txt`…) | every 10 min during the training cell, and at its end (interruption included) | thread of the cell, `colab.sync_outputs` |
| folder of a run (`runs/<run>/` or the root of `_train`) + `summary.json` | as soon as each `tools/train.py` ends, or is interrupted | `colab.autosync`, called by `commandes.run` |
| folder of an evaluation (`map_float*.md`, `dets*/`, PR) + `summary.json` | as soon as each `tools/eval_voc.py` / `eval_quant.py` ends, sweeps **and** `_infer` | same |

`summary.json` (`runs.write_summary`) sums up the folder without the notebook: `run.json`,
iterations done, mean loss and components over the last 50 iterations, duration (sum of
`seconds` of `loss.csv`), s/image, mAP and AP per class of each `map_float*.md`. A
disconnected session therefore only loses the command in progress since its last copy (at
most 10 min of training), never a finished evaluation; the figures stay on the Drive even
if the outputs of the notebook are not saved. `EMBEDDEDCV_AUTOSYNC=0` switches off the copy
at the end of each command.

After a disconnect: reconnect a Colab kernel and rerun "Run All". `colab.restore_outputs`
copies the folder back, the training resumes from the last checkpoint (`--resume`), and
`_sweep` skips the finished runs (`SKIP_DONE`); their log is then no longer in the outputs
of the notebook, but in `loss.csv` on the Drive (see "Fetching"). An exact resume assumes
the same backend (T12.11).

## Following from the PC

```bash
rclone lsd gdrive:EmbeddedCV/runs/build/notebooks/kitti/tiny-yolov3-kitti/runs
rclone cat gdrive:EmbeddedCV/runs/build/notebooks/kitti/tiny-yolov3-kitti/loss.csv | tail
rclone lsl gdrive:EmbeddedCV/runs/build/notebooks/kitti/tiny-yolov3-kitti/ | grep weights
```

A run is finished when its `final.weights` exists. The `loss.csv` on the Drive is at most as
old as the last copy (10 min).

## Fetching and comparing

Harvest at the end of a session, on the PC (rclone configured,
[donnees-drive.md](donnees-drive.md)):

```bash
make harvest   # RUNS_REMOTE=gdrive:EmbeddedCV/runs/build/notebooks by default
```

`make harvest` copies into `build/notebooks/` the `summary.json`, `run.json`, `loss.csv`,
`log.txt`, `map_float*.md` tables and PNG figures of all the datasets, without weights or
checkpoints (a few MB), then prints the summary of the sweeps
(`python -m tools.notebooks.balayages --runs`). The analysis notebook
([analyse_balayages.ipynb](../../notebooks/analyse_balayages.ipynb)) picks up these logs
for the runs missing from the outputs of their notebook. To evaluate locally, the weights
are also needed:

```bash
rclone copy gdrive:EmbeddedCV/runs/build/notebooks/kitti build/notebooks/kitti
```

Then, locally, the inference notebook of the model
([kitti/tiny-yolov3-kitti_infer.ipynb](../../notebooks/kitti/tiny-yolov3-kitti_infer.ipynb))
takes the run `RUN` (None: the most recent), and `COMPARE = True` puts the runs in a single
table (T14.11). `run.json` gives the backend (`device`). An mAP on `SUBSET` images ranks
the runs, it is not publishable (M12).

## Cell outputs and GitHub

Under VS Code, the outputs of a Colab kernel (text, figures) are saved in the `.ipynb` of
the PC. GitHub shows them if the file is pushed with them, images included (PNG in
base64), but not the JavaScript or the widgets, and the preview fails on large notebooks.
Repository rule ([M14](M14-notebooks.md#rules)):

- notebook **executed in full without error**, template cells intact: it is committed with
  its outputs, and `make notebooks` keeps it;
- partial or failed run (e.g. dataset missing): `make ci` refuses it; clear the outputs
  ("Clear All Outputs" in VS Code, or `make notebooks`) before the commit, or rerun to the
  end;
- the parameters are changed without a commit, and an added cell (e.g. `!nvidia-smi`) is
  removed before the commit.

The files of `build/notebooks/<dataset>/<model>/` (copied to the Drive) remain the record
of the runs, and the publishable figures go into `results/` through the M12 scripts.
