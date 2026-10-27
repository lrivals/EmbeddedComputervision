# Datasets on Google Drive

The datasets do not go through git (`data/` is 33 GB). To find them on another machine or
on Colab, each ready dataset is archived as a tar and copied to Google Drive, in
`MyDrive/EmbeddedCV/data/`. The notebooks and `tools/notebooks/colab.py:prepare_data` know
how to fetch it from there ("Access" section below).

```
MyDrive/EmbeddedCV/
  data/   <dataset>.tar   one uncompressed tar per dataset, links followed (tree of data/)
  runs/                   outputs and checkpoints of the notebooks (colab.sync_outputs; M14, T14.8)
```

The training runs follow the same path as in the repository ("Training runs" section
below).

## Setting up rclone (once per PC)

```bash
sudo apt install rclone
rclone config create gdrive drive scope=drive   # opens the browser: Google sign-in
rclone lsd gdrive:                              # lists the folders of the Drive
rclone about gdrive:                            # free space
```

The remote is called `gdrive`, and the archives go into `gdrive:EmbeddedCV/data`. To change
it, set the environment variable `DRIVE_REMOTE` (for example
`DRIVE_REMOTE=other:Folder/data`), read by `tools/get_datasets.sh` and
`colab.prepare_data`.

## Exporting a dataset

On the PC that has the dataset (marker file present, `tools/get_datasets.sh check`):

```bash
tools/get_datasets.sh push voc visdrone crowdhuman exdark
```

`push` runs `pack` (`data/<root>` → `data_archives/<dataset>.tar`, symbolic links followed,
`*.zip`, `*.tar` and `*.part` excluded), then `rclone copy` to `gdrive:EmbeddedCV/data`. The
archive stays in `data_archives/`, which git ignores, and can be deleted once the upload is
checked:

```bash
rclone ls gdrive:EmbeddedCV/data     # sizes on the Drive
ls -l data_archives/                 # must be identical
```

| Dataset | Size | Marker file (`data/…`) | On the Drive | Remark |
|---|---|---|---|---|
| voc | 2.9 GB | `VOCdevkit/VOC2007/ImageSets/Main/test.txt` | ✓ (2026-10-06) | also through `tools/get_voc.sh` |
| exdark | 1.5 GB | `exdark/imageclasslist.txt` | ✓ (2026-10-06) | the only way: annotations missing from Kaggle |
| visdrone | 1.9 GB | `visdrone/VisDrone2019-DET-val/annotations` | ✓ (2026-10-06) | also through the Kaggle API |
| crowdhuman | 11 GB | `crowdhuman/annotation_val.odgt` | ✓ (2026-10-06) | also through the Kaggle API; the longest to upload |
| coco | ≈ 1.3 GB | `coco/annotations/instances_val2017.json` | — | missing from the PC; can be downloaded again (`get_datasets.sh coco`) |
| kitti | ≈ 12 GB | `kitti/training/image_2/000000.png` | — | missing from the PC; can be downloaded again (`get_datasets.sh kitti`) |
| auair | 2.5 GB | `auair/annotations.json` | ✓ (2026-10-08) | M18: link to `AU-AIR dataset/` (`get_datasets.sh kaggle`) |
| dronevehicle | 2.2 GB | `dronevehicle/test/labels` | ✓ (2026-10-08) | M18: output of `tools/prep_datasets.py dronevehicle` (cropped) |
| hituav | 0.2 GB | `hituav/labels/test` | ✓ (2026-10-08) | M18: link to `HIT-UAV: …/hit-uav` |
| uavdt | 6.5 GB | `uavdt/annotations_test.json` | ✓ (2026-10-08) | M18: output of `tools/prep_datasets.py uavdt` (M sequences only) |
| flir | 12 GB | `flir/images_thermal_val/coco.json` | ✓ (2026-10-07) | also through the Kaggle API (mirror `samdazel/teledyne-flir-adas-thermal-dataset-v2`); RGB and videos included (thermal images alone: 4.9 GB) |
| asl | 1.2 GB | `asl/annotations_val.json` | ✓ (2026-10-27) | M21: output of `tools/prep_datasets.py asl` (hand pseudo-boxes, hard links, settings of T21.3); 1,214,136,320 bytes, same size on the Drive |
| prewid | 1.1 GB | `prewid/annotations_val.json` | ✓ (2026-10-26) | M38: output of `tools/prep_datasets.py prewid` (hard links, all splits including `lab` and leave-one-camera-out); the lab mp4 stays out |

The sizes are those of `du -shL` on the PC. Together, the ten datasets uploaded make about
43 GB. That is more than the free Drive (15 GB), but the project account has 100 GB. Check
the space with `rclone about gdrive:` before a large upload. After a new upload, update the
"On the Drive" column.

On 2026-10-08, the sizes of the nine archives of the Drive were compared with those of
`data_archives/`, which was then emptied. crowdhuman, exdark and flir are now only on the
Drive (missing from `data/`): `tools/get_datasets.sh pull <dataset>` fetches them.

## Access

The rule is the same everywhere: **a dataset that is present is used as it is; otherwise,
it is fetched from the Drive**; as a last resort, it is downloaded, but only on Colab.

| Where | Dataset present | Dataset missing |
|---|---|---|
| PC, command line | nothing to do | `tools/get_datasets.sh pull <dataset>` |
| PC, notebook | used as it is | `env.prepare` runs `pull` (rclone); otherwise, `Prerequis` with the commands |
| Colab | used as it is | Drive mounted + `unpack`; otherwise, direct download (VOC, COCO, KITTI, Kaggle: CrowdHuman, VisDrone, FLIR) |

`pull` does nothing if the marker file is present. Otherwise, it copies
`gdrive:EmbeddedCV/data/<dataset>.tar` into `data_archives/`, then extracts it into `data/`
(`unpack`). The exit code is non-zero if rclone, the remote or the archive is missing, and
the message says what to do.

In the code, `colab.prepare_data` tries the ways in order and returns the one that worked:

```python
from tools.notebooks import colab

colab.prepare_data("exdark")   # "ready" | "drive" | "rclone" | "download"
```

1. `ready`: the marker file is present;
2. `drive`: `<drive_dir>/data/<dataset>.tar` on the mounted Drive (Colab,
   `/content/drive`);
3. `rclone`: the same archive through `get_datasets.sh pull` (PC with rclone;
   `remote=None` disables this way);
4. `download`: the direct download (`download=False` disables it).

If no way works, the function raises `RuntimeError` with the `push` command to run on the
PC that has the dataset. Locally, the notebooks call `prepare_data` without a download
(`env.prepare`): a download of several GB stays an explicit command.

## Training runs

On Colab, with `DRIVE_DIR` (default `/content/drive/MyDrive/EmbeddedCV`), the `_train` and
`_sweep` notebooks copy their folder every 10 min, then at the end of the cell
(`colab.sync_outputs`). The copy goes into `runs/` under the same relative path as in the
repository:

```
MyDrive/EmbeddedCV/runs/build/notebooks/<dataset>/<model>/
  final.weights, loss.csv, run.json …   "train" run (_train notebook)
  runs/b16-sall/, runs/b8-s500/ …       batch × subset sweep (_sweep notebook)
  eval/<run>/                           evaluations of the inference notebooks
```

At the start of a session, `colab.restore_outputs` copies them back into
`build/notebooks/`. An interrupted training thus resumes from the last copied checkpoint
(`--resume`), and the inference notebook finds the runs (`RUN`, `COMPARE`; M14, T14.11). To
get the runs trained on Colab onto the PC:

```bash
rclone copy gdrive:EmbeddedCV/runs/build/notebooks build/notebooks   # all the runs
rclone copy gdrive:EmbeddedCV/runs/build/notebooks/kitti build/notebooks/kitti
```

The local inference notebooks then read these runs like those trained on the PC. A run
trained on GPU says so in `run.json` (`device`).

## Pitfalls

- Only the tar is read on the Drive, in one go. The images are never read through the Drive
  mount, which is slow for thousands of small files.
- `pack` first writes `<dataset>.tar.part`, then renames it: a tar that is present is
  complete. A `.part` left after an interruption can be deleted.
- ExDark: the link `data/exdark/ExDark` points to the whole of `data/ExDark Dataset/`. The
  archive therefore also contains the annotations under `exdark/ExDark/ExDark_Annno/`. They
  weigh a few MB and are not read.
- FLIR on the PC: the manual download lands in `data/Teledyne FLIR ADAS Thermal Dataset
  v2/FLIR_ADAS_v2/`. Link it by hand, as `kaggle` does for the mirror:
  `ln -s "Teledyne FLIR ADAS Thermal Dataset v2/FLIR_ADAS_v2" data/flir`.
- Large upload (> 10 GB): the file goes up in full, then Drive takes several minutes to
  finalise it. rclone may then give up and resend everything without an error message: the
  `--progress` total grows by one archive size at each attempt (23.8 then 35.7 GiB for the
  11.9 GiB of FLIR), and the upload sometimes freezes (`0/s`). Do not rerun `push`, which
  rebuilds the archive: resend the one in `data_archives/` (`--timeout 30m` to let Drive
  finish, an untested option), then compare the sizes.

  ```bash
  rclone copy data_archives/flir.tar gdrive:EmbeddedCV/data --progress --timeout 30m
  ```

- A dataset modified locally (new annotations, subset) must be sent again with `push`.
  `pull` never replaces a dataset that is already present.
