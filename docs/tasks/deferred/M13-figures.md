# M13 — Figures: deferred items

Parent: [M13-figures.md](../M13-figures.md). The figure code is complete; these items are
data that only Vitis or the board produce. Each figure picks the data up by itself on the
next `make figures`. Index: [README.md](README.md).

### [ ] T13.24 — Solid resource bars
- **Condition**: Vitis · **Needs first**: T6.5, T9.2.4, T10.8, M23 synthesis
- **To do**: the Vitis syntheses `make hls-synth`, `hls-synth-pow2`, `hls-synth-pack`,
  `hls-synth-stream`; the figure reads
  `hls/proj_kv260_<step>/sol/syn/report/csynth.xml` (`tools/hls_report.read_synth`).
- **Acceptance**: the hatched bars (estimates) become solid in
  `figures/resultats/ressources.png`.

### [ ] T13.12, T13.13, T13.22, T13.23 — Board stage of the result figures
- **Condition**: Board · **Needs first**: T8.1, T8.2
- **To do**: once the M8 measurements exist, regenerate `make figures` and check that:
  - T13.12 (`chaine_verif.png`): the "board" arrow turns green (read from the Progress
    table);
  - T13.13 (`map_stades.png`): the "board" bar appears;
  - T13.22 (`roofline_couches_<board>.png`): the "measured" point appears;
  - T13.23 (`etat_art.png`): the markers of this work become filled, power of this work
    filled in.
