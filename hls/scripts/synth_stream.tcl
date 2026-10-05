# Synthèse du top streaming `yolo_stream` (T10.8) : poids en ROM, DATAFLOW de 9 étages.
# Prérequis : build/hls/stream_rom (make hls-synth-stream le génère).
set step synth_stream
set top yolo_stream
set tb tb_stream
set default_tb_args {--image 000001}
source [file join [file dirname [info script]] common.tcl]
csynth_design
exit
