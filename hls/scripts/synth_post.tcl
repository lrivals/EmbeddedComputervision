# Synthèse du post-traitement matériel `yolo_post` (T9.1.2).
set step synth_post
set top yolo_post
set tb tb_post
set default_tb_args {}
source [file join [file dirname [info script]] common.tcl]
csynth_design
exit
