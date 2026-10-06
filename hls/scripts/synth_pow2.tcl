# Synthèse de la PE à décalages (T9.2.4, REQ-YOLO ; T13.24) : rapports dans
# proj_<carte>_synth_pow2/sol/syn/report/, lus par tools/figures (ressources.png).
set step synth_pow2
set tb tb_net
set default_tb_args {}
set extra_defs -DACC_WMODE_POW2
source [file join [file dirname [info script]] common.tcl]
csynth_design
exit
