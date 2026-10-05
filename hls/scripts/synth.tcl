# Synthèse du noyau (T6.5) : rapports dans proj_<carte>_synth/sol/syn/report/.
set step synth
set tb tb_net
set default_tb_args {}
source [file join [file dirname [info script]] common.tcl]
csynth_design
exit
