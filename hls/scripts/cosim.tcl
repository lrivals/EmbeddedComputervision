# Co-simulation RTL sur une image complète (T6.2, T6.5) : le testbench réseau appelle le
# noyau RTL une fois par conv et compare chaque couche au golden et aux dumps.
# Traces de ports (recouvrement chargement / calcul) : proj_<carte>_cosim/sol/sim/.
# Durée : l'estimation C-sim est ~4·10⁷ cycles par image ; la simulation RTL peut prendre
# plusieurs heures. Autre image ou réseau : -tclargs kv260 --net tiny-yolov3-coco --image 000002
set step cosim
set tb tb_net
set default_tb_args {--net tiny-yolov2-voc --image 000001}
source [file join [file dirname [info script]] common.tcl]
csynth_design
cosim_design -argv $argv_tb -ldflags "-lstdc++fs" -trace_level port
exit
