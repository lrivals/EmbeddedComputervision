# Projet Vitis HLS commun aux étapes csim / synth / cosim / export (T6.5).
#
# Appelé par les scripts d'étape, depuis hls/ :
#   vitis_hls -f scripts/<étape>.tcl -tclargs [carte] [arguments du testbench…]
# La carte (défaut kv260) choisit hls/configs/<carte>.tcl : part, horloge, tuiles.
# Avant `source`, le script d'étape fixe `step` (nom du projet) et `tb` (testbench).

set board kv260
set tb_args {}
if {[info exists argv] && [llength $argv] > 0} {
  set board [lindex $argv 0]
  set tb_args [lrange $argv 1 end]
}
set root [file normalize [file join [file dirname [info script]] .. ..]]
source [file join $root hls configs $board.tcl]

# Noyau : C++14 (défaut Vitis), tuiles de la carte. Testbench : C++17 (golden, driver).
set inc "-I$root/hls/kernels -I$root/cpp/golden/include -I$root/sw/driver"
set defs "-DACC_TM=$TM -DACC_TN=$TN -DACC_TR=$TR -DACC_TC=$TC"
set kflags "$inc $defs"
set tbflags "-std=c++17 $inc $defs"
if {[llength $tb_args] == 0} { set tb_args $default_tb_args }
set argv_tb "--model $root/model $tb_args"

open_project -reset proj_${board}_$step
set_top yolo_conv
add_files [file join $root hls kernels conv_pe.cpp] -cflags $kflags
foreach f {
  cpp/golden/src/golden.cpp cpp/golden/src/json.cpp cpp/golden/src/npy.cpp
  cpp/golden/src/model.cpp cpp/golden/src/engine.cpp sw/driver/program.cpp
} {
  add_files -tb [file join $root $f] -cflags $tbflags
}
add_files -tb [file join $root hls tb $tb.cpp] -cflags $tbflags

open_solution -reset sol -flow_target vivado
set_part $PART
create_clock -period $CLOCK_NS -name default
config_interface -m_axi_addr64
