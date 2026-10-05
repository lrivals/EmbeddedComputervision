# Export de l'IP pour Vivado (T6.5, intégration M7) : build/hls/ip/yolo_conv_<carte>.zip.
set step export
set tb tb_net
set default_tb_args {}
source [file join [file dirname [info script]] common.tcl]
csynth_design
file mkdir [file join $root build hls ip]
export_design -format ip_catalog -rtl verilog -vendor yolo-embarque -library hls \
  -version 1.0 -output [file join $root build hls ip yolo_conv_$board.zip]
exit
