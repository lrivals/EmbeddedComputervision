# Export de l'IP `yolo_post` (T9.1.3) : build/hls/ip/yolo_post_<carte>.zip.
set step export_post
set top yolo_post
set tb tb_post
set default_tb_args {}
source [file join [file dirname [info script]] common.tcl]
csynth_design
file mkdir [file join $root build hls ip]
export_design -format ip_catalog -rtl verilog -vendor yolo-embarque -library hls \
  -version 1.0 -output [file join $root build hls ip yolo_post_$board.zip]
exit
