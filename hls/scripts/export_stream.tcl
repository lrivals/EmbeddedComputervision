# Export de l'IP `yolo_stream` (T10.9) : build/hls/ip/yolo_stream_<carte>.zip.
set step export_stream
set top yolo_stream
set tb tb_stream
set default_tb_args {--image 000001}
source [file join [file dirname [info script]] common.tcl]
csynth_design
file mkdir [file join $root build hls ip]
export_design -format ip_catalog -rtl verilog -vendor yolo-embarque -library hls \
  -version 1.0 -output [file join $root build hls ip yolo_stream_$board.zip]
exit
