# Co-simulation RTL de `yolo_stream` (T10.8) : DATAFLOW sans interblocage, profondeurs des
# FIFO (#pragma HLS STREAM de yolo_stream.cpp). Long : une image.
set step cosim_stream
set top yolo_stream
set tb tb_stream
set default_tb_args {--image 000001}
source [file join [file dirname [info script]] common.tcl]
csynth_design
cosim_design -trace_level none
exit
