# C-simulation du noyau, couche par couche (T6.1, T6.3) : make csim BOARD=kv260.
# Sans Vitis, même testbench avec g++ : make csim-gcc.
set step csim
set tb tb_conv
set default_tb_args {}
source [file join [file dirname [info script]] common.tcl]
csim_design -argv $argv_tb -ldflags "-lstdc++fs"
exit
