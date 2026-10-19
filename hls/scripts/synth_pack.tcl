# Synthesis of the packed PE, two int8 products per DSP48E2 (ACC_PACK=2, T23.2): DSP, LUT and
# timing at 200 MHz in proj_<board>_synth_pack/sol/syn/report/ (results/dsp_packing.md).
set step synth_pack
set tb tb_net
set default_tb_args {}
set extra_defs -DACC_PACK=2
source [file join [file dirname [info script]] common.tcl]
csynth_design
exit
