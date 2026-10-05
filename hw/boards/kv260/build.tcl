# Block design Vivado de la KV260 (T7.1) : PS Zynq UltraScale+, IP `yolo_conv` (M6),
# interconnexions AXI, horloge 200 MHz, reset, interruption ; bitstream + .xsa.
#
# Reproductible depuis ce seul script (mode batch), après `make hls-export BOARD=kv260` :
#   vivado -mode batch -source hw/boards/kv260/build.tcl -tclargs [jobs]   (make vivado-build)
# Sorties : build/vivado/kv260/{yolo.bit, yolo.xsa, timing.rpt, utilization.rpt}.
# Échoue si le timing n'est pas tenu (WNS < 0 ou TNS < 0, WHS < 0).
#
# Carte des adresses (reprise par pl.dtsi et sw/driver/regmap.hpp) :
#   s_axi_control  0xA000_0000, 64 Ko, via M_AXI_HPM0_FPD
#   gmem_in, gmem_out → S_AXI_HP0_FPD ; gmem_w, gmem_p → S_AXI_HP1_FPD (DDR entière, 64 bits)
#   interrupt → pl_ps_irq0[0] (GIC SPI 89)
#   yolo_post (T9.1, si build/hls/ip/yolo_post_kv260.zip existe : make hls-export-post) :
#   s_axi_control 0xA001_0000, gmem_in/gmem_out → HP0, gmem_p → HP1, interrupt →
#   pl_ps_irq0[1] (GIC SPI 90)
#
# Moteur streaming (T10.9) : -tclargs <jobs> stream (make vivado-build ENGINE=stream), après
# make hls-export-stream. `yolo_stream` à la place de `yolo_conv` (s_axi_control
# 0xA000_0000, gmem_w et gmem_p → HP1, interrupt → pl_ps_irq0[0]) et un AXI DMA en mode
# direct, flux de 8 bits (S_AXI_LITE 0xA001_0000, M_AXI_MM2S et M_AXI_S2MM → HP0,
# s2mm_introut → pl_ps_irq0[1]) ; device tree : pl_stream.dtsi. Non exécuté (Vivado absent) :
# les noms des ports AXI-Stream de l'IP (in_r, out_r) sont à vérifier après l'export.

set jobs 8
set engine conv
if {[info exists argv] && [llength $argv] > 0} { set jobs [lindex $argv 0] }
if {[info exists argv] && [llength $argv] > 1} { set engine [lindex $argv 1] }
if {$engine ni {conv stream}} { error "moteur : conv ou stream" }

set root [file normalize [file join [file dirname [info script]] .. .. ..]]
source [file join $root hls configs kv260.tcl]   ;# PART, CLOCK_NS (mêmes que la synthèse HLS)
set out [file join $root build vivado kv260]
set top [expr {$engine eq "stream" ? "yolo_stream" : "yolo_conv"}]
set ip_zip [file join $root build hls ip ${top}_kv260.zip]
set freq_mhz [expr {round(1000.0 / $CLOCK_NS)}]
set post_zip [file join $root build hls ip yolo_post_kv260.zip]
set with_post [expr {$engine eq "conv" && [file exists $post_zip]}]

if {![file exists $ip_zip]} {
  error "IP absente : $ip_zip (make hls-export[expr {$engine eq "stream" ? "-stream" : ""}] BOARD=kv260)"
}

# --- Projet ------------------------------------------------------------------------------
file delete -force $out
file mkdir $out
create_project yolo_kv260 [file join $out proj] -part $PART

# Fichiers de carte Kria (xhub) : SOM + connecteur de la carrière KV260.
set som [lindex [lsort -dictionary [get_board_parts -quiet *kv260_som*]] end]
if {$som eq ""} {
  error "fichiers de carte KV260 absents : xhub::refresh_catalog \[xhub::get_xstores xilinx_board_store\] ; xhub::install \[xhub::get_xitems *kv260*\]"
}
set_property board_part $som [current_project]
set carrier [lindex [lsort -dictionary [get_board_parts -quiet *kv260_carrier*]] end]
if {$carrier ne ""} {
  # xilinx.com:kv260_carrier:part0:1.3 → xilinx.com:kv260_carrier:som240_1_connector:1.3
  lassign [split $carrier :] vendor name _ version
  set_property board_connections \
    "som240_1_connector $vendor:$name:som240_1_connector:$version" [current_project]
}

# Dépôt IP : l'archive exportée par Vitis HLS (hls/scripts/export.tcl).
set repo [file join $out ip_repo]
file mkdir $repo
exec unzip -o -q $ip_zip -d [file join $repo $top]
if {$with_post} { exec unzip -o -q $post_zip -d [file join $repo yolo_post] }
set_property ip_repo_paths $repo [current_project]
update_ip_catalog

# --- Block design ------------------------------------------------------------------------
create_bd_design yolo
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
  -config {apply_board_preset "1"} $ps

# Un seul maître PL (HPM0_FPD), deux ports esclaves HP, une horloge, une interruption.
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $freq_mhz \
  CONFIG.PSU__USE__IRQ0 {1} \
] $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl0]
if {$engine eq "stream"} {
  # --- Streaming (T10.9) : yolo_stream + AXI DMA ; le reste du script reprend après. ------
  set accel [create_bd_cell -type ip -vlnv yolo-embarque:hls:yolo_stream:1.0 yolo_stream_0]
  set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0]
  set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_addr_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
    CONFIG.c_s_axis_s2mm_tdata_width {8} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_s2mm_burst_size {64} \
  ] $dma
  set sc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_ctrl]
  set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $sc_ctrl
  set sc_hp0 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_hp0]
  set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $sc_hp0
  set sc_hp1 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_hp1]
  set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $sc_hp1
  connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $sc_ctrl/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins $sc_ctrl/M00_AXI] [get_bd_intf_pins $accel/s_axi_control]
  connect_bd_intf_net [get_bd_intf_pins $sc_ctrl/M01_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]
  connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $accel/in_r]
  connect_bd_intf_net [get_bd_intf_pins $accel/out_r] [get_bd_intf_pins $dma/S_AXIS_S2MM]
  connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $sc_hp0/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $sc_hp0/S01_AXI]
  connect_bd_intf_net [get_bd_intf_pins $sc_hp0/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0_FPD]
  connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_w] [get_bd_intf_pins $sc_hp1/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_p] [get_bd_intf_pins $sc_hp1/S01_AXI]
  connect_bd_intf_net [get_bd_intf_pins $sc_hp1/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP1_FPD]
  set clk [get_bd_pins $ps/pl_clk0]
  connect_bd_net $clk [get_bd_pins $ps/maxihpm0_fpd_aclk] [get_bd_pins $ps/saxihp0_fpd_aclk] \
    [get_bd_pins $ps/saxihp1_fpd_aclk] [get_bd_pins $accel/ap_clk] \
    [get_bd_pins $dma/s_axi_lite_aclk] [get_bd_pins $dma/m_axi_mm2s_aclk] \
    [get_bd_pins $dma/m_axi_s2mm_aclk] [get_bd_pins $rst/slowest_sync_clk] \
    [get_bd_pins $sc_ctrl/aclk] [get_bd_pins $sc_hp0/aclk] [get_bd_pins $sc_hp1/aclk]
  connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
  connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $accel/ap_rst_n] \
    [get_bd_pins $dma/axi_resetn] [get_bd_pins $sc_ctrl/aresetn] [get_bd_pins $sc_hp0/aresetn] \
    [get_bd_pins $sc_hp1/aresetn]
  set irq [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat irq_concat]
  set_property CONFIG.NUM_PORTS {2} $irq
  connect_bd_net [get_bd_pins $accel/interrupt] [get_bd_pins $irq/In0]
  connect_bd_net [get_bd_pins $dma/s2mm_introut] [get_bd_pins $irq/In1]
  connect_bd_net [get_bd_pins $irq/dout] [get_bd_pins $ps/pl_ps_irq0]
  assign_bd_address -offset 0xA0000000 -range 0x10000 \
    -target_address_space [get_bd_addr_spaces $ps/Data] [get_bd_addr_segs $accel/s_axi_control/Reg]
  assign_bd_address -offset 0xA0010000 -range 0x10000 \
    -target_address_space [get_bd_addr_spaces $ps/Data] [get_bd_addr_segs $dma/S_AXI_LITE/Reg]
  assign_bd_address
} else {
set accel [create_bd_cell -type ip -vlnv yolo-embarque:hls:yolo_conv:1.0 yolo_conv_0]

# Contrôle : HPM0_FPD → s_axi_control.
set sc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc_ctrl
# Données : deux bundles d'activations sur HP0 (chargement et stockage recouverts, T6.2),
# poids et paramètres sur HP1.
set sc_hp0 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_hp0]
set_property -dict [list CONFIG.NUM_SI [expr {$with_post ? 4 : 2}] CONFIG.NUM_MI {1}] $sc_hp0
set sc_hp1 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_hp1]
set_property -dict [list CONFIG.NUM_SI [expr {$with_post ? 3 : 2}] CONFIG.NUM_MI {1}] $sc_hp1
if {$with_post} { set_property CONFIG.NUM_MI {2} $sc_ctrl }

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $sc_ctrl/M00_AXI] [get_bd_intf_pins $accel/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_in] [get_bd_intf_pins $sc_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_out] [get_bd_intf_pins $sc_hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $sc_hp0/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_w] [get_bd_intf_pins $sc_hp1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi_gmem_p] [get_bd_intf_pins $sc_hp1/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $sc_hp1/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP1_FPD]

# Horloge unique pl_clk0, reset synchronisé.
set clk [get_bd_pins $ps/pl_clk0]
connect_bd_net $clk [get_bd_pins $ps/maxihpm0_fpd_aclk] [get_bd_pins $ps/saxihp0_fpd_aclk] \
  [get_bd_pins $ps/saxihp1_fpd_aclk] [get_bd_pins $accel/ap_clk] [get_bd_pins $rst/slowest_sync_clk] \
  [get_bd_pins $sc_ctrl/aclk] [get_bd_pins $sc_hp0/aclk] [get_bd_pins $sc_hp1/aclk]
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $accel/ap_rst_n] \
  [get_bd_pins $sc_ctrl/aresetn] [get_bd_pins $sc_hp0/aresetn] [get_bd_pins $sc_hp1/aresetn]
if {$with_post} {
  # Post-traitement matériel (T9.1) : même horloge et reset, deux interruptions concaténées.
  set post [create_bd_cell -type ip -vlnv yolo-embarque:hls:yolo_post:1.0 yolo_post_0]
  connect_bd_intf_net [get_bd_intf_pins $sc_ctrl/M01_AXI] [get_bd_intf_pins $post/s_axi_control]
  connect_bd_intf_net [get_bd_intf_pins $post/m_axi_gmem_in] [get_bd_intf_pins $sc_hp0/S02_AXI]
  connect_bd_intf_net [get_bd_intf_pins $post/m_axi_gmem_out] [get_bd_intf_pins $sc_hp0/S03_AXI]
  connect_bd_intf_net [get_bd_intf_pins $post/m_axi_gmem_p] [get_bd_intf_pins $sc_hp1/S02_AXI]
  connect_bd_net $clk [get_bd_pins $post/ap_clk]
  connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $post/ap_rst_n]
  set irq [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat irq_concat]
  set_property CONFIG.NUM_PORTS {2} $irq
  connect_bd_net [get_bd_pins $accel/interrupt] [get_bd_pins $irq/In0]
  connect_bd_net [get_bd_pins $post/interrupt] [get_bd_pins $irq/In1]
  connect_bd_net [get_bd_pins $irq/dout] [get_bd_pins $ps/pl_ps_irq0]
} else {
  connect_bd_net [get_bd_pins $accel/interrupt] [get_bd_pins $ps/pl_ps_irq0]
}

# Adresses : registres à 0xA000_0000 ; les ports m_axi voient toute la DDR.
assign_bd_address -offset 0xA0000000 -range 0x10000 \
  -target_address_space [get_bd_addr_spaces $ps/Data] [get_bd_addr_segs $accel/s_axi_control/Reg]
if {$with_post} {
  assign_bd_address -offset 0xA0010000 -range 0x10000 \
    -target_address_space [get_bd_addr_spaces $ps/Data] [get_bd_addr_segs $post/s_axi_control/Reg]
}
assign_bd_address
}  ;# moteur conv

validate_bd_design
save_bd_design

# --- Synthèse, implémentation, bitstream -------------------------------------------------
set bd_file [get_files yolo.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top yolo_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "échec de la synthèse" }
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "échec de l'implémentation" }

open_run impl_1
report_timing_summary -file [file join $out timing.rpt]
report_utilization -file [file join $out utilization.rpt]
report_utilization -hierarchical -file [file join $out utilization_hier.rpt]
report_power -file [file join $out power.rpt]   ;# puissance puce estimée (M8)

set impl_dir [get_property DIRECTORY [get_runs impl_1]]
file copy -force [file join $impl_dir yolo_wrapper.bit] [file join $out yolo.bit]
write_hw_platform -fixed -include_bit -force [file join $out yolo.xsa]

# Timing tenu : critère d'acceptation de T7.1.
set wns [get_property STATS.WNS [get_runs impl_1]]
set tns [get_property STATS.TNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]
puts "timing : WNS $wns ns, TNS $tns ns, WHS $whs ns à $freq_mhz MHz"
if {$wns < 0 || $tns < 0 || $whs < 0} { error "timing non tenu à $freq_mhz MHz" }
puts "OK : [file join $out yolo.bit], [file join $out yolo.xsa]"
