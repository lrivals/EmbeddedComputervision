# ADR 0003 — Choix de la carte FPGA

- Statut : **à décider** (tâche T6.0)

## Contexte

L'architecture « moteur unique couche par couche » (§10.1) convient à un SoC avec ARM
(Zynq-7000, Zynq UltraScale+/Kria). Le choix fixe les ressources (DSP, BRAM, bande passante
DDR) et donc les tuiles Tm, Tn, Tr, Tc.

## Critères

| Critère | Mesure |
|---|---|
| Faisabilité | tuiles trouvées par `tools/roofline.py` (T5.6) tenant dans la BRAM |
| Performance estimée | ms/image au point roofline retenu |
| Processeur | ARM dur disponible pour le driver et le post-traitement |
| Outils | version de Vivado/Vitis supportée, licence gratuite |
| Coût et disponibilité | prix, livraison |
| Comparabilité | cartes utilisées par la base (Zynq 7035, Zybo Z7-20, KV260) |

## Candidats

Kria KV260, Zynq-7000 (Zybo Z7-20, PYNQ-Z2), Ultra96-V2.

## Décision

À compléter.
