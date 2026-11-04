/*
 * Copyright (c) 2026 Andrew Pullin
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Embed every .pte with explicit alignment.
 *
 * EMBED_FILES resolves symbols but only guarantees 4-byte alignment, and
 * Program::load hard-requires kMinimumAlignment (program.cpp:146-153) or it returns
 * InvalidArgument. Aligning in flash costs no RAM; copying to an aligned buffer
 * would cost the size of each program.
 */
        .section .rodata.embedded, "a"

        .macro embed_pte name, file
        .balign 16
        .global \name\()_start
\name\()_start:
        .incbin "\file"
        .global \name\()_end
\name\()_end:
        .byte 0
        .endm

#ifndef ET_PLE_ONLY
        embed_pte mlp_pte,     "mlp.pte"
        embed_pte bmmhalf_pte, "bmmhalf.pte"
        embed_pte mlpdel_pte,  "mlp_delegated.pte"
        embed_pte mlpf32_pte,  "mlp_fp32_backend.pte"
        embed_pte mlpper_pte,  "mlp_perop.pte"
        /* the same MLP down the other two rungs of the weight-format ladder */
        embed_pte mlpi4_pte,   "mlp_i4.pte"
        /* embedding gathers down both non-i8 format rungs */
        embed_pte embi4_pte,   "embed_i4.pte"
        embed_pte embi16_pte,  "embed_i16.pte"
        embed_pte selhead_pte, "selected_head.pte"
        embed_pte embedlin_pte, "embed_linear.pte"
        embed_pte grpi4_pte,   "grouped_i4.pte"
        embed_pte grpi8_pte,   "grouped_i8.pte"
        embed_pte grpi4w_pte,  "grouped_i4_wide.pte"
        embed_pte convd_pte,   "convd_delegated.pte"
        embed_pte convdp_pte,  "convd.pte"
        embed_pte c1dd_pte,    "conv1d_delegated.pte"
        embed_pte c1ddp_pte,   "conv1dd.pte"
        embed_pte c3dd_pte,    "conv3d_delegated.pte"
        embed_pte dwc_pte,     "dwconv_delegated.pte"
        embed_pte dwi4_pte,    "dwconv_i4.pte"
        embed_pte dwgi4_pte,   "dwconv_grouped_i4.pte"
        embed_pte dwi16_pte,   "dwconv_i16.pte"
        embed_pte dwcp_pte,    "dwconvd.pte"
        embed_pte dwres_pte,   "dwconvres_delegated.pte"
        embed_pte dwresp_pte,  "dwconvresd.pte"
        embed_pte dw1_pte,     "dwconv1d_delegated.pte"
        embed_pte dw1p_pte,    "dwconv1dd.pte"
        embed_pte cvr_pte,     "convrelu_delegated.pte"
        embed_pte cvs_pte,     "convsilu_delegated.pte"
        embed_pte cvt_pte,     "convt_delegated.pte"
        embed_pte cvti4_pte,   "convt_i4.pte"
        embed_pte cvti16_pte,  "convt_i16.pte"
        embed_pte cvtp_pte,    "convtd.pte"
        embed_pte cvtop_pte,   "convtop_delegated.pte"
        embed_pte cvtopp_pte,  "convtopd.pte"
        embed_pte cvtdil_pte,  "convtdil_delegated.pte"
        embed_pte cvtdil16_pte, "convtdil_i16.pte"
        embed_pte cvtdilp_pte, "convtdild.pte"
        embed_pte grpp_pte,    "grpconvd.pte"
        embed_pte grpep_pte,   "grpconvedged.pte"
        embed_pte grp8p_pte,   "grpconv8d.pte"
        embed_pte grp_pte,     "grpconv_delegated.pte"
        embed_pte grpe_pte,    "grpconvedge_delegated.pte"
        embed_pte grp8_pte,    "grpconv8_delegated.pte"
        embed_pte grpr_pte,    "grpconvrelu_delegated.pte"
        embed_pte grprp_pte,   "grpconvrelud.pte"
        embed_pte cvst_pte,    "conv_static.pte"
        embed_pte ln2d_pte,    "layernorm2dd.pte"
        embed_pte ln2dx_pte,   "layernorm2dx.pte"
        embed_pte cvres_pte,   "convres_delegated.pte"
        embed_pte cvrv_pte,    "convresview_delegated.pte"
        embed_pte cvrvp_pte,   "convresviewd.pte"
        embed_pte cvresp_pte,  "convresd.pte"
        embed_pte cvresq_pte,  "convres_i16.pte"
        embed_pte cvi4_pte,    "conv_i4.pte"
        embed_pte cvi16_pte,   "conv_i16.pte"
        embed_pte g8i4_pte,    "grpconv8_i4.pte"
        embed_pte g8i16_pte,   "grpconv8_i16.pte"
        embed_pte mlpml_pte,   "mlp_minlat.pte"
        embed_pte mlppt_pte,   "mlp_pt2e.pte"
        embed_pte cvpt_pte,    "convnet_pt2e.pte"
        embed_pte actshare_pte, "actshare.pte"
        embed_pte pt2eties_pte, "pt2e_ties.pte"
        embed_pte blkwide_pte, "block_wide.pte"
        embed_pte mlpi16_pte,  "mlp_i16.pte"
        /* calibrated static activation scales (FLAG_STATIC_ACT) */
        embed_pte mlpstat_pte, "mlp_static.pte"
        /* single-head attention core, portable and with opt-in int8 q@kT */
        embed_pte attnqk_pte,  "attnqk.pte"
        embed_pte attnqkq_pte, "attnqk_qk.pte"
        /* ... with its keys kept as a persistent int8 K cache */
        embed_pte attnkc8_pte,  "attnkc8.pte"
        embed_pte attnkc8q_pte, "attnkc8_qk.pte"
        /* both fates of a fusable relu: final fused, mid-chain portable */
        embed_pte mlpr_pte,    "mlpr.pte"
        embed_pte mlprdel_pte, "mlpr_delegated.pte"
        /* silu diamond + exact gelu fused into the delegate */
        embed_pte mlps_pte,    "mlps.pte"
        embed_pte mlpsdel_pte, "mlps_delegated.pte"
        /* fused epilogues: fused multiply (pair + gate), one-op activations */
        embed_pte epiglu_pte,  "epiglu_delegated.pte"
        embed_pte epiglum1_pte, "epiglu_m1.pte"
        embed_pte epiacts_pte, "epiacts_delegated.pte"
        embed_pte blkdel_pte,  "block_delegated.pte"
        embed_pte blki4_pte,   "block_i4.pte"
        embed_pte blki16_pte,  "block_i16.pte"
        /* One PTE, one PTD and one mutable arena across prefill -> decode. */
        embed_pte glimmermulti_pte, "glimmer_multimethod.pte"
        /* The same, with a persistent int8 K cache; same glimmerw named data. */
        embed_pte glimmermultikc8_pte, "glimmer_multimethod_kc8.pte"
        embed_pte decode4_pte, "decode_i4.pte"
        embed_pte convnet_pte, "convnet.pte"
        embed_pte block_pte,   "block.pte"
        /* coverage zoo: recurrence, unrolled by export into gate ops */
        embed_pte lstm_pte,    "lstm.pte"
        embed_pte gru_pte,     "gru.pte"
        /* conv/pool ranks: 1-D rides on the 2-D kernels via unsqueeze */
        embed_pte conv1d_pte,  "conv1d.pte"
        embed_pte pool_pte,    "pool.pte"
        /* architecture coverage: ssm, kan, shared-weight loop, activations, norms */
        embed_pte mamba_pte,  "mamba.pte"
        embed_pte kan_pte,    "kan.pte"
        embed_pte looped_pte, "looped.pte"
        embed_pte acts_pte,   "acts.pte"
        embed_pte norms_pte,  "norms.pte"
        embed_pte memplan_pte, "memplan.pte"
        embed_pte ropefull_pte, "ropefull.pte"
        /* the fused decode attention tail, held to its host mirror */
        embed_pte attntail_pte, "attntail.pte"
        /* message passing: scatter-accumulate with duplicate indices */
        embed_pte gcn_pte,     "gcn.pte"
        /* program-data separated pair: program only, plus its weights */
        embed_pte block_split_pte, "block_split.pte"
        /* the asymmetric refusal check: the carrier and its synthetic cases
           (asym_blob_cases.py; main.cpp run_asym_refusal_check) */
        embed_pte asymcarrier_pte, "asym_carrier.pte"
        embed_pte asymcases_blob,  "asym_refusal_cases.bin"
#endif /* !ET_PLE_ONLY */
