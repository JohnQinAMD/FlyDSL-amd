# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Self-contained gfx950 fixed-register MLA prefill phase engine.

The opaque call owns Q/O initialization, the N16x2 backedge, the final
quadrant-sum reduction, normalization, and stores.  The instruction stream is
an independently assembled FlyDSL inline-ISA reimplementation of the
gfx950 aiter phase topology; it neither loads nor calls an external code
object.  Keeping the full state lifetime inside one call prevents LLVM from
creating a 256-f32 O PHI/copy boundary.
"""

import re

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


_R25_FIXED_ENGINE_BODY = r"""s_mul_i32 s80, s2, 1
	s_sub_u32 s81, s78, s79
	s_cmp_le_u32 s81, s80
	s_cbranch_scc1 .Lr25_label_3059
	s_mov_b32 s69, 0
	s_lshr_b32 s44, 16, s69
	s_mul_i32 s73, s44, 4
	s_mul_i32 s73, s73, s67
	s_mul_i32 s45, s4, s44
	s_sub_u32 s50, s46, s47
	s_lshl_b32 s56, s50, s69
	s_sub_u32 s82, s56, s81
	s_mul_i32 s58, s2, 1
	s_add_u32 s82, s82, s58
	s_add_u32 s57, s82, 1
	s_min_u32 s56, s56, s57
	s_lshr_b32 s50, s56, s69
	s_lshl_b32 s56, s45, s69
	s_add_u32 s83, s56, 15
	s_mul_i32 s84, s67, 16
	s_cmp_le_u32 s50, s45
	s_cbranch_scc1 .Lr25_label_3059
	s_mul_i32 s56, s50, 4
	s_mov_b32 s26, s56
	s_mul_i32 s56, s47, 4
	s_add_u32 s24, s56, s24
	s_addc_u32 s25, 0, s25
	s_mov_b32 s70, 0
	s_sub_u32 s71, s50, s45
	s_mul_i32 s39, s67, s44
	s_mov_b32 s38, s71
	v_cvt_f32_u32_e32 v20, s39
	s_sub_i32 s56, 0, s39
	v_rcp_iflag_f32_e32 v20, v20
	s_nop 0
	v_mul_f32_e32 v20, 0x4f7ffffe, v20
	v_cvt_u32_f32_e32 v20, v20
	v_mul_lo_u32 v21, s56, v20
	v_mul_hi_u32 v21, v20, v21
	v_add_u32_e32 v20, v20, v21
	v_mul_hi_u32 v20, s38, v20
	v_mul_lo_u32 v21, v20, s39
	v_sub_u32_e32 v23, s38, v21
	v_add_u32_e32 v22, 1, v20
	v_cmp_le_u32_e32 vcc, s39, v23
	v_subrev_u32_e32 v21, s39, v23
	s_nop 0
	v_cndmask_b32_e32 v20, v20, v22, vcc
	v_cndmask_b32_e32 v23, v23, v21, vcc
	v_add_u32_e32 v21, 1, v20
	v_cmp_le_u32_e32 vcc, s39, v23
	s_nop 1
	v_cndmask_b32_e32 v23, v20, v21, vcc
	s_nop 3
	v_readfirstlane_b32 s40, v23
	s_nop 3
	s_mov_b32 s71, s40
	s_mul_i32 s56, s71, s39
	s_sub_u32 s56, s38, s56
	s_mov_b32 s57, 0
	s_cmp_lt_u32 s56, s44
	s_cselect_b32 s57, s57, 1
	s_add_u32 s71, s57, s71
	s_cmpk_eq_u32 s57, 0x1
	s_cselect_b32 s49, 0, s56
	s_mov_b32 s48, s49
	v_lshrrev_b32_e32 v20, 3, v0
	v_and_b32_e32 v21, 1, v20
	v_lshlrev_b32_e32 v8, 3, v21
	v_and_b32_e32 v20, 4, v20
	v_add_u32_e32 v8, v8, v20
	v_add_u32_e64 v8, v8, s7
	v_add_u32_e32 v8, s45, v8
	v_lshlrev_b32_e32 v8, 2, v8
	buffer_load_dword v10, v8, s[24:27], 0 offen
	v_add_u32_e32 v8, s73, v8
	buffer_load_dword v11, v8, s[24:27], 0 offen
	v_add_u32_e32 v8, s73, v8
	s_add_u32 s56, s80, s79
	v_mov_b32_e32 v20, s56
	v_mul_lo_u32 v21, s74, v20
	v_mul_hi_u32 v22, s74, v20
	s_nop 2
	v_readfirstlane_b32 s56, v21
	v_readfirstlane_b32 s57, v22
	s_nop 4
	s_add_u32 s16, s56, s16
	s_addc_u32 s17, s57, s17
	s_sub_u32 s56, s81, s80
	s_mul_i32 s56, s56, s74
	s_mov_b32 s18, s56
	s_mul_i32 s56, s7, 0x480
	v_lshlrev_b32_e32 v30, 2, v0
	v_add_u32_e32 v30, s56, v30
	s_mul_i32 s56, s7, 0x1420
	s_add_u32 s34, 0, s56
	s_add_u32 s35, 0x5080, s34
	s_add_u32 s36, 0x5080, s35
	v_lshrrev_b32_e32 v20, 4, v0
	v_lshlrev_b32_e32 v21, 2, v20
	v_and_b32_e32 v20, 15, v0
	v_lshrrev_b32_e32 v22, 2, v20
	v_mul_i32_i24_e32 v22, 0x140, v22
	v_add_u32_e32 v21, v22, v21
	v_and_b32_e32 v20, 3, v0
	v_mul_i32_i24_e32 v22, 0x508, v20
	v_add_u32_e32 v21, v22, v21
	v_lshlrev_b32_e32 v29, 2, v21
	s_mov_b32 m0, s34
	v_add_u32_e32 v28, 0, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_mov_b32 m0, s35
	v_add_u32_e32 v28, 0x4800, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_waitcnt vmcnt(20)
	s_barrier
	s_mov_b32 m0, s36
	v_add_u32_e32 v28, 0x9000, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_cmp_eq_i32 s7, 0
	s_cbranch_scc0 .Lr25_label_01C1
	ds_read_b128 a[0:3], v29
	ds_read_b128 a[4:7], v29 offset:64
	ds_read_b128 a[8:11], v29 offset:128
	ds_read_b128 a[12:15], v29 offset:192
	ds_read_b128 a[16:19], v29 offset:256
	ds_read_b128 a[20:23], v29 offset:320
	ds_read_b128 a[24:27], v29 offset:384
	ds_read_b128 a[28:31], v29 offset:448
	ds_read_b128 a[32:35], v29 offset:512
	ds_read_b128 a[36:39], v29 offset:576
	ds_read_b128 a[40:43], v29 offset:640
	ds_read_b128 a[44:47], v29 offset:704
	ds_read_b128 a[48:51], v29 offset:768
	ds_read_b128 a[52:55], v29 offset:832
	ds_read_b128 a[56:59], v29 offset:896
	ds_read_b128 a[60:63], v29 offset:960
	ds_read_b128 a[64:67], v29 offset:1024
	ds_read_b128 a[68:71], v29 offset:1088
	s_waitcnt lgkmcnt(0)
	.Lr25_label_01C1:
	s_waitcnt vmcnt(20)
	s_barrier
	s_mov_b32 m0, s34
	v_add_u32_e32 v28, 0xd800, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_cmp_eq_i32 s7, 1
	s_cbranch_scc0 .Lr25_label_0225
	ds_read_b128 a[0:3], v29 offset:20608
	ds_read_b128 a[4:7], v29 offset:20672
	ds_read_b128 a[8:11], v29 offset:20736
	ds_read_b128 a[12:15], v29 offset:20800
	ds_read_b128 a[16:19], v29 offset:20864
	ds_read_b128 a[20:23], v29 offset:20928
	ds_read_b128 a[24:27], v29 offset:20992
	ds_read_b128 a[28:31], v29 offset:21056
	ds_read_b128 a[32:35], v29 offset:21120
	ds_read_b128 a[36:39], v29 offset:21184
	ds_read_b128 a[40:43], v29 offset:21248
	ds_read_b128 a[44:47], v29 offset:21312
	ds_read_b128 a[48:51], v29 offset:21376
	ds_read_b128 a[52:55], v29 offset:21440
	ds_read_b128 a[56:59], v29 offset:21504
	ds_read_b128 a[60:63], v29 offset:21568
	ds_read_b128 a[64:67], v29 offset:21632
	ds_read_b128 a[68:71], v29 offset:21696
	s_waitcnt lgkmcnt(0)
	.Lr25_label_0225:
	s_waitcnt vmcnt(20)
	s_barrier
	s_mov_b32 m0, s35
	v_add_u32_e32 v28, 0x12000, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_cmp_eq_i32 s7, 2
	s_cbranch_scc0 .Lr25_label_0289
	ds_read_b128 a[0:3], v29 offset:41216
	ds_read_b128 a[4:7], v29 offset:41280
	ds_read_b128 a[8:11], v29 offset:41344
	ds_read_b128 a[12:15], v29 offset:41408
	ds_read_b128 a[16:19], v29 offset:41472
	ds_read_b128 a[20:23], v29 offset:41536
	ds_read_b128 a[24:27], v29 offset:41600
	ds_read_b128 a[28:31], v29 offset:41664
	ds_read_b128 a[32:35], v29 offset:41728
	ds_read_b128 a[36:39], v29 offset:41792
	ds_read_b128 a[40:43], v29 offset:41856
	ds_read_b128 a[44:47], v29 offset:41920
	ds_read_b128 a[48:51], v29 offset:41984
	ds_read_b128 a[52:55], v29 offset:42048
	ds_read_b128 a[56:59], v29 offset:42112
	ds_read_b128 a[60:63], v29 offset:42176
	ds_read_b128 a[64:67], v29 offset:42240
	ds_read_b128 a[68:71], v29 offset:42304
	s_waitcnt lgkmcnt(0)
	.Lr25_label_0289:
	s_waitcnt vmcnt(20)
	s_barrier
	s_mov_b32 m0, s36
	v_add_u32_e32 v28, 0x16800, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_cmp_eq_i32 s7, 3
	s_cbranch_scc0 .Lr25_label_02ED
	ds_read_b128 a[0:3], v29
	ds_read_b128 a[4:7], v29 offset:64
	ds_read_b128 a[8:11], v29 offset:128
	ds_read_b128 a[12:15], v29 offset:192
	ds_read_b128 a[16:19], v29 offset:256
	ds_read_b128 a[20:23], v29 offset:320
	ds_read_b128 a[24:27], v29 offset:384
	ds_read_b128 a[28:31], v29 offset:448
	ds_read_b128 a[32:35], v29 offset:512
	ds_read_b128 a[36:39], v29 offset:576
	ds_read_b128 a[40:43], v29 offset:640
	ds_read_b128 a[44:47], v29 offset:704
	ds_read_b128 a[48:51], v29 offset:768
	ds_read_b128 a[52:55], v29 offset:832
	ds_read_b128 a[56:59], v29 offset:896
	ds_read_b128 a[60:63], v29 offset:960
	ds_read_b128 a[64:67], v29 offset:1024
	ds_read_b128 a[68:71], v29 offset:1088
	s_waitcnt lgkmcnt(0)
	.Lr25_label_02ED:
	s_waitcnt vmcnt(20)
	s_barrier
	s_mov_b32 m0, s34
	v_add_u32_e32 v28, 0x1b000, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_cmp_eq_i32 s7, 0
	s_cbranch_scc0 .Lr25_label_0351
	ds_read_b128 a[72:75], v29 offset:20608
	ds_read_b128 a[76:79], v29 offset:20672
	ds_read_b128 a[80:83], v29 offset:20736
	ds_read_b128 a[84:87], v29 offset:20800
	ds_read_b128 a[88:91], v29 offset:20864
	ds_read_b128 a[92:95], v29 offset:20928
	ds_read_b128 a[96:99], v29 offset:20992
	ds_read_b128 a[100:103], v29 offset:21056
	ds_read_b128 a[104:107], v29 offset:21120
	ds_read_b128 a[108:111], v29 offset:21184
	ds_read_b128 a[112:115], v29 offset:21248
	ds_read_b128 a[116:119], v29 offset:21312
	ds_read_b128 a[120:123], v29 offset:21376
	ds_read_b128 a[124:127], v29 offset:21440
	ds_read_b128 a[128:131], v29 offset:21504
	ds_read_b128 a[132:135], v29 offset:21568
	ds_read_b128 a[136:139], v29 offset:21632
	ds_read_b128 a[140:143], v29 offset:21696
	s_waitcnt lgkmcnt(0)
	.Lr25_label_0351:
	s_waitcnt vmcnt(20)
	s_barrier
	s_mov_b32 m0, s35
	v_add_u32_e32 v28, 0x1f800, v30
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	buffer_load_dword v28, s[16:19], 0 offen lds
	buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:512 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:768 lds
	buffer_load_dword v28, s[16:19], 0 offen offset:1024 lds
	s_add_u32 m0, m0, 0x500
	v_add_u32_e32 v28, 0x1200, v28
	s_cmp_eq_i32 s7, 1
	s_cbranch_scc0 .Lr25_label_03B5
	ds_read_b128 a[72:75], v29 offset:41216
	ds_read_b128 a[76:79], v29 offset:41280
	ds_read_b128 a[80:83], v29 offset:41344
	ds_read_b128 a[84:87], v29 offset:41408
	ds_read_b128 a[88:91], v29 offset:41472
	ds_read_b128 a[92:95], v29 offset:41536
	ds_read_b128 a[96:99], v29 offset:41600
	ds_read_b128 a[100:103], v29 offset:41664
	ds_read_b128 a[104:107], v29 offset:41728
	ds_read_b128 a[108:111], v29 offset:41792
	ds_read_b128 a[112:115], v29 offset:41856
	ds_read_b128 a[116:119], v29 offset:41920
	ds_read_b128 a[120:123], v29 offset:41984
	ds_read_b128 a[124:127], v29 offset:42048
	ds_read_b128 a[128:131], v29 offset:42112
	ds_read_b128 a[132:135], v29 offset:42176
	ds_read_b128 a[136:139], v29 offset:42240
	ds_read_b128 a[140:143], v29 offset:42304
	s_waitcnt lgkmcnt(0)
	.Lr25_label_03B5:
	s_waitcnt vmcnt(20)
	s_barrier
	s_cmp_eq_i32 s7, 2
	s_cbranch_scc0 .Lr25_label_03DE
	ds_read_b128 a[72:75], v29
	ds_read_b128 a[76:79], v29 offset:64
	ds_read_b128 a[80:83], v29 offset:128
	ds_read_b128 a[84:87], v29 offset:192
	ds_read_b128 a[88:91], v29 offset:256
	ds_read_b128 a[92:95], v29 offset:320
	ds_read_b128 a[96:99], v29 offset:384
	ds_read_b128 a[100:103], v29 offset:448
	ds_read_b128 a[104:107], v29 offset:512
	ds_read_b128 a[108:111], v29 offset:576
	ds_read_b128 a[112:115], v29 offset:640
	ds_read_b128 a[116:119], v29 offset:704
	ds_read_b128 a[120:123], v29 offset:768
	ds_read_b128 a[124:127], v29 offset:832
	ds_read_b128 a[128:131], v29 offset:896
	ds_read_b128 a[132:135], v29 offset:960
	ds_read_b128 a[136:139], v29 offset:1024
	ds_read_b128 a[140:143], v29 offset:1088
	s_waitcnt lgkmcnt(0)
	.Lr25_label_03DE:
	s_waitcnt vmcnt(0)
	s_barrier
	s_cmp_eq_i32 s7, 3
	s_cbranch_scc0 .Lr25_label_0407
	ds_read_b128 a[72:75], v29 offset:20608
	ds_read_b128 a[76:79], v29 offset:20672
	ds_read_b128 a[80:83], v29 offset:20736
	ds_read_b128 a[84:87], v29 offset:20800
	ds_read_b128 a[88:91], v29 offset:20864
	ds_read_b128 a[92:95], v29 offset:20928
	ds_read_b128 a[96:99], v29 offset:20992
	ds_read_b128 a[100:103], v29 offset:21056
	ds_read_b128 a[104:107], v29 offset:21120
	ds_read_b128 a[108:111], v29 offset:21184
	ds_read_b128 a[112:115], v29 offset:21248
	ds_read_b128 a[116:119], v29 offset:21312
	ds_read_b128 a[120:123], v29 offset:21376
	ds_read_b128 a[124:127], v29 offset:21440
	ds_read_b128 a[128:131], v29 offset:21504
	ds_read_b128 a[132:135], v29 offset:21568
	ds_read_b128 a[136:139], v29 offset:21632
	ds_read_b128 a[140:143], v29 offset:21696
	s_waitcnt lgkmcnt(0)
	.Lr25_label_0407:
	s_waitcnt vmcnt(0)
	s_barrier
	s_mov_b32 s52, 0x7060302
	s_mov_b32 s53, 0x5040100
	s_mov_b32 s6, 0x3fb8aa3b
	v_mov_b32_e32 v21, s6
	v_mov_b32_e32 v20, s64
	v_mul_f32_e32 v20, s6, v20
	v_rcp_f32_e32 v21, v21
	v_mov_b32_e32 v12, 0xff7fffff
	v_mov_b32_e32 v13, 0xff7fffff
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v9, s68
	v_readfirstlane_b32 s5, v20
	v_readfirstlane_b32 s63, v21
	v_and_b32_e32 v2, 15, v0
	v_lshlrev_b32_e32 v2, 2, v2
	s_mul_i32 s56, 0x100, s7
	v_add_u32_e32 v2, s56, v2
	v_lshlrev_b32_e32 v3, 2, v0
	s_mul_i32 s56, 0x100, s7
	v_add_u32_e32 v3, s56, v3
	v_and_b32_e32 v20, 31, v0
	v_lshlrev_b32_e32 v1, 2, v20
	s_mul_i32 s34, s7, 0x1220
	s_add_u32 s34, 0, s34
	s_add_u32 s35, 0x900, s34
	s_add_u32 s36, 0x4880, s34
	s_add_u32 s37, 0x4880, s35
	s_waitcnt vmcnt(0)
	v_mul_u32_u24_dpp v18, v10, v9 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mul_u32_u24_dpp v19, v10, v9 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_add_u32_e32 v18, v18, v1
	v_add_u32_e32 v19, v19, v1
	s_mov_b32 m0, s34
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_u32 m0, 0, s35
	buffer_load_dword v19, s[20:23], 0 offen lds
	s_add_u32 m0, 0x80, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x80, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x100, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x100, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x180, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x180, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x200, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x200, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x280, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x280, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x300, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x300, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x380, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x380, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x400, s34
	buffer_load_dword v18, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x400, s35
	buffer_load_dword v19, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x480, s34
	buffer_load_dword v10, v8, s[24:27], 0 offen
	v_add_u32_e32 v8, s73, v8
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v21, 4, v20
	v_and_b32_e32 v20, 15, v0
	v_and_b32_e32 v22, 3, v20
	v_mul_i32_i24_e32 v22, 0x488, v22
	v_add_u32_e32 v4, v22, v21
	v_lshrrev_b32_e32 v20, 2, v20
	v_and_b32_e32 v21, 1, v20
	v_mul_i32_i24_e32 v21, 32, v21
	v_add_u32_e32 v4, v4, v21
	v_and_b32_e32 v21, 2, v20
	v_mul_i32_i24_e32 v21, 0x120, v21
	v_add_u32_e32 v4, v4, v21
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshrrev_b32_e32 v20, 4, v0
	v_and_b32_e32 v21, 1, v20
	v_mul_i32_i24_e32 v5, 32, v21
	v_and_b32_e32 v21, 2, v20
	v_mul_i32_i24_e32 v21, 0x120, v21
	v_add_u32_e32 v5, v5, v21
	v_and_b32_e32 v20, 15, v0
	v_mul_i32_i24_e32 v21, 2, v20
	v_add_u32_e32 v5, v5, v21
	s_mul_i32 s56, 64, s7
	v_add_u32_e64 v5, v5, s56
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v6, 2, v0
	s_mul_i32 s56, 0x200, s7
	v_add_u32_e64 v6, v6, s56
	v_lshlrev_b32_e32 v6, 2, v6
	v_lshlrev_b32_e32 v7, 4, v0
	v_mul_u32_u24_dpp v18, v11, v9 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mul_u32_u24_dpp v19, v11, v9 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_add_u32_e32 v18, v18, v1
	v_add_u32_e32 v19, v19, v1
	s_mov_b32 m0, s36
	buffer_load_dword v11, v8, s[24:27], 0 offen
	v_add_u32_e32 v8, s73, v8
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_u32 m0, 0, s37
	buffer_load_dword v19, s[20:23], 0 offen lds
	s_add_u32 m0, 0x80, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x80, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x100, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x100, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x180, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x180, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x200, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x200, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x280, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x280, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x300, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x300, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x380, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x380, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x400, s36
	buffer_load_dword v18, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x400, s37
	buffer_load_dword v19, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x480, s36
	s_waitcnt vmcnt(19) lgkmcnt(0)
	s_barrier
	v_mul_u32_u24_dpp v18, v10, v9 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mul_u32_u24_dpp v19, v10, v9 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_add_u32_e32 v18, v18, v1
	v_add_u32_e32 v19, v19, v1
	s_mov_b32 m0, s34
	ds_read_b128 a[144:147], v4
	ds_read_b128 a[148:151], v4 offset:64
	ds_read_b128 a[152:155], v4 offset:256
	ds_read_b128 a[156:159], v4 offset:320
	ds_read_b128 a[160:163], v4 offset:512
	ds_read_b128 a[164:167], v4 offset:576
	ds_read_b128 a[168:171], v4 offset:768
	ds_read_b128 a[172:175], v4 offset:832
	ds_read_b128 a[176:179], v4 offset:1024
	ds_read_b128 a[180:183], v4 offset:1088
	ds_read_b128 a[184:187], v4 offset:1280
	ds_read_b128 a[188:191], v4 offset:1344
	ds_read_b128 a[192:195], v4 offset:1536
	ds_read_b128 a[196:199], v4 offset:1600
	ds_read_b128 a[200:203], v4 offset:1792
	ds_read_b128 a[204:207], v4 offset:1856
	ds_read_b128 a[208:211], v4 offset:2048
	ds_read_b128 a[212:215], v4 offset:2112
	ds_read_b64 v[20:21], v5
	ds_read_b64 v[22:23], v5 offset:4640
	ds_read_b64 v[24:25], v5 offset:9280
	ds_read_b64 v[26:27], v5 offset:13920
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	ds_write_b128 v6, v[28:31] offset:37120
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	ds_write_b128 v6, v[28:31] offset:38144
	ds_read_b64 v[20:21], v5 offset:1024
	ds_read_b64 v[22:23], v5 offset:5664
	ds_read_b64 v[24:25], v5 offset:10304
	ds_read_b64 v[26:27], v5 offset:14944
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	ds_write_b128 v6, v[28:31] offset:45312
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	ds_write_b128 v6, v[28:31] offset:46336
	s_cmp_lt_u32 s71, 1
	s_cbranch_scc1 .Lr25_label_18AC
	s_cmp_lt_i32 s7, 2
	s_cbranch_scc0 .Lr25_label_0F9D
	.Lr25_label_068D:
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0
	ds_read_b128 a[176:179], v4 offset:1024
	ds_read_b128 a[180:183], v4 offset:1088
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]
	buffer_load_dword v10, v8, s[24:27], 0 offen
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]
	ds_read_b128 a[184:187], v4 offset:1280
	ds_read_b128 a[188:191], v4 offset:1344
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]
	ds_read_b128 a[192:195], v4 offset:1536
	ds_read_b128 a[196:199], v4 offset:1600
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]
	ds_read_b128 a[200:203], v4 offset:1792
	ds_read_b128 a[204:207], v4 offset:1856
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]
	ds_read_b128 a[208:211], v4 offset:2048
	ds_read_b128 a[212:215], v4 offset:2112
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_u32 m0, 0, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]
	ds_write_b128 v6, v[28:31] offset:45312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen lds
	s_add_u32 m0, 0x80, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x80, s35
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]
	ds_write_b128 v6, v[28:31] offset:46336
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x100, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x100, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x180, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x180, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x200, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x200, s35
	v_add_u32_e32 v8, s73, v8
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_074C
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	.Lr25_label_074C:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[36:39], a[144:145], a[72:73], 0
	v_mfma_f32_16x16x16_bf16 v[36:39], a[146:147], a[74:75], v[36:39]
	v_max3_f32 v24, v32, v33, v32
	v_max3_f32 v24, v34, v35, v24
	ds_write_b32 v3, v24 offset:53504
	v_mfma_f32_16x16x16_bf16 v[36:39], a[148:149], a[76:77], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[150:151], a[78:79], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x280, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[152:153], a[80:81], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[154:155], a[82:83], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x280, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[156:157], a[84:85], v[36:39]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:53504
	ds_read_b32 v21, v2 offset:53568
	v_mfma_f32_16x16x16_bf16 v[36:39], a[158:159], a[86:87], v[36:39]
	ds_read_b32 v22, v2 offset:53632
	ds_read_b32 v23, v2 offset:53696
	v_mfma_f32_16x16x16_bf16 v[36:39], a[160:161], a[88:89], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[162:163], a[90:91], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x300, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[164:165], a[92:93], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[166:167], a[94:95], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x300, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[168:169], a[96:97], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[170:171], a[98:99], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x380, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[172:173], a[100:101], v[36:39]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mfma_f32_16x16x16_bf16 v[36:39], a[174:175], a[102:103], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[176:177], a[104:105], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[178:179], a[106:107], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x380, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[180:181], a[108:109], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[182:183], a[110:111], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x400, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[184:185], a[112:113], v[36:39]
	ds_read_b128 a[144:147], v7 offset:37120
	ds_read_b128 a[148:151], v7 offset:38144
	v_mfma_f32_16x16x16_bf16 v[36:39], a[186:187], a[114:115], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x400, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[188:189], a[116:117], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[190:191], a[118:119], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[192:193], a[120:121], v[36:39]
	ds_read_b128 a[152:155], v7 offset:39168
	ds_read_b128 a[156:159], v7 offset:40192
	v_mfma_f32_16x16x16_bf16 v[36:39], a[194:195], a[122:123], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x480, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[196:197], a[124:125], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[198:199], a[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[200:201], a[128:129], v[36:39]
	ds_read_b128 a[160:163], v7 offset:41216
	ds_read_b128 a[164:167], v7 offset:42240
	v_mfma_f32_16x16x16_bf16 v[36:39], a[202:203], a[130:131], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[204:205], a[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[206:207], a[134:135], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[208:209], a[136:137], v[36:39]
	ds_read_b128 a[168:171], v7 offset:43264
	ds_read_b128 a[172:175], v7 offset:44288
	v_mfma_f32_16x16x16_bf16 v[36:39], a[210:211], a[138:139], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[212:213], a[140:141], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[214:215], a[142:143], v[36:39]
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v12
	v_max_f32_e32 v20, v24, v12
	v_sub_f32_e32 v16, v12, v20
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]
	v_mov_b32_e32 v12, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v16, s5, v16
	v_exp_f32_e32 v16, v16
	v_fma_f32 v32, v32, s5, -v21
	v_fma_f32 v33, v33, s5, -v21
	v_fma_f32 v34, v34, s5, -v21
	v_fma_f32 v35, v35, s5, -v21
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_mul_f32_e32 v14, v16, v14
	v_mov_b32_e32 v22, v32
	v_add_f32_e32 v22, v33, v22
	v_add_f32_e32 v22, v34, v22
	v_add_f32_e32 v22, v35, v22
	v_add_f32_e32 v14, v22, v14
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v32, v32
	v_add3_u32 v28, v32, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v33, v33
	v_add3_u32 v28, v33, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v32, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v34, v34
	v_add3_u32 v28, v34, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v35, v35
	v_add3_u32 v28, v35, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v33, v21, v20, s52
	s_nop 2
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_0843
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	.Lr25_label_0843:
	s_add_u32 s83, s84, s83
	s_nop 0
	v_mul_u32_u24_dpp v18, v11, v9 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mul_u32_u24_dpp v19, v11, v9 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_add_u32_e32 v18, v18, v1
	v_add_u32_e32 v19, v19, v1
	s_mov_b32 m0, s36
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v23, v16
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a216, v20
	v_accvgpr_write_b32 a217, v21
	v_accvgpr_read_b32 v20, a218
	v_accvgpr_read_b32 v21, a219
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a218, v20
	v_accvgpr_write_b32 a219, v21
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a220, v20
	v_accvgpr_write_b32 a221, v21
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a223
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a223, v21
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a224, v20
	v_accvgpr_write_b32 a225, v21
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a226, v20
	v_accvgpr_write_b32 a227, v21
	v_accvgpr_read_b32 v20, a228
	v_accvgpr_read_b32 v21, a229
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a228, v20
	v_accvgpr_write_b32 a229, v21
	v_accvgpr_read_b32 v20, a230
	v_accvgpr_read_b32 v21, a231
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a230, v20
	v_accvgpr_write_b32 a231, v21
	v_accvgpr_read_b32 v20, a232
	v_accvgpr_read_b32 v21, a233
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a232, v20
	v_accvgpr_write_b32 a233, v21
	v_accvgpr_read_b32 v20, a234
	v_accvgpr_read_b32 v21, a235
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a234, v20
	v_accvgpr_write_b32 a235, v21
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]
	ds_read_b128 a[176:179], v7 offset:45312
	ds_read_b128 a[180:183], v7 offset:46336
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]
	ds_read_b128 a[184:187], v7 offset:47360
	ds_read_b128 a[188:191], v7 offset:48384
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]
	ds_read_b128 a[192:195], v7 offset:49408
	ds_read_b128 a[196:199], v7 offset:50432
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]
	ds_read_b128 a[200:203], v7 offset:51456
	ds_read_b128 a[204:207], v7 offset:52480
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]
	v_max3_f32 v24, v36, v37, v36
	v_max3_f32 v24, v38, v39, v24
	ds_write_b32 v3, v24 offset:54528
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:54528
	ds_read_b32 v21, v2 offset:54592
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]
	ds_read_b32 v22, v2 offset:54656
	ds_read_b32 v23, v2 offset:54720
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]
	v_mfma_f32_16x16x16_bf16 a[216:219], a[198:199], v[32:33], a[216:219]
	v_mfma_f32_16x16x16_bf16 a[220:223], a[200:201], v[32:33], a[220:223]
	v_mfma_f32_16x16x16_bf16 a[224:227], a[202:203], v[32:33], a[224:227]
	v_mfma_f32_16x16x16_bf16 a[228:231], a[204:205], v[32:33], a[228:231]
	v_mfma_f32_16x16x16_bf16 a[232:235], a[206:207], v[32:33], a[232:235]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v13
	v_max_f32_e32 v20, v24, v13
	v_sub_f32_e32 v17, v13, v20
	v_cndmask_b32_e64 v17, v17, 0, s[38:39]
	v_mov_b32_e32 v13, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v17, s5, v17
	v_exp_f32_e32 v17, v17
	v_fma_f32 v36, v36, s5, -v21
	v_fma_f32 v37, v37, s5, -v21
	v_fma_f32 v38, v38, s5, -v21
	v_fma_f32 v39, v39, s5, -v21
	v_exp_f32_e32 v36, v36
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	v_mul_f32_e32 v15, v17, v15
	v_mov_b32_e32 v22, v36
	v_add_f32_e32 v22, v37, v22
	v_add_f32_e32 v22, v38, v22
	v_add_f32_e32 v22, v39, v22
	v_add_f32_e32 v15, v22, v15
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v36, v36
	v_add3_u32 v28, v36, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v37, v37
	v_add3_u32 v28, v37, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v36, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v38, v38
	v_add3_u32 v28, v38, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v39, v39
	v_add3_u32 v28, v39, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v37, v21, v20, s52
	s_nop 2
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]
	v_pk_mul_f32 v[168:169], v[22:23], v[168:169]
	v_pk_mul_f32 v[170:171], v[22:23], v[170:171]
	v_pk_mul_f32 v[172:173], v[22:23], v[172:173]
	v_pk_mul_f32 v[174:175], v[22:23], v[174:175]
	v_pk_mul_f32 v[176:177], v[22:23], v[176:177]
	v_pk_mul_f32 v[178:179], v[22:23], v[178:179]
	v_pk_mul_f32 v[180:181], v[22:23], v[180:181]
	v_pk_mul_f32 v[182:183], v[22:23], v[182:183]
	v_pk_mul_f32 v[184:185], v[22:23], v[184:185]
	v_pk_mul_f32 v[186:187], v[22:23], v[186:187]
	v_pk_mul_f32 v[188:189], v[22:23], v[188:189]
	v_pk_mul_f32 v[190:191], v[22:23], v[190:191]
	v_pk_mul_f32 v[192:193], v[22:23], v[192:193]
	v_pk_mul_f32 v[194:195], v[22:23], v[194:195]
	v_pk_mul_f32 v[196:197], v[22:23], v[196:197]
	v_pk_mul_f32 v[198:199], v[22:23], v[198:199]
	v_pk_mul_f32 v[200:201], v[22:23], v[200:201]
	v_pk_mul_f32 v[202:203], v[22:23], v[202:203]
	v_pk_mul_f32 v[204:205], v[22:23], v[204:205]
	v_pk_mul_f32 v[206:207], v[22:23], v[206:207]
	v_pk_mul_f32 v[208:209], v[22:23], v[208:209]
	v_pk_mul_f32 v[210:211], v[22:23], v[210:211]
	v_pk_mul_f32 v[212:213], v[22:23], v[212:213]
	v_pk_mul_f32 v[214:215], v[22:23], v[214:215]
	v_pk_mul_f32 v[216:217], v[22:23], v[216:217]
	v_pk_mul_f32 v[218:219], v[22:23], v[218:219]
	v_pk_mul_f32 v[220:221], v[22:23], v[220:221]
	v_pk_mul_f32 v[222:223], v[22:23], v[222:223]
	v_pk_mul_f32 v[224:225], v[22:23], v[224:225]
	v_pk_mul_f32 v[226:227], v[22:23], v[226:227]
	v_pk_mul_f32 v[228:229], v[22:23], v[228:229]
	v_pk_mul_f32 v[230:231], v[22:23], v[230:231]
	v_pk_mul_f32 v[232:233], v[22:23], v[232:233]
	v_pk_mul_f32 v[234:235], v[22:23], v[234:235]
	v_pk_mul_f32 v[236:237], v[22:23], v[236:237]
	v_pk_mul_f32 v[238:239], v[22:23], v[238:239]
	v_pk_mul_f32 v[240:241], v[22:23], v[240:241]
	v_pk_mul_f32 v[242:243], v[22:23], v[242:243]
	v_pk_mul_f32 v[244:245], v[22:23], v[244:245]
	v_pk_mul_f32 v[246:247], v[22:23], v[246:247]
	v_pk_mul_f32 v[248:249], v[22:23], v[248:249]
	v_pk_mul_f32 v[250:251], v[22:23], v[250:251]
	v_pk_mul_f32 v[252:253], v[22:23], v[252:253]
	v_pk_mul_f32 v[254:255], v[22:23], v[254:255]
	v_accvgpr_read_b32 v20, a236
	v_accvgpr_read_b32 v21, a237
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a236, v20
	v_accvgpr_write_b32 a237, v21
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a238, v20
	v_accvgpr_write_b32 a239, v21
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a241
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a240, v20
	v_accvgpr_write_b32 a241, v21
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a242, v20
	v_accvgpr_write_b32 a243, v21
	v_accvgpr_read_b32 v20, a244
	v_accvgpr_read_b32 v21, a245
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a244, v20
	v_accvgpr_write_b32 a245, v21
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a246, v20
	v_accvgpr_write_b32 a247, v21
	v_accvgpr_read_b32 v20, a248
	v_accvgpr_read_b32 v21, a249
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a248, v20
	v_accvgpr_write_b32 a249, v21
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a250, v20
	v_accvgpr_write_b32 a251, v21
	v_accvgpr_read_b32 v20, a252
	v_accvgpr_read_b32 v21, a253
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a252, v20
	v_accvgpr_write_b32 a253, v21
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a254, v20
	v_accvgpr_write_b32 a255, v21
	s_waitcnt vmcnt(18) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[148:151], a[144:145], v[36:37], v[148:151]
	v_mfma_f32_16x16x16_bf16 v[152:155], a[146:147], v[36:37], v[152:155]
	ds_read_b64 v[20:21], v5 offset:18560
	ds_read_b64 v[22:23], v5 offset:23200
	ds_read_b64 v[24:25], v5 offset:27840
	ds_read_b64 v[26:27], v5 offset:32480
	v_mfma_f32_16x16x16_bf16 v[156:159], a[148:149], v[36:37], v[156:159]
	v_mfma_f32_16x16x16_bf16 v[160:163], a[150:151], v[36:37], v[160:163]
	v_mfma_f32_16x16x16_bf16 v[164:167], a[152:153], v[36:37], v[164:167]
	v_mfma_f32_16x16x16_bf16 v[168:171], a[154:155], v[36:37], v[168:171]
	ds_read_b128 a[144:147], v4 offset:18560
	ds_read_b128 a[148:151], v4 offset:18624
	v_mfma_f32_16x16x16_bf16 v[172:175], a[156:157], v[36:37], v[172:175]
	v_mfma_f32_16x16x16_bf16 v[176:179], a[158:159], v[36:37], v[176:179]
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[180:183], a[160:161], v[36:37], v[180:183]
	v_mfma_f32_16x16x16_bf16 v[184:187], a[162:163], v[36:37], v[184:187]
	ds_write_b128 v6, v[28:31] offset:37120
	v_mfma_f32_16x16x16_bf16 v[188:191], a[164:165], v[36:37], v[188:191]
	v_mfma_f32_16x16x16_bf16 v[192:195], a[166:167], v[36:37], v[192:195]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[196:199], a[168:169], v[36:37], v[196:199]
	v_mfma_f32_16x16x16_bf16 v[200:203], a[170:171], v[36:37], v[200:203]
	ds_write_b128 v6, v[28:31] offset:38144
	v_mfma_f32_16x16x16_bf16 v[204:207], a[172:173], v[36:37], v[204:207]
	v_mfma_f32_16x16x16_bf16 v[208:211], a[174:175], v[36:37], v[208:211]
	v_mfma_f32_16x16x16_bf16 v[212:215], a[176:177], v[36:37], v[212:215]
	v_mfma_f32_16x16x16_bf16 v[216:219], a[178:179], v[36:37], v[216:219]
	ds_read_b64 v[20:21], v5 offset:19584
	ds_read_b64 v[22:23], v5 offset:24224
	v_mfma_f32_16x16x16_bf16 v[220:223], a[180:181], v[36:37], v[220:223]
	v_mfma_f32_16x16x16_bf16 v[224:227], a[182:183], v[36:37], v[224:227]
	ds_read_b64 v[24:25], v5 offset:28864
	ds_read_b64 v[26:27], v5 offset:33504
	v_mfma_f32_16x16x16_bf16 v[228:231], a[184:185], v[36:37], v[228:231]
	v_mfma_f32_16x16x16_bf16 v[232:235], a[186:187], v[36:37], v[232:235]
	ds_read_b128 a[152:155], v4 offset:18816
	ds_read_b128 a[156:159], v4 offset:18880
	v_mfma_f32_16x16x16_bf16 v[236:239], a[188:189], v[36:37], v[236:239]
	v_mfma_f32_16x16x16_bf16 v[240:243], a[190:191], v[36:37], v[240:243]
	v_mfma_f32_16x16x16_bf16 v[244:247], a[192:193], v[36:37], v[244:247]
	v_mfma_f32_16x16x16_bf16 v[248:251], a[194:195], v[36:37], v[248:251]
	ds_read_b128 a[160:163], v4 offset:19072
	ds_read_b128 a[164:167], v4 offset:19136
	v_mfma_f32_16x16x16_bf16 v[252:255], a[196:197], v[36:37], v[252:255]
	v_mfma_f32_16x16x16_bf16 a[236:239], a[198:199], v[36:37], a[236:239]
	v_mfma_f32_16x16x16_bf16 a[240:243], a[200:201], v[36:37], a[240:243]
	v_mfma_f32_16x16x16_bf16 a[244:247], a[202:203], v[36:37], a[244:247]
	ds_read_b128 a[168:171], v4 offset:19328
	ds_read_b128 a[172:175], v4 offset:19392
	v_mfma_f32_16x16x16_bf16 a[248:251], a[204:205], v[36:37], a[248:251]
	v_mfma_f32_16x16x16_bf16 a[252:255], a[206:207], v[36:37], a[252:255]
	s_nop 0
	s_addk_i32 s70, 0x1
	s_cmp_lt_i32 s70, s71
	s_cbranch_scc0 .Lr25_label_0F9A
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0
	ds_read_b128 a[176:179], v4 offset:19584
	ds_read_b128 a[180:183], v4 offset:19648
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]
	buffer_load_dword v11, v8, s[24:27], 0 offen
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]
	ds_read_b128 a[184:187], v4 offset:19840
	ds_read_b128 a[188:191], v4 offset:19904
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]
	ds_read_b128 a[192:195], v4 offset:20096
	ds_read_b128 a[196:199], v4 offset:20160
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]
	ds_read_b128 a[200:203], v4 offset:20352
	ds_read_b128 a[204:207], v4 offset:20416
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]
	ds_read_b128 a[208:211], v4 offset:20608
	ds_read_b128 a[212:215], v4 offset:20672
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_u32 m0, 0, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]
	ds_write_b128 v6, v[28:31] offset:45312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen lds
	s_add_u32 m0, 0x80, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x80, s37
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]
	ds_write_b128 v6, v[28:31] offset:46336
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x100, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x100, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x180, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x180, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x200, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x200, s37
	v_add_u32_e32 v8, s73, v8
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_0BD2
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	.Lr25_label_0BD2:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[36:39], a[144:145], a[72:73], 0
	v_mfma_f32_16x16x16_bf16 v[36:39], a[146:147], a[74:75], v[36:39]
	v_max3_f32 v24, v32, v33, v32
	v_max3_f32 v24, v34, v35, v24
	ds_write_b32 v3, v24 offset:53504
	v_mfma_f32_16x16x16_bf16 v[36:39], a[148:149], a[76:77], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[150:151], a[78:79], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x280, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[152:153], a[80:81], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[154:155], a[82:83], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x280, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[156:157], a[84:85], v[36:39]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:53504
	ds_read_b32 v21, v2 offset:53568
	v_mfma_f32_16x16x16_bf16 v[36:39], a[158:159], a[86:87], v[36:39]
	ds_read_b32 v22, v2 offset:53632
	ds_read_b32 v23, v2 offset:53696
	v_mfma_f32_16x16x16_bf16 v[36:39], a[160:161], a[88:89], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[162:163], a[90:91], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x300, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[164:165], a[92:93], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[166:167], a[94:95], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x300, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[168:169], a[96:97], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[170:171], a[98:99], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x380, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[172:173], a[100:101], v[36:39]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mfma_f32_16x16x16_bf16 v[36:39], a[174:175], a[102:103], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[176:177], a[104:105], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[178:179], a[106:107], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x380, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[180:181], a[108:109], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[182:183], a[110:111], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x400, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[184:185], a[112:113], v[36:39]
	ds_read_b128 a[144:147], v7 offset:37120
	ds_read_b128 a[148:151], v7 offset:38144
	v_mfma_f32_16x16x16_bf16 v[36:39], a[186:187], a[114:115], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x400, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[188:189], a[116:117], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[190:191], a[118:119], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[192:193], a[120:121], v[36:39]
	ds_read_b128 a[152:155], v7 offset:39168
	ds_read_b128 a[156:159], v7 offset:40192
	v_mfma_f32_16x16x16_bf16 v[36:39], a[194:195], a[122:123], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x480, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[196:197], a[124:125], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[198:199], a[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[200:201], a[128:129], v[36:39]
	ds_read_b128 a[160:163], v7 offset:41216
	ds_read_b128 a[164:167], v7 offset:42240
	v_mfma_f32_16x16x16_bf16 v[36:39], a[202:203], a[130:131], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[204:205], a[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[206:207], a[134:135], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[208:209], a[136:137], v[36:39]
	ds_read_b128 a[168:171], v7 offset:43264
	ds_read_b128 a[172:175], v7 offset:44288
	v_mfma_f32_16x16x16_bf16 v[36:39], a[210:211], a[138:139], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[212:213], a[140:141], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[214:215], a[142:143], v[36:39]
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v12
	v_max_f32_e32 v20, v24, v12
	v_sub_f32_e32 v16, v12, v20
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]
	v_mov_b32_e32 v12, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v16, s5, v16
	v_exp_f32_e32 v16, v16
	v_fma_f32 v32, v32, s5, -v21
	v_fma_f32 v33, v33, s5, -v21
	v_fma_f32 v34, v34, s5, -v21
	v_fma_f32 v35, v35, s5, -v21
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_mul_f32_e32 v14, v16, v14
	v_mov_b32_e32 v22, v32
	v_add_f32_e32 v22, v33, v22
	v_add_f32_e32 v22, v34, v22
	v_add_f32_e32 v22, v35, v22
	v_add_f32_e32 v14, v22, v14
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v32, v32
	v_add3_u32 v28, v32, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v33, v33
	v_add3_u32 v28, v33, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v32, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v34, v34
	v_add3_u32 v28, v34, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v35, v35
	v_add3_u32 v28, v35, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v33, v21, v20, s52
	s_nop 2
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_0CC9
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	.Lr25_label_0CC9:
	s_add_u32 s83, s84, s83
	s_nop 0
	v_mul_u32_u24_dpp v18, v10, v9 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mul_u32_u24_dpp v19, v10, v9 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_add_u32_e32 v18, v18, v1
	v_add_u32_e32 v19, v19, v1
	s_mov_b32 m0, s34
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v23, v16
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a216, v20
	v_accvgpr_write_b32 a217, v21
	v_accvgpr_read_b32 v20, a218
	v_accvgpr_read_b32 v21, a219
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a218, v20
	v_accvgpr_write_b32 a219, v21
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a220, v20
	v_accvgpr_write_b32 a221, v21
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a223
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a223, v21
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a224, v20
	v_accvgpr_write_b32 a225, v21
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a226, v20
	v_accvgpr_write_b32 a227, v21
	v_accvgpr_read_b32 v20, a228
	v_accvgpr_read_b32 v21, a229
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a228, v20
	v_accvgpr_write_b32 a229, v21
	v_accvgpr_read_b32 v20, a230
	v_accvgpr_read_b32 v21, a231
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a230, v20
	v_accvgpr_write_b32 a231, v21
	v_accvgpr_read_b32 v20, a232
	v_accvgpr_read_b32 v21, a233
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a232, v20
	v_accvgpr_write_b32 a233, v21
	v_accvgpr_read_b32 v20, a234
	v_accvgpr_read_b32 v21, a235
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a234, v20
	v_accvgpr_write_b32 a235, v21
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]
	ds_read_b128 a[176:179], v7 offset:45312
	ds_read_b128 a[180:183], v7 offset:46336
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]
	ds_read_b128 a[184:187], v7 offset:47360
	ds_read_b128 a[188:191], v7 offset:48384
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]
	ds_read_b128 a[192:195], v7 offset:49408
	ds_read_b128 a[196:199], v7 offset:50432
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]
	ds_read_b128 a[200:203], v7 offset:51456
	ds_read_b128 a[204:207], v7 offset:52480
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]
	v_max3_f32 v24, v36, v37, v36
	v_max3_f32 v24, v38, v39, v24
	ds_write_b32 v3, v24 offset:54528
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:54528
	ds_read_b32 v21, v2 offset:54592
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]
	ds_read_b32 v22, v2 offset:54656
	ds_read_b32 v23, v2 offset:54720
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]
	v_mfma_f32_16x16x16_bf16 a[216:219], a[198:199], v[32:33], a[216:219]
	v_mfma_f32_16x16x16_bf16 a[220:223], a[200:201], v[32:33], a[220:223]
	v_mfma_f32_16x16x16_bf16 a[224:227], a[202:203], v[32:33], a[224:227]
	v_mfma_f32_16x16x16_bf16 a[228:231], a[204:205], v[32:33], a[228:231]
	v_mfma_f32_16x16x16_bf16 a[232:235], a[206:207], v[32:33], a[232:235]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v13
	v_max_f32_e32 v20, v24, v13
	v_sub_f32_e32 v17, v13, v20
	v_cndmask_b32_e64 v17, v17, 0, s[38:39]
	v_mov_b32_e32 v13, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v17, s5, v17
	v_exp_f32_e32 v17, v17
	v_fma_f32 v36, v36, s5, -v21
	v_fma_f32 v37, v37, s5, -v21
	v_fma_f32 v38, v38, s5, -v21
	v_fma_f32 v39, v39, s5, -v21
	v_exp_f32_e32 v36, v36
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	v_mul_f32_e32 v15, v17, v15
	v_mov_b32_e32 v22, v36
	v_add_f32_e32 v22, v37, v22
	v_add_f32_e32 v22, v38, v22
	v_add_f32_e32 v22, v39, v22
	v_add_f32_e32 v15, v22, v15
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v36, v36
	v_add3_u32 v28, v36, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v37, v37
	v_add3_u32 v28, v37, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v36, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v38, v38
	v_add3_u32 v28, v38, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v39, v39
	v_add3_u32 v28, v39, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v37, v21, v20, s52
	s_nop 2
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]
	v_pk_mul_f32 v[168:169], v[22:23], v[168:169]
	v_pk_mul_f32 v[170:171], v[22:23], v[170:171]
	v_pk_mul_f32 v[172:173], v[22:23], v[172:173]
	v_pk_mul_f32 v[174:175], v[22:23], v[174:175]
	v_pk_mul_f32 v[176:177], v[22:23], v[176:177]
	v_pk_mul_f32 v[178:179], v[22:23], v[178:179]
	v_pk_mul_f32 v[180:181], v[22:23], v[180:181]
	v_pk_mul_f32 v[182:183], v[22:23], v[182:183]
	v_pk_mul_f32 v[184:185], v[22:23], v[184:185]
	v_pk_mul_f32 v[186:187], v[22:23], v[186:187]
	v_pk_mul_f32 v[188:189], v[22:23], v[188:189]
	v_pk_mul_f32 v[190:191], v[22:23], v[190:191]
	v_pk_mul_f32 v[192:193], v[22:23], v[192:193]
	v_pk_mul_f32 v[194:195], v[22:23], v[194:195]
	v_pk_mul_f32 v[196:197], v[22:23], v[196:197]
	v_pk_mul_f32 v[198:199], v[22:23], v[198:199]
	v_pk_mul_f32 v[200:201], v[22:23], v[200:201]
	v_pk_mul_f32 v[202:203], v[22:23], v[202:203]
	v_pk_mul_f32 v[204:205], v[22:23], v[204:205]
	v_pk_mul_f32 v[206:207], v[22:23], v[206:207]
	v_pk_mul_f32 v[208:209], v[22:23], v[208:209]
	v_pk_mul_f32 v[210:211], v[22:23], v[210:211]
	v_pk_mul_f32 v[212:213], v[22:23], v[212:213]
	v_pk_mul_f32 v[214:215], v[22:23], v[214:215]
	v_pk_mul_f32 v[216:217], v[22:23], v[216:217]
	v_pk_mul_f32 v[218:219], v[22:23], v[218:219]
	v_pk_mul_f32 v[220:221], v[22:23], v[220:221]
	v_pk_mul_f32 v[222:223], v[22:23], v[222:223]
	v_pk_mul_f32 v[224:225], v[22:23], v[224:225]
	v_pk_mul_f32 v[226:227], v[22:23], v[226:227]
	v_pk_mul_f32 v[228:229], v[22:23], v[228:229]
	v_pk_mul_f32 v[230:231], v[22:23], v[230:231]
	v_pk_mul_f32 v[232:233], v[22:23], v[232:233]
	v_pk_mul_f32 v[234:235], v[22:23], v[234:235]
	v_pk_mul_f32 v[236:237], v[22:23], v[236:237]
	v_pk_mul_f32 v[238:239], v[22:23], v[238:239]
	v_pk_mul_f32 v[240:241], v[22:23], v[240:241]
	v_pk_mul_f32 v[242:243], v[22:23], v[242:243]
	v_pk_mul_f32 v[244:245], v[22:23], v[244:245]
	v_pk_mul_f32 v[246:247], v[22:23], v[246:247]
	v_pk_mul_f32 v[248:249], v[22:23], v[248:249]
	v_pk_mul_f32 v[250:251], v[22:23], v[250:251]
	v_pk_mul_f32 v[252:253], v[22:23], v[252:253]
	v_pk_mul_f32 v[254:255], v[22:23], v[254:255]
	v_accvgpr_read_b32 v20, a236
	v_accvgpr_read_b32 v21, a237
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a236, v20
	v_accvgpr_write_b32 a237, v21
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a238, v20
	v_accvgpr_write_b32 a239, v21
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a241
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a240, v20
	v_accvgpr_write_b32 a241, v21
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a242, v20
	v_accvgpr_write_b32 a243, v21
	v_accvgpr_read_b32 v20, a244
	v_accvgpr_read_b32 v21, a245
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a244, v20
	v_accvgpr_write_b32 a245, v21
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a246, v20
	v_accvgpr_write_b32 a247, v21
	v_accvgpr_read_b32 v20, a248
	v_accvgpr_read_b32 v21, a249
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a248, v20
	v_accvgpr_write_b32 a249, v21
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a250, v20
	v_accvgpr_write_b32 a251, v21
	v_accvgpr_read_b32 v20, a252
	v_accvgpr_read_b32 v21, a253
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a252, v20
	v_accvgpr_write_b32 a253, v21
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a254, v20
	v_accvgpr_write_b32 a255, v21
	s_waitcnt vmcnt(18) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[148:151], a[144:145], v[36:37], v[148:151]
	v_mfma_f32_16x16x16_bf16 v[152:155], a[146:147], v[36:37], v[152:155]
	ds_read_b64 v[20:21], v5
	ds_read_b64 v[22:23], v5 offset:4640
	ds_read_b64 v[24:25], v5 offset:9280
	ds_read_b64 v[26:27], v5 offset:13920
	v_mfma_f32_16x16x16_bf16 v[156:159], a[148:149], v[36:37], v[156:159]
	v_mfma_f32_16x16x16_bf16 v[160:163], a[150:151], v[36:37], v[160:163]
	v_mfma_f32_16x16x16_bf16 v[164:167], a[152:153], v[36:37], v[164:167]
	v_mfma_f32_16x16x16_bf16 v[168:171], a[154:155], v[36:37], v[168:171]
	ds_read_b128 a[144:147], v4
	ds_read_b128 a[148:151], v4 offset:64
	v_mfma_f32_16x16x16_bf16 v[172:175], a[156:157], v[36:37], v[172:175]
	v_mfma_f32_16x16x16_bf16 v[176:179], a[158:159], v[36:37], v[176:179]
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[180:183], a[160:161], v[36:37], v[180:183]
	v_mfma_f32_16x16x16_bf16 v[184:187], a[162:163], v[36:37], v[184:187]
	ds_write_b128 v6, v[28:31] offset:37120
	v_mfma_f32_16x16x16_bf16 v[188:191], a[164:165], v[36:37], v[188:191]
	v_mfma_f32_16x16x16_bf16 v[192:195], a[166:167], v[36:37], v[192:195]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[196:199], a[168:169], v[36:37], v[196:199]
	v_mfma_f32_16x16x16_bf16 v[200:203], a[170:171], v[36:37], v[200:203]
	ds_write_b128 v6, v[28:31] offset:38144
	v_mfma_f32_16x16x16_bf16 v[204:207], a[172:173], v[36:37], v[204:207]
	v_mfma_f32_16x16x16_bf16 v[208:211], a[174:175], v[36:37], v[208:211]
	v_mfma_f32_16x16x16_bf16 v[212:215], a[176:177], v[36:37], v[212:215]
	v_mfma_f32_16x16x16_bf16 v[216:219], a[178:179], v[36:37], v[216:219]
	ds_read_b64 v[20:21], v5 offset:1024
	ds_read_b64 v[22:23], v5 offset:5664
	v_mfma_f32_16x16x16_bf16 v[220:223], a[180:181], v[36:37], v[220:223]
	v_mfma_f32_16x16x16_bf16 v[224:227], a[182:183], v[36:37], v[224:227]
	ds_read_b64 v[24:25], v5 offset:10304
	ds_read_b64 v[26:27], v5 offset:14944
	v_mfma_f32_16x16x16_bf16 v[228:231], a[184:185], v[36:37], v[228:231]
	v_mfma_f32_16x16x16_bf16 v[232:235], a[186:187], v[36:37], v[232:235]
	ds_read_b128 a[152:155], v4 offset:256
	ds_read_b128 a[156:159], v4 offset:320
	v_mfma_f32_16x16x16_bf16 v[236:239], a[188:189], v[36:37], v[236:239]
	v_mfma_f32_16x16x16_bf16 v[240:243], a[190:191], v[36:37], v[240:243]
	v_mfma_f32_16x16x16_bf16 v[244:247], a[192:193], v[36:37], v[244:247]
	v_mfma_f32_16x16x16_bf16 v[248:251], a[194:195], v[36:37], v[248:251]
	ds_read_b128 a[160:163], v4 offset:512
	ds_read_b128 a[164:167], v4 offset:576
	v_mfma_f32_16x16x16_bf16 v[252:255], a[196:197], v[36:37], v[252:255]
	v_mfma_f32_16x16x16_bf16 a[236:239], a[198:199], v[36:37], a[236:239]
	v_mfma_f32_16x16x16_bf16 a[240:243], a[200:201], v[36:37], a[240:243]
	v_mfma_f32_16x16x16_bf16 a[244:247], a[202:203], v[36:37], a[244:247]
	ds_read_b128 a[168:171], v4 offset:768
	ds_read_b128 a[172:175], v4 offset:832
	v_mfma_f32_16x16x16_bf16 a[248:251], a[204:205], v[36:37], a[248:251]
	v_mfma_f32_16x16x16_bf16 a[252:255], a[206:207], v[36:37], a[252:255]
	s_nop 0
	s_addk_i32 s70, 0x1
	s_cmp_lt_i32 s70, s71
	s_cbranch_scc0 .Lr25_label_0F9A
	s_branch .Lr25_label_068D
	.Lr25_label_0F9A:
	s_nop 0
	s_nop 0
	s_branch .Lr25_label_18AA
	.Lr25_label_0F9D:
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0
	buffer_load_dword v10, v8, s[24:27], 0 offen
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]
	ds_read_b128 a[176:179], v4 offset:1024
	ds_read_b128 a[180:183], v4 offset:1088
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]
	ds_read_b128 a[184:187], v4 offset:1280
	ds_read_b128 a[188:191], v4 offset:1344
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]
	ds_read_b128 a[192:195], v4 offset:1536
	ds_read_b128 a[196:199], v4 offset:1600
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]
	ds_read_b128 a[200:203], v4 offset:1792
	ds_read_b128 a[204:207], v4 offset:1856
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]
	ds_read_b128 a[208:211], v4 offset:2048
	ds_read_b128 a[212:215], v4 offset:2112
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_u32 m0, 0, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen lds
	s_add_u32 m0, 0x80, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]
	ds_write_b128 v6, v[28:31] offset:45312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x80, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x100, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]
	ds_write_b128 v6, v[28:31] offset:46336
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x100, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x180, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x180, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x200, s34
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x200, s35
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]
	v_add_u32_e32 v8, s73, v8
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_105C
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	.Lr25_label_105C:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[36:39], a[144:145], a[72:73], 0
	v_mfma_f32_16x16x16_bf16 v[36:39], a[146:147], a[74:75], v[36:39]
	v_max3_f32 v24, v32, v33, v32
	v_max3_f32 v24, v34, v35, v24
	ds_write_b32 v3, v24 offset:53504
	v_mfma_f32_16x16x16_bf16 v[36:39], a[148:149], a[76:77], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x280, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[150:151], a[78:79], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[152:153], a[80:81], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x280, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[154:155], a[82:83], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[156:157], a[84:85], v[36:39]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:53504
	ds_read_b32 v21, v2 offset:53568
	v_mfma_f32_16x16x16_bf16 v[36:39], a[158:159], a[86:87], v[36:39]
	ds_read_b32 v22, v2 offset:53632
	ds_read_b32 v23, v2 offset:53696
	v_mfma_f32_16x16x16_bf16 v[36:39], a[160:161], a[88:89], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x300, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[162:163], a[90:91], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[164:165], a[92:93], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x300, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[166:167], a[94:95], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[168:169], a[96:97], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x380, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[170:171], a[98:99], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[172:173], a[100:101], v[36:39]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mfma_f32_16x16x16_bf16 v[36:39], a[174:175], a[102:103], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[176:177], a[104:105], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x380, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[178:179], a[106:107], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[180:181], a[108:109], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x400, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[182:183], a[110:111], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[184:185], a[112:113], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x400, s35
	v_mfma_f32_16x16x16_bf16 v[36:39], a[186:187], a[114:115], v[36:39]
	ds_read_b128 a[144:147], v7 offset:37120
	ds_read_b128 a[148:151], v7 offset:38144
	v_mfma_f32_16x16x16_bf16 v[36:39], a[188:189], a[116:117], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[190:191], a[118:119], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[192:193], a[120:121], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x480, s34
	v_mfma_f32_16x16x16_bf16 v[36:39], a[194:195], a[122:123], v[36:39]
	ds_read_b128 a[152:155], v7 offset:39168
	ds_read_b128 a[156:159], v7 offset:40192
	v_mfma_f32_16x16x16_bf16 v[36:39], a[196:197], a[124:125], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[198:199], a[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[200:201], a[128:129], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[202:203], a[130:131], v[36:39]
	ds_read_b128 a[160:163], v7 offset:41216
	ds_read_b128 a[164:167], v7 offset:42240
	v_mfma_f32_16x16x16_bf16 v[36:39], a[204:205], a[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[206:207], a[134:135], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[208:209], a[136:137], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[210:211], a[138:139], v[36:39]
	ds_read_b128 a[168:171], v7 offset:43264
	ds_read_b128 a[172:175], v7 offset:44288
	v_mfma_f32_16x16x16_bf16 v[36:39], a[212:213], a[140:141], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[214:215], a[142:143], v[36:39]
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v12
	v_max_f32_e32 v20, v24, v12
	v_sub_f32_e32 v16, v12, v20
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]
	v_mov_b32_e32 v12, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v16, s5, v16
	v_exp_f32_e32 v16, v16
	v_fma_f32 v32, v32, s5, -v21
	v_fma_f32 v33, v33, s5, -v21
	v_fma_f32 v34, v34, s5, -v21
	v_fma_f32 v35, v35, s5, -v21
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_mul_f32_e32 v14, v16, v14
	v_mov_b32_e32 v22, v32
	v_add_f32_e32 v22, v33, v22
	v_add_f32_e32 v22, v34, v22
	v_add_f32_e32 v22, v35, v22
	v_add_f32_e32 v14, v22, v14
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v32, v32
	v_add3_u32 v28, v32, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v33, v33
	v_add3_u32 v28, v33, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v32, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v34, v34
	v_add3_u32 v28, v34, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v35, v35
	v_add3_u32 v28, v35, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v33, v21, v20, s52
	s_nop 2
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_1153
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	.Lr25_label_1153:
	s_add_u32 s83, s84, s83
	s_nop 0
	v_mul_u32_u24_dpp v18, v11, v9 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mul_u32_u24_dpp v19, v11, v9 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_add_u32_e32 v18, v18, v1
	v_add_u32_e32 v19, v19, v1
	s_mov_b32 m0, s36
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v23, v16
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a216, v20
	v_accvgpr_write_b32 a217, v21
	v_accvgpr_read_b32 v20, a218
	v_accvgpr_read_b32 v21, a219
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a218, v20
	v_accvgpr_write_b32 a219, v21
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a220, v20
	v_accvgpr_write_b32 a221, v21
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a223
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a223, v21
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a224, v20
	v_accvgpr_write_b32 a225, v21
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a226, v20
	v_accvgpr_write_b32 a227, v21
	v_accvgpr_read_b32 v20, a228
	v_accvgpr_read_b32 v21, a229
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a228, v20
	v_accvgpr_write_b32 a229, v21
	v_accvgpr_read_b32 v20, a230
	v_accvgpr_read_b32 v21, a231
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a230, v20
	v_accvgpr_write_b32 a231, v21
	v_accvgpr_read_b32 v20, a232
	v_accvgpr_read_b32 v21, a233
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a232, v20
	v_accvgpr_write_b32 a233, v21
	v_accvgpr_read_b32 v20, a234
	v_accvgpr_read_b32 v21, a235
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a234, v20
	v_accvgpr_write_b32 a235, v21
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]
	ds_read_b128 a[176:179], v7 offset:45312
	ds_read_b128 a[180:183], v7 offset:46336
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]
	ds_read_b128 a[184:187], v7 offset:47360
	ds_read_b128 a[188:191], v7 offset:48384
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]
	ds_read_b128 a[192:195], v7 offset:49408
	ds_read_b128 a[196:199], v7 offset:50432
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]
	ds_read_b128 a[200:203], v7 offset:51456
	ds_read_b128 a[204:207], v7 offset:52480
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]
	v_max3_f32 v24, v36, v37, v36
	v_max3_f32 v24, v38, v39, v24
	ds_write_b32 v3, v24 offset:54528
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:54528
	ds_read_b32 v21, v2 offset:54592
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]
	ds_read_b32 v22, v2 offset:54656
	ds_read_b32 v23, v2 offset:54720
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]
	v_mfma_f32_16x16x16_bf16 a[216:219], a[198:199], v[32:33], a[216:219]
	v_mfma_f32_16x16x16_bf16 a[220:223], a[200:201], v[32:33], a[220:223]
	v_mfma_f32_16x16x16_bf16 a[224:227], a[202:203], v[32:33], a[224:227]
	v_mfma_f32_16x16x16_bf16 a[228:231], a[204:205], v[32:33], a[228:231]
	v_mfma_f32_16x16x16_bf16 a[232:235], a[206:207], v[32:33], a[232:235]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v13
	v_max_f32_e32 v20, v24, v13
	v_sub_f32_e32 v17, v13, v20
	v_cndmask_b32_e64 v17, v17, 0, s[38:39]
	v_mov_b32_e32 v13, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v17, s5, v17
	v_exp_f32_e32 v17, v17
	v_fma_f32 v36, v36, s5, -v21
	v_fma_f32 v37, v37, s5, -v21
	v_fma_f32 v38, v38, s5, -v21
	v_fma_f32 v39, v39, s5, -v21
	v_exp_f32_e32 v36, v36
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	v_mul_f32_e32 v15, v17, v15
	v_mov_b32_e32 v22, v36
	v_add_f32_e32 v22, v37, v22
	v_add_f32_e32 v22, v38, v22
	v_add_f32_e32 v22, v39, v22
	v_add_f32_e32 v15, v22, v15
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v36, v36
	v_add3_u32 v28, v36, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v37, v37
	v_add3_u32 v28, v37, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v36, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v38, v38
	v_add3_u32 v28, v38, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v39, v39
	v_add3_u32 v28, v39, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v37, v21, v20, s52
	s_nop 2
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]
	v_pk_mul_f32 v[168:169], v[22:23], v[168:169]
	v_pk_mul_f32 v[170:171], v[22:23], v[170:171]
	v_pk_mul_f32 v[172:173], v[22:23], v[172:173]
	v_pk_mul_f32 v[174:175], v[22:23], v[174:175]
	v_pk_mul_f32 v[176:177], v[22:23], v[176:177]
	v_pk_mul_f32 v[178:179], v[22:23], v[178:179]
	v_pk_mul_f32 v[180:181], v[22:23], v[180:181]
	v_pk_mul_f32 v[182:183], v[22:23], v[182:183]
	v_pk_mul_f32 v[184:185], v[22:23], v[184:185]
	v_pk_mul_f32 v[186:187], v[22:23], v[186:187]
	v_pk_mul_f32 v[188:189], v[22:23], v[188:189]
	v_pk_mul_f32 v[190:191], v[22:23], v[190:191]
	v_pk_mul_f32 v[192:193], v[22:23], v[192:193]
	v_pk_mul_f32 v[194:195], v[22:23], v[194:195]
	v_pk_mul_f32 v[196:197], v[22:23], v[196:197]
	v_pk_mul_f32 v[198:199], v[22:23], v[198:199]
	v_pk_mul_f32 v[200:201], v[22:23], v[200:201]
	v_pk_mul_f32 v[202:203], v[22:23], v[202:203]
	v_pk_mul_f32 v[204:205], v[22:23], v[204:205]
	v_pk_mul_f32 v[206:207], v[22:23], v[206:207]
	v_pk_mul_f32 v[208:209], v[22:23], v[208:209]
	v_pk_mul_f32 v[210:211], v[22:23], v[210:211]
	v_pk_mul_f32 v[212:213], v[22:23], v[212:213]
	v_pk_mul_f32 v[214:215], v[22:23], v[214:215]
	v_pk_mul_f32 v[216:217], v[22:23], v[216:217]
	v_pk_mul_f32 v[218:219], v[22:23], v[218:219]
	v_pk_mul_f32 v[220:221], v[22:23], v[220:221]
	v_pk_mul_f32 v[222:223], v[22:23], v[222:223]
	v_pk_mul_f32 v[224:225], v[22:23], v[224:225]
	v_pk_mul_f32 v[226:227], v[22:23], v[226:227]
	v_pk_mul_f32 v[228:229], v[22:23], v[228:229]
	v_pk_mul_f32 v[230:231], v[22:23], v[230:231]
	v_pk_mul_f32 v[232:233], v[22:23], v[232:233]
	v_pk_mul_f32 v[234:235], v[22:23], v[234:235]
	v_pk_mul_f32 v[236:237], v[22:23], v[236:237]
	v_pk_mul_f32 v[238:239], v[22:23], v[238:239]
	v_pk_mul_f32 v[240:241], v[22:23], v[240:241]
	v_pk_mul_f32 v[242:243], v[22:23], v[242:243]
	v_pk_mul_f32 v[244:245], v[22:23], v[244:245]
	v_pk_mul_f32 v[246:247], v[22:23], v[246:247]
	v_pk_mul_f32 v[248:249], v[22:23], v[248:249]
	v_pk_mul_f32 v[250:251], v[22:23], v[250:251]
	v_pk_mul_f32 v[252:253], v[22:23], v[252:253]
	v_pk_mul_f32 v[254:255], v[22:23], v[254:255]
	v_accvgpr_read_b32 v20, a236
	v_accvgpr_read_b32 v21, a237
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a236, v20
	v_accvgpr_write_b32 a237, v21
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a238, v20
	v_accvgpr_write_b32 a239, v21
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a241
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a240, v20
	v_accvgpr_write_b32 a241, v21
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a242, v20
	v_accvgpr_write_b32 a243, v21
	v_accvgpr_read_b32 v20, a244
	v_accvgpr_read_b32 v21, a245
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a244, v20
	v_accvgpr_write_b32 a245, v21
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a246, v20
	v_accvgpr_write_b32 a247, v21
	v_accvgpr_read_b32 v20, a248
	v_accvgpr_read_b32 v21, a249
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a248, v20
	v_accvgpr_write_b32 a249, v21
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a250, v20
	v_accvgpr_write_b32 a251, v21
	v_accvgpr_read_b32 v20, a252
	v_accvgpr_read_b32 v21, a253
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a252, v20
	v_accvgpr_write_b32 a253, v21
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a254, v20
	v_accvgpr_write_b32 a255, v21
	s_waitcnt vmcnt(18) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[148:151], a[144:145], v[36:37], v[148:151]
	ds_read_b64 v[20:21], v5 offset:18560
	ds_read_b64 v[22:23], v5 offset:23200
	ds_read_b64 v[24:25], v5 offset:27840
	ds_read_b64 v[26:27], v5 offset:32480
	v_mfma_f32_16x16x16_bf16 v[152:155], a[146:147], v[36:37], v[152:155]
	v_mfma_f32_16x16x16_bf16 v[156:159], a[148:149], v[36:37], v[156:159]
	v_mfma_f32_16x16x16_bf16 v[160:163], a[150:151], v[36:37], v[160:163]
	v_mfma_f32_16x16x16_bf16 v[164:167], a[152:153], v[36:37], v[164:167]
	ds_read_b128 a[144:147], v4 offset:18560
	ds_read_b128 a[148:151], v4 offset:18624
	v_mfma_f32_16x16x16_bf16 v[168:171], a[154:155], v[36:37], v[168:171]
	v_mfma_f32_16x16x16_bf16 v[172:175], a[156:157], v[36:37], v[172:175]
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[176:179], a[158:159], v[36:37], v[176:179]
	v_mfma_f32_16x16x16_bf16 v[180:183], a[160:161], v[36:37], v[180:183]
	ds_write_b128 v6, v[28:31] offset:37120
	v_mfma_f32_16x16x16_bf16 v[184:187], a[162:163], v[36:37], v[184:187]
	v_mfma_f32_16x16x16_bf16 v[188:191], a[164:165], v[36:37], v[188:191]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[192:195], a[166:167], v[36:37], v[192:195]
	v_mfma_f32_16x16x16_bf16 v[196:199], a[168:169], v[36:37], v[196:199]
	ds_write_b128 v6, v[28:31] offset:38144
	v_mfma_f32_16x16x16_bf16 v[200:203], a[170:171], v[36:37], v[200:203]
	v_mfma_f32_16x16x16_bf16 v[204:207], a[172:173], v[36:37], v[204:207]
	v_mfma_f32_16x16x16_bf16 v[208:211], a[174:175], v[36:37], v[208:211]
	v_mfma_f32_16x16x16_bf16 v[212:215], a[176:177], v[36:37], v[212:215]
	ds_read_b64 v[20:21], v5 offset:19584
	ds_read_b64 v[22:23], v5 offset:24224
	v_mfma_f32_16x16x16_bf16 v[216:219], a[178:179], v[36:37], v[216:219]
	v_mfma_f32_16x16x16_bf16 v[220:223], a[180:181], v[36:37], v[220:223]
	ds_read_b64 v[24:25], v5 offset:28864
	ds_read_b64 v[26:27], v5 offset:33504
	v_mfma_f32_16x16x16_bf16 v[224:227], a[182:183], v[36:37], v[224:227]
	v_mfma_f32_16x16x16_bf16 v[228:231], a[184:185], v[36:37], v[228:231]
	ds_read_b128 a[152:155], v4 offset:18816
	ds_read_b128 a[156:159], v4 offset:18880
	v_mfma_f32_16x16x16_bf16 v[232:235], a[186:187], v[36:37], v[232:235]
	v_mfma_f32_16x16x16_bf16 v[236:239], a[188:189], v[36:37], v[236:239]
	v_mfma_f32_16x16x16_bf16 v[240:243], a[190:191], v[36:37], v[240:243]
	v_mfma_f32_16x16x16_bf16 v[244:247], a[192:193], v[36:37], v[244:247]
	ds_read_b128 a[160:163], v4 offset:19072
	ds_read_b128 a[164:167], v4 offset:19136
	v_mfma_f32_16x16x16_bf16 v[248:251], a[194:195], v[36:37], v[248:251]
	v_mfma_f32_16x16x16_bf16 v[252:255], a[196:197], v[36:37], v[252:255]
	v_mfma_f32_16x16x16_bf16 a[236:239], a[198:199], v[36:37], a[236:239]
	v_mfma_f32_16x16x16_bf16 a[240:243], a[200:201], v[36:37], a[240:243]
	ds_read_b128 a[168:171], v4 offset:19328
	ds_read_b128 a[172:175], v4 offset:19392
	v_mfma_f32_16x16x16_bf16 a[244:247], a[202:203], v[36:37], a[244:247]
	v_mfma_f32_16x16x16_bf16 a[248:251], a[204:205], v[36:37], a[248:251]
	v_mfma_f32_16x16x16_bf16 a[252:255], a[206:207], v[36:37], a[252:255]
	s_nop 0
	s_addk_i32 s70, 0x1
	s_cmp_lt_i32 s70, s71
	s_cbranch_scc0 .Lr25_label_0F9A
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0
	buffer_load_dword v11, v8, s[24:27], 0 offen
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]
	ds_read_b128 a[176:179], v4 offset:19584
	ds_read_b128 a[180:183], v4 offset:19648
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]
	ds_read_b128 a[184:187], v4 offset:19840
	ds_read_b128 a[188:191], v4 offset:19904
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]
	ds_read_b128 a[192:195], v4 offset:20096
	ds_read_b128 a[196:199], v4 offset:20160
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]
	ds_read_b128 a[200:203], v4 offset:20352
	ds_read_b128 a[204:207], v4 offset:20416
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]
	ds_read_b128 a[208:211], v4 offset:20608
	ds_read_b128 a[212:215], v4 offset:20672
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_u32 m0, 0, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen lds
	s_add_u32 m0, 0x80, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]
	ds_write_b128 v6, v[28:31] offset:45312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x80, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:128 lds
	s_add_u32 m0, 0x100, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]
	ds_write_b128 v6, v[28:31] offset:46336
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x100, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:256 lds
	s_add_u32 m0, 0x180, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x180, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]
	buffer_load_dword v19, s[20:23], 0 offen offset:384 lds
	s_add_u32 m0, 0x200, s36
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]
	buffer_load_dword v18, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x200, s37
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]
	v_add_u32_e32 v8, s73, v8
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_14E2
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	.Lr25_label_14E2:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[36:39], a[144:145], a[72:73], 0
	v_mfma_f32_16x16x16_bf16 v[36:39], a[146:147], a[74:75], v[36:39]
	v_max3_f32 v24, v32, v33, v32
	v_max3_f32 v24, v34, v35, v24
	ds_write_b32 v3, v24 offset:53504
	v_mfma_f32_16x16x16_bf16 v[36:39], a[148:149], a[76:77], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:512 lds
	s_add_u32 m0, 0x280, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[150:151], a[78:79], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[152:153], a[80:81], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x280, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[154:155], a[82:83], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[156:157], a[84:85], v[36:39]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:53504
	ds_read_b32 v21, v2 offset:53568
	v_mfma_f32_16x16x16_bf16 v[36:39], a[158:159], a[86:87], v[36:39]
	ds_read_b32 v22, v2 offset:53632
	ds_read_b32 v23, v2 offset:53696
	v_mfma_f32_16x16x16_bf16 v[36:39], a[160:161], a[88:89], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:640 lds
	s_add_u32 m0, 0x300, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[162:163], a[90:91], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[164:165], a[92:93], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x300, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[166:167], a[94:95], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[168:169], a[96:97], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:768 lds
	s_add_u32 m0, 0x380, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[170:171], a[98:99], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[172:173], a[100:101], v[36:39]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mfma_f32_16x16x16_bf16 v[36:39], a[174:175], a[102:103], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[176:177], a[104:105], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x380, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[178:179], a[106:107], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[180:181], a[108:109], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:896 lds
	s_add_u32 m0, 0x400, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[182:183], a[110:111], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[184:185], a[112:113], v[36:39]
	buffer_load_dword v18, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x400, s37
	v_mfma_f32_16x16x16_bf16 v[36:39], a[186:187], a[114:115], v[36:39]
	ds_read_b128 a[144:147], v7 offset:37120
	ds_read_b128 a[148:151], v7 offset:38144
	v_mfma_f32_16x16x16_bf16 v[36:39], a[188:189], a[116:117], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[190:191], a[118:119], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[192:193], a[120:121], v[36:39]
	buffer_load_dword v19, s[20:23], 0 offen offset:1024 lds
	s_add_u32 m0, 0x480, s36
	v_mfma_f32_16x16x16_bf16 v[36:39], a[194:195], a[122:123], v[36:39]
	ds_read_b128 a[152:155], v7 offset:39168
	ds_read_b128 a[156:159], v7 offset:40192
	v_mfma_f32_16x16x16_bf16 v[36:39], a[196:197], a[124:125], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[198:199], a[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[200:201], a[128:129], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[202:203], a[130:131], v[36:39]
	ds_read_b128 a[160:163], v7 offset:41216
	ds_read_b128 a[164:167], v7 offset:42240
	v_mfma_f32_16x16x16_bf16 v[36:39], a[204:205], a[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[206:207], a[134:135], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[208:209], a[136:137], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[210:211], a[138:139], v[36:39]
	ds_read_b128 a[168:171], v7 offset:43264
	ds_read_b128 a[172:175], v7 offset:44288
	v_mfma_f32_16x16x16_bf16 v[36:39], a[212:213], a[140:141], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[214:215], a[142:143], v[36:39]
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v12
	v_max_f32_e32 v20, v24, v12
	v_sub_f32_e32 v16, v12, v20
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]
	v_mov_b32_e32 v12, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v16, s5, v16
	v_exp_f32_e32 v16, v16
	v_fma_f32 v32, v32, s5, -v21
	v_fma_f32 v33, v33, s5, -v21
	v_fma_f32 v34, v34, s5, -v21
	v_fma_f32 v35, v35, s5, -v21
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_mul_f32_e32 v14, v16, v14
	v_mov_b32_e32 v22, v32
	v_add_f32_e32 v22, v33, v22
	v_add_f32_e32 v22, v34, v22
	v_add_f32_e32 v22, v35, v22
	v_add_f32_e32 v14, v22, v14
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v32, v32
	v_add3_u32 v28, v32, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v33, v33
	v_add3_u32 v28, v33, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v32, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v34, v34
	v_add3_u32 v28, v34, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v35, v35
	v_add3_u32 v28, v35, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v33, v21, v20, s52
	s_nop 2
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_15D9
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	.Lr25_label_15D9:
	s_add_u32 s83, s84, s83
	s_nop 0
	v_mul_u32_u24_dpp v18, v10, v9 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mul_u32_u24_dpp v19, v10, v9 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_add_u32_e32 v18, v18, v1
	v_add_u32_e32 v19, v19, v1
	s_mov_b32 m0, s34
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v23, v16
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a216, v20
	v_accvgpr_write_b32 a217, v21
	v_accvgpr_read_b32 v20, a218
	v_accvgpr_read_b32 v21, a219
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a218, v20
	v_accvgpr_write_b32 a219, v21
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a220, v20
	v_accvgpr_write_b32 a221, v21
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a223
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a223, v21
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a224, v20
	v_accvgpr_write_b32 a225, v21
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a226, v20
	v_accvgpr_write_b32 a227, v21
	v_accvgpr_read_b32 v20, a228
	v_accvgpr_read_b32 v21, a229
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a228, v20
	v_accvgpr_write_b32 a229, v21
	v_accvgpr_read_b32 v20, a230
	v_accvgpr_read_b32 v21, a231
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a230, v20
	v_accvgpr_write_b32 a231, v21
	v_accvgpr_read_b32 v20, a232
	v_accvgpr_read_b32 v21, a233
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a232, v20
	v_accvgpr_write_b32 a233, v21
	v_accvgpr_read_b32 v20, a234
	v_accvgpr_read_b32 v21, a235
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a234, v20
	v_accvgpr_write_b32 a235, v21
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]
	ds_read_b128 a[176:179], v7 offset:45312
	ds_read_b128 a[180:183], v7 offset:46336
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]
	ds_read_b128 a[184:187], v7 offset:47360
	ds_read_b128 a[188:191], v7 offset:48384
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]
	ds_read_b128 a[192:195], v7 offset:49408
	ds_read_b128 a[196:199], v7 offset:50432
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]
	ds_read_b128 a[200:203], v7 offset:51456
	ds_read_b128 a[204:207], v7 offset:52480
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]
	v_max3_f32 v24, v36, v37, v36
	v_max3_f32 v24, v38, v39, v24
	ds_write_b32 v3, v24 offset:54528
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:54528
	ds_read_b32 v21, v2 offset:54592
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]
	ds_read_b32 v22, v2 offset:54656
	ds_read_b32 v23, v2 offset:54720
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]
	v_mfma_f32_16x16x16_bf16 a[216:219], a[198:199], v[32:33], a[216:219]
	v_mfma_f32_16x16x16_bf16 a[220:223], a[200:201], v[32:33], a[220:223]
	v_mfma_f32_16x16x16_bf16 a[224:227], a[202:203], v[32:33], a[224:227]
	v_mfma_f32_16x16x16_bf16 a[228:231], a[204:205], v[32:33], a[228:231]
	v_mfma_f32_16x16x16_bf16 a[232:235], a[206:207], v[32:33], a[232:235]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v13
	v_max_f32_e32 v20, v24, v13
	v_sub_f32_e32 v17, v13, v20
	v_cndmask_b32_e64 v17, v17, 0, s[38:39]
	v_mov_b32_e32 v13, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v17, s5, v17
	v_exp_f32_e32 v17, v17
	v_fma_f32 v36, v36, s5, -v21
	v_fma_f32 v37, v37, s5, -v21
	v_fma_f32 v38, v38, s5, -v21
	v_fma_f32 v39, v39, s5, -v21
	v_exp_f32_e32 v36, v36
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	v_mul_f32_e32 v15, v17, v15
	v_mov_b32_e32 v22, v36
	v_add_f32_e32 v22, v37, v22
	v_add_f32_e32 v22, v38, v22
	v_add_f32_e32 v22, v39, v22
	v_add_f32_e32 v15, v22, v15
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v36, v36
	v_add3_u32 v28, v36, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v37, v37
	v_add3_u32 v28, v37, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v36, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v38, v38
	v_add3_u32 v28, v38, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v39, v39
	v_add3_u32 v28, v39, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v37, v21, v20, s52
	s_nop 2
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]
	v_pk_mul_f32 v[168:169], v[22:23], v[168:169]
	v_pk_mul_f32 v[170:171], v[22:23], v[170:171]
	v_pk_mul_f32 v[172:173], v[22:23], v[172:173]
	v_pk_mul_f32 v[174:175], v[22:23], v[174:175]
	v_pk_mul_f32 v[176:177], v[22:23], v[176:177]
	v_pk_mul_f32 v[178:179], v[22:23], v[178:179]
	v_pk_mul_f32 v[180:181], v[22:23], v[180:181]
	v_pk_mul_f32 v[182:183], v[22:23], v[182:183]
	v_pk_mul_f32 v[184:185], v[22:23], v[184:185]
	v_pk_mul_f32 v[186:187], v[22:23], v[186:187]
	v_pk_mul_f32 v[188:189], v[22:23], v[188:189]
	v_pk_mul_f32 v[190:191], v[22:23], v[190:191]
	v_pk_mul_f32 v[192:193], v[22:23], v[192:193]
	v_pk_mul_f32 v[194:195], v[22:23], v[194:195]
	v_pk_mul_f32 v[196:197], v[22:23], v[196:197]
	v_pk_mul_f32 v[198:199], v[22:23], v[198:199]
	v_pk_mul_f32 v[200:201], v[22:23], v[200:201]
	v_pk_mul_f32 v[202:203], v[22:23], v[202:203]
	v_pk_mul_f32 v[204:205], v[22:23], v[204:205]
	v_pk_mul_f32 v[206:207], v[22:23], v[206:207]
	v_pk_mul_f32 v[208:209], v[22:23], v[208:209]
	v_pk_mul_f32 v[210:211], v[22:23], v[210:211]
	v_pk_mul_f32 v[212:213], v[22:23], v[212:213]
	v_pk_mul_f32 v[214:215], v[22:23], v[214:215]
	v_pk_mul_f32 v[216:217], v[22:23], v[216:217]
	v_pk_mul_f32 v[218:219], v[22:23], v[218:219]
	v_pk_mul_f32 v[220:221], v[22:23], v[220:221]
	v_pk_mul_f32 v[222:223], v[22:23], v[222:223]
	v_pk_mul_f32 v[224:225], v[22:23], v[224:225]
	v_pk_mul_f32 v[226:227], v[22:23], v[226:227]
	v_pk_mul_f32 v[228:229], v[22:23], v[228:229]
	v_pk_mul_f32 v[230:231], v[22:23], v[230:231]
	v_pk_mul_f32 v[232:233], v[22:23], v[232:233]
	v_pk_mul_f32 v[234:235], v[22:23], v[234:235]
	v_pk_mul_f32 v[236:237], v[22:23], v[236:237]
	v_pk_mul_f32 v[238:239], v[22:23], v[238:239]
	v_pk_mul_f32 v[240:241], v[22:23], v[240:241]
	v_pk_mul_f32 v[242:243], v[22:23], v[242:243]
	v_pk_mul_f32 v[244:245], v[22:23], v[244:245]
	v_pk_mul_f32 v[246:247], v[22:23], v[246:247]
	v_pk_mul_f32 v[248:249], v[22:23], v[248:249]
	v_pk_mul_f32 v[250:251], v[22:23], v[250:251]
	v_pk_mul_f32 v[252:253], v[22:23], v[252:253]
	v_pk_mul_f32 v[254:255], v[22:23], v[254:255]
	v_accvgpr_read_b32 v20, a236
	v_accvgpr_read_b32 v21, a237
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a236, v20
	v_accvgpr_write_b32 a237, v21
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a238, v20
	v_accvgpr_write_b32 a239, v21
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a241
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a240, v20
	v_accvgpr_write_b32 a241, v21
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a242, v20
	v_accvgpr_write_b32 a243, v21
	v_accvgpr_read_b32 v20, a244
	v_accvgpr_read_b32 v21, a245
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a244, v20
	v_accvgpr_write_b32 a245, v21
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a246, v20
	v_accvgpr_write_b32 a247, v21
	v_accvgpr_read_b32 v20, a248
	v_accvgpr_read_b32 v21, a249
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a248, v20
	v_accvgpr_write_b32 a249, v21
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a250, v20
	v_accvgpr_write_b32 a251, v21
	v_accvgpr_read_b32 v20, a252
	v_accvgpr_read_b32 v21, a253
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a252, v20
	v_accvgpr_write_b32 a253, v21
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a254, v20
	v_accvgpr_write_b32 a255, v21
	s_waitcnt vmcnt(18) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[148:151], a[144:145], v[36:37], v[148:151]
	ds_read_b64 v[20:21], v5
	ds_read_b64 v[22:23], v5 offset:4640
	ds_read_b64 v[24:25], v5 offset:9280
	ds_read_b64 v[26:27], v5 offset:13920
	v_mfma_f32_16x16x16_bf16 v[152:155], a[146:147], v[36:37], v[152:155]
	v_mfma_f32_16x16x16_bf16 v[156:159], a[148:149], v[36:37], v[156:159]
	v_mfma_f32_16x16x16_bf16 v[160:163], a[150:151], v[36:37], v[160:163]
	v_mfma_f32_16x16x16_bf16 v[164:167], a[152:153], v[36:37], v[164:167]
	ds_read_b128 a[144:147], v4
	ds_read_b128 a[148:151], v4 offset:64
	v_mfma_f32_16x16x16_bf16 v[168:171], a[154:155], v[36:37], v[168:171]
	v_mfma_f32_16x16x16_bf16 v[172:175], a[156:157], v[36:37], v[172:175]
	s_waitcnt lgkmcnt(2)
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[176:179], a[158:159], v[36:37], v[176:179]
	v_mfma_f32_16x16x16_bf16 v[180:183], a[160:161], v[36:37], v[180:183]
	ds_write_b128 v6, v[28:31] offset:37120
	v_mfma_f32_16x16x16_bf16 v[184:187], a[162:163], v[36:37], v[184:187]
	v_mfma_f32_16x16x16_bf16 v[188:191], a[164:165], v[36:37], v[188:191]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[192:195], a[166:167], v[36:37], v[192:195]
	v_mfma_f32_16x16x16_bf16 v[196:199], a[168:169], v[36:37], v[196:199]
	ds_write_b128 v6, v[28:31] offset:38144
	v_mfma_f32_16x16x16_bf16 v[200:203], a[170:171], v[36:37], v[200:203]
	v_mfma_f32_16x16x16_bf16 v[204:207], a[172:173], v[36:37], v[204:207]
	v_mfma_f32_16x16x16_bf16 v[208:211], a[174:175], v[36:37], v[208:211]
	v_mfma_f32_16x16x16_bf16 v[212:215], a[176:177], v[36:37], v[212:215]
	ds_read_b64 v[20:21], v5 offset:1024
	ds_read_b64 v[22:23], v5 offset:5664
	v_mfma_f32_16x16x16_bf16 v[216:219], a[178:179], v[36:37], v[216:219]
	v_mfma_f32_16x16x16_bf16 v[220:223], a[180:181], v[36:37], v[220:223]
	ds_read_b64 v[24:25], v5 offset:10304
	ds_read_b64 v[26:27], v5 offset:14944
	v_mfma_f32_16x16x16_bf16 v[224:227], a[182:183], v[36:37], v[224:227]
	v_mfma_f32_16x16x16_bf16 v[228:231], a[184:185], v[36:37], v[228:231]
	ds_read_b128 a[152:155], v4 offset:256
	ds_read_b128 a[156:159], v4 offset:320
	v_mfma_f32_16x16x16_bf16 v[232:235], a[186:187], v[36:37], v[232:235]
	v_mfma_f32_16x16x16_bf16 v[236:239], a[188:189], v[36:37], v[236:239]
	v_mfma_f32_16x16x16_bf16 v[240:243], a[190:191], v[36:37], v[240:243]
	v_mfma_f32_16x16x16_bf16 v[244:247], a[192:193], v[36:37], v[244:247]
	ds_read_b128 a[160:163], v4 offset:512
	ds_read_b128 a[164:167], v4 offset:576
	v_mfma_f32_16x16x16_bf16 v[248:251], a[194:195], v[36:37], v[248:251]
	v_mfma_f32_16x16x16_bf16 v[252:255], a[196:197], v[36:37], v[252:255]
	v_mfma_f32_16x16x16_bf16 a[236:239], a[198:199], v[36:37], a[236:239]
	v_mfma_f32_16x16x16_bf16 a[240:243], a[200:201], v[36:37], a[240:243]
	ds_read_b128 a[168:171], v4 offset:768
	ds_read_b128 a[172:175], v4 offset:832
	v_mfma_f32_16x16x16_bf16 a[244:247], a[202:203], v[36:37], a[244:247]
	v_mfma_f32_16x16x16_bf16 a[248:251], a[204:205], v[36:37], a[248:251]
	v_mfma_f32_16x16x16_bf16 a[252:255], a[206:207], v[36:37], a[252:255]
	s_nop 0
	s_addk_i32 s70, 0x1
	s_cmp_lt_i32 s70, s71
	s_cbranch_scc0 .Lr25_label_0F9A
	s_branch .Lr25_label_0F9D
	.Lr25_label_18AA:
	s_cmp_eq_i32 s48, 0
	s_cbranch_scc1 .Lr25_label_2133
	.Lr25_label_18AC:
	s_and_b32 s56, s71, 1
	s_cmp_eq_i32 s56, 1
	s_cbranch_scc1 .Lr25_label_1CF1
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0
	ds_read_b128 a[176:179], v4 offset:1024
	ds_read_b128 a[180:183], v4 offset:1088
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]
	ds_read_b128 a[184:187], v4 offset:1280
	ds_read_b128 a[188:191], v4 offset:1344
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]
	ds_read_b128 a[192:195], v4 offset:1536
	ds_read_b128 a[196:199], v4 offset:1600
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]
	ds_read_b128 a[200:203], v4 offset:1792
	ds_read_b128 a[204:207], v4 offset:1856
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]
	ds_read_b128 a[208:211], v4 offset:2048
	ds_read_b128 a[212:215], v4 offset:2112
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]
	ds_write_b128 v6, v[28:31] offset:45312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]
	ds_write_b128 v6, v[28:31] offset:46336
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_1948
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	.Lr25_label_1948:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v25, 0xff800000
	s_and_b32 s56, s48, 0xff
	v_mov_b32_e32 v24, s56
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_lt_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[144:145], a[72:73], 0
	v_mfma_f32_16x16x16_bf16 v[36:39], a[146:147], a[74:75], v[36:39]
	v_max3_f32 v24, v32, v33, v32
	v_max3_f32 v24, v34, v35, v24
	ds_write_b32 v3, v24 offset:53504
	v_mfma_f32_16x16x16_bf16 v[36:39], a[148:149], a[76:77], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[150:151], a[78:79], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[152:153], a[80:81], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[154:155], a[82:83], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[156:157], a[84:85], v[36:39]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:53504
	ds_read_b32 v21, v2 offset:53568
	v_mfma_f32_16x16x16_bf16 v[36:39], a[158:159], a[86:87], v[36:39]
	ds_read_b32 v22, v2 offset:53632
	ds_read_b32 v23, v2 offset:53696
	v_mfma_f32_16x16x16_bf16 v[36:39], a[160:161], a[88:89], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[162:163], a[90:91], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[164:165], a[92:93], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[166:167], a[94:95], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[168:169], a[96:97], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[170:171], a[98:99], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[172:173], a[100:101], v[36:39]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mfma_f32_16x16x16_bf16 v[36:39], a[174:175], a[102:103], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[176:177], a[104:105], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[178:179], a[106:107], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[180:181], a[108:109], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[182:183], a[110:111], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[184:185], a[112:113], v[36:39]
	ds_read_b128 a[144:147], v7 offset:37120
	ds_read_b128 a[148:151], v7 offset:38144
	v_mfma_f32_16x16x16_bf16 v[36:39], a[186:187], a[114:115], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[188:189], a[116:117], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[190:191], a[118:119], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[192:193], a[120:121], v[36:39]
	ds_read_b128 a[152:155], v7 offset:39168
	ds_read_b128 a[156:159], v7 offset:40192
	v_mfma_f32_16x16x16_bf16 v[36:39], a[194:195], a[122:123], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[196:197], a[124:125], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[198:199], a[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[200:201], a[128:129], v[36:39]
	ds_read_b128 a[160:163], v7 offset:41216
	ds_read_b128 a[164:167], v7 offset:42240
	v_mfma_f32_16x16x16_bf16 v[36:39], a[202:203], a[130:131], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[204:205], a[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[206:207], a[134:135], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[208:209], a[136:137], v[36:39]
	ds_read_b128 a[168:171], v7 offset:43264
	ds_read_b128 a[172:175], v7 offset:44288
	v_mfma_f32_16x16x16_bf16 v[36:39], a[210:211], a[138:139], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[212:213], a[140:141], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[214:215], a[142:143], v[36:39]
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v12
	v_max_f32_e32 v20, v24, v12
	v_sub_f32_e32 v16, v12, v20
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]
	v_mov_b32_e32 v12, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v16, s5, v16
	v_exp_f32_e32 v16, v16
	v_fma_f32 v32, v32, s5, -v21
	v_fma_f32 v33, v33, s5, -v21
	v_fma_f32 v34, v34, s5, -v21
	v_fma_f32 v35, v35, s5, -v21
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_mul_f32_e32 v14, v16, v14
	v_mov_b32_e32 v22, v32
	v_add_f32_e32 v22, v33, v22
	v_add_f32_e32 v22, v34, v22
	v_add_f32_e32 v22, v35, v22
	v_add_f32_e32 v14, v22, v14
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v32, v32
	v_add3_u32 v28, v32, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v33, v33
	v_add3_u32 v28, v33, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v32, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v34, v34
	v_add3_u32 v28, v34, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v35, v35
	v_add3_u32 v28, v35, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v33, v21, v20, s52
	s_nop 2
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_1A3D
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	.Lr25_label_1A3D:
	s_add_u32 s83, s84, s83
	s_nop 0
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v23, v16
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a216, v20
	v_accvgpr_write_b32 a217, v21
	v_accvgpr_read_b32 v20, a218
	v_accvgpr_read_b32 v21, a219
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a218, v20
	v_accvgpr_write_b32 a219, v21
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a220, v20
	v_accvgpr_write_b32 a221, v21
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a223
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a223, v21
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a224, v20
	v_accvgpr_write_b32 a225, v21
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a226, v20
	v_accvgpr_write_b32 a227, v21
	v_accvgpr_read_b32 v20, a228
	v_accvgpr_read_b32 v21, a229
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a228, v20
	v_accvgpr_write_b32 a229, v21
	v_accvgpr_read_b32 v20, a230
	v_accvgpr_read_b32 v21, a231
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a230, v20
	v_accvgpr_write_b32 a231, v21
	v_accvgpr_read_b32 v20, a232
	v_accvgpr_read_b32 v21, a233
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a232, v20
	v_accvgpr_write_b32 a233, v21
	v_accvgpr_read_b32 v20, a234
	v_accvgpr_read_b32 v21, a235
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a234, v20
	v_accvgpr_write_b32 a235, v21
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v25, 0xff800000
	s_and_b32 s56, s48, 0xff
	v_mov_b32_e32 v24, s56
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_lt_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]
	ds_read_b128 a[176:179], v7 offset:45312
	ds_read_b128 a[180:183], v7 offset:46336
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]
	ds_read_b128 a[184:187], v7 offset:47360
	ds_read_b128 a[188:191], v7 offset:48384
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]
	ds_read_b128 a[192:195], v7 offset:49408
	ds_read_b128 a[196:199], v7 offset:50432
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]
	ds_read_b128 a[200:203], v7 offset:51456
	ds_read_b128 a[204:207], v7 offset:52480
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]
	v_max3_f32 v24, v36, v37, v36
	v_max3_f32 v24, v38, v39, v24
	ds_write_b32 v3, v24 offset:54528
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:54528
	ds_read_b32 v21, v2 offset:54592
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]
	ds_read_b32 v22, v2 offset:54656
	ds_read_b32 v23, v2 offset:54720
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]
	v_mfma_f32_16x16x16_bf16 a[216:219], a[198:199], v[32:33], a[216:219]
	v_mfma_f32_16x16x16_bf16 a[220:223], a[200:201], v[32:33], a[220:223]
	v_mfma_f32_16x16x16_bf16 a[224:227], a[202:203], v[32:33], a[224:227]
	v_mfma_f32_16x16x16_bf16 a[228:231], a[204:205], v[32:33], a[228:231]
	v_mfma_f32_16x16x16_bf16 a[232:235], a[206:207], v[32:33], a[232:235]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v13
	v_max_f32_e32 v20, v24, v13
	v_sub_f32_e32 v17, v13, v20
	v_cndmask_b32_e64 v17, v17, 0, s[38:39]
	v_mov_b32_e32 v13, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v17, s5, v17
	v_exp_f32_e32 v17, v17
	v_fma_f32 v36, v36, s5, -v21
	v_fma_f32 v37, v37, s5, -v21
	v_fma_f32 v38, v38, s5, -v21
	v_fma_f32 v39, v39, s5, -v21
	v_exp_f32_e32 v36, v36
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	v_mul_f32_e32 v15, v17, v15
	v_mov_b32_e32 v22, v36
	v_add_f32_e32 v22, v37, v22
	v_add_f32_e32 v22, v38, v22
	v_add_f32_e32 v22, v39, v22
	v_add_f32_e32 v15, v22, v15
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v36, v36
	v_add3_u32 v28, v36, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v37, v37
	v_add3_u32 v28, v37, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v36, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v38, v38
	v_add3_u32 v28, v38, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v39, v39
	v_add3_u32 v28, v39, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v37, v21, v20, s52
	s_nop 2
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]
	v_pk_mul_f32 v[168:169], v[22:23], v[168:169]
	v_pk_mul_f32 v[170:171], v[22:23], v[170:171]
	v_pk_mul_f32 v[172:173], v[22:23], v[172:173]
	v_pk_mul_f32 v[174:175], v[22:23], v[174:175]
	v_pk_mul_f32 v[176:177], v[22:23], v[176:177]
	v_pk_mul_f32 v[178:179], v[22:23], v[178:179]
	v_pk_mul_f32 v[180:181], v[22:23], v[180:181]
	v_pk_mul_f32 v[182:183], v[22:23], v[182:183]
	v_pk_mul_f32 v[184:185], v[22:23], v[184:185]
	v_pk_mul_f32 v[186:187], v[22:23], v[186:187]
	v_pk_mul_f32 v[188:189], v[22:23], v[188:189]
	v_pk_mul_f32 v[190:191], v[22:23], v[190:191]
	v_pk_mul_f32 v[192:193], v[22:23], v[192:193]
	v_pk_mul_f32 v[194:195], v[22:23], v[194:195]
	v_pk_mul_f32 v[196:197], v[22:23], v[196:197]
	v_pk_mul_f32 v[198:199], v[22:23], v[198:199]
	v_pk_mul_f32 v[200:201], v[22:23], v[200:201]
	v_pk_mul_f32 v[202:203], v[22:23], v[202:203]
	v_pk_mul_f32 v[204:205], v[22:23], v[204:205]
	v_pk_mul_f32 v[206:207], v[22:23], v[206:207]
	v_pk_mul_f32 v[208:209], v[22:23], v[208:209]
	v_pk_mul_f32 v[210:211], v[22:23], v[210:211]
	v_pk_mul_f32 v[212:213], v[22:23], v[212:213]
	v_pk_mul_f32 v[214:215], v[22:23], v[214:215]
	v_pk_mul_f32 v[216:217], v[22:23], v[216:217]
	v_pk_mul_f32 v[218:219], v[22:23], v[218:219]
	v_pk_mul_f32 v[220:221], v[22:23], v[220:221]
	v_pk_mul_f32 v[222:223], v[22:23], v[222:223]
	v_pk_mul_f32 v[224:225], v[22:23], v[224:225]
	v_pk_mul_f32 v[226:227], v[22:23], v[226:227]
	v_pk_mul_f32 v[228:229], v[22:23], v[228:229]
	v_pk_mul_f32 v[230:231], v[22:23], v[230:231]
	v_pk_mul_f32 v[232:233], v[22:23], v[232:233]
	v_pk_mul_f32 v[234:235], v[22:23], v[234:235]
	v_pk_mul_f32 v[236:237], v[22:23], v[236:237]
	v_pk_mul_f32 v[238:239], v[22:23], v[238:239]
	v_pk_mul_f32 v[240:241], v[22:23], v[240:241]
	v_pk_mul_f32 v[242:243], v[22:23], v[242:243]
	v_pk_mul_f32 v[244:245], v[22:23], v[244:245]
	v_pk_mul_f32 v[246:247], v[22:23], v[246:247]
	v_pk_mul_f32 v[248:249], v[22:23], v[248:249]
	v_pk_mul_f32 v[250:251], v[22:23], v[250:251]
	v_pk_mul_f32 v[252:253], v[22:23], v[252:253]
	v_pk_mul_f32 v[254:255], v[22:23], v[254:255]
	v_accvgpr_read_b32 v20, a236
	v_accvgpr_read_b32 v21, a237
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a236, v20
	v_accvgpr_write_b32 a237, v21
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a238, v20
	v_accvgpr_write_b32 a239, v21
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a241
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a240, v20
	v_accvgpr_write_b32 a241, v21
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a242, v20
	v_accvgpr_write_b32 a243, v21
	v_accvgpr_read_b32 v20, a244
	v_accvgpr_read_b32 v21, a245
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a244, v20
	v_accvgpr_write_b32 a245, v21
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a246, v20
	v_accvgpr_write_b32 a247, v21
	v_accvgpr_read_b32 v20, a248
	v_accvgpr_read_b32 v21, a249
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a248, v20
	v_accvgpr_write_b32 a249, v21
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a250, v20
	v_accvgpr_write_b32 a251, v21
	v_accvgpr_read_b32 v20, a252
	v_accvgpr_read_b32 v21, a253
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a252, v20
	v_accvgpr_write_b32 a253, v21
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a254, v20
	v_accvgpr_write_b32 a255, v21
	s_waitcnt vmcnt(18) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[148:151], a[144:145], v[36:37], v[148:151]
	v_mfma_f32_16x16x16_bf16 v[152:155], a[146:147], v[36:37], v[152:155]
	v_mfma_f32_16x16x16_bf16 v[156:159], a[148:149], v[36:37], v[156:159]
	v_mfma_f32_16x16x16_bf16 v[160:163], a[150:151], v[36:37], v[160:163]
	v_mfma_f32_16x16x16_bf16 v[164:167], a[152:153], v[36:37], v[164:167]
	v_mfma_f32_16x16x16_bf16 v[168:171], a[154:155], v[36:37], v[168:171]
	v_mfma_f32_16x16x16_bf16 v[172:175], a[156:157], v[36:37], v[172:175]
	v_mfma_f32_16x16x16_bf16 v[176:179], a[158:159], v[36:37], v[176:179]
	v_mfma_f32_16x16x16_bf16 v[180:183], a[160:161], v[36:37], v[180:183]
	v_mfma_f32_16x16x16_bf16 v[184:187], a[162:163], v[36:37], v[184:187]
	v_mfma_f32_16x16x16_bf16 v[188:191], a[164:165], v[36:37], v[188:191]
	v_mfma_f32_16x16x16_bf16 v[192:195], a[166:167], v[36:37], v[192:195]
	v_mfma_f32_16x16x16_bf16 v[196:199], a[168:169], v[36:37], v[196:199]
	v_mfma_f32_16x16x16_bf16 v[200:203], a[170:171], v[36:37], v[200:203]
	v_mfma_f32_16x16x16_bf16 v[204:207], a[172:173], v[36:37], v[204:207]
	v_mfma_f32_16x16x16_bf16 v[208:211], a[174:175], v[36:37], v[208:211]
	v_mfma_f32_16x16x16_bf16 v[212:215], a[176:177], v[36:37], v[212:215]
	v_mfma_f32_16x16x16_bf16 v[216:219], a[178:179], v[36:37], v[216:219]
	v_mfma_f32_16x16x16_bf16 v[220:223], a[180:181], v[36:37], v[220:223]
	v_mfma_f32_16x16x16_bf16 v[224:227], a[182:183], v[36:37], v[224:227]
	v_mfma_f32_16x16x16_bf16 v[228:231], a[184:185], v[36:37], v[228:231]
	v_mfma_f32_16x16x16_bf16 v[232:235], a[186:187], v[36:37], v[232:235]
	v_mfma_f32_16x16x16_bf16 v[236:239], a[188:189], v[36:37], v[236:239]
	v_mfma_f32_16x16x16_bf16 v[240:243], a[190:191], v[36:37], v[240:243]
	v_mfma_f32_16x16x16_bf16 v[244:247], a[192:193], v[36:37], v[244:247]
	v_mfma_f32_16x16x16_bf16 v[248:251], a[194:195], v[36:37], v[248:251]
	v_mfma_f32_16x16x16_bf16 v[252:255], a[196:197], v[36:37], v[252:255]
	v_mfma_f32_16x16x16_bf16 a[236:239], a[198:199], v[36:37], a[236:239]
	v_mfma_f32_16x16x16_bf16 a[240:243], a[200:201], v[36:37], a[240:243]
	v_mfma_f32_16x16x16_bf16 a[244:247], a[202:203], v[36:37], a[244:247]
	v_mfma_f32_16x16x16_bf16 a[248:251], a[204:205], v[36:37], a[248:251]
	v_mfma_f32_16x16x16_bf16 a[252:255], a[206:207], v[36:37], a[252:255]
	s_nop 8
	s_branch .Lr25_label_2133
	.Lr25_label_1CF1:
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[144:145], a[0:1], 0
	ds_read_b128 a[176:179], v4 offset:19584
	ds_read_b128 a[180:183], v4 offset:19648
	v_mfma_f32_16x16x16_bf16 v[32:35], a[146:147], a[2:3], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[148:149], a[4:5], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[150:151], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[152:153], a[8:9], v[32:35]
	ds_read_b128 a[184:187], v4 offset:19840
	ds_read_b128 a[188:191], v4 offset:19904
	v_mfma_f32_16x16x16_bf16 v[32:35], a[154:155], a[10:11], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[156:157], a[12:13], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[158:159], a[14:15], v[32:35]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[160:161], a[16:17], v[32:35]
	ds_read_b128 a[192:195], v4 offset:20096
	ds_read_b128 a[196:199], v4 offset:20160
	v_mfma_f32_16x16x16_bf16 v[32:35], a[162:163], a[18:19], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[164:165], a[20:21], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[166:167], a[22:23], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[168:169], a[24:25], v[32:35]
	ds_read_b128 a[200:203], v4 offset:20352
	ds_read_b128 a[204:207], v4 offset:20416
	v_mfma_f32_16x16x16_bf16 v[32:35], a[170:171], a[26:27], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[172:173], a[28:29], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[174:175], a[30:31], v[32:35]
	s_waitcnt lgkmcnt(4)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[32:35], a[176:177], a[32:33], v[32:35]
	ds_read_b128 a[208:211], v4 offset:20608
	ds_read_b128 a[212:215], v4 offset:20672
	v_mfma_f32_16x16x16_bf16 v[32:35], a[178:179], a[34:35], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[180:181], a[36:37], v[32:35]
	v_perm_b32 v28, v22, v20, s53
	v_perm_b32 v30, v22, v20, s52
	v_perm_b32 v29, v26, v24, s53
	v_perm_b32 v31, v26, v24, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[182:183], a[38:39], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[184:185], a[40:41], v[32:35]
	ds_write_b128 v6, v[28:31] offset:45312
	v_mfma_f32_16x16x16_bf16 v[32:35], a[186:187], a[42:43], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[188:189], a[44:45], v[32:35]
	v_perm_b32 v28, v23, v21, s53
	v_perm_b32 v30, v23, v21, s52
	v_perm_b32 v29, v27, v25, s53
	v_perm_b32 v31, v27, v25, s52
	v_mfma_f32_16x16x16_bf16 v[32:35], a[190:191], a[46:47], v[32:35]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 v[32:35], a[192:193], a[48:49], v[32:35]
	ds_write_b128 v6, v[28:31] offset:46336
	v_mfma_f32_16x16x16_bf16 v[32:35], a[194:195], a[50:51], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[196:197], a[52:53], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[198:199], a[54:55], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[200:201], a[56:57], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[202:203], a[58:59], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[204:205], a[60:61], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[206:207], a[62:63], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[208:209], a[64:65], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[210:211], a[66:67], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[212:213], a[68:69], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[32:35], a[214:215], a[70:71], v[32:35]
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_1D8A
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	.Lr25_label_1D8A:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v25, 0xff800000
	s_and_b32 s56, s48, 0xff
	v_mov_b32_e32 v24, s56
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_lt_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v32, v25, v32, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v33, v25, v33, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v34, v25, v34, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v35, v25, v35, s[38:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[144:145], a[72:73], 0
	v_mfma_f32_16x16x16_bf16 v[36:39], a[146:147], a[74:75], v[36:39]
	v_max3_f32 v24, v32, v33, v32
	v_max3_f32 v24, v34, v35, v24
	ds_write_b32 v3, v24 offset:53504
	v_mfma_f32_16x16x16_bf16 v[36:39], a[148:149], a[76:77], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[150:151], a[78:79], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[152:153], a[80:81], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[154:155], a[82:83], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[156:157], a[84:85], v[36:39]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:53504
	ds_read_b32 v21, v2 offset:53568
	v_mfma_f32_16x16x16_bf16 v[36:39], a[158:159], a[86:87], v[36:39]
	ds_read_b32 v22, v2 offset:53632
	ds_read_b32 v23, v2 offset:53696
	v_mfma_f32_16x16x16_bf16 v[36:39], a[160:161], a[88:89], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[162:163], a[90:91], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[164:165], a[92:93], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[166:167], a[94:95], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[168:169], a[96:97], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[170:171], a[98:99], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[172:173], a[100:101], v[36:39]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mfma_f32_16x16x16_bf16 v[36:39], a[174:175], a[102:103], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[176:177], a[104:105], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[178:179], a[106:107], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[180:181], a[108:109], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[182:183], a[110:111], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[184:185], a[112:113], v[36:39]
	ds_read_b128 a[144:147], v7 offset:37120
	ds_read_b128 a[148:151], v7 offset:38144
	v_mfma_f32_16x16x16_bf16 v[36:39], a[186:187], a[114:115], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[188:189], a[116:117], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[190:191], a[118:119], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[192:193], a[120:121], v[36:39]
	ds_read_b128 a[152:155], v7 offset:39168
	ds_read_b128 a[156:159], v7 offset:40192
	v_mfma_f32_16x16x16_bf16 v[36:39], a[194:195], a[122:123], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[196:197], a[124:125], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[198:199], a[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[200:201], a[128:129], v[36:39]
	ds_read_b128 a[160:163], v7 offset:41216
	ds_read_b128 a[164:167], v7 offset:42240
	v_mfma_f32_16x16x16_bf16 v[36:39], a[202:203], a[130:131], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[204:205], a[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[206:207], a[134:135], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[208:209], a[136:137], v[36:39]
	ds_read_b128 a[168:171], v7 offset:43264
	ds_read_b128 a[172:175], v7 offset:44288
	v_mfma_f32_16x16x16_bf16 v[36:39], a[210:211], a[138:139], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[212:213], a[140:141], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[36:39], a[214:215], a[142:143], v[36:39]
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v12
	v_max_f32_e32 v20, v24, v12
	v_sub_f32_e32 v16, v12, v20
	v_cndmask_b32_e64 v16, v16, 0, s[38:39]
	v_mov_b32_e32 v12, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v16, s5, v16
	v_exp_f32_e32 v16, v16
	v_fma_f32 v32, v32, s5, -v21
	v_fma_f32 v33, v33, s5, -v21
	v_fma_f32 v34, v34, s5, -v21
	v_fma_f32 v35, v35, s5, -v21
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_mul_f32_e32 v14, v16, v14
	v_mov_b32_e32 v22, v32
	v_add_f32_e32 v22, v33, v22
	v_add_f32_e32 v22, v34, v22
	v_add_f32_e32 v22, v35, v22
	v_add_f32_e32 v14, v22, v14
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v32, v32
	v_add3_u32 v28, v32, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v33, v33
	v_add3_u32 v28, v33, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v32, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v34, v34
	v_add3_u32 v28, v34, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v35, v35
	v_add3_u32 v28, v35, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v33, v21, v20, s52
	s_nop 2
	s_cmp_le_i32 s83, s82
	s_cbranch_scc1 .Lr25_label_1E7F
	v_mov_b32_e32 v25, 0xff800000
	v_mov_b32_e32 v24, s82
	s_sub_u32 s56, s83, 15
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v20, s56, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_le_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_le_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	.Lr25_label_1E7F:
	s_add_u32 s83, s84, s83
	s_nop 0
	v_mov_b32_e32 v22, v16
	v_mov_b32_e32 v23, v16
	v_pk_mul_f32 v[40:41], v[22:23], v[40:41]
	v_pk_mul_f32 v[42:43], v[22:23], v[42:43]
	v_pk_mul_f32 v[44:45], v[22:23], v[44:45]
	v_pk_mul_f32 v[46:47], v[22:23], v[46:47]
	v_pk_mul_f32 v[48:49], v[22:23], v[48:49]
	v_pk_mul_f32 v[50:51], v[22:23], v[50:51]
	v_pk_mul_f32 v[52:53], v[22:23], v[52:53]
	v_pk_mul_f32 v[54:55], v[22:23], v[54:55]
	v_pk_mul_f32 v[56:57], v[22:23], v[56:57]
	v_pk_mul_f32 v[58:59], v[22:23], v[58:59]
	v_pk_mul_f32 v[60:61], v[22:23], v[60:61]
	v_pk_mul_f32 v[62:63], v[22:23], v[62:63]
	v_pk_mul_f32 v[64:65], v[22:23], v[64:65]
	v_pk_mul_f32 v[66:67], v[22:23], v[66:67]
	v_pk_mul_f32 v[68:69], v[22:23], v[68:69]
	v_pk_mul_f32 v[70:71], v[22:23], v[70:71]
	v_pk_mul_f32 v[72:73], v[22:23], v[72:73]
	v_pk_mul_f32 v[74:75], v[22:23], v[74:75]
	v_pk_mul_f32 v[76:77], v[22:23], v[76:77]
	v_pk_mul_f32 v[78:79], v[22:23], v[78:79]
	v_pk_mul_f32 v[80:81], v[22:23], v[80:81]
	v_pk_mul_f32 v[82:83], v[22:23], v[82:83]
	v_pk_mul_f32 v[84:85], v[22:23], v[84:85]
	v_pk_mul_f32 v[86:87], v[22:23], v[86:87]
	v_pk_mul_f32 v[88:89], v[22:23], v[88:89]
	v_pk_mul_f32 v[90:91], v[22:23], v[90:91]
	v_pk_mul_f32 v[92:93], v[22:23], v[92:93]
	v_pk_mul_f32 v[94:95], v[22:23], v[94:95]
	v_pk_mul_f32 v[96:97], v[22:23], v[96:97]
	v_pk_mul_f32 v[98:99], v[22:23], v[98:99]
	v_pk_mul_f32 v[100:101], v[22:23], v[100:101]
	v_pk_mul_f32 v[102:103], v[22:23], v[102:103]
	v_pk_mul_f32 v[104:105], v[22:23], v[104:105]
	v_pk_mul_f32 v[106:107], v[22:23], v[106:107]
	v_pk_mul_f32 v[108:109], v[22:23], v[108:109]
	v_pk_mul_f32 v[110:111], v[22:23], v[110:111]
	v_pk_mul_f32 v[112:113], v[22:23], v[112:113]
	v_pk_mul_f32 v[114:115], v[22:23], v[114:115]
	v_pk_mul_f32 v[116:117], v[22:23], v[116:117]
	v_pk_mul_f32 v[118:119], v[22:23], v[118:119]
	v_pk_mul_f32 v[120:121], v[22:23], v[120:121]
	v_pk_mul_f32 v[122:123], v[22:23], v[122:123]
	v_pk_mul_f32 v[124:125], v[22:23], v[124:125]
	v_pk_mul_f32 v[126:127], v[22:23], v[126:127]
	v_pk_mul_f32 v[128:129], v[22:23], v[128:129]
	v_pk_mul_f32 v[130:131], v[22:23], v[130:131]
	v_pk_mul_f32 v[132:133], v[22:23], v[132:133]
	v_pk_mul_f32 v[134:135], v[22:23], v[134:135]
	v_pk_mul_f32 v[136:137], v[22:23], v[136:137]
	v_pk_mul_f32 v[138:139], v[22:23], v[138:139]
	v_pk_mul_f32 v[140:141], v[22:23], v[140:141]
	v_pk_mul_f32 v[142:143], v[22:23], v[142:143]
	v_pk_mul_f32 v[144:145], v[22:23], v[144:145]
	v_pk_mul_f32 v[146:147], v[22:23], v[146:147]
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a216, v20
	v_accvgpr_write_b32 a217, v21
	v_accvgpr_read_b32 v20, a218
	v_accvgpr_read_b32 v21, a219
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a218, v20
	v_accvgpr_write_b32 a219, v21
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a220, v20
	v_accvgpr_write_b32 a221, v21
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a223
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a223, v21
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a224, v20
	v_accvgpr_write_b32 a225, v21
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a226, v20
	v_accvgpr_write_b32 a227, v21
	v_accvgpr_read_b32 v20, a228
	v_accvgpr_read_b32 v21, a229
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a228, v20
	v_accvgpr_write_b32 a229, v21
	v_accvgpr_read_b32 v20, a230
	v_accvgpr_read_b32 v21, a231
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a230, v20
	v_accvgpr_write_b32 a231, v21
	v_accvgpr_read_b32 v20, a232
	v_accvgpr_read_b32 v21, a233
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a232, v20
	v_accvgpr_write_b32 a233, v21
	v_accvgpr_read_b32 v20, a234
	v_accvgpr_read_b32 v21, a235
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a234, v20
	v_accvgpr_write_b32 a235, v21
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v25, 0xff800000
	s_and_b32 s56, s48, 0xff
	v_mov_b32_e32 v24, s56
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 4, v20
	v_add_u32_e32 v21, 1, v20
	v_add_u32_e32 v22, 2, v20
	v_add_u32_e32 v23, 3, v20
	v_cmp_lt_u32_e64 s[38:39], v20, v24
	v_add_u32_e32 v20, 64, v20
	s_nop 0
	v_cndmask_b32_e64 v36, v25, v36, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v21, v24
	v_add_u32_e32 v21, 64, v21
	s_nop 0
	v_cndmask_b32_e64 v37, v25, v37, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v22, v24
	v_add_u32_e32 v22, 64, v22
	s_nop 0
	v_cndmask_b32_e64 v38, v25, v38, s[38:39]
	v_cmp_lt_u32_e64 s[38:39], v23, v24
	v_add_u32_e32 v23, 64, v23
	s_nop 0
	v_cndmask_b32_e64 v39, v25, v39, s[38:39]
	v_mfma_f32_16x16x16_bf16 v[40:43], a[144:145], v[32:33], v[40:43]
	ds_read_b128 a[176:179], v7 offset:45312
	ds_read_b128 a[180:183], v7 offset:46336
	v_mfma_f32_16x16x16_bf16 v[44:47], a[146:147], v[32:33], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[48:51], a[148:149], v[32:33], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], a[150:151], v[32:33], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], a[152:153], v[32:33], v[56:59]
	ds_read_b128 a[184:187], v7 offset:47360
	ds_read_b128 a[188:191], v7 offset:48384
	v_mfma_f32_16x16x16_bf16 v[60:63], a[154:155], v[32:33], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[64:67], a[156:157], v[32:33], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], a[158:159], v[32:33], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], a[160:161], v[32:33], v[72:75]
	ds_read_b128 a[192:195], v7 offset:49408
	ds_read_b128 a[196:199], v7 offset:50432
	v_mfma_f32_16x16x16_bf16 v[76:79], a[162:163], v[32:33], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[80:83], a[164:165], v[32:33], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], a[166:167], v[32:33], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], a[168:169], v[32:33], v[88:91]
	ds_read_b128 a[200:203], v7 offset:51456
	ds_read_b128 a[204:207], v7 offset:52480
	v_mfma_f32_16x16x16_bf16 v[92:95], a[170:171], v[32:33], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[96:99], a[172:173], v[32:33], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], a[174:175], v[32:33], v[100:103]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x16_bf16 v[104:107], a[176:177], v[32:33], v[104:107]
	v_max3_f32 v24, v36, v37, v36
	v_max3_f32 v24, v38, v39, v24
	ds_write_b32 v3, v24 offset:54528
	v_mfma_f32_16x16x16_bf16 v[108:111], a[178:179], v[32:33], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], a[180:181], v[32:33], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], a[182:183], v[32:33], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], a[184:185], v[32:33], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], a[186:187], v[32:33], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], a[188:189], v[32:33], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[132:135], a[190:191], v[32:33], v[132:135]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 v[136:139], a[192:193], v[32:33], v[136:139]
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:54528
	ds_read_b32 v21, v2 offset:54592
	v_mfma_f32_16x16x16_bf16 v[140:143], a[194:195], v[32:33], v[140:143]
	ds_read_b32 v22, v2 offset:54656
	ds_read_b32 v23, v2 offset:54720
	v_mfma_f32_16x16x16_bf16 v[144:147], a[196:197], v[32:33], v[144:147]
	v_mfma_f32_16x16x16_bf16 a[216:219], a[198:199], v[32:33], a[216:219]
	v_mfma_f32_16x16x16_bf16 a[220:223], a[200:201], v[32:33], a[220:223]
	v_mfma_f32_16x16x16_bf16 a[224:227], a[202:203], v[32:33], a[224:227]
	v_mfma_f32_16x16x16_bf16 a[228:231], a[204:205], v[32:33], a[228:231]
	v_mfma_f32_16x16x16_bf16 a[232:235], a[206:207], v[32:33], a[232:235]
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v24, v20, v21, v24
	v_max3_f32 v24, v22, v23, v24
	v_mov_b32_e32 v25, 0xff7fffff
	v_cmp_eq_u32_e64 s[38:39], v25, v13
	v_max_f32_e32 v20, v24, v13
	v_sub_f32_e32 v17, v13, v20
	v_cndmask_b32_e64 v17, v17, 0, s[38:39]
	v_mov_b32_e32 v13, v20
	v_mul_f32_e32 v21, s5, v20
	v_mul_f32_e32 v17, s5, v17
	v_exp_f32_e32 v17, v17
	v_fma_f32 v36, v36, s5, -v21
	v_fma_f32 v37, v37, s5, -v21
	v_fma_f32 v38, v38, s5, -v21
	v_fma_f32 v39, v39, s5, -v21
	v_exp_f32_e32 v36, v36
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	v_mul_f32_e32 v15, v17, v15
	v_mov_b32_e32 v22, v36
	v_add_f32_e32 v22, v37, v22
	v_add_f32_e32 v22, v38, v22
	v_add_f32_e32 v22, v39, v22
	v_add_f32_e32 v15, v22, v15
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	v_cmp_u_f32_e64 s[38:39], v36, v36
	v_add3_u32 v28, v36, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v37, v37
	v_add3_u32 v28, v37, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v36, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v38, v38
	v_add3_u32 v28, v38, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v39, v39
	v_add3_u32 v28, v39, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v37, v21, v20, s52
	s_nop 2
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_pk_mul_f32 v[148:149], v[22:23], v[148:149]
	v_pk_mul_f32 v[150:151], v[22:23], v[150:151]
	v_pk_mul_f32 v[152:153], v[22:23], v[152:153]
	v_pk_mul_f32 v[154:155], v[22:23], v[154:155]
	v_pk_mul_f32 v[156:157], v[22:23], v[156:157]
	v_pk_mul_f32 v[158:159], v[22:23], v[158:159]
	v_pk_mul_f32 v[160:161], v[22:23], v[160:161]
	v_pk_mul_f32 v[162:163], v[22:23], v[162:163]
	v_pk_mul_f32 v[164:165], v[22:23], v[164:165]
	v_pk_mul_f32 v[166:167], v[22:23], v[166:167]
	v_pk_mul_f32 v[168:169], v[22:23], v[168:169]
	v_pk_mul_f32 v[170:171], v[22:23], v[170:171]
	v_pk_mul_f32 v[172:173], v[22:23], v[172:173]
	v_pk_mul_f32 v[174:175], v[22:23], v[174:175]
	v_pk_mul_f32 v[176:177], v[22:23], v[176:177]
	v_pk_mul_f32 v[178:179], v[22:23], v[178:179]
	v_pk_mul_f32 v[180:181], v[22:23], v[180:181]
	v_pk_mul_f32 v[182:183], v[22:23], v[182:183]
	v_pk_mul_f32 v[184:185], v[22:23], v[184:185]
	v_pk_mul_f32 v[186:187], v[22:23], v[186:187]
	v_pk_mul_f32 v[188:189], v[22:23], v[188:189]
	v_pk_mul_f32 v[190:191], v[22:23], v[190:191]
	v_pk_mul_f32 v[192:193], v[22:23], v[192:193]
	v_pk_mul_f32 v[194:195], v[22:23], v[194:195]
	v_pk_mul_f32 v[196:197], v[22:23], v[196:197]
	v_pk_mul_f32 v[198:199], v[22:23], v[198:199]
	v_pk_mul_f32 v[200:201], v[22:23], v[200:201]
	v_pk_mul_f32 v[202:203], v[22:23], v[202:203]
	v_pk_mul_f32 v[204:205], v[22:23], v[204:205]
	v_pk_mul_f32 v[206:207], v[22:23], v[206:207]
	v_pk_mul_f32 v[208:209], v[22:23], v[208:209]
	v_pk_mul_f32 v[210:211], v[22:23], v[210:211]
	v_pk_mul_f32 v[212:213], v[22:23], v[212:213]
	v_pk_mul_f32 v[214:215], v[22:23], v[214:215]
	v_pk_mul_f32 v[216:217], v[22:23], v[216:217]
	v_pk_mul_f32 v[218:219], v[22:23], v[218:219]
	v_pk_mul_f32 v[220:221], v[22:23], v[220:221]
	v_pk_mul_f32 v[222:223], v[22:23], v[222:223]
	v_pk_mul_f32 v[224:225], v[22:23], v[224:225]
	v_pk_mul_f32 v[226:227], v[22:23], v[226:227]
	v_pk_mul_f32 v[228:229], v[22:23], v[228:229]
	v_pk_mul_f32 v[230:231], v[22:23], v[230:231]
	v_pk_mul_f32 v[232:233], v[22:23], v[232:233]
	v_pk_mul_f32 v[234:235], v[22:23], v[234:235]
	v_pk_mul_f32 v[236:237], v[22:23], v[236:237]
	v_pk_mul_f32 v[238:239], v[22:23], v[238:239]
	v_pk_mul_f32 v[240:241], v[22:23], v[240:241]
	v_pk_mul_f32 v[242:243], v[22:23], v[242:243]
	v_pk_mul_f32 v[244:245], v[22:23], v[244:245]
	v_pk_mul_f32 v[246:247], v[22:23], v[246:247]
	v_pk_mul_f32 v[248:249], v[22:23], v[248:249]
	v_pk_mul_f32 v[250:251], v[22:23], v[250:251]
	v_pk_mul_f32 v[252:253], v[22:23], v[252:253]
	v_pk_mul_f32 v[254:255], v[22:23], v[254:255]
	v_accvgpr_read_b32 v20, a236
	v_accvgpr_read_b32 v21, a237
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a236, v20
	v_accvgpr_write_b32 a237, v21
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a238, v20
	v_accvgpr_write_b32 a239, v21
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a241
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a240, v20
	v_accvgpr_write_b32 a241, v21
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a242, v20
	v_accvgpr_write_b32 a243, v21
	v_accvgpr_read_b32 v20, a244
	v_accvgpr_read_b32 v21, a245
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a244, v20
	v_accvgpr_write_b32 a245, v21
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a246, v20
	v_accvgpr_write_b32 a247, v21
	v_accvgpr_read_b32 v20, a248
	v_accvgpr_read_b32 v21, a249
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a248, v20
	v_accvgpr_write_b32 a249, v21
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a250, v20
	v_accvgpr_write_b32 a251, v21
	v_accvgpr_read_b32 v20, a252
	v_accvgpr_read_b32 v21, a253
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a252, v20
	v_accvgpr_write_b32 a253, v21
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	v_accvgpr_write_b32 a254, v20
	v_accvgpr_write_b32 a255, v21
	s_waitcnt vmcnt(18) lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x16_bf16 v[148:151], a[144:145], v[36:37], v[148:151]
	v_mfma_f32_16x16x16_bf16 v[152:155], a[146:147], v[36:37], v[152:155]
	v_mfma_f32_16x16x16_bf16 v[156:159], a[148:149], v[36:37], v[156:159]
	v_mfma_f32_16x16x16_bf16 v[160:163], a[150:151], v[36:37], v[160:163]
	v_mfma_f32_16x16x16_bf16 v[164:167], a[152:153], v[36:37], v[164:167]
	v_mfma_f32_16x16x16_bf16 v[168:171], a[154:155], v[36:37], v[168:171]
	v_mfma_f32_16x16x16_bf16 v[172:175], a[156:157], v[36:37], v[172:175]
	v_mfma_f32_16x16x16_bf16 v[176:179], a[158:159], v[36:37], v[176:179]
	v_mfma_f32_16x16x16_bf16 v[180:183], a[160:161], v[36:37], v[180:183]
	v_mfma_f32_16x16x16_bf16 v[184:187], a[162:163], v[36:37], v[184:187]
	v_mfma_f32_16x16x16_bf16 v[188:191], a[164:165], v[36:37], v[188:191]
	v_mfma_f32_16x16x16_bf16 v[192:195], a[166:167], v[36:37], v[192:195]
	v_mfma_f32_16x16x16_bf16 v[196:199], a[168:169], v[36:37], v[196:199]
	v_mfma_f32_16x16x16_bf16 v[200:203], a[170:171], v[36:37], v[200:203]
	v_mfma_f32_16x16x16_bf16 v[204:207], a[172:173], v[36:37], v[204:207]
	v_mfma_f32_16x16x16_bf16 v[208:211], a[174:175], v[36:37], v[208:211]
	v_mfma_f32_16x16x16_bf16 v[212:215], a[176:177], v[36:37], v[212:215]
	v_mfma_f32_16x16x16_bf16 v[216:219], a[178:179], v[36:37], v[216:219]
	v_mfma_f32_16x16x16_bf16 v[220:223], a[180:181], v[36:37], v[220:223]
	v_mfma_f32_16x16x16_bf16 v[224:227], a[182:183], v[36:37], v[224:227]
	v_mfma_f32_16x16x16_bf16 v[228:231], a[184:185], v[36:37], v[228:231]
	v_mfma_f32_16x16x16_bf16 v[232:235], a[186:187], v[36:37], v[232:235]
	v_mfma_f32_16x16x16_bf16 v[236:239], a[188:189], v[36:37], v[236:239]
	v_mfma_f32_16x16x16_bf16 v[240:243], a[190:191], v[36:37], v[240:243]
	v_mfma_f32_16x16x16_bf16 v[244:247], a[192:193], v[36:37], v[244:247]
	v_mfma_f32_16x16x16_bf16 v[248:251], a[194:195], v[36:37], v[248:251]
	v_mfma_f32_16x16x16_bf16 v[252:255], a[196:197], v[36:37], v[252:255]
	v_mfma_f32_16x16x16_bf16 a[236:239], a[198:199], v[36:37], a[236:239]
	v_mfma_f32_16x16x16_bf16 a[240:243], a[200:201], v[36:37], a[240:243]
	v_mfma_f32_16x16x16_bf16 a[244:247], a[202:203], v[36:37], a[244:247]
	v_mfma_f32_16x16x16_bf16 a[248:251], a[204:205], v[36:37], a[248:251]
	v_mfma_f32_16x16x16_bf16 a[252:255], a[206:207], v[36:37], a[252:255]
	s_nop 8
	s_branch .Lr25_label_2133
	.Lr25_label_2133:
	ds_write_b32 v3, v14 offset:55552
	ds_write_b32 v3, v15 offset:56576
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v20, v2 offset:55552
	ds_read_b32 v21, v2 offset:55616
	ds_read_b32 v22, v2 offset:55680
	ds_read_b32 v23, v2 offset:55744
	ds_read_b32 v24, v2 offset:56576
	ds_read_b32 v25, v2 offset:56640
	ds_read_b32 v26, v2 offset:56704
	ds_read_b32 v27, v2 offset:56768
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_add_f32_e32 v14, v20, v14
	v_add_f32_e32 v15, v24, v15
	v_add_f32_e32 v14, v21, v14
	v_add_f32_e32 v15, v25, v15
	v_add_f32_e32 v14, v22, v14
	v_add_f32_e32 v15, v26, v15
	v_add_f32_e32 v14, v23, v14
	v_add_f32_e32 v15, v27, v15
	v_mov_b32_e32 v20, 0
	v_cmp_eq_u32_e64 s[38:39], v20, v14
	v_cmp_eq_u32_e64 s[40:41], v20, v15
	v_mul_f32_e64 v20, v12, s64
	v_mul_f32_e64 v22, v13, s64
	v_log_f32_e32 v21, v14
	v_log_f32_e32 v23, v15
	v_cndmask_b32_e64 v14, v14, 1.0, s[38:39]
	v_cndmask_b32_e64 v15, v15, 1.0, s[40:41]
	s_nop 1
	v_rcp_f32_e32 v14, v14
	v_rcp_f32_e32 v15, v15
	s_nop 1
	v_fma_f32 v24, v21, s63, v20
	v_fma_f32 v25, v23, s63, v22
	v_mul_f32_e32 v40, v14, v40
	v_mul_f32_e32 v41, v14, v41
	v_mul_f32_e32 v42, v14, v42
	v_mul_f32_e32 v43, v14, v43
	v_mul_f32_e32 v44, v14, v44
	v_mul_f32_e32 v45, v14, v45
	v_mul_f32_e32 v46, v14, v46
	v_mul_f32_e32 v47, v14, v47
	v_mul_f32_e32 v48, v14, v48
	v_mul_f32_e32 v49, v14, v49
	v_mul_f32_e32 v50, v14, v50
	v_mul_f32_e32 v51, v14, v51
	v_mul_f32_e32 v52, v14, v52
	v_mul_f32_e32 v53, v14, v53
	v_mul_f32_e32 v54, v14, v54
	v_mul_f32_e32 v55, v14, v55
	v_mul_f32_e32 v56, v14, v56
	v_mul_f32_e32 v57, v14, v57
	v_mul_f32_e32 v58, v14, v58
	v_mul_f32_e32 v59, v14, v59
	v_mul_f32_e32 v60, v14, v60
	v_mul_f32_e32 v61, v14, v61
	v_mul_f32_e32 v62, v14, v62
	v_mul_f32_e32 v63, v14, v63
	v_mul_f32_e32 v64, v14, v64
	v_mul_f32_e32 v65, v14, v65
	v_mul_f32_e32 v66, v14, v66
	v_mul_f32_e32 v67, v14, v67
	v_mul_f32_e32 v68, v14, v68
	v_mul_f32_e32 v69, v14, v69
	v_mul_f32_e32 v70, v14, v70
	v_mul_f32_e32 v71, v14, v71
	v_mul_f32_e32 v72, v14, v72
	v_mul_f32_e32 v73, v14, v73
	v_mul_f32_e32 v74, v14, v74
	v_mul_f32_e32 v75, v14, v75
	v_mul_f32_e32 v76, v14, v76
	v_mul_f32_e32 v77, v14, v77
	v_mul_f32_e32 v78, v14, v78
	v_mul_f32_e32 v79, v14, v79
	v_mul_f32_e32 v80, v14, v80
	v_mul_f32_e32 v81, v14, v81
	v_mul_f32_e32 v82, v14, v82
	v_mul_f32_e32 v83, v14, v83
	v_mul_f32_e32 v84, v14, v84
	v_mul_f32_e32 v85, v14, v85
	v_mul_f32_e32 v86, v14, v86
	v_mul_f32_e32 v87, v14, v87
	v_mul_f32_e32 v88, v14, v88
	v_mul_f32_e32 v89, v14, v89
	v_mul_f32_e32 v90, v14, v90
	v_mul_f32_e32 v91, v14, v91
	v_mul_f32_e32 v92, v14, v92
	v_mul_f32_e32 v93, v14, v93
	v_mul_f32_e32 v94, v14, v94
	v_mul_f32_e32 v95, v14, v95
	v_mul_f32_e32 v96, v14, v96
	v_mul_f32_e32 v97, v14, v97
	v_mul_f32_e32 v98, v14, v98
	v_mul_f32_e32 v99, v14, v99
	v_mul_f32_e32 v100, v14, v100
	v_mul_f32_e32 v101, v14, v101
	v_mul_f32_e32 v102, v14, v102
	v_mul_f32_e32 v103, v14, v103
	v_mul_f32_e32 v104, v14, v104
	v_mul_f32_e32 v105, v14, v105
	v_mul_f32_e32 v106, v14, v106
	v_mul_f32_e32 v107, v14, v107
	v_mul_f32_e32 v108, v14, v108
	v_mul_f32_e32 v109, v14, v109
	v_mul_f32_e32 v110, v14, v110
	v_mul_f32_e32 v111, v14, v111
	v_mul_f32_e32 v112, v14, v112
	v_mul_f32_e32 v113, v14, v113
	v_mul_f32_e32 v114, v14, v114
	v_mul_f32_e32 v115, v14, v115
	v_mul_f32_e32 v116, v14, v116
	v_mul_f32_e32 v117, v14, v117
	v_mul_f32_e32 v118, v14, v118
	v_mul_f32_e32 v119, v14, v119
	v_mul_f32_e32 v120, v14, v120
	v_mul_f32_e32 v121, v14, v121
	v_mul_f32_e32 v122, v14, v122
	v_mul_f32_e32 v123, v14, v123
	v_mul_f32_e32 v124, v14, v124
	v_mul_f32_e32 v125, v14, v125
	v_mul_f32_e32 v126, v14, v126
	v_mul_f32_e32 v127, v14, v127
	v_mul_f32_e32 v128, v14, v128
	v_mul_f32_e32 v129, v14, v129
	v_mul_f32_e32 v130, v14, v130
	v_mul_f32_e32 v131, v14, v131
	v_mul_f32_e32 v132, v14, v132
	v_mul_f32_e32 v133, v14, v133
	v_mul_f32_e32 v134, v14, v134
	v_mul_f32_e32 v135, v14, v135
	v_mul_f32_e32 v136, v14, v136
	v_mul_f32_e32 v137, v14, v137
	v_mul_f32_e32 v138, v14, v138
	v_mul_f32_e32 v139, v14, v139
	v_mul_f32_e32 v140, v14, v140
	v_mul_f32_e32 v141, v14, v141
	v_mul_f32_e32 v142, v14, v142
	v_mul_f32_e32 v143, v14, v143
	v_mul_f32_e32 v144, v14, v144
	v_mul_f32_e32 v145, v14, v145
	v_mul_f32_e32 v146, v14, v146
	v_mul_f32_e32 v147, v14, v147
	v_accvgpr_read_b32 v20, a216
	v_accvgpr_read_b32 v21, a217
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a216, v20
	v_accvgpr_write_b32 a217, v21
	v_accvgpr_read_b32 v20, a218
	v_accvgpr_read_b32 v21, a219
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a218, v20
	v_accvgpr_write_b32 a219, v21
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a221
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a220, v20
	v_accvgpr_write_b32 a221, v21
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a223
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a222, v20
	v_accvgpr_write_b32 a223, v21
	v_accvgpr_read_b32 v20, a224
	v_accvgpr_read_b32 v21, a225
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a224, v20
	v_accvgpr_write_b32 a225, v21
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a226, v20
	v_accvgpr_write_b32 a227, v21
	v_accvgpr_read_b32 v20, a228
	v_accvgpr_read_b32 v21, a229
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a228, v20
	v_accvgpr_write_b32 a229, v21
	v_accvgpr_read_b32 v20, a230
	v_accvgpr_read_b32 v21, a231
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a230, v20
	v_accvgpr_write_b32 a231, v21
	v_accvgpr_read_b32 v20, a232
	v_accvgpr_read_b32 v21, a233
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a232, v20
	v_accvgpr_write_b32 a233, v21
	v_accvgpr_read_b32 v20, a234
	v_accvgpr_read_b32 v21, a235
	v_mul_f32_e32 v20, v14, v20
	v_mul_f32_e32 v21, v14, v21
	v_accvgpr_write_b32 a234, v20
	v_accvgpr_write_b32 a235, v21
	v_mul_f32_e32 v148, v15, v148
	v_mul_f32_e32 v149, v15, v149
	v_mul_f32_e32 v150, v15, v150
	v_mul_f32_e32 v151, v15, v151
	v_mul_f32_e32 v152, v15, v152
	v_mul_f32_e32 v153, v15, v153
	v_mul_f32_e32 v154, v15, v154
	v_mul_f32_e32 v155, v15, v155
	v_mul_f32_e32 v156, v15, v156
	v_mul_f32_e32 v157, v15, v157
	v_mul_f32_e32 v158, v15, v158
	v_mul_f32_e32 v159, v15, v159
	v_mul_f32_e32 v160, v15, v160
	v_mul_f32_e32 v161, v15, v161
	v_mul_f32_e32 v162, v15, v162
	v_mul_f32_e32 v163, v15, v163
	v_mul_f32_e32 v164, v15, v164
	v_mul_f32_e32 v165, v15, v165
	v_mul_f32_e32 v166, v15, v166
	v_mul_f32_e32 v167, v15, v167
	v_mul_f32_e32 v168, v15, v168
	v_mul_f32_e32 v169, v15, v169
	v_mul_f32_e32 v170, v15, v170
	v_mul_f32_e32 v171, v15, v171
	v_mul_f32_e32 v172, v15, v172
	v_mul_f32_e32 v173, v15, v173
	v_mul_f32_e32 v174, v15, v174
	v_mul_f32_e32 v175, v15, v175
	v_mul_f32_e32 v176, v15, v176
	v_mul_f32_e32 v177, v15, v177
	v_mul_f32_e32 v178, v15, v178
	v_mul_f32_e32 v179, v15, v179
	v_mul_f32_e32 v180, v15, v180
	v_mul_f32_e32 v181, v15, v181
	v_mul_f32_e32 v182, v15, v182
	v_mul_f32_e32 v183, v15, v183
	v_mul_f32_e32 v184, v15, v184
	v_mul_f32_e32 v185, v15, v185
	v_mul_f32_e32 v186, v15, v186
	v_mul_f32_e32 v187, v15, v187
	v_mul_f32_e32 v188, v15, v188
	v_mul_f32_e32 v189, v15, v189
	v_mul_f32_e32 v190, v15, v190
	v_mul_f32_e32 v191, v15, v191
	v_mul_f32_e32 v192, v15, v192
	v_mul_f32_e32 v193, v15, v193
	v_mul_f32_e32 v194, v15, v194
	v_mul_f32_e32 v195, v15, v195
	v_mul_f32_e32 v196, v15, v196
	v_mul_f32_e32 v197, v15, v197
	v_mul_f32_e32 v198, v15, v198
	v_mul_f32_e32 v199, v15, v199
	v_mul_f32_e32 v200, v15, v200
	v_mul_f32_e32 v201, v15, v201
	v_mul_f32_e32 v202, v15, v202
	v_mul_f32_e32 v203, v15, v203
	v_mul_f32_e32 v204, v15, v204
	v_mul_f32_e32 v205, v15, v205
	v_mul_f32_e32 v206, v15, v206
	v_mul_f32_e32 v207, v15, v207
	v_mul_f32_e32 v208, v15, v208
	v_mul_f32_e32 v209, v15, v209
	v_mul_f32_e32 v210, v15, v210
	v_mul_f32_e32 v211, v15, v211
	v_mul_f32_e32 v212, v15, v212
	v_mul_f32_e32 v213, v15, v213
	v_mul_f32_e32 v214, v15, v214
	v_mul_f32_e32 v215, v15, v215
	v_mul_f32_e32 v216, v15, v216
	v_mul_f32_e32 v217, v15, v217
	v_mul_f32_e32 v218, v15, v218
	v_mul_f32_e32 v219, v15, v219
	v_mul_f32_e32 v220, v15, v220
	v_mul_f32_e32 v221, v15, v221
	v_mul_f32_e32 v222, v15, v222
	v_mul_f32_e32 v223, v15, v223
	v_mul_f32_e32 v224, v15, v224
	v_mul_f32_e32 v225, v15, v225
	v_mul_f32_e32 v226, v15, v226
	v_mul_f32_e32 v227, v15, v227
	v_mul_f32_e32 v228, v15, v228
	v_mul_f32_e32 v229, v15, v229
	v_mul_f32_e32 v230, v15, v230
	v_mul_f32_e32 v231, v15, v231
	v_mul_f32_e32 v232, v15, v232
	v_mul_f32_e32 v233, v15, v233
	v_mul_f32_e32 v234, v15, v234
	v_mul_f32_e32 v235, v15, v235
	v_mul_f32_e32 v236, v15, v236
	v_mul_f32_e32 v237, v15, v237
	v_mul_f32_e32 v238, v15, v238
	v_mul_f32_e32 v239, v15, v239
	v_mul_f32_e32 v240, v15, v240
	v_mul_f32_e32 v241, v15, v241
	v_mul_f32_e32 v242, v15, v242
	v_mul_f32_e32 v243, v15, v243
	v_mul_f32_e32 v244, v15, v244
	v_mul_f32_e32 v245, v15, v245
	v_mul_f32_e32 v246, v15, v246
	v_mul_f32_e32 v247, v15, v247
	v_mul_f32_e32 v248, v15, v248
	v_mul_f32_e32 v249, v15, v249
	v_mul_f32_e32 v250, v15, v250
	v_mul_f32_e32 v251, v15, v251
	v_mul_f32_e32 v252, v15, v252
	v_mul_f32_e32 v253, v15, v253
	v_mul_f32_e32 v254, v15, v254
	v_mul_f32_e32 v255, v15, v255
	v_accvgpr_read_b32 v20, a236
	v_accvgpr_read_b32 v21, a237
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a236, v20
	v_accvgpr_write_b32 a237, v21
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a238, v20
	v_accvgpr_write_b32 a239, v21
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a241
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a240, v20
	v_accvgpr_write_b32 a241, v21
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a243
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a242, v20
	v_accvgpr_write_b32 a243, v21
	v_accvgpr_read_b32 v20, a244
	v_accvgpr_read_b32 v21, a245
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a244, v20
	v_accvgpr_write_b32 a245, v21
	v_accvgpr_read_b32 v20, a246
	v_accvgpr_read_b32 v21, a247
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a246, v20
	v_accvgpr_write_b32 a247, v21
	v_accvgpr_read_b32 v20, a248
	v_accvgpr_read_b32 v21, a249
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a248, v20
	v_accvgpr_write_b32 a249, v21
	v_accvgpr_read_b32 v20, a250
	v_accvgpr_read_b32 v21, a251
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a250, v20
	v_accvgpr_write_b32 a251, v21
	v_accvgpr_read_b32 v20, a252
	v_accvgpr_read_b32 v21, a253
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a252, v20
	v_accvgpr_write_b32 a253, v21
	v_accvgpr_read_b32 v20, a254
	v_accvgpr_read_b32 v21, a255
	v_mul_f32_e32 v20, v15, v20
	v_mul_f32_e32 v21, v15, v21
	v_accvgpr_write_b32 a254, v20
	v_accvgpr_write_b32 a255, v21
	s_cmp_le_u32 s67, 1
	s_cbranch_scc0 .Lr25_label_2CFA
	s_mul_i32 s75, 0x400, s65
	s_mul_i32 s76, s67, s75
	s_add_u32 s56, s80, s79
	v_mov_b32_e32 v20, s56
	v_mul_lo_u32 v21, s76, v20
	v_mul_hi_u32 v22, s76, v20
	s_nop 2
	v_readfirstlane_b32 s56, v21
	v_readfirstlane_b32 s57, v22
	s_nop 4
	s_add_u32 s8, s56, s8
	s_addc_u32 s9, s57, s9
	s_sub_u32 s56, s81, s80
	s_mul_i32 s56, s56, s76
	s_mov_b32 s10, s56
	v_and_b32_e32 v20, 7, v0
	v_lshlrev_b32_e32 v18, 4, v20
	v_lshrrev_b32_e32 v20, 3, v0
	v_mul_i32_i24_e32 v20, 0x400, v20
	v_add_u32_e32 v18, v18, v20
	s_mul_i32 s56, s4, s75
	v_add_u32_e64 v18, v18, s56
	s_mul_i32 s56, s7, 0x4000
	v_add_u32_e64 v18, v18, s56
	v_mov_b32_e32 v19, v18
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_mul_i32 s75, 0x400, s65
	s_mul_i32 s76, s67, s75
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v5, 0x48, v20
	v_and_b32_e32 v20, 15, v0
	v_mul_i32_i24_e32 v20, 2, v20
	v_add_u32_e32 v5, v20, v5
	s_mul_i32 s56, s7, 0x480
	v_add_u32_e32 v5, s56, v5
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshrrev_b32_e32 v20, 3, v0
	v_mul_i32_i24_e32 v4, 2, v20
	v_and_b32_e32 v20, 7, v0
	v_mul_i32_i24_e32 v20, 36, v20
	v_add_u32_e32 v4, v20, v4
	s_mul_i32 s56, s7, 0x480
	v_add_u32_e32 v4, s56, v4
	v_lshlrev_b32_e32 v4, 2, v4
	v_mov_b32_e32 v29, 0xffff0000
	v_mov_b32_e32 v30, 0x7fff0000
	v_mov_b32_e32 v31, 0x7fff
	s_mov_b32 s56, 0
	v_add_u32_e64 v19, v19, s56
	v_mov_b32_e32 v24, v40
	v_mov_b32_e32 v25, v44
	v_mov_b32_e32 v26, v48
	v_mov_b32_e32 v27, v52
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v41
	v_mov_b32_e32 v25, v45
	v_mov_b32_e32 v26, v49
	v_mov_b32_e32 v27, v53
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v42
	v_mov_b32_e32 v25, v46
	v_mov_b32_e32 v26, v50
	v_mov_b32_e32 v27, v54
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v43
	v_mov_b32_e32 v25, v47
	v_mov_b32_e32 v26, v51
	v_mov_b32_e32 v27, v55
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_mov_b32_e32 v24, v56
	v_mov_b32_e32 v25, v60
	v_mov_b32_e32 v26, v64
	v_mov_b32_e32 v27, v68
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_mov_b32_e32 v24, v57
	v_mov_b32_e32 v25, v61
	v_mov_b32_e32 v26, v65
	v_mov_b32_e32 v27, v69
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_mov_b32_e32 v24, v58
	v_mov_b32_e32 v25, v62
	v_mov_b32_e32 v26, v66
	v_mov_b32_e32 v27, v70
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_mov_b32_e32 v24, v59
	v_mov_b32_e32 v25, v63
	v_mov_b32_e32 v26, v67
	v_mov_b32_e32 v27, v71
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:128
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:128
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v24, v72
	v_mov_b32_e32 v25, v76
	v_mov_b32_e32 v26, v80
	v_mov_b32_e32 v27, v84
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v73
	v_mov_b32_e32 v25, v77
	v_mov_b32_e32 v26, v81
	v_mov_b32_e32 v27, v85
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v74
	v_mov_b32_e32 v25, v78
	v_mov_b32_e32 v26, v82
	v_mov_b32_e32 v27, v86
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v75
	v_mov_b32_e32 v25, v79
	v_mov_b32_e32 v26, v83
	v_mov_b32_e32 v27, v87
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_mov_b32_e32 v24, v88
	v_mov_b32_e32 v25, v92
	v_mov_b32_e32 v26, v96
	v_mov_b32_e32 v27, v100
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_mov_b32_e32 v24, v89
	v_mov_b32_e32 v25, v93
	v_mov_b32_e32 v26, v97
	v_mov_b32_e32 v27, v101
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_mov_b32_e32 v24, v90
	v_mov_b32_e32 v25, v94
	v_mov_b32_e32 v26, v98
	v_mov_b32_e32 v27, v102
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_mov_b32_e32 v24, v91
	v_mov_b32_e32 v25, v95
	v_mov_b32_e32 v26, v99
	v_mov_b32_e32 v27, v103
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:256
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:384
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:256
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:384
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v24, v104
	v_mov_b32_e32 v25, v108
	v_mov_b32_e32 v26, v112
	v_mov_b32_e32 v27, v116
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v105
	v_mov_b32_e32 v25, v109
	v_mov_b32_e32 v26, v113
	v_mov_b32_e32 v27, v117
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v106
	v_mov_b32_e32 v25, v110
	v_mov_b32_e32 v26, v114
	v_mov_b32_e32 v27, v118
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v107
	v_mov_b32_e32 v25, v111
	v_mov_b32_e32 v26, v115
	v_mov_b32_e32 v27, v119
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_mov_b32_e32 v24, v120
	v_mov_b32_e32 v25, v124
	v_mov_b32_e32 v26, v128
	v_mov_b32_e32 v27, v132
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_mov_b32_e32 v24, v121
	v_mov_b32_e32 v25, v125
	v_mov_b32_e32 v26, v129
	v_mov_b32_e32 v27, v133
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_mov_b32_e32 v24, v122
	v_mov_b32_e32 v25, v126
	v_mov_b32_e32 v26, v130
	v_mov_b32_e32 v27, v134
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_mov_b32_e32 v24, v123
	v_mov_b32_e32 v25, v127
	v_mov_b32_e32 v26, v131
	v_mov_b32_e32 v27, v135
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:640
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:640
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v24, v136
	v_mov_b32_e32 v25, v140
	v_mov_b32_e32 v26, v144
	v_accvgpr_read_b32 v27, a216
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v137
	v_mov_b32_e32 v25, v141
	v_mov_b32_e32 v26, v145
	v_accvgpr_read_b32 v27, a217
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v138
	v_mov_b32_e32 v25, v142
	v_mov_b32_e32 v26, v146
	v_accvgpr_read_b32 v27, a218
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v139
	v_mov_b32_e32 v25, v143
	v_mov_b32_e32 v26, v147
	v_accvgpr_read_b32 v27, a219
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_accvgpr_read_b32 v24, a220
	v_accvgpr_read_b32 v25, a224
	v_accvgpr_read_b32 v26, a228
	v_accvgpr_read_b32 v27, a232
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_accvgpr_read_b32 v24, a221
	v_accvgpr_read_b32 v25, a225
	v_accvgpr_read_b32 v26, a229
	v_accvgpr_read_b32 v27, a233
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_accvgpr_read_b32 v24, a222
	v_accvgpr_read_b32 v25, a226
	v_accvgpr_read_b32 v26, a230
	v_accvgpr_read_b32 v27, a234
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_accvgpr_read_b32 v24, a223
	v_accvgpr_read_b32 v25, a227
	v_accvgpr_read_b32 v26, a231
	v_accvgpr_read_b32 v27, a235
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:768
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:896
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:768
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:896
	v_add_u32_e32 v18, 0x2000, v18
	s_mov_b32 s56, 0x10000
	v_add_u32_e64 v19, v19, s56
	v_mov_b32_e32 v24, v148
	v_mov_b32_e32 v25, v152
	v_mov_b32_e32 v26, v156
	v_mov_b32_e32 v27, v160
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v149
	v_mov_b32_e32 v25, v153
	v_mov_b32_e32 v26, v157
	v_mov_b32_e32 v27, v161
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v150
	v_mov_b32_e32 v25, v154
	v_mov_b32_e32 v26, v158
	v_mov_b32_e32 v27, v162
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v151
	v_mov_b32_e32 v25, v155
	v_mov_b32_e32 v26, v159
	v_mov_b32_e32 v27, v163
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_mov_b32_e32 v24, v164
	v_mov_b32_e32 v25, v168
	v_mov_b32_e32 v26, v172
	v_mov_b32_e32 v27, v176
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_mov_b32_e32 v24, v165
	v_mov_b32_e32 v25, v169
	v_mov_b32_e32 v26, v173
	v_mov_b32_e32 v27, v177
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_mov_b32_e32 v24, v166
	v_mov_b32_e32 v25, v170
	v_mov_b32_e32 v26, v174
	v_mov_b32_e32 v27, v178
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_mov_b32_e32 v24, v167
	v_mov_b32_e32 v25, v171
	v_mov_b32_e32 v26, v175
	v_mov_b32_e32 v27, v179
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:128
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:128
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v24, v180
	v_mov_b32_e32 v25, v184
	v_mov_b32_e32 v26, v188
	v_mov_b32_e32 v27, v192
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v181
	v_mov_b32_e32 v25, v185
	v_mov_b32_e32 v26, v189
	v_mov_b32_e32 v27, v193
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v182
	v_mov_b32_e32 v25, v186
	v_mov_b32_e32 v26, v190
	v_mov_b32_e32 v27, v194
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v183
	v_mov_b32_e32 v25, v187
	v_mov_b32_e32 v26, v191
	v_mov_b32_e32 v27, v195
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_mov_b32_e32 v24, v196
	v_mov_b32_e32 v25, v200
	v_mov_b32_e32 v26, v204
	v_mov_b32_e32 v27, v208
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_mov_b32_e32 v24, v197
	v_mov_b32_e32 v25, v201
	v_mov_b32_e32 v26, v205
	v_mov_b32_e32 v27, v209
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_mov_b32_e32 v24, v198
	v_mov_b32_e32 v25, v202
	v_mov_b32_e32 v26, v206
	v_mov_b32_e32 v27, v210
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_mov_b32_e32 v24, v199
	v_mov_b32_e32 v25, v203
	v_mov_b32_e32 v26, v207
	v_mov_b32_e32 v27, v211
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:256
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:384
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:256
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:384
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v24, v212
	v_mov_b32_e32 v25, v216
	v_mov_b32_e32 v26, v220
	v_mov_b32_e32 v27, v224
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v213
	v_mov_b32_e32 v25, v217
	v_mov_b32_e32 v26, v221
	v_mov_b32_e32 v27, v225
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v214
	v_mov_b32_e32 v25, v218
	v_mov_b32_e32 v26, v222
	v_mov_b32_e32 v27, v226
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v215
	v_mov_b32_e32 v25, v219
	v_mov_b32_e32 v26, v223
	v_mov_b32_e32 v27, v227
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_mov_b32_e32 v24, v228
	v_mov_b32_e32 v25, v232
	v_mov_b32_e32 v26, v236
	v_mov_b32_e32 v27, v240
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_mov_b32_e32 v24, v229
	v_mov_b32_e32 v25, v233
	v_mov_b32_e32 v26, v237
	v_mov_b32_e32 v27, v241
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_mov_b32_e32 v24, v230
	v_mov_b32_e32 v25, v234
	v_mov_b32_e32 v26, v238
	v_mov_b32_e32 v27, v242
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_mov_b32_e32 v24, v231
	v_mov_b32_e32 v25, v235
	v_mov_b32_e32 v26, v239
	v_mov_b32_e32 v27, v243
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:640
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:640
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v24, v244
	v_mov_b32_e32 v25, v248
	v_mov_b32_e32 v26, v252
	v_accvgpr_read_b32 v27, a236
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25]
	v_mov_b32_e32 v24, v245
	v_mov_b32_e32 v25, v249
	v_mov_b32_e32 v26, v253
	v_accvgpr_read_b32 v27, a237
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1152
	v_mov_b32_e32 v24, v246
	v_mov_b32_e32 v25, v250
	v_mov_b32_e32 v26, v254
	v_accvgpr_read_b32 v27, a238
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:144
	v_mov_b32_e32 v24, v247
	v_mov_b32_e32 v25, v251
	v_mov_b32_e32 v26, v255
	v_accvgpr_read_b32 v27, a239
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:1296
	v_accvgpr_read_b32 v24, a240
	v_accvgpr_read_b32 v25, a244
	v_accvgpr_read_b32 v26, a248
	v_accvgpr_read_b32 v27, a252
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2304
	v_accvgpr_read_b32 v24, a241
	v_accvgpr_read_b32 v25, a245
	v_accvgpr_read_b32 v26, a249
	v_accvgpr_read_b32 v27, a253
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3456
	v_accvgpr_read_b32 v24, a242
	v_accvgpr_read_b32 v25, a246
	v_accvgpr_read_b32 v26, a250
	v_accvgpr_read_b32 v27, a254
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:2448
	v_accvgpr_read_b32 v24, a243
	v_accvgpr_read_b32 v25, a247
	v_accvgpr_read_b32 v26, a251
	v_accvgpr_read_b32 v27, a255
	v_cmp_u_f32_e64 s[38:39], v24, v24
	v_add3_u32 v28, v24, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v25, v25
	v_add3_u32 v28, v25, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v24, v21, v20, s52
	v_cmp_u_f32_e64 s[38:39], v26, v26
	v_add3_u32 v28, v26, v31, 1
	v_cndmask_b32_e64 v20, v28, v30, s[38:39]
	v_cmp_u_f32_e64 s[38:39], v27, v27
	v_add3_u32 v28, v27, v31, 1
	v_cndmask_b32_e64 v21, v28, v30, s[38:39]
	v_perm_b32 v25, v21, v20, s52
	ds_write_b64 v5, v[24:25] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[40:41], v4
	ds_read_b64 v[44:45], v4 offset:64
	ds_read_b64 v[42:43], v4 offset:1152
	ds_read_b64 v[46:47], v4 offset:1216
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[48:49], v4 offset:2304
	ds_read_b64 v[52:53], v4 offset:2368
	ds_read_b64 v[50:51], v4 offset:3456
	ds_read_b64 v[54:55], v4 offset:3520
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:768
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:896
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:768
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:896
	v_add_u32_e32 v18, 0x2000, v18
	s_branch .Lr25_label_3059
	.Lr25_label_2CFA:
	s_mul_i32 s76, s67, s75
	s_add_u32 s56, s80, s79
	v_mov_b32_e32 v20, s56
	v_mul_lo_u32 v21, s76, v20
	v_mul_hi_u32 v22, s76, v20
	s_nop 2
	v_readfirstlane_b32 s56, v21
	v_readfirstlane_b32 s57, v22
	s_nop 4
	s_add_u32 s8, s56, s8
	s_addc_u32 s9, s57, s9
	s_sub_u32 s56, s81, s80
	s_mul_i32 s56, s56, s76
	s_mov_b32 s10, s56
	v_and_b32_e32 v20, 15, v0
	v_lshlrev_b32_e32 v18, 4, v20
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v20, 0x800, v20
	v_add_u32_e32 v18, v18, v20
	s_mul_i32 s56, s4, s75
	v_add_u32_e64 v18, v18, s56
	s_mul_i32 s56, s7, 0x8000
	v_add_u32_e64 v18, v18, s56
	v_mov_b32_e32 v19, v18
	s_mul_i32 s58, 4, s65
	s_mul_i32 s77, s67, s58
	s_add_u32 s56, s80, s79
	s_mul_i32 s56, s56, s77
	s_add_u32 s12, s56, s12
	s_addc_u32 s13, 0, s13
	s_sub_u32 s56, s81, s80
	s_mul_i32 s56, s56, s77
	s_mov_b32 s14, s56
	v_and_b32_e32 v26, 15, v0
	v_lshlrev_b32_e32 v26, 2, v26
	s_mul_i32 s56, s4, s58
	v_add_u32_e64 v26, v26, s56
	s_mul_i32 s56, s7, 64
	v_add_u32_e64 v26, v26, s56
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_lshlrev_b32_e32 v5, 2, v0
	s_mul_i32 s56, s7, 0x840
	v_add_u32_e32 v5, s56, v5
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshrrev_b32_e32 v20, 4, v0
	v_mul_i32_i24_e32 v4, 4, v20
	v_and_b32_e32 v20, 3, v0
	v_mul_i32_i24_e32 v20, 0x108, v20
	v_add_u32_e32 v4, v20, v4
	v_and_b32_e32 v20, 15, v0
	v_lshrrev_b32_e32 v20, 2, v20
	v_mul_i32_i24_e32 v20, 64, v20
	v_add_u32_e32 v4, v20, v4
	s_mul_i32 s56, s7, 0x840
	v_add_u32_e32 v4, s56, v4
	v_lshlrev_b32_e32 v4, 2, v4
	s_mov_b32 s56, 0
	v_add_u32_e64 v19, v19, s56
	v_mov_b32_e32 v20, v40
	v_mov_b32_e32 v21, v44
	v_mov_b32_e32 v22, v48
	v_mov_b32_e32 v23, v52
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v41
	v_mov_b32_e32 v21, v45
	v_mov_b32_e32 v22, v49
	v_mov_b32_e32 v23, v53
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v42
	v_mov_b32_e32 v21, v46
	v_mov_b32_e32 v22, v50
	v_mov_b32_e32 v23, v54
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v43
	v_mov_b32_e32 v21, v47
	v_mov_b32_e32 v22, v51
	v_mov_b32_e32 v23, v55
	ds_write_b128 v5, v[20:23] offset:3168
	v_mov_b32_e32 v20, v56
	v_mov_b32_e32 v21, v60
	v_mov_b32_e32 v22, v64
	v_mov_b32_e32 v23, v68
	ds_write_b128 v5, v[20:23] offset:4224
	v_mov_b32_e32 v20, v57
	v_mov_b32_e32 v21, v61
	v_mov_b32_e32 v22, v65
	v_mov_b32_e32 v23, v69
	ds_write_b128 v5, v[20:23] offset:5280
	v_mov_b32_e32 v20, v58
	v_mov_b32_e32 v21, v62
	v_mov_b32_e32 v22, v66
	v_mov_b32_e32 v23, v70
	ds_write_b128 v5, v[20:23] offset:6336
	v_mov_b32_e32 v20, v59
	v_mov_b32_e32 v21, v63
	v_mov_b32_e32 v22, v67
	v_mov_b32_e32 v23, v71
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v20, v72
	v_mov_b32_e32 v21, v76
	v_mov_b32_e32 v22, v80
	v_mov_b32_e32 v23, v84
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v73
	v_mov_b32_e32 v21, v77
	v_mov_b32_e32 v22, v81
	v_mov_b32_e32 v23, v85
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v74
	v_mov_b32_e32 v21, v78
	v_mov_b32_e32 v22, v82
	v_mov_b32_e32 v23, v86
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v75
	v_mov_b32_e32 v21, v79
	v_mov_b32_e32 v22, v83
	v_mov_b32_e32 v23, v87
	ds_write_b128 v5, v[20:23] offset:3168
	v_mov_b32_e32 v20, v88
	v_mov_b32_e32 v21, v92
	v_mov_b32_e32 v22, v96
	v_mov_b32_e32 v23, v100
	ds_write_b128 v5, v[20:23] offset:4224
	v_mov_b32_e32 v20, v89
	v_mov_b32_e32 v21, v93
	v_mov_b32_e32 v22, v97
	v_mov_b32_e32 v23, v101
	ds_write_b128 v5, v[20:23] offset:5280
	v_mov_b32_e32 v20, v90
	v_mov_b32_e32 v21, v94
	v_mov_b32_e32 v22, v98
	v_mov_b32_e32 v23, v102
	ds_write_b128 v5, v[20:23] offset:6336
	v_mov_b32_e32 v20, v91
	v_mov_b32_e32 v21, v95
	v_mov_b32_e32 v22, v99
	v_mov_b32_e32 v23, v103
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v20, v104
	v_mov_b32_e32 v21, v108
	v_mov_b32_e32 v22, v112
	v_mov_b32_e32 v23, v116
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v105
	v_mov_b32_e32 v21, v109
	v_mov_b32_e32 v22, v113
	v_mov_b32_e32 v23, v117
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v106
	v_mov_b32_e32 v21, v110
	v_mov_b32_e32 v22, v114
	v_mov_b32_e32 v23, v118
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v107
	v_mov_b32_e32 v21, v111
	v_mov_b32_e32 v22, v115
	v_mov_b32_e32 v23, v119
	ds_write_b128 v5, v[20:23] offset:3168
	v_mov_b32_e32 v20, v120
	v_mov_b32_e32 v21, v124
	v_mov_b32_e32 v22, v128
	v_mov_b32_e32 v23, v132
	ds_write_b128 v5, v[20:23] offset:4224
	v_mov_b32_e32 v20, v121
	v_mov_b32_e32 v21, v125
	v_mov_b32_e32 v22, v129
	v_mov_b32_e32 v23, v133
	ds_write_b128 v5, v[20:23] offset:5280
	v_mov_b32_e32 v20, v122
	v_mov_b32_e32 v21, v126
	v_mov_b32_e32 v22, v130
	v_mov_b32_e32 v23, v134
	ds_write_b128 v5, v[20:23] offset:6336
	v_mov_b32_e32 v20, v123
	v_mov_b32_e32 v21, v127
	v_mov_b32_e32 v22, v131
	v_mov_b32_e32 v23, v135
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v20, v136
	v_mov_b32_e32 v21, v140
	v_mov_b32_e32 v22, v144
	v_accvgpr_read_b32 v23, a216
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v137
	v_mov_b32_e32 v21, v141
	v_mov_b32_e32 v22, v145
	v_accvgpr_read_b32 v23, a217
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v138
	v_mov_b32_e32 v21, v142
	v_mov_b32_e32 v22, v146
	v_accvgpr_read_b32 v23, a218
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v139
	v_mov_b32_e32 v21, v143
	v_mov_b32_e32 v22, v147
	v_accvgpr_read_b32 v23, a219
	ds_write_b128 v5, v[20:23] offset:3168
	v_accvgpr_read_b32 v20, a220
	v_accvgpr_read_b32 v21, a224
	v_accvgpr_read_b32 v22, a228
	v_accvgpr_read_b32 v23, a232
	ds_write_b128 v5, v[20:23] offset:4224
	v_accvgpr_read_b32 v20, a221
	v_accvgpr_read_b32 v21, a225
	v_accvgpr_read_b32 v22, a229
	v_accvgpr_read_b32 v23, a233
	ds_write_b128 v5, v[20:23] offset:5280
	v_accvgpr_read_b32 v20, a222
	v_accvgpr_read_b32 v21, a226
	v_accvgpr_read_b32 v22, a230
	v_accvgpr_read_b32 v23, a234
	ds_write_b128 v5, v[20:23] offset:6336
	v_accvgpr_read_b32 v20, a223
	v_accvgpr_read_b32 v21, a227
	v_accvgpr_read_b32 v22, a231
	v_accvgpr_read_b32 v23, a235
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	s_mov_b32 s56, 0x100
	v_add_u32_e64 v26, v26, s56
	s_mov_b32 s56, 0x20000
	v_add_u32_e64 v19, v19, s56
	v_mov_b32_e32 v20, v148
	v_mov_b32_e32 v21, v152
	v_mov_b32_e32 v22, v156
	v_mov_b32_e32 v23, v160
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v149
	v_mov_b32_e32 v21, v153
	v_mov_b32_e32 v22, v157
	v_mov_b32_e32 v23, v161
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v150
	v_mov_b32_e32 v21, v154
	v_mov_b32_e32 v22, v158
	v_mov_b32_e32 v23, v162
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v151
	v_mov_b32_e32 v21, v155
	v_mov_b32_e32 v22, v159
	v_mov_b32_e32 v23, v163
	ds_write_b128 v5, v[20:23] offset:3168
	v_mov_b32_e32 v20, v164
	v_mov_b32_e32 v21, v168
	v_mov_b32_e32 v22, v172
	v_mov_b32_e32 v23, v176
	ds_write_b128 v5, v[20:23] offset:4224
	v_mov_b32_e32 v20, v165
	v_mov_b32_e32 v21, v169
	v_mov_b32_e32 v22, v173
	v_mov_b32_e32 v23, v177
	ds_write_b128 v5, v[20:23] offset:5280
	v_mov_b32_e32 v20, v166
	v_mov_b32_e32 v21, v170
	v_mov_b32_e32 v22, v174
	v_mov_b32_e32 v23, v178
	ds_write_b128 v5, v[20:23] offset:6336
	v_mov_b32_e32 v20, v167
	v_mov_b32_e32 v21, v171
	v_mov_b32_e32 v22, v175
	v_mov_b32_e32 v23, v179
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:256
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v20, v180
	v_mov_b32_e32 v21, v184
	v_mov_b32_e32 v22, v188
	v_mov_b32_e32 v23, v192
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v181
	v_mov_b32_e32 v21, v185
	v_mov_b32_e32 v22, v189
	v_mov_b32_e32 v23, v193
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v182
	v_mov_b32_e32 v21, v186
	v_mov_b32_e32 v22, v190
	v_mov_b32_e32 v23, v194
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v183
	v_mov_b32_e32 v21, v187
	v_mov_b32_e32 v22, v191
	v_mov_b32_e32 v23, v195
	ds_write_b128 v5, v[20:23] offset:3168
	v_mov_b32_e32 v20, v196
	v_mov_b32_e32 v21, v200
	v_mov_b32_e32 v22, v204
	v_mov_b32_e32 v23, v208
	ds_write_b128 v5, v[20:23] offset:4224
	v_mov_b32_e32 v20, v197
	v_mov_b32_e32 v21, v201
	v_mov_b32_e32 v22, v205
	v_mov_b32_e32 v23, v209
	ds_write_b128 v5, v[20:23] offset:5280
	v_mov_b32_e32 v20, v198
	v_mov_b32_e32 v21, v202
	v_mov_b32_e32 v22, v206
	v_mov_b32_e32 v23, v210
	ds_write_b128 v5, v[20:23] offset:6336
	v_mov_b32_e32 v20, v199
	v_mov_b32_e32 v21, v203
	v_mov_b32_e32 v22, v207
	v_mov_b32_e32 v23, v211
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:512
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:768
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v20, v212
	v_mov_b32_e32 v21, v216
	v_mov_b32_e32 v22, v220
	v_mov_b32_e32 v23, v224
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v213
	v_mov_b32_e32 v21, v217
	v_mov_b32_e32 v22, v221
	v_mov_b32_e32 v23, v225
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v214
	v_mov_b32_e32 v21, v218
	v_mov_b32_e32 v22, v222
	v_mov_b32_e32 v23, v226
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v215
	v_mov_b32_e32 v21, v219
	v_mov_b32_e32 v22, v223
	v_mov_b32_e32 v23, v227
	ds_write_b128 v5, v[20:23] offset:3168
	v_mov_b32_e32 v20, v228
	v_mov_b32_e32 v21, v232
	v_mov_b32_e32 v22, v236
	v_mov_b32_e32 v23, v240
	ds_write_b128 v5, v[20:23] offset:4224
	v_mov_b32_e32 v20, v229
	v_mov_b32_e32 v21, v233
	v_mov_b32_e32 v22, v237
	v_mov_b32_e32 v23, v241
	ds_write_b128 v5, v[20:23] offset:5280
	v_mov_b32_e32 v20, v230
	v_mov_b32_e32 v21, v234
	v_mov_b32_e32 v22, v238
	v_mov_b32_e32 v23, v242
	ds_write_b128 v5, v[20:23] offset:6336
	v_mov_b32_e32 v20, v231
	v_mov_b32_e32 v21, v235
	v_mov_b32_e32 v22, v239
	v_mov_b32_e32 v23, v243
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:1024
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:1280
	v_add_u32_e32 v18, 0x2000, v18
	v_mov_b32_e32 v20, v244
	v_mov_b32_e32 v21, v248
	v_mov_b32_e32 v22, v252
	v_accvgpr_read_b32 v23, a236
	ds_write_b128 v5, v[20:23]
	v_mov_b32_e32 v20, v245
	v_mov_b32_e32 v21, v249
	v_mov_b32_e32 v22, v253
	v_accvgpr_read_b32 v23, a237
	ds_write_b128 v5, v[20:23] offset:1056
	v_mov_b32_e32 v20, v246
	v_mov_b32_e32 v21, v250
	v_mov_b32_e32 v22, v254
	v_accvgpr_read_b32 v23, a238
	ds_write_b128 v5, v[20:23] offset:2112
	v_mov_b32_e32 v20, v247
	v_mov_b32_e32 v21, v251
	v_mov_b32_e32 v22, v255
	v_accvgpr_read_b32 v23, a239
	ds_write_b128 v5, v[20:23] offset:3168
	v_accvgpr_read_b32 v20, a240
	v_accvgpr_read_b32 v21, a244
	v_accvgpr_read_b32 v22, a248
	v_accvgpr_read_b32 v23, a252
	ds_write_b128 v5, v[20:23] offset:4224
	v_accvgpr_read_b32 v20, a241
	v_accvgpr_read_b32 v21, a245
	v_accvgpr_read_b32 v22, a249
	v_accvgpr_read_b32 v23, a253
	ds_write_b128 v5, v[20:23] offset:5280
	v_accvgpr_read_b32 v20, a242
	v_accvgpr_read_b32 v21, a246
	v_accvgpr_read_b32 v22, a250
	v_accvgpr_read_b32 v23, a254
	ds_write_b128 v5, v[20:23] offset:6336
	v_accvgpr_read_b32 v20, a243
	v_accvgpr_read_b32 v21, a247
	v_accvgpr_read_b32 v22, a251
	v_accvgpr_read_b32 v23, a255
	ds_write_b128 v5, v[20:23] offset:7392
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[40:43], v4
	ds_read_b128 v[44:47], v4 offset:64
	ds_read_b128 v[48:51], v4 offset:128
	ds_read_b128 v[52:55], v4 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[56:59], v4 offset:4224
	ds_read_b128 v[60:63], v4 offset:4288
	ds_read_b128 v[64:67], v4 offset:4352
	ds_read_b128 v[68:71], v4 offset:4416
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v18, v19
	buffer_store_dwordx4 v[40:43], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[56:59], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[60:63], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[48:51], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[64:67], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	buffer_store_dwordx4 v[52:55], v18, s[8:11], 0 offen offset:1536
	buffer_store_dwordx4 v[68:71], v18, s[8:11], 0 offen offset:1792
	v_add_u32_e32 v18, 0x2000, v18
	s_mov_b32 s56, 0x100
	v_add_u32_e64 v26, v26, s56
	.Lr25_label_3059:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"""


def _patch_native_normalize_and_pack(body):
    """Use gfx950 packed FP32 normalization and native BF16 conversion.

    The imported engine's selected s67=1 output path normalized 256 scalar
    values, wrote 40 normalized AGPR values back to the accumulator file, and
    then software-emulated every f32-to-bf16 conversion.  Keep the exact LDS
    write/read/store mapping, but normalize ordinary O in packed pairs and
    normalize AGPR O only after its final read at the corresponding pack site.
    """
    epilogue = body.index(".Lr25_label_2133:")
    normalize_start_token = "\tv_mul_f32_e64 v20, v12, s64\n"
    normalize_start = body.index(normalize_start_token, epilogue)
    output_branch = body.index("\ts_cmp_le_u32 s67, 1\n", normalize_start)
    old_normalize = body[normalize_start:output_branch]
    assert old_normalize.count("\tv_mul_f32_e32 v") == 256
    assert old_normalize.count("\tv_accvgpr_read_b32") == 40
    assert old_normalize.count("\tv_accvgpr_write_b32") == 40

    normalize = [
        "v_cndmask_b32_e64 v14, v14, 1.0, s[38:39]",
        "v_cndmask_b32_e64 v15, v15, 1.0, s[40:41]",
        "s_nop 1",
        "v_rcp_f32_e32 v14, v14",
        "v_rcp_f32_e32 v15, v15",
        "s_nop 1",
        "v_mov_b32_e32 v28, v14",
        "v_mov_b32_e32 v29, v14",
        "v_mov_b32_e32 v30, v15",
        "v_mov_b32_e32 v31, v15",
    ]
    for first, last, factor in ((40, 148, 28), (148, 256, 30)):
        for reg in range(first, last, 2):
            normalize.append(
                f"v_pk_mul_f32 v[{reg}:{reg + 1}], "
                f"v[{factor}:{factor + 1}], v[{reg}:{reg + 1}]"
            )
    new_normalize = "\n".join(f"\t{line}" for line in normalize) + "\n"
    body = (
        body[:normalize_start]
        + new_normalize
        + body[output_branch:]
    )

    selected_start = body.index(
        "\ts_cbranch_scc0 .Lr25_label_2CFA\n", epilogue
    )
    selected_end = body.index(
        "\ts_branch .Lr25_label_3059\n", selected_start
    )
    selected = body[selected_start:selected_end]
    assert selected.count("\tv_cmp_u_f32_e64") == 256
    assert selected.count("\tv_add3_u32") == 256
    assert selected.count("\tv_cndmask_b32_e64") == 256
    assert selected.count("\tv_perm_b32") == 128
    assert selected.count("\tds_write_b64 v5, v[24:25]") == 64
    assert selected.count("\tv_accvgpr_read_b32") == 40

    # The old three software-pack constants immediately precede the first
    # quartet.  v28:v31 now retain replicated reciprocal factors instead.
    for constant in (
        "\tv_mov_b32_e32 v29, 0xffff0000\n",
        "\tv_mov_b32_e32 v30, 0x7fff0000\n",
        "\tv_mov_b32_e32 v31, 0x7fff\n",
    ):
        assert selected.count(constant) == 1
        selected = selected.replace(constant, "", 1)

    lines = selected.splitlines()
    rewritten = []
    quartet_count = 0

    def parse_quartet(pos):
        sources = []
        for lane in range(4):
            line = lines[pos + lane]
            mov = re.fullmatch(
                rf"\tv_mov_b32_e32 v{24 + lane}, v(\d+)", line
            )
            acc = re.fullmatch(
                rf"\tv_accvgpr_read_b32 v{24 + lane}, a(\d+)", line
            )
            assert mov or acc, (pos, lane, line)
            sources.append(
                ("v", int(mov.group(1)))
                if mov
                else ("a", int(acc.group(1)))
            )
        assert lines[pos + 4].startswith("\tv_cmp_u_f32_e64")
        assert lines[pos + 17].startswith("\tv_perm_b32 v25")
        assert lines[pos + 18].startswith(
            "\tds_write_b64 v5, v[24:25]"
        )
        return sources, lines[pos + 18]

    def factor_for(agpr):
        assert 216 <= agpr <= 255
        return 28 if agpr < 236 else 30

    i = 0
    while i < len(lines):
        if not lines[i].startswith("\tv_mov_b32_e32 v24, v") and not (
            lines[i].startswith("\tv_accvgpr_read_b32 v24, a")
        ):
            rewritten.append(lines[i])
            i += 1
            continue

        first_sources, first_write = parse_quartet(i)
        second_sources, second_write = parse_quartet(i + 19)
        quartet_count += 2
        agpr_counts = (
            sum(kind == "a" for kind, _ in first_sources),
            sum(kind == "a" for kind, _ in second_sources),
        )
        assert agpr_counts in ((0, 0), (1, 1), (4, 4)), agpr_counts

        converted = [None, None]
        if agpr_counts == (1, 1):
            agpr_sources = [
                source
                for quartet in (first_sources, second_sources)
                for source in quartet
                if source[0] == "a"
            ]
            assert len(agpr_sources) == 2
            factors = {factor_for(reg) for _, reg in agpr_sources}
            assert len(factors) == 1
            factor = factors.pop()
            for scratch, (_, reg) in enumerate(agpr_sources, start=20):
                rewritten.append(
                    f"\tv_accvgpr_read_b32 v{scratch}, a{reg}"
                )
            rewritten.append(
                f"\tv_pk_mul_f32 v[20:21], "
                f"v[{factor}:{factor + 1}], v[20:21]"
            )
            scratch_iter = iter((20, 21))
            for q, sources in enumerate(
                (first_sources, second_sources)
            ):
                converted[q] = [
                    ("v", next(scratch_iter))
                    if kind == "a"
                    else (kind, reg)
                    for kind, reg in sources
                ]
        elif agpr_counts == (4, 4):
            for q, sources in enumerate(
                (first_sources, second_sources)
            ):
                factor = factor_for(sources[0][1])
                assert all(
                    kind == "a" and factor_for(reg) == factor
                    for kind, reg in sources
                )
                for scratch, (_, reg) in enumerate(sources, start=20):
                    rewritten.append(
                        f"\tv_accvgpr_read_b32 v{scratch}, a{reg}"
                    )
                rewritten.append(
                    f"\tv_pk_mul_f32 v[20:21], "
                    f"v[{factor}:{factor + 1}], v[20:21]"
                )
                rewritten.append(
                    f"\tv_pk_mul_f32 v[22:23], "
                    f"v[{factor}:{factor + 1}], v[22:23]"
                )
                converted[q] = [("v", reg) for reg in range(20, 24)]
                dst = 24 + 2 * q
                rewritten.append(
                    f"\tv_cvt_pk_bf16_f32 v{dst}, "
                    f"v{converted[q][0][1]}, v{converted[q][1][1]}"
                )
                rewritten.append(
                    f"\tv_cvt_pk_bf16_f32 v{dst + 1}, "
                    f"v{converted[q][2][1]}, v{converted[q][3][1]}"
                )
        else:
            converted = [first_sources, second_sources]

        if agpr_counts != (4, 4):
            for q, sources in enumerate(converted):
                dst = 24 + 2 * q
                assert all(kind == "v" for kind, _ in sources)
                rewritten.append(
                    f"\tv_cvt_pk_bf16_f32 v{dst}, "
                    f"v{sources[0][1]}, v{sources[1][1]}"
                )
                rewritten.append(
                    f"\tv_cvt_pk_bf16_f32 v{dst + 1}, "
                    f"v{sources[2][1]}, v{sources[3][1]}"
                )
        rewritten.append(first_write)
        rewritten.append(second_write.replace("v[24:25]", "v[26:27]", 1))
        i += 38

    assert quartet_count == 64
    selected = "\n".join(rewritten) + "\n"
    assert selected.count("\tv_cvt_pk_bf16_f32") == 128
    assert selected.count("\tv_pk_mul_f32") == 20
    assert selected.count("\tv_accvgpr_read_b32") == 40
    assert selected.count("\tv_accvgpr_write_b32") == 0
    assert selected.count("\tv_cmp_u_f32_e64") == 0
    assert selected.count("\tv_add3_u32") == 0
    assert selected.count("\tv_cndmask_b32_e64") == 0
    assert selected.count("\tv_perm_b32") == 0
    assert selected.count("\tds_write_b64 v5, v[24:25]") == 32
    assert selected.count("\tds_write_b64 v5, v[26:27]") == 32

    body = body[:selected_start] + selected + body[selected_end:]
    selected_epilogue = body[
        body.index(".Lr25_label_2133:"):
        body.index(".Lr25_label_2CFA:")
    ]
    assert selected_epilogue.count("\tv_pk_mul_f32") == 128
    assert selected_epilogue.count("\tv_cvt_pk_bf16_f32") == 128
    return body


_R25_FIXED_ENGINE_BODY = _patch_native_normalize_and_pack(
    _R25_FIXED_ENGINE_BODY
)


def _patch_threshold8_cold_site(
    body,
    cursor,
    limit_label,
    site,
    anchor,
    corr,
    last_agpr,
):
    """Split one eager correction site into a fallthrough fast path and cold rebase.

    The score and anchor registers are lane/head-local.  Only the branch
    decision is wave-uniform: if any of the wave's 32 heads exceeds eight
    log2-exponent units, every lane executes the original lane-local
    max/correction math.  This preserves exact per-head state on the rare
    path while the common path leaves the reference anchor and O/L state
    unchanged.  The cold block is returned separately so callers can place it
    beyond the static body's unconditional exit/backedge region.
    """
    limit = body.index(limit_label, cursor)
    start_token = "\tv_mov_b32_e32 v25, 0xff7fffff\n"
    start = body.index(start_token, cursor)
    assert start < limit, (site, start, limit)
    corr_exp = f"\tv_exp_f32_e32 v{corr}, v{corr}\n"
    corr_end = body.index(corr_exp, start) + len(corr_exp)
    scale_token = (
        f"\tv_mov_b32_e32 v22, v{corr}\n"
        f"\tv_mov_b32_e32 v23, v{corr}\n"
    )
    scale_start = body.index(scale_token, corr_end)
    scale_end_token = f"\tv_accvgpr_write_b32 a{last_agpr}, v21\n"
    scale_end = body.index(scale_end_token, scale_start) + len(scale_end_token)
    assert scale_end < limit, (site, scale_end, limit)

    original_corr = body[start:corr_end]
    shared_probability_and_feed = body[corr_end:scale_start]
    original_o_rebase = body[scale_start:scale_end]
    sum_reg = anchor + 2
    l_rebase = f"\tv_mul_f32_e32 v{sum_reg}, v{corr}, v{sum_reg}\n"
    assert shared_probability_and_feed.count(l_rebase) == 1, site
    fast_probability_and_feed = shared_probability_and_feed.replace(
        l_rebase, "", 1
    )

    slow_label = f".Lr31_lazy_slow_{site}"
    probability_label = f".Lr31_lazy_probability_{site}"
    hot = (
        f"\tv_sub_f32_e32 v25, v24, v{anchor}\n"
        "\tv_mul_f32_e32 v25, s5, v25\n"
        "\tv_cmp_nle_f32_e64 s[38:39], v25, s85\n"
        f"\tv_mul_f32_e32 v21, s5, v{anchor}\n"
        "\ts_or_b32 s38, s38, s39\n"
        f"\ts_cbranch_scc1 {slow_label}\n"
        f"\t{probability_label}:\n"
        + fast_probability_and_feed
    )
    cold = (
        f"\t{slow_label}:\n"
        + original_corr
        + f"\tv_mul_f32_e32 v21, s5, v{anchor}\n"
        + "\tv_mov_b32_e32 v25, v24\n"
        + original_o_rebase
        + l_rebase
        + f"\tv_mul_f32_e32 v21, s5, v{anchor}\n"
        + f"\ts_branch {probability_label}\n"
    )
    replacement = hot
    return (
        body[:start] + replacement + body[scale_end:],
        start + len(replacement),
        cold,
    )


_lazy_specs = (
    (12, 16, 235),
    (13, 17, 255),
    (12, 16, 235),
    (13, 17, 255),
)

# Waves 0/1 dispatch through 068D.  Patch from a cursor rooted in that body,
# then place its cold islands after the body's unconditional exit branch and
# before the duplicated 0F9D body.
_lazy_cursor = _R25_FIXED_ENGINE_BODY.index(".Lr25_label_068D:")
_lazy_cold = []
for _lazy_site, _lazy_spec in enumerate(_lazy_specs):
    _R25_FIXED_ENGINE_BODY, _lazy_cursor, _cold = _patch_threshold8_cold_site(
        _R25_FIXED_ENGINE_BODY,
        _lazy_cursor,
        ".Lr25_label_0F9D:",
        _lazy_site,
        *_lazy_spec,
    )
    _lazy_cold.append(_cold)
_body0_cold_anchor = (
    "\ts_branch .Lr25_label_18AA\n"
    "\t.Lr25_label_0F9D:\n"
)
assert _R25_FIXED_ENGINE_BODY.count(_body0_cold_anchor) == 1
_R25_FIXED_ENGINE_BODY = _R25_FIXED_ENGINE_BODY.replace(
    _body0_cold_anchor,
    "\ts_branch .Lr25_label_18AA\n"
    + "".join(_lazy_cold)
    + "\t.Lr25_label_0F9D:\n",
    1,
)

# Waves 2/3 dispatch through the duplicated 0F9D body.  Use a fresh cursor;
# its cold islands live after that body's unconditional backedge and before
# the common 18AA exit/tail region.
_lazy_cursor = _R25_FIXED_ENGINE_BODY.index(".Lr25_label_0F9D:")
_lazy_cold = []
for _local_site, _lazy_spec in enumerate(_lazy_specs):
    _lazy_site = 4 + _local_site
    _R25_FIXED_ENGINE_BODY, _lazy_cursor, _cold = _patch_threshold8_cold_site(
        _R25_FIXED_ENGINE_BODY,
        _lazy_cursor,
        ".Lr25_label_18AA:",
        _lazy_site,
        *_lazy_spec,
    )
    _lazy_cold.append(_cold)
_body1_cold_anchor = (
    "\ts_branch .Lr25_label_0F9D\n"
    "\t.Lr25_label_18AA:\n"
)
assert _R25_FIXED_ENGINE_BODY.count(_body1_cold_anchor) == 1
_R25_FIXED_ENGINE_BODY = _R25_FIXED_ENGINE_BODY.replace(
    _body1_cold_anchor,
    "\ts_branch .Lr25_label_0F9D\n"
    + "".join(_lazy_cold)
    + "\t.Lr25_label_18AA:\n",
    1,
)

assert _R25_FIXED_ENGINE_BODY.count("\t.Lr31_lazy_slow_") == 8
assert _R25_FIXED_ENGINE_BODY.count("\t.Lr31_lazy_probability_") == 8
assert _R25_FIXED_ENGINE_BODY.count(
    "\ts_cbranch_scc1 .Lr31_lazy_slow_"
) == 8
assert _R25_FIXED_ENGINE_BODY.count(
    "\ts_branch .Lr31_lazy_probability_"
) == 8
del (
    _body0_cold_anchor,
    _body1_cold_anchor,
    _cold,
    _lazy_cold,
    _lazy_cursor,
    _lazy_site,
    _local_site,
    _lazy_spec,
    _lazy_specs,
)

_R41_QK_X16_RE = re.compile(
    r"^\tv_mfma_f32_16x16x16_bf16 "
    r"v\[(32:35|36:39)\], "
    r"a\[(\d+):(\d+)\], a\[(\d+):(\d+)\], "
    r"(0|v\[(?:32:35|36:39)\])$"
)


def _patch_qk_x32_short_body(body):
    """Fuse each aligned pair of production QK x16 events into one x32 event.

    Non-MFMA lines stay byte-for-byte and in the same positions.  The fused
    event occupies the first old event.  A source-definition audit proves the
    full K/Q quartet is already defined there.  If an LDS wait occurs between
    old events, the chain is head 1 and the same K quartet has already been
    consumed by the immediately preceding head-0 chain, independently proving
    readiness before the wait.
    """
    lines = body.splitlines(keepends=True)
    events = []
    for index, line in enumerate(lines):
        match = _R41_QK_X16_RE.fullmatch(line.rstrip("\n"))
        if match:
            events.append((index, match))
    assert len(events) == 432

    chains = []
    for event in events:
        if int(event[1].group(2)) == 144:
            chains.append([])
        chains[-1].append(event)
    assert len(chains) == 12
    assert all(len(chain) == 36 for chain in chains)
    assert [chain[0][1].group(1) for chain in chains] == [
        "32:35", "36:39",
    ] * 6

    old_non_mfma = [line for line in lines if "v_mfma_" not in line]
    old_pv = [
        line
        for line in lines
        if "v_mfma_f32_16x16x16_bf16" in line
        and not _R41_QK_X16_RE.fullmatch(line.rstrip("\n"))
    ]
    assert len(old_pv) == 384

    reg_def_re = re.compile(
        r"(?:ds_read\w*|buffer_load\w*|v_accvgpr_write_b32)\s+"
        r"a(?:\[(\d+):(\d+)\]|(\d+))"
    )
    transformed = 0
    placement_proof = []
    for chain_index, chain in enumerate(chains):
        dest = chain[0][1].group(1)
        qbase = 0 if dest == "32:35" else 72
        for r in range(18):
            first_index, first = chain[2 * r]
            second_index, second = chain[2 * r + 1]
            expected_k0 = 144 + 4 * r
            expected_q0 = qbase + 4 * r
            assert tuple(map(int, first.group(2, 3, 4, 5))) == (
                expected_k0,
                expected_k0 + 1,
                expected_q0,
                expected_q0 + 1,
            )
            assert tuple(map(int, second.group(2, 3, 4, 5))) == (
                expected_k0 + 2,
                expected_k0 + 3,
                expected_q0 + 2,
                expected_q0 + 3,
            )
            assert first.group(1) == second.group(1) == dest

            source_regs = set(range(expected_k0, expected_k0 + 4))
            source_regs.update(range(expected_q0, expected_q0 + 4))
            intervening_defs = []
            intervening_waits = []
            intervening_dest_refs = []
            for line_index in range(first_index + 1, second_index):
                line = lines[line_index]
                for definition in reg_def_re.finditer(line):
                    lo = int(definition.group(1) or definition.group(3))
                    hi = int(definition.group(2) or definition.group(3))
                    if source_regs.intersection(range(lo, hi + 1)):
                        intervening_defs.append((line_index, line.rstrip()))
                if "s_waitcnt lgkmcnt" in line:
                    intervening_waits.append((line_index, line.rstrip()))
                for reg in range(
                    32 if dest == "32:35" else 36,
                    36 if dest == "32:35" else 40,
                ):
                    if re.search(rf"\bv{reg}\b", line):
                        intervening_dest_refs.append((line_index, line.rstrip()))
            assert not intervening_defs, (
                chain_index,
                r,
                intervening_defs,
            )
            assert not intervening_dest_refs, (
                chain_index,
                r,
                intervening_dest_refs,
            )
            if intervening_waits:
                # Only head-1 chains may cross an unrelated reduction/VT wait.
                # Their identical K quartet was consumed by the preceding
                # head-0 chain, hence it was ready before this head-1 event.
                assert dest == "36:39"
                prior = chains[chain_index - 1][2 * r][1]
                assert prior.group(1) == "32:35"
                assert prior.group(2, 3) == first.group(2, 3)

            c = "0" if r == 0 else f"v[{dest}]"
            lines[first_index] = (
                f"\tv_mfma_f32_16x16x32_bf16 v[{dest}], "
                f"a[{expected_k0}:{expected_k0 + 3}], "
                f"a[{expected_q0}:{expected_q0 + 3}], {c}\n"
            )
            lines[second_index] = ""
            transformed += 1
            placement_proof.append(
                (chain_index, r, first_index, second_index, len(intervening_waits))
            )

    result = "".join(lines)
    assert transformed == 216
    assert result.count("v_mfma_f32_16x16x32_bf16") == 216
    assert len(_R41_QK_X16_RE.findall(result)) == 0
    assert result.count("v_mfma_f32_16x16x16_bf16") == 384
    assert result.count("v_mfma_f32_") == 600
    new_lines = result.splitlines(keepends=True)
    assert [line for line in new_lines if "v_mfma_" not in line] == old_non_mfma
    new_pv = [
        line
        for line in new_lines
        if "v_mfma_f32_16x16x16_bf16" in line
    ]
    assert new_pv == old_pv
    assert len(placement_proof) == 216
    return result


_R41_QK_X32_SHORT_BODY = _patch_qk_x32_short_body(
    _R25_FIXED_ENGINE_BODY
)



def _r43_software_probability_pack_block(base):
    return "\n".join([
        "\tv_mov_b32_e32 v29, 0xffff0000",
        "\tv_mov_b32_e32 v30, 0x7fff0000",
        "\tv_mov_b32_e32 v31, 0x7fff",
        f"\tv_cmp_u_f32_e64 s[38:39], v{base}, v{base}",
        f"\tv_add3_u32 v28, v{base}, v31, 1",
        "\tv_cndmask_b32_e64 v20, v28, v30, s[38:39]",
        f"\tv_cmp_u_f32_e64 s[38:39], v{base + 1}, v{base + 1}",
        f"\tv_add3_u32 v28, v{base + 1}, v31, 1",
        "\tv_cndmask_b32_e64 v21, v28, v30, s[38:39]",
        f"\tv_perm_b32 v{base}, v21, v20, s52",
        f"\tv_cmp_u_f32_e64 s[38:39], v{base + 2}, v{base + 2}",
        f"\tv_add3_u32 v28, v{base + 2}, v31, 1",
        "\tv_cndmask_b32_e64 v20, v28, v30, s[38:39]",
        f"\tv_cmp_u_f32_e64 s[38:39], v{base + 3}, v{base + 3}",
        f"\tv_add3_u32 v28, v{base + 3}, v31, 1",
        "\tv_cndmask_b32_e64 v21, v28, v30, s[38:39]",
        f"\tv_perm_b32 v{base + 1}, v21, v20, s52",
    ])


def _patch_native_probability_pack_r43(body):
    """Use native BF16 packing while preserving every surrounding line."""
    original_lines = body.splitlines(keepends=True)
    result = body
    replacements = 0
    for base in (32, 36):
        old = _r43_software_probability_pack_block(base)
        expected = 6
        actual = result.count(old)
        assert actual == expected, (base, actual, expected)
        new = "\n".join([
            f"\tv_cvt_pk_bf16_f32 v{base}, v{base}, v{base + 1}",
            f"\tv_cvt_pk_bf16_f32 v{base + 1}, v{base + 2}, v{base + 3}",
        ])
        result = result.replace(old, new)
        replacements += actual

    assert replacements == 12
    assert result.count("\tv_cvt_pk_bf16_f32") == 152
    assert result.count("\tv_mfma_f32_16x16x32_bf16") == 216
    assert result.count("\tv_mfma_f32_16x16x16_bf16") == 384
    assert result.count("\tv_mfma_f32_") == 600

    # Exactly the twelve 17-op software blocks were replaced by 2-op blocks.
    new_lines = result.splitlines(keepends=True)
    assert len(original_lines) - len(new_lines) == 12 * 15
    old_without_pack = [
        line
        for line in original_lines
        if line not in {
            "\tv_mov_b32_e32 v29, 0xffff0000\n",
            "\tv_mov_b32_e32 v30, 0x7fff0000\n",
            "\tv_mov_b32_e32 v31, 0x7fff\n",
        }
    ]
    # The stronger emitted-ISA normalized diff is performed after compiling;
    # here the exact block count and surrounding string replacement guarantee
    # source locality without disturbing the retained trailing s_nop 2.
    assert len(old_without_pack) == len(original_lines) - 36
    return result


_R43_NATIVE_PROB_PACK_SHORT_BODY = _patch_native_probability_pack_r43(
    _R41_QK_X32_SHORT_BODY
)



def whole_loop_fixed_clobber(
    tid,
    q_token,
    out_rsrc,
    query_rsrc,
    kv_rsrc,
    index_rsrc,
    sm_scale,
    seq_len,
    kv_base,
    lds_base,
    preserve_fixed_inputs=False,
    persistent_total_q=None,
):
    """Execute the complete fixed-register absorbed-MLA prefill engine."""
    if preserve_fixed_inputs and persistent_total_q is not None:
        raise ValueError("persistent assembly owns fixed-input preservation")

    prelude = [
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_lshrrev_b32 v3, 6, v0",
        "v_and_b32 v0, 63, v0",
        "v_readfirstlane_b32 s7, v3",
        "s_mov_b32 s3, 0",
        "s_mov_b32 s4, 0",
        "s_mov_b32 s65, 128",
        "s_mov_b32 s67, 1",
        "s_mov_b32 s68, 1152",
        "s_mov_b32 s69, 0",
        "s_mov_b32 s74, 147456",
        "s_mov_b32 s78, s46",
        "s_mov_b32 s79, 0",
        "s_mov_b32 s85, 0x41000000",
        "s_mov_b32 m0, s84",
    ]
    persistent_init = []
    postlude = []
    if persistent_total_q is not None:
        # The high scalar window survives the imported per-token engine:
        #   s86 workgroup i, s87 pass, s88 total_q,
        #   s89:s94 pristine descriptor bases, s95 pristine LDS base.
        # Descriptor size/config dwords are invariant or rebuilt by the token
        # body.  Keeping the pass loop in this one asm avoids a compiler-
        # visible fixed-register boundary and emits one 816-MFMA engine body.
        persistent_init = [
            "s_mov_b32 s86, s2",
            "s_mov_b32 s87, 0",
            "s_mov_b32 s88, s3",
            "s_mov_b32 s89, s8",
            "s_mov_b32 s90, s9",
            "s_mov_b32 s91, s16",
            "s_mov_b32 s92, s17",
            "s_mov_b32 s93, s24",
            "s_mov_b32 s94, s25",
            "s_mov_b32 s95, s84",
            ".Lr35_persistent_token:",
        ]
        postlude = [
            # The body reaches here only after its full VMEM/EXP/LDS drain.
            # Compute the next token before synchronizing so the final pass
            # exits without the former unconditional barrier.
            "s_add_u32 s87, s87, 1",
            "s_min_u32 s58, s88, 0x100",
            "s_mul_i32 s56, s87, s58",
            "s_sub_u32 s57, s58, 1",
            "s_sub_u32 s57, s57, s86",
            "s_and_b32 s58, s87, 1",
            "s_cmp_eq_u32 s58, 0",
            "s_cselect_b32 s2, s86, s57",
            "s_add_u32 s2, s56, s2",
            "s_cmp_lt_u32 s2, s88",
            "s_cbranch_scc0 .Lr35_persistent_done",
            "s_barrier",
            # Restore the bases mutated by the token engine.  The third
            # descriptor dwords are reconstructed before their first use.
            "s_mov_b32 s8, s89",
            "s_mov_b32 s9, s90",
            "s_mov_b32 s10, -1",
            "s_mov_b32 s16, s91",
            "s_mov_b32 s17, s92",
            "s_mov_b32 s18, -1",
            "s_mov_b32 s24, s93",
            "s_mov_b32 s25, s94",
            "s_mov_b32 s26, -1",
            "s_mov_b32 s84, s95",
            # v0 remains lane_id; s7 remains the original wave id.
            "v_mov_b32_e32 v1, s7",
            "v_lshlrev_b32_e32 v1, 6, v1",
            "v_add_u32_e32 v0, v0, v1",
            "s_branch .Lr35_persistent_token",
            ".Lr35_persistent_done:",
        ]
    elif preserve_fixed_inputs:
        # The imported single-query engine advances the fixed output/query/
        # page-index resource descriptors in place.  A single invocation did
        # not need to expose that implementation detail to LLVM, but the
        # persistent short route re-enters the same asm from one SCF backedge.
        # Restore the descriptors before crossing that boundary so the
        # read-only fixed-input contract remains true.  s66 is dead throughout
        # the engine and retains the original shared-memory base.
        prelude.insert(0, "s_mov_b32 s66, s84")
        postlude = [
            "s_mul_i32 s56, s2, s74",
            "s_sub_u32 s16, s16, s56",
            "s_subb_u32 s17, s17, 0",
            "s_mov_b32 s18, -1",
            "s_lshl_b32 s56, s2, 17",
            "s_sub_u32 s8, s8, s56",
            "s_subb_u32 s9, s9, 0",
            "s_mov_b32 s10, -1",
            "s_lshl_b32 s56, s47, 2",
            "s_sub_u32 s24, s24, s56",
            "s_subb_u32 s25, s25, 0",
            "s_mov_b32 s26, -1",
            "s_mov_b32 s84, s66",
            "v_mov_b32_e32 v1, s7",
            "v_lshlrev_b32_e32 v1, 6, v1",
            "v_add_u32_e32 v0, v0, v1",
        ]
    engine_body = (
        _R43_NATIVE_PROB_PACK_SHORT_BODY
        if persistent_total_q is not None
        else _R25_FIXED_ENGINE_BODY
    )
    fixed_s = {
        2,
        *range(8, 12),
        *range(16, 20),
        *range(20, 24),
        *range(24, 28),
        46,
        47,
        64,
        84,
    }
    if persistent_total_q is not None:
        fixed_s.add(3)
    used_s = set(map(int, re.findall(r"\bs(\d+)\b", engine_body)))
    for match in re.finditer(r"\bs\[(\d+):(\d+)\]", engine_body):
        used_s.update(range(int(match.group(1)), int(match.group(2)) + 1))

    clobbers = [f"~{{v{i}}}" for i in range(1, 256)]
    clobbers += [f"~{{a{i}}}" for i in range(256)]
    clobbers += [
        f"~{{s{i}}}" for i in sorted(used_s - fixed_s) if i <= 85
    ]
    # The per-call prelude initializes s3 even though the imported body does
    # not otherwise mention it.  Declare that write so an enclosing persistent
    # loop cannot allocate its forward-query induction state in s3.
    if 3 not in used_s and persistent_total_q is None:
        clobbers += ["~{s3}"]
    if preserve_fixed_inputs:
        clobbers += ["~{s66}"]
    if persistent_total_q is not None:
        clobbers += [f"~{{s{i}}}" for i in range(86, 96)]
    clobbers += ["~{scc}", "~{vcc}", "~{memory}"]
    constraints = [
        "{v0}",
        "{s2}",
        *(["{s3}"] if persistent_total_q is not None else []),
        "{s[8:11]}",
        "{s[16:19]}",
        "{s[20:23]}",
        "{s[24:27]}",
        "{s64}",
        "{s46}",
        "{s47}",
        "{s84}",
    ]
    return llvm.inline_asm(
        None,
        [
            _raw(fx.Int32(tid)),
            _raw(fx.Int32(q_token)),
            *(
                [_raw(fx.Int32(persistent_total_q))]
                if persistent_total_q is not None
                else []
            ),
            _raw(out_rsrc),
            _raw(query_rsrc),
            _raw(kv_rsrc),
            _raw(index_rsrc),
            _raw(fx.Float32(sm_scale)),
            _raw(fx.Int32(seq_len)),
            _raw(fx.Int32(kv_base)),
            _raw(fx.Int32(lds_base)),
        ],
        "\n\t".join(persistent_init + prelude)
        + "\n\t"
        + engine_body
        + ("\n\t" + "\n\t".join(postlude) if postlude else ""),
        ",".join(constraints + clobbers),
        has_side_effects=True,
    )
