--筛选候选调整规则
with base_rule as
(
	select
		t1.product_no,t1.apply_dt,t1.apply_id,t1.refuse_rule_log,t1.rule
		,max(case when t2.model_no='XW_PBOC_013'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as XW_PBOC_013
		,max(case when t2.model_no='GR_A_YSD_PBOC_2202_001'then cast(get_json_object(t2.outputs,"$.final_score") as double) end) as ysd_pboc_model
		
		,max(case when t2.model_no='GR_PBOC_005'then cast(get_json_object(outputs,"$.final_socre") as double) end ) as GR_PBOC_005
		,max(case when t2.model_no='GR_SF_018'then cast(get_json_object(outputs,"$.final_score") as double) end ) as GR_SF_018
		,max(case when t2.model_no='GR_SF_022' then cast(get_json_object(outputs,"$.final_score") as double) end ) as GR_SF_022
		,max(case when t2.model_no='GR-PBOC-009'then cast(get_json_object(outputs,"$.final_score") as double) end ) as GR_PBOC_009
		,max(case when t2.model_no='GR_A_WP_PBOC_2203_001'then cast(get_json_object(outputs,"$.final_score") as double) end) as GR_A_WP_PBOC_2203_001
	from
	(
		select distinct
			product_no,substr(create_date,1,7) as apply_dt
			,apply_id,refuse_rule_log
			,split(refuse_rule_log,',')[0] as rule
		from risk.scene_log_rule_result
		where product_no in ('8010110182')
		and scene in ('060401','050401') --('060501','050501')
		and create_date>'2022-09-05 09:10:00'  --优酷第一次通过率提升上线
	)t1
	left join risk_decision.v_sp_risk_extinfo_risk_apply_center_model_score_info_a t2
	on t1.apply_id=t2.client_flow_no
	group by t1.product_no,t1.apply_dt,t1.apply_id,t1.refuse_rule_log,t1.rule
)
select
	t1.product_no,t1.apply_dt,t1.rule
	,t1.cnt,t2.apply_cnt
	,t1.cnt/t2.apply_cnt as trg_rate
	,t1.xw_pboc_013,t1.ysd_pboc_model,t1.gr_pboc_005,t1.gr_sf_018,t1.gr_sf_022,t1.gr_pboc_009,t1.wp_pboc_001
	,t2.xw_pboc_013,t2.ysd_pboc_model,t2.gr_pboc_005,t2.gr_sf_018,t2.gr_sf_022,t2.gr_pboc_009,t2.wp_pboc_001
	,t2.passed_avg_pboc_005,t2.passed_avg_sf_018,t2.passed_avg_sf_022,t2.passed_avg_pboc_009,t2.passed_avg_pboc_001
	,t3.cnt as sig_trg
	,t3.cnt/t2.apply_cnt as sig_rate
from
(	--规则触碰量
	select
		product_no,apply_dt,vt.rule
		,count(distinct apply_id) as cnt
		,avg(XW_PBOC_013) as xw_pboc_013
		,avg(ysd_pboc_model) as ysd_pboc_model

		,avg(GR_PBOC_005) as gr_pboc_005
		,avg(if(GR_SF_018>=0,GR_SF_018,0)) as gr_sf_018
		,avg(if(GR_SF_022>=0,GR_SF_018,0)) as gr_sf_022
		,avg(GR_PBOC_009) as gr_pboc_009
		,avg(GR_A_WP_PBOC_2203_001) as wp_pboc_001
	from base_rule
	lateral view explode(split(refuse_rule_log,',')) vt as rule
	group by product_no,apply_dt,vt.rule
)t1
left join
(	--总申请
	select
		product_no,apply_dt
		,count(distinct apply_id) as apply_cnt
		,avg(XW_PBOC_013) as xw_pboc_013
		,avg(ysd_pboc_model) as ysd_pboc_model
		
		,avg(GR_PBOC_005) as gr_pboc_005
		,avg(if(GR_SF_018>=0,GR_SF_018,0)) as gr_sf_018
		,avg(if(GR_SF_022>=0,GR_SF_018,0)) as gr_sf_022
		,avg(GR_PBOC_009) as gr_pboc_009
		,avg(GR_A_WP_PBOC_2203_001) as wp_pboc_001

		,avg(if(refuse_rule_log=='', GR_PBOC_005, null)) as passed_avg_pboc_005
		,avg(if(refuse_rule_log=='', if(GR_SF_018>=0,GR_SF_018,0), null)) as passed_avg_sf_018
		,avg(if(refuse_rule_log=='', if(GR_SF_022>=0,GR_SF_022,0), null)) as passed_avg_sf_022
		,avg(if(refuse_rule_log=='', GR_PBOC_009, null)) as passed_avg_pboc_009
		,avg(if(refuse_rule_log=='', GR_A_WP_PBOC_2203_001, null)) as passed_avg_pboc_001
	from base_rule
	group by product_no,apply_dt
)t2
on t1.product_no=t2.product_no and t1.apply_dt=t2.apply_dt
left join
(
	select
		product_no,apply_dt,rule
		,count(distinct apply_id) as cnt
	from base_rule
	where size(split(refuse_rule_log,','))<=1
	group by product_no,apply_dt,rule
)t3
on t1.product_no=t3.product_no and t1.apply_dt=t3.apply_dt and t1.rule=t3.rule





with base_rule as
(
	select
		t1.product_no,t1.apply_dt,t1.apply_id,t1.refuse_rule_log,t1.rule
		,max(case when t2.model_no='XW_PBOC_013'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as XW_PBOC_013
		,max(case when t2.model_no='GR_A_YSD_PBOC_2202_001'then cast(get_json_object(t2.outputs,"$.final_score") as double) end) as ysd_pboc_model
	from
	(
		select distinct
			product_no,substr(create_date,1,7) as apply_dt
			,apply_id,refuse_rule_log
			,split(refuse_rule_log,',')[0] as rule
		from risk.scene_log_rule_result
		where product_no in ('8010110182')
		and scene in ('060401','050401') --('060501','050501')
		and create_date>'2022-09-05 09:10:00'  --优酷第一次通过率提升上线
	)t1
	left join risk_decision.v_sp_risk_extinfo_risk_apply_center_model_score_info_a t2
	on t1.apply_id=t2.client_flow_no
	group by t1.product_no,t1.apply_dt,t1.apply_id,t1.refuse_rule_log,t1.rule
)
select
	t1.product_no,t1.apply_dt,t1.refuse_rule_log
	,t1.cnt,t2.apply_cnt
	,t1.cnt/t2.apply_cnt as trg_rate
	,t1.xw_pboc_013,t1.ysd_pboc_model
	,t2.xw_pboc_013,t2.ysd_pboc_model
from
(	--规则触碰量
	select
		product_no,apply_dt,refuse_rule_log
		,count(distinct apply_id) as cnt
		,avg(XW_PBOC_013) as xw_pboc_013
		,avg(ysd_pboc_model) as ysd_pboc_model
	from base_rule
	group by product_no,apply_dt,refuse_rule_log
)t1
left join
(	--总申请
	select
		product_no,apply_dt
		,count(distinct apply_id) as apply_cnt
		,avg(XW_PBOC_013) as xw_pboc_013
		,avg(ysd_pboc_model) as ysd_pboc_model
	from base_rule
	group by product_no,apply_dt
)t2
on t1.product_no=t2.product_no and t1.apply_dt=t2.apply_dt







--申请apply_id规则逻辑变量明细
drop table if exists risk_analysis.jc_temp_youku_var_resize20220907;
create table risk_analysis.jc_temp_youku_var_resize20220907 as
with base_rule as 
(
	select distinct
		product_no,apply_id,refuse_rule_log
		,split(refuse_rule_log,',')[0] as rule
		,substr(create_date,1,7) as date_bin
	from risk.scene_log_rule_result
	where product_no in ('8010110182')
	and scene in ('060401','050401')
	and create_date>'2022-09-05 09:10:00'
)
-- ,var_base as (
	select
		t1.product_no,t1.apply_id,t1.date_bin,t1.refuse_rule_log
		--单一触碰高
		-- ,t3.bj_tz_app_use_num_m12
		-- ,max(case when t2.model_no='GR_SF_009'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as gr_sf_009
		--合并触碰高
		,t4.rh_m3_lnaprv_allorgcnt
		,max(case when t2.model_no='GR_PBOC_005'then cast(get_json_object(t2.outputs,"$.final_socre") as double) end ) as gr_pboc_005
		,t5.br_als_m3_id_nbank_allnum
		,t6.homeln_cur_cnt
		,max(case when t2.model_no='GR_SF_018'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as gr_sf_018
		,t7.rh_card_accnum
		,max(case when t2.model_no='XW_PBOC_013'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as XW_PBOC_013
		,max(case when t2.model_no='GR-PBOC-009'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as gr_pboc_009
		,t5.br_als_m12_id_nbank_avg_monnum
		,max(case when t2.model_no='GR_SF_022'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as GR_SF_022
		,t5.br_als_m6_id_nbank_orgnum,t5.br_als_m12_cell_bank_allnum,t5.br_als_m12_id_nbank_orgnum,t5.br_apply_loan_str_flag
		,t5.br_als_m1_id_nbank_allnum
		,t8.rh_d15_lnaprv_allorgcnt
		,t5.br_apply3m_id_orgnum
		,t4.rh_querycnt_1m
		,max(case when t2.model_no='GR_A_YSD_PBOC_2202_001'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as ysd_pboc_model
		,t8.rh_d15_crdtcrd_lnaprv_enqrcnt
		,max(case when t2.model_no='GR_SF_009'then cast(get_json_object(t2.outputs,"$.final_score") as double) end ) as GR_SF_009
		,max(case when t2.model_no='GR_A_WP_PBOC_2203_001'then cast(get_json_object(outputs,"$.final_score") as double) end) as GR_A_WP_PBOC_2203_001
	from base_rule t1
	left join risk_decision.v_sp_risk_extinfo_risk_apply_center_model_score_info_a t2
	on t1.apply_id=t2.client_flow_no
	left join risk_decision.data_standard_grpintrfccd_extinfo_sp_risk_var_risk_var_calculate_result_i_ice___realtimeservice t3
	on t1.apply_id=t3.apply_id
	left join risk_decision.data_standard_grpintrfccd_extinfo_sp_risk_var_risk_var_calculate_result_i_rh___creditreport_fifth t4
	on t1.apply_id=t4.apply_id
	left join risk_decision.data_standard_grpintrfccd_extinfo_sp_risk_var_risk_var_calculate_result_i_bsapi___applyloanstr t5
	on t1.apply_id=t5.apply_id
	left join risk_decision.data_standard_grpintrfccd_extinfo_sp_risk_var_risk_var_calculate_result_i_rh___creditreport t6
	on t1.apply_id=t6.apply_id
	left join risk_decision.data_standard_grpintrfccd_extinfo_sp_risk_var_risk_var_calculate_result_i_rh___creditreport_third t7
	on t1.apply_id=t7.apply_id
	left join risk_decision.data_standard_grpintrfccd_extinfo_sp_risk_var_risk_var_calculate_result_i_rh___creditreport_forth t8
	on t1.apply_id=t8.apply_id

	-- where size(split(t1.refuse_rule_log,','))<=1
	group by t1.product_no,t1.apply_id,t1.date_bin,t1.refuse_rule_log
		,t4.rh_m3_lnaprv_allorgcnt
		,t5.br_als_m3_id_nbank_allnum
		,t6.homeln_cur_cnt
		,t7.rh_card_accnum
		,t5.br_als_m12_id_nbank_avg_monnum
		,t5.br_als_m6_id_nbank_orgnum,t5.br_als_m12_cell_bank_allnum,t5.br_als_m12_id_nbank_orgnum,t5.br_apply_loan_str_flag
		,t5.br_als_m1_id_nbank_allnum
		,t8.rh_d15_lnaprv_allorgcnt
		,t5.br_apply3m_id_orgnum
		,t4.rh_querycnt_1m
		,t8.rh_d15_crdtcrd_lnaprv_enqrcnt
-- )




--调整后规则触碰: 新旧两版规则中，重复的规则以调整后的为准，统计通过率。
select
	-- 调整前后通过率变动
	-- t1.date_bin
	-- ,count(t1.apply_id) as total_cnt
	-- ,sum(if(t1.refuse_rule_log is null or t1.refuse_rule_log='',1,0)) as old_pass
	-- ,sum(if(t1.refuse_rule_log is null or t1.refuse_rule_log='',1,0))/count(t1.apply_id) as old_pass_rate
	-- -- ,sum(if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_PBOC_004+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_021_002+t1.OR_C_PBOC_003_002+t1.OR_C_BR_MULT_010+t1.OR_C_PBOC_030+t1.OR_C_PBOC_027+t1.OR_C_PBOC_026+t1.OR_C_BJ_TZ_002>0,0,1))/count(t1.apply_id) as pass_rate
	-- -- ,sum(if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_010+t1.OR_C_BJ_TZ_002=0,0,1)) as new_pass
	-- -- ,sum(if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_010+t1.OR_C_BJ_TZ_002=0,0,1))/count(t1.apply_id) as pass_rate
	-- ,sum(if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_PBOC_004+t1.OR_C_BR_MULT_021_002+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_020+t1.OR_C_PBOC_003_002+t1.OR_C_BR_MULT_010+t1.OR_C_PBOC_027+t1.OR_C_PBOC_030+t1.OR_C_BR_MULT_017+t1.OR_C_PBOC_044_002=0,1,0)) as new_pass
	-- ,sum(if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_PBOC_004+t1.OR_C_BR_MULT_021_002+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_020+t1.OR_C_PBOC_003_002+t1.OR_C_BR_MULT_010+t1.OR_C_PBOC_027+t1.OR_C_PBOC_030+t1.OR_C_BR_MULT_017+t1.OR_C_PBOC_044_002=0,1,0))/count(t1.apply_id) as pass_rate


	-- --调整前后通过拒绝单量分布
	t1.date_bin
	,if(t1.refuse_rule_log is null or t1.refuse_rule_log='', 'Y', 'N') as old_pass
	-- ,if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_010+t1.OR_C_PBOC_027+t1.OR_C_BJ_TZ_002>0,'N','Y') as new_pass
	,if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_PBOC_004+t1.OR_C_BR_MULT_021_002+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_020+t1.OR_C_PBOC_003_002+t1.OR_C_BR_MULT_010+t1.OR_C_PBOC_027+t1.OR_C_PBOC_030+t1.OR_C_BR_MULT_017+t1.OR_C_PBOC_044_002>0,'N','Y') as new_pass
	,count(t1.apply_id) as cnt
	,avg(XW_PBOC_013) as avg_xw_pboc_013
	,avg(ysd_pboc_model) as avg_ysd_pboc_model


	-- -- --调整的规则：调整后触发量、单一触发量
	-- t1.date_bin
	-- ,count(t1.apply_id) as apply_cnt
	-- ,sum(if(t1.OR_C_PBOC_031>0,1,0)) as OR_C_PBOC_031_tol
	-- ,sum(if(t1.OR_C_PBOC_031>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_PBOC_031_sig
	-- ,sum(if(t1.OR_C_PBOC_004>0,1,0)) as OR_C_PBOC_004_tol
	-- ,sum(if(t1.OR_C_PBOC_004>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_PBOC_004_sig
	-- ,sum(if(t1.OR_C_BR_MULT_021_002>0,1,0)) as OR_C_BR_MULT_021_002_tol
	-- ,sum(if(t1.OR_C_BR_MULT_021_002>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_BR_MULT_021_002_sig
	-- ,sum(if(t1.OR_C_BR_MULT_009>0,1,0)) asOR_C_BR_MULT_009_tol
	-- ,sum(if(t1.OR_C_BR_MULT_009>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_BR_MULT_009_sig
	-- ,sum(if(t1.OR_C_BR_MULT_020>0,1,0)) as OR_C_BR_MULT_020_tol
	-- ,sum(if(t1.OR_C_BR_MULT_020>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_BR_MULT_020_sig
	-- ,sum(if(t1.OR_C_PBOC_003_002>0,1,0)) as OR_C_PBOC_003_002_tol
	-- ,sum(if(t1.OR_C_PBOC_003_002>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_PBOC_003_002_sig
	-- -- ,sum(if(t1.OR_C_BR_MULT_012>0,1,0)) as OR_C_BR_MULT_012_tol
	-- -- ,sum(if(t1.OR_C_BR_MULT_012>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_012+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_BR_MULT_012_sig
	-- ,sum(if(t1.OR_C_BR_MULT_010>0,1,0)) as OR_C_BR_MULT_010_tol
	-- ,sum(if(t1.OR_C_BR_MULT_010>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_BR_MULT_010_sig
	-- ,sum(if(t1.OR_C_PBOC_027>0,1,0)) as OR_C_PBOC_027_tol
	-- ,sum(if(t1.OR_C_PBOC_027>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_PBOC_027_sig
	-- ,sum(if(t1.OR_C_PBOC_030>0,1,0)) as OR_C_PBOC_030_tol
	-- ,sum(if(t1.OR_C_PBOC_030>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_PBOC_030_sig
	-- ,sum(if(t1.OR_C_BR_MULT_017>0,1,0)) as OR_C_BR_MULT_017_tol
	-- ,sum(if(t1.OR_C_BR_MULT_017>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_BR_MULT_017_sig
	-- ,sum(if(t1.OR_C_PBOC_044_002>0,1,0)) as OR_C_PBOC_044_002_tol
	-- ,sum(if(t1.OR_C_PBOC_044_002>0 and t2.if_refuse+OR_C_PBOC_031+OR_C_PBOC_004+OR_C_BR_MULT_021_002+OR_C_BR_MULT_009+OR_C_BR_MULT_020+OR_C_PBOC_003_002+OR_C_BR_MULT_010+OR_C_PBOC_027+OR_C_PBOC_030+OR_C_BR_MULT_017+OR_C_PBOC_044_002=1,1,0)) as OR_C_PBOC_044_002_sig
from 
(--新规则
	select
		*
		-- ,case when rh_m3_lnaprv_allorgcnt>7 then 1 else 0 end as OR_C_PBOC_031  --初始为6
		-- -- ,case when gr_pboc_005>=0 and gr_pboc_005<546 then 1 else 0 end as OR_C_PBOC_004  --初始为546 生效70%
		-- ,case when br_als_m3_id_nbank_allnum>24 then 1 else 0 end as OR_C_BR_MULT_009  --初始为22
		-- ,case when br_als_m1_id_nbank_allnum>12 then 1 else 0 end as OR_C_BR_MULT_010  --初始为11
		-- ,case when gr_sf_009>=0 and gr_sf_009<358 then 1 else 0 end as OR_C_BJ_TZ_002  --初始为372

		,case when rh_m3_lnaprv_allorgcnt>12 then 1 else 0 end as OR_C_PBOC_031  --7
		,case when gr_pboc_005>=0 and gr_pboc_005<530 then 1 else 0 end as OR_C_PBOC_004   --546
		,case when homeln_cur_cnt<=0 and GR_SF_018>=0 and GR_SF_018<475 then 1 else 0 end as OR_C_BR_MULT_021_002  --初始为487
		,case when br_als_m3_id_nbank_allnum>28 then 1 else 0 end as OR_C_BR_MULT_009  --初始为24
		,case when br_als_m12_id_nbank_avg_monnum>3.5 and GR_SF_022>=0 and GR_SF_022<680 then 1 else 0 end as OR_C_BR_MULT_020  --2.75; 682
		,case when homeln_cur_cnt<=0 and gr_pboc_009>=0 and gr_pboc_009<562 then 1 else 0 end as OR_C_PBOC_003_002  --初始为575 生效50%
		-- ,case when br_apply_loan_str_flag=1 and br_als_m6_id_nbank_orgnum>3.5 and br_als_m12_cell_bank_allnum<=4.5 and br_als_m12_id_nbank_orgnum>9.5 then 1 else 0 end as OR_C_BR_MULT_012 -- 3.5; 4.5; 9.5
		,case when br_als_m1_id_nbank_allnum>14 then 1 else 0 end as OR_C_BR_MULT_010  --初始为12
		,case when rh_querycnt_1m>9 then 1 else 0 end as OR_C_PBOC_027  --初始为6
		,case when rh_d15_lnaprv_allorgcnt>5 then 1 else 0 end as OR_C_PBOC_030  --初始为3
		,case when br_apply3m_id_orgnum>=13 and GR_SF_022>=0 and GR_SF_022<682 then 1 else 0 end as OR_C_BR_MULT_017 -- 12; 682
		,case when homeln_cur_cnt<=0 and GR_A_WP_PBOC_2203_001>=0 and GR_A_WP_PBOC_2203_001<525 then 1 else 0 end as OR_C_PBOC_044_002 --535
	from risk_analysis.jc_temp_youku_var_resize20220907
)t1
left join
(--旧规则
	select
		apply_id,if(concat_ws(',',refuse_rule_log)<>'',1,0) if_refuse,refuse_rule_log
	from
	(
		select
			apply_id,collect_list(case when t.v not in (
				-- 'OR_C_PBOC_031'
				-- ,'OR_C_BR_MULT_009'
				-- ,'OR_C_BR_MULT_010'
				-- ,'OR_C_BJ_TZ_002'

				'OR_C_PBOC_031'
				,'OR_C_PBOC_004'
				,'OR_C_BR_MULT_021_002'
				,'OR_C_BR_MULT_009'
				,'OR_C_BR_MULT_020'
				,'OR_C_PBOC_003_002'
				-- ,'OR_C_BR_MULT_012'
				,'OR_C_BR_MULT_010'
				,'OR_C_PBOC_027'
				,'OR_C_PBOC_030'
				,'OR_C_BR_MULT_017'
				,'OR_C_PBOC_044_002'
			) then t.v end) as refuse_rule_log
		from risk_analysis.jc_temp_youku_var_resize20220907
		lateral view explode(split(refuse_rule_log,',')) t as v
		group by apply_id
	)base
)t2
on t1.apply_id=t2.apply_id
-- group by date_bin
group by t1.date_bin
	,if(t1.refuse_rule_log is null or t1.refuse_rule_log='', 'Y', 'N')
	-- ,if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_010+t1.OR_C_PBOC_027+t1.OR_C_BJ_TZ_002>0,'N','Y')
	,if(t2.if_refuse+t1.OR_C_PBOC_031+t1.OR_C_PBOC_004+t1.OR_C_BR_MULT_021_002+t1.OR_C_BR_MULT_009+t1.OR_C_BR_MULT_020+t1.OR_C_PBOC_003_002+t1.OR_C_BR_MULT_010+t1.OR_C_PBOC_027+t1.OR_C_PBOC_030+t1.OR_C_BR_MULT_017+t1.OR_C_PBOC_044_002>0,'N','Y')






--调整的规则：调整前触发量、单一触发量
with base_rule as
(
	select distinct
		product_no,substr(create_date,1,7) as apply_dt
		,apply_id,refuse_rule_log
		,split(refuse_rule_log,',')[0] as rule
	from risk.scene_log_rule_result
	where product_no in ('8010110182')
	-- and scene in ('060501','050501')
	and scene in ('060401','050401')
	and to_date(create_date)>='2022-07-01'
)
select
	t1.product_no,t1.apply_dt,t1.rule
	,t2.apply_cnt
	,t1.cnt as total_trg
	,t1.cnt/t2.apply_cnt as total_trg_rate
	,t3.cnt as sig_trg
	,t3.cnt/t2.apply_cnt as sig_trg_rate
from
(
	select
		product_no,apply_dt,vt.rule,count(apply_id) as cnt
	from base_rule
	lateral view explode(split(refuse_rule_log,',')) vt as rule
	group by product_no,apply_dt,vt.rule
)t1
left join
(
	select product_no,apply_dt,count(apply_id) as apply_cnt
	from base_rule
	group by product_no,apply_dt
)t2
on t1.product_no=t2.product_no and t1.apply_dt=t2.apply_dt
left join
(
	select
		product_no,apply_dt,rule
		,count(distinct apply_id) as cnt
	from base_rule
	where size(split(refuse_rule_log,','))<=1
	group by product_no,apply_dt,rule
)t3
on t1.product_no=t3.product_no and t1.apply_dt=t3.apply_dt and t1.rule=t3.rule
where t1.rule in (
				'OR_C_PBOC_031'
				-- ,'OR_C_PBOC_004'
				,'OR_C_BR_MULT_009'
				-- ,'OR_C_BR_MULT_021_002'
				-- ,'OR_C_PBOC_003_002'
				,'OR_C_BR_MULT_010'
				-- ,'OR_C_PBOC_030'
				,'OR_C_PBOC_027'
				-- ,'OR_C_PBOC_026'
				,'OR_C_BJ_TZ_002'
			)






select
  decimal2(snglprdc_histovrd_max_days, -999) as snglprdc_histovrd_max_days,
  decimal2(snglprdc_cur_ovrd_unpd_amt, -999) as snglprdc_cur_ovrd_unpd_amt,
  decimal2(snglprdc_loan_cptl_bal, -999) as snglprdc_loan_cptl_bal,
  decimal2(snglprdc_histovrd_cptl_cnt, -999) as snglprdc_histovrd_cptl_cnt,
  decimal2(snglprdc_histovrd_cptl_intr_cnt, -999) as snglprdc_histovrd_cptl_intr_cnt,
  decimal2(allprdc_histovrd_max_days, -999) as allprdc_histovrd_max_days,
  decimal2(allprdc_cur_ovrd_unpd_amt, -999) as allprdc_cur_ovrd_unpd_amt,
  decimal2(allprdc_loan_cptl_bal, -999) as allprdc_loan_cptl_bal,
  decimal2(allprdc_histovrd_cptl_cnt, -999) as allprdc_histovrd_cptl_cnt,
  decimal2(allprdc_histovrd_cptl_intr_cnt, -999) as allprdc_histovrd_cptl_intr_cnt
from
(
  select
    case when snglprdc_histovrd_max_days = -999999997 then null else snglprdc_histovrd_max_days end as snglprdc_histovrd_max_days,
    case when snglprdc_cur_ovrd_unpd_amt = -999999997 then null else snglprdc_cur_ovrd_unpd_amt end as snglprdc_cur_ovrd_unpd_amt,
    case when snglprdc_loan_cptl_bal = -999999997 then null else snglprdc_loan_cptl_bal end as snglprdc_loan_cptl_bal,
    case when snglprdc_histovrd_cptl_cnt = -999999997 then null else snglprdc_histovrd_cptl_cnt end as snglprdc_histovrd_cptl_cnt,
    case when snglprdc_histovrd_cptl_intr_cnt = -999999997 then null else snglprdc_histovrd_cptl_intr_cnt end as snglprdc_histovrd_cptl_intr_cnt,
    case when allprdc_histovrd_max_days = -999999997 then null else allprdc_histovrd_max_days end as allprdc_histovrd_max_days,
    case when allprdc_cur_ovrd_unpd_amt = -999999997 then null else allprdc_cur_ovrd_unpd_amt end as allprdc_cur_ovrd_unpd_amt,
    case when allprdc_loan_cptl_bal = -999999997 then null else allprdc_loan_cptl_bal end as allprdc_loan_cptl_bal,
    case when allprdc_histovrd_cptl_cnt = -999999997 then null else allprdc_histovrd_cptl_cnt end as allprdc_histovrd_cptl_cnt,
    case when allprdc_histovrd_cptl_intr_cnt = -999999997 then null else allprdc_histovrd_cptl_intr_cnt end as allprdc_histovrd_cptl_intr_cnt
  from
    on_loan_history_overdue_record
)