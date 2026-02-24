/**
 * Frontend sizing contracts for TP/PP/EP/network-aware inference roofline.
 *
 * @typedef {Object} MoESpec
 * @property {number} num_experts          — total expert count (e.g. 8 for Mixtral)
 * @property {number} experts_per_token    — top-K experts activated per token
 * @property {number} expert_ffn_dim       — FFN hidden dim inside each expert
 * @property {number} capacity_factor      — capacity multiplier (>1 = buffer for imbalance)
 * @property {number} moe_layer_freq       — every N layers is a MoE layer (1 = all layers)
 *
 * @typedef {Object} WorkloadSpec
 * @property {"prefill"|"decode"} phase
 * @property {number} batch
 * @property {number} seq_len
 * @property {number=} prefill_tokens
 * @property {number=} decode_tokens
 * @property {{L:number,H:number,nh:number,nkv:number,dh:number,dff:number,V:number,gate:boolean}} model
 * @property {{w:string,a:string,kv:string,computeAs:string}} precision
 * @property {MoESpec=} moe               — if present, treat matching layers as MoE
 *
 * @typedef {Object} HardwareSpec
 * @property {string} name
 * @property {Object.<string, number>} peak_tflops
 * @property {number} mem_bw_gbs
 * @property {{mode:"single"|"two_tier",onchip_bw_gbs?:number,offchip_bw_gbs?:number,onchip_hit_rate?:number}=} memory_model
 *
 * @typedef {Object} ParallelSpec
 * @property {number} tp
 * @property {number} pp
 * @property {number=} ep                 — expert parallelism degree (default 1)
 * @property {number=} max_asics
 *
 * @typedef {Object} NetworkSpec
 * @property {number=} tp_link_bw_gbs
 * @property {number=} tp_link_latency_us
 * @property {number=} pp_link_bw_gbs
 * @property {number=} pp_link_latency_us
 * @property {number=} overlap_fraction
 * @property {number=} intranode_bw_gbs   — NVLink BW within a node (default 900 GB/s)
 * @property {number=} intranode_lat_us   — NVLink latency within a node (default 1 us)
 * @property {number=} internode_bw_gbs   — IB/Ethernet BW between nodes (default 25 GB/s)
 * @property {number=} internode_lat_us   — IB latency between nodes (default 5 us)
 * @property {number=} gpus_per_node      — GPUs in a single node (default 8)
 *
 * @typedef {Object} SizingRequest
 * @property {WorkloadSpec} workload
 * @property {HardwareSpec} hardware
 * @property {ParallelSpec} parallel
 * @property {NetworkSpec} network
 * @property {number=} target_latency_ms
 *
 * @typedef {Object} SizingResult
 * @property {{flops:number,bytes:number,ai:number}} totals
 * @property {{tp_allreduce_count:number,tp_allreduce_bytes:number,pp_send_count:number,pp_send_bytes:number,ep_alltoall_count:number,ep_alltoall_bytes:number}} collective
 * @property {{compute_ms:number,memory_ms:number,network_ms:number,kernel_ms:number,end_to_end_ms:number,tokens_per_s:number}} time
 * @property {"compute"|"memory"|"network"} bottleneck
 * @property {Array<{layer:number,stage:number,input_bytes:number,output_bytes:number,weight_bytes:number,tp_sync_bytes:number,pp_boundary_send_bytes:number}>} layer_io
 * @property {Array<{tp:number,pp:number,ep:number,asics:number,latency_ms:number,bottleneck:string,note:string,pareto_optimal:boolean}>} recommendations
 * @property {{network_bw_gbs?:number,mem_bw_gbs?:number,offchip_bw_gbs?:number,peak_tflops?:number,intranode_bw_gbs?:number,internode_bw_gbs?:number}=} required_to_debottleneck
 */

export const SIZING_TYPES_VERSION = "2.0.0";
