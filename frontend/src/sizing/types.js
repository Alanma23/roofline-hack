/**
 * Frontend sizing contracts for TP/PP/network-aware inference roofline.
 *
 * @typedef {Object} WorkloadSpec
 * @property {"prefill"|"decode"} phase
 * @property {number} batch
 * @property {number} seq_len
 * @property {number=} prefill_tokens
 * @property {number=} decode_tokens
 * @property {{L:number,H:number,nh:number,nkv:number,dh:number,dff:number,V:number,gate:boolean}} model
 * @property {{w:string,a:string,kv:string,computeAs:string}} precision
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
 * @property {number=} max_asics
 *
 * @typedef {Object} NetworkSpec
 * @property {number=} tp_link_bw_gbs
 * @property {number=} tp_link_latency_us
 * @property {number=} pp_link_bw_gbs
 * @property {number=} pp_link_latency_us
 * @property {number=} overlap_fraction
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
 * @property {{tp_allreduce_count:number,tp_allreduce_bytes:number,pp_send_count:number,pp_send_bytes:number}} collective
 * @property {{compute_ms:number,memory_ms:number,network_ms:number,kernel_ms:number,end_to_end_ms:number,tokens_per_s:number}} time
 * @property {"compute"|"memory"|"network"} bottleneck
 * @property {Array<{layer:number,stage:number,input_bytes:number,output_bytes:number,weight_bytes:number,tp_sync_bytes:number,pp_boundary_send_bytes:number}>} layer_io
 * @property {Array<{tp:number,pp:number,asics:number,latency_ms:number,bottleneck:string,note:string}>} recommendations
 * @property {{network_bw_gbs?:number,mem_bw_gbs?:number,offchip_bw_gbs?:number,peak_tflops?:number}=} required_to_debottleneck
 */

export const SIZING_TYPES_VERSION = "1.0.0";
