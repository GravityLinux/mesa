#include "util/u_debug.h"
/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_physical_device.h"

#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "util/disk_cache.h"
#include "util/mesa-blake3.h"
#include "util/os_misc.h"
#include "git_sha1.h"
#include "hk_buffer.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_instance.h"
#include "hk_private.h"
#include "hk_shader.h"
#include "hk_wsi.h"

#include "util/simple_mtx.h"
#include "vulkan/vulkan_core.h"
#include "vulkan/wsi/wsi_common.h"
#include "vk_drm_syncobj.h"
#include "vk_physical_device.h"
#include "vk_shader_module.h"

#include <fcntl.h>
#include <string.h>
#include <xf86drm.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

static uint32_t
hk_get_vk_version()
{
   /* Version override takes priority */
   const uint32_t version_override = vk_get_version_override();
   if (version_override)
      return version_override;

   return VK_MAKE_VERSION(1, 4, VK_HEADER_VERSION);
}

static void
hk_get_device_extensions(const struct hk_instance *instance,
                         struct vk_device_extension_table *ext)
{
   *ext = (struct vk_device_extension_table){
      .KHR_8bit_storage = true,
      .KHR_16bit_storage = true,
      .KHR_bind_memory2 = true,
      .KHR_buffer_device_address = true,
      .KHR_calibrated_timestamps = true,
      .KHR_copy_commands2 = true,
      .KHR_create_renderpass2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_depth_stencil_resolve = true,
      .KHR_descriptor_update_template = true,
      .KHR_device_group = true,
      .KHR_draw_indirect_count = true,
      .KHR_driver_properties = true,
      .KHR_dynamic_rendering = true,
      .KHR_dynamic_rendering_local_read = true,
      .KHR_external_fence = true,
      .KHR_external_fence_fd = true,
      .KHR_external_memory = true,
      .KHR_external_memory_fd = true,
      .KHR_external_semaphore = true,
      .KHR_external_semaphore_fd = true,
      .KHR_format_feature_flags2 = true,
      .KHR_fragment_shader_barycentric = false,
      .KHR_get_memory_requirements2 = true,
      .KHR_global_priority = true,
      .KHR_image_format_list = true,
      .KHR_imageless_framebuffer = true,
#ifdef HK_USE_WSI_PLATFORM
      .KHR_incremental_present = true,
#endif
      .KHR_index_type_uint8 = true,
      .KHR_line_rasterization = true,
      .KHR_load_store_op_none = true,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_maintenance4 = true,
      .KHR_maintenance5 = true,
      .KHR_maintenance6 = true,
      .KHR_maintenance7 = true,
      .KHR_maintenance8 = true,
      .KHR_maintenance9 = true,
      .KHR_map_memory2 = true,
      .KHR_multiview = true,
      .KHR_pipeline_binary = true,
      .KHR_pipeline_executable_properties = true,
      .KHR_pipeline_library = true,
#ifdef HK_USE_WSI_PLATFORM
      .KHR_present_id = true,
      .KHR_present_id2 = true,
      .KHR_present_wait = true,
      .KHR_present_wait2 = true,
#endif
      .KHR_push_descriptor = true,
      .KHR_relaxed_block_layout = true,
      .KHR_robustness2 = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_sampler_ycbcr_conversion = true,
      .KHR_separate_depth_stencil_layouts = true,
      .KHR_shader_atomic_int64 = false,
      .KHR_shader_clock = false,
      .KHR_shader_draw_parameters = true,
      .KHR_shader_expect_assume = true,
      .KHR_shader_float_controls = true,
      .KHR_shader_subgroup_extended_types = true,
      .EXT_shader_viewport_index_layer = false,
      .KHR_shader_float_controls2 = true,
      .KHR_shader_float16_int8 = true,
      .KHR_shader_integer_dot_product = true,
      .KHR_shader_maximal_reconvergence = true,
      .KHR_shader_non_semantic_info = true,
      .KHR_shader_quad_control = true,
      .KHR_shader_relaxed_extended_instruction = true,
      .KHR_shader_subgroup_extended_types = true,
      .KHR_shader_subgroup_rotate = true,
      .KHR_shader_subgroup_uniform_control_flow = true,
      .KHR_shader_terminate_invocation = true,
      .KHR_spirv_1_4 = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_timeline_semaphore = true,
#ifdef HK_USE_WSI_PLATFORM
      .KHR_swapchain = true,
      .KHR_swapchain_maintenance1 = true,
      .KHR_swapchain_mutable_format = true,
#endif
      .KHR_synchronization2 = true,
      .KHR_uniform_buffer_standard_layout = true,
      .KHR_variable_pointers = true,
      .KHR_vertex_attribute_divisor = true,
      .KHR_vulkan_memory_model = true,
      .KHR_workgroup_memory_explicit_layout = true,
      .KHR_zero_initialize_workgroup_memory = true,
      .EXT_4444_formats = true,
      .EXT_attachment_feedback_loop_layout = true,
      .EXT_blend_operation_advanced = true,
      .EXT_border_color_swizzle = true,
      .EXT_buffer_device_address = true,
      .EXT_calibrated_timestamps = true,
      .EXT_conditional_rendering = false,
      .EXT_color_write_enable = true,
      .EXT_custom_border_color = true,
      .EXT_debug_marker = true,
      .EXT_depth_bias_control = true,
      .EXT_depth_clip_control = false,
      .EXT_depth_clip_enable = true,
      .EXT_descriptor_indexing = true,
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
      .EXT_display_control = false,
#endif
      .EXT_dynamic_rendering_unused_attachments = true,
      .EXT_extended_dynamic_state = true,
      .EXT_extended_dynamic_state2 = true,
      .EXT_extended_dynamic_state3 = true,
      .EXT_external_memory_dma_buf = true,
      .EXT_global_priority = true,
      .EXT_global_priority_query = true,
      .EXT_graphics_pipeline_library = true,
      .EXT_hdr_metadata = true,
      .EXT_host_query_reset = true,
      .EXT_host_image_copy = true,
      .EXT_image_2d_view_of_3d = true,
      .EXT_image_drm_format_modifier = true,
      .EXT_image_robustness = true,
      .EXT_image_sliced_view_of_3d = false,
      .EXT_image_view_min_lod = instance->drirc.misc.image_view_min_lod,
      .EXT_index_type_uint8 = true,
      .EXT_inline_uniform_block = true,
      .EXT_line_rasterization = true,
      .EXT_load_store_op_none = true,
      .EXT_map_memory_placed = true,
      .EXT_memory_budget = false,
      .EXT_multi_draw = true,
      .EXT_mutable_descriptor_type = true,
      .EXT_non_seamless_cube_map = true,
      .EXT_pipeline_creation_cache_control = true,
      .EXT_pipeline_creation_feedback = true,
      .EXT_pipeline_protected_access = true,
      .EXT_pipeline_robustness = true,
      .EXT_physical_device_drm = true,
#ifdef HK_USE_WSI_PLATFORM
      .EXT_present_timing = true,
#endif
      .EXT_primitive_topology_list_restart = true,
      .EXT_private_data = true,
      .EXT_primitives_generated_query = false,
      .EXT_provoking_vertex = true,
      .EXT_queue_family_foreign = true,
      .EXT_robustness2 = true,
      .EXT_sample_locations = true,
      .EXT_sampler_filter_minmax = false,
      .EXT_scalar_block_layout = true,
      .EXT_separate_stencil_usage = true,
      .EXT_shader_image_atomic_int64 = false,
      .EXT_shader_demote_to_helper_invocation = true,
      .EXT_shader_module_identifier = true,
      .EXT_shader_object = true,
      .EXT_shader_replicated_composites = true,
      .EXT_shader_stencil_export = true,
      .EXT_shader_subgroup_ballot = true,
      .EXT_shader_subgroup_vote = true,
      .EXT_shader_uniform_buffer_unsized_array = true,
      .EXT_shader_viewport_index_layer = true,
      .EXT_subgroup_size_control = true,
#ifdef HK_USE_WSI_PLATFORM
      .EXT_swapchain_maintenance1 = true,
#endif
      .EXT_texel_buffer_alignment = true,
      .EXT_tooling_info = true,
      .EXT_transform_feedback = true,
      .EXT_vertex_attribute_divisor = true,
      .EXT_vertex_input_dynamic_state = true,
      .EXT_ycbcr_2plane_444_formats = false,
      .EXT_ycbcr_image_arrays = false,
      .GOOGLE_decorate_string = true,
#ifdef HK_USE_WSI_PLATFORM
      .GOOGLE_display_timing = wsi_instance_supports_google_display_timing(&instance->vk, &instance->drirc.options),
#endif
      .GOOGLE_hlsl_functionality1 = true,
      .GOOGLE_user_type = true,
      .VALVE_mutable_descriptor_type = true,
   };
}

static void
hk_get_device_features(
   const struct agx_device *dev, const struct hk_instance *instance,
   const struct vk_device_extension_table *supported_extensions,
   struct vk_features *features)
{
   *features = (struct vk_features){
      /* Vulkan 1.0 */
      .robustBufferAccess = true,
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .geometryShader = true,
      .tessellationShader = true,
      .sampleRateShading = true,
      .dualSrcBlend = true,
      .logicOp = true,
      .multiDrawIndirect = true,
      .drawIndirectFirstInstance = true,
      .depthClamp = true,
      .depthBiasClamp = true,
      .fillModeNonSolid = true,
      .depthBounds = false,
      .wideLines = true,
      .largePoints = true,
      .alphaToOne = true,
      .multiViewport = true,
      .samplerAnisotropy = true,
      .textureCompressionETC2 = true,
      .textureCompressionBC = true,
      .textureCompressionASTC_LDR = true,
      .occlusionQueryPrecise = true,
      .pipelineStatisticsQuery = true,
      .vertexPipelineStoresAndAtomics = instance->drirc.misc.enable_vertex_pipeline_stores_atomics,
      .fragmentStoresAndAtomics = true,
      .shaderTessellationAndGeometryPointSize = true,
      .shaderImageGatherExtended = true,
      .shaderStorageImageExtendedFormats = true,
      /* TODO: hitting the vertex shader timeout in CTS, but should work */
      .shaderStorageImageMultisample = false,
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      .shaderClipDistance = true,
      .shaderCullDistance = true,
      .shaderFloat64 = false,
      .shaderInt64 = true,
      .shaderInt16 = true,
      .shaderResourceResidency = true,
      .shaderResourceMinLod = true,
      .sparseBinding = true,

      /* We probably could advertise multisampled sparse but we don't have a use
       * case yet and it isn't trivial.
       */
      .sparseResidency2Samples = false,
      .sparseResidency4Samples = false,
      .sparseResidency8Samples = false,
      .sparseResidencyAliased = true,
      .sparseResidencyImage2D = true,
      .sparseResidencyBuffer = true,

      /* This needs investigation. */
      .sparseResidencyImage3D = false,
      .variableMultisampleRate = false,
      .inheritedQueries = true,

      /* Vulkan 1.1 */
      .storageBuffer16BitAccess = true,
      .uniformAndStorageBuffer16BitAccess = true,
      .storagePushConstant16 = true,
      .storageInputOutput16 = false,
      .multiview = true,
      .multiviewGeometryShader = false,
      .multiviewTessellationShader = false,
      .variablePointersStorageBuffer = true,
      .variablePointers = true,
      .shaderDrawParameters = true,
      .samplerYcbcrConversion = true,

      /* Vulkan 1.2 */
      .samplerMirrorClampToEdge = true,
      .drawIndirectCount = true,
      .storageBuffer8BitAccess = true,
      .uniformAndStorageBuffer8BitAccess = true,
      .storagePushConstant8 = true,
      .shaderBufferInt64Atomics = false,
      .shaderSharedInt64Atomics = false,
      .shaderFloat16 = true,
      .shaderInt8 = true,
      .descriptorIndexing = true,
      .shaderInputAttachmentArrayDynamicIndexing = true,
      .shaderUniformTexelBufferArrayDynamicIndexing = true,
      .shaderStorageTexelBufferArrayDynamicIndexing = true,
      .shaderUniformBufferArrayNonUniformIndexing = true,
      .shaderSampledImageArrayNonUniformIndexing = true,
      .shaderStorageBufferArrayNonUniformIndexing = true,
      .shaderStorageImageArrayNonUniformIndexing = true,
      .shaderInputAttachmentArrayNonUniformIndexing = true,
      .shaderUniformTexelBufferArrayNonUniformIndexing = true,
      .shaderStorageTexelBufferArrayNonUniformIndexing = true,
      .descriptorBindingUniformBufferUpdateAfterBind = true,
      .descriptorBindingSampledImageUpdateAfterBind = true,
      .descriptorBindingStorageImageUpdateAfterBind = true,
      .descriptorBindingStorageBufferUpdateAfterBind = true,
      .descriptorBindingUniformTexelBufferUpdateAfterBind = true,
      .descriptorBindingStorageTexelBufferUpdateAfterBind = true,
      .descriptorBindingUpdateUnusedWhilePending = true,
      .descriptorBindingPartiallyBound = true,
      .descriptorBindingVariableDescriptorCount = true,
      .runtimeDescriptorArray = true,
      .samplerFilterMinmax = instance->drirc.misc.fake_minmax,
      .scalarBlockLayout = true,
      .imagelessFramebuffer = true,
      .uniformBufferStandardLayout = true,
      .shaderSubgroupExtendedTypes = true,
      .separateDepthStencilLayouts = true,
      .hostQueryReset = true,
      .timelineSemaphore = true,
      .bufferDeviceAddress = true,
      .bufferDeviceAddressCaptureReplay = true,
      .bufferDeviceAddressMultiDevice = false,
      .vulkanMemoryModel = true,
      .vulkanMemoryModelDeviceScope = true,
      .vulkanMemoryModelAvailabilityVisibilityChains = false,
      .shaderOutputViewportIndex = true,
      .shaderOutputLayer = true,
      .subgroupBroadcastDynamicId = true,

      /* Vulkan 1.3 */
      .robustImageAccess = true,
      .inlineUniformBlock = true,
      .descriptorBindingInlineUniformBlockUpdateAfterBind = true,
      .pipelineCreationCacheControl = true,
      .privateData = true,
      .shaderDemoteToHelperInvocation = true,
      .shaderTerminateInvocation = true,
      .subgroupSizeControl = true,
      .computeFullSubgroups = true,
      .synchronization2 = true,
      .shaderZeroInitializeWorkgroupMemory = true,
      .dynamicRendering = true,
      .shaderIntegerDotProduct = true,
      .maintenance4 = true,

      /* Vulkan 1.4 */
      .pushDescriptor = true,

      /* VK_KHR_dynamic_rendering_local_read */
      .dynamicRenderingLocalRead = true,

      /* VK_KHR_fragment_shader_barycentric */
      .fragmentShaderBarycentric = false,

      /* VK_KHR_global_priority */
      .globalPriorityQuery = true,

      /* VK_KHR_index_type_uint8 */
      .indexTypeUint8 = true,

      /* VK_KHR_line_rasterization */
      .rectangularLines = false,
      .bresenhamLines = true,
      .smoothLines = false,
      .stippledRectangularLines = false,
      .stippledBresenhamLines = false,
      .stippledSmoothLines = false,

      /* VK_KHR_maintenance5 */
      .maintenance5 = true,

      /* VK_KHR_maintenance6 */
      .maintenance6 = true,

      /* VK_KHR_maintenance7 */
      .maintenance7 = true,

      /* VK_KHR_maintenance8 */
      .maintenance8 = true,

      /* VK_KHR_maintenance9 */
      .maintenance9 = true,

      /* VK_KHR_pipeline_binary */
      .pipelineBinaries = true,

      /* VK_KHR_pipeline_executable_properties */
      .pipelineExecutableInfo = true,

      /* VK_KHR_present_id */
      .presentId = true,

      /* VK_KHR_present_id2 */
      .presentId2 = true,

      /* VK_KHR_present_wait */
      .presentWait = true,

      /* VK_KHR_present_wait2 */
      .presentWait2 = true,

      /* VK_KHR_shader_clock */
      .shaderSubgroupClock = false,
      .shaderDeviceClock = false,

      /* VK_KHR_shader_expect_assume */
      .shaderExpectAssume = true,

      /* VK_KHR_shader_float_controls2 */
      .shaderFloatControls2 = true,

      /* VK_KHR_shader_maximal_reconvergence */
      .shaderMaximalReconvergence = true,

      /* VK_KHR_shader_quad_control */
      .shaderQuadControl = true,

      /* VK_KHR_shader_subgroup_rotate */
      .shaderSubgroupRotate = true,
      .shaderSubgroupRotateClustered = true,

      /* VK_KHR_vertex_attribute_divisor */
      .vertexAttributeInstanceRateDivisor = true,
      .vertexAttributeInstanceRateZeroDivisor = true,

      /* VK_KHR_workgroup_memory_explicit_layout */
      .workgroupMemoryExplicitLayout = true,
      .workgroupMemoryExplicitLayoutScalarBlockLayout = true,
      .workgroupMemoryExplicitLayout8BitAccess = true,
      .workgroupMemoryExplicitLayout16BitAccess = true,

      /* VK_EXT_4444_formats */
      .formatA4R4G4B4 = true,
      .formatA4B4G4R4 = true,

      /* VK_EXT_attachment_feedback_loop_layout */
      .attachmentFeedbackLoopLayout = true,

      /* VK_EXT_border_color_swizzle */
      .borderColorSwizzle = true,
      .borderColorSwizzleFromImage = false,

      /* VK_EXT_buffer_device_address */
      .bufferDeviceAddressCaptureReplayEXT = true,

      /* VK_EXT_color_write_enable */
      .colorWriteEnable = true,

      /* VK_EXT_conditional_rendering */
      .conditionalRendering = false,
      .inheritedConditionalRendering = false,

      /* VK_EXT_custom_border_color */
      .customBorderColors = true,
      .customBorderColorWithoutFormat = true,

      /* VK_EXT_depth_bias_control */
      .depthBiasControl = true,
      .leastRepresentableValueForceUnormRepresentation = true,
      .floatRepresentation = false,
      .depthBiasExact = true,

      /* VK_EXT_depth_clip_control */
      .depthClipControl = false,

      /* VK_EXT_depth_clip_enable */
      .depthClipEnable = true,

      /* VK_EXT_dynamic_rendering_unused_attachments */
      .dynamicRenderingUnusedAttachments = true,

      /* VK_EXT_extended_dynamic_state */
      .extendedDynamicState = true,

      /* VK_EXT_extended_dynamic_state2 */
      .extendedDynamicState2 = true,
      .extendedDynamicState2LogicOp = true,
      .extendedDynamicState2PatchControlPoints = true,

      /* VK_EXT_extended_dynamic_state3 */
      .extendedDynamicState3TessellationDomainOrigin = true,
      .extendedDynamicState3DepthClampEnable = true,
      .extendedDynamicState3PolygonMode = true,
      .extendedDynamicState3RasterizationSamples = true,
      .extendedDynamicState3SampleMask = true,
      .extendedDynamicState3AlphaToCoverageEnable = true,
      .extendedDynamicState3AlphaToOneEnable = true,
      .extendedDynamicState3LogicOpEnable = true,
      .extendedDynamicState3ColorBlendEnable = true,
      .extendedDynamicState3ColorBlendEquation = true,
      .extendedDynamicState3ColorWriteMask = true,
      .extendedDynamicState3RasterizationStream = true,
      .extendedDynamicState3ConservativeRasterizationMode = false,
      .extendedDynamicState3ExtraPrimitiveOverestimationSize = false,
      .extendedDynamicState3DepthClipEnable = true,
      .extendedDynamicState3SampleLocationsEnable = true,
      .extendedDynamicState3ColorBlendAdvanced = false,
      .extendedDynamicState3ProvokingVertexMode = true,
      .extendedDynamicState3LineRasterizationMode = true,
      .extendedDynamicState3LineStippleEnable = false,
      .extendedDynamicState3DepthClipNegativeOneToOne = false,
      .extendedDynamicState3ViewportWScalingEnable = false,
      .extendedDynamicState3ViewportSwizzle = false,
      .extendedDynamicState3CoverageToColorEnable = false,
      .extendedDynamicState3CoverageToColorLocation = false,
      .extendedDynamicState3CoverageModulationMode = false,
      .extendedDynamicState3CoverageModulationTableEnable = false,
      .extendedDynamicState3CoverageModulationTable = false,
      .extendedDynamicState3CoverageReductionMode = false,
      .extendedDynamicState3RepresentativeFragmentTestEnable = false,
      .extendedDynamicState3ShadingRateImageEnable = false,

      /* VK_EXT_graphics_pipeline_library */
      .graphicsPipelineLibrary = true,

      /* VK_EXT_host_image_copy */
      .hostImageCopy = true,

      /* VK_EXT_image_2d_view_of_3d */
      .image2DViewOf3D = true,
      .sampler2DViewOf3D = true,

      /* VK_EXT_image_sliced_view_of_3d */
      .imageSlicedViewOf3D = false,

#ifdef HK_USE_WSI_PLATFORM
      /* VK_KHR_swapchain_maintenance1 */
      .swapchainMaintenance1 = true,
#endif

      /* VK_EXT_image_view_min_lod */
      .minLod = instance->drirc.misc.image_view_min_lod,

      /* VK_EXT_map_memory_placed */
      .memoryMapPlaced = true,
      .memoryMapRangePlaced = false,
      .memoryUnmapReserve = true,

      /* VK_EXT_multi_draw */
      .multiDraw = true,

      /* VK_EXT_mutable_descriptor_type */
      .mutableDescriptorType = true,

      /* VK_EXT_non_seamless_cube_map */
      .nonSeamlessCubeMap = true,

      /* VK_EXT_pipeline_protected_access */
      .pipelineProtectedAccess = true,

      /* VK_EXT_pipeline_robustness */
      .pipelineRobustness = true,

      /* VK_EXT_primitive_topology_list_restart */
      .primitiveTopologyListRestart = true,
      .primitiveTopologyPatchListRestart = false,

      /* VK_EXT_primitives_generated_query */
      .primitivesGeneratedQuery = false,
      .primitivesGeneratedQueryWithNonZeroStreams = false,
      .primitivesGeneratedQueryWithRasterizerDiscard = false,

      /* VK_EXT_provoking_vertex */
      .provokingVertexLast = true,
      .transformFeedbackPreservesProvokingVertex = true,

      /* VK_KHR_robustness2 */
      .robustBufferAccess2 = true,
      .robustImageAccess2 = true,
      .nullDescriptor = true,

      /* VK_EXT_shader_image_atomic_int64 */
      .shaderImageInt64Atomics = false,
      .sparseImageInt64Atomics = false,

      /* VK_EXT_shader_module_identifier */
      .shaderModuleIdentifier = true,

      /* VK_EXT_shader_object */
      .shaderObject = true,

      /* VK_EXT_shader_replicated_composites */
      .shaderReplicatedComposites = true,

      /* VK_KHR_shader_subgroup_uniform_control_flow */
      .shaderSubgroupUniformControlFlow = true,

      /* VK_EXT_shader_uniform_buffer_unsized_array */
      .shaderUniformBufferUnsizedArray = true,

      /* VK_EXT_texel_buffer_alignment */
      .texelBufferAlignment = true,

      /* VK_EXT_transform_feedback */
      .transformFeedback = true,
      .geometryStreams = true,

      /* VK_EXT_vertex_input_dynamic_state */
      .vertexInputDynamicState = true,

      /* VK_EXT_ycbcr_2plane_444_formats */
      .ycbcr2plane444Formats = false,

      /* VK_EXT_ycbcr_image_arrays */
      .ycbcrImageArrays = false,

      /* VK_KHR_shader_relaxed_extended_instruction */
      .shaderRelaxedExtendedInstruction = true,

      /* VK_EXT_blend_operation_advanced */
      .advancedBlendCoherentOperations = true,

#ifdef HK_USE_WSI_PLATFORM
      /* VK_EXT_present_timing */
      .presentTiming = true,
      .presentAtRelativeTime = true,
      .presentAtAbsoluteTime = true,
#endif
   };
}

static void
hk_get_device_properties(const struct agx_device *dev,
                         const struct hk_instance *instance,
                         struct vk_properties *properties)
{
   /* Note: update nir_shader_compiler_options.max_samples when changing this. */
   const VkSampleCountFlagBits sample_counts =
      VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT |
      /* Native Apple9 texture/tilebuffer paths currently implement 1x/4x. */
      (dev->params.gpu_generation >= 16 ? 0 : VK_SAMPLE_COUNT_2_BIT);

   uint64_t os_page_size = 16384;
   os_get_page_size(&os_page_size);

   /* The hardware limit is 128. However, we need to set a lower limit to
    * account for duplicated system values: clip/cull and viewport targets.
    * There are up to 8 combined clip/cull planes and only 1 viewport target,
    * giving a theoretical limit of 119. We round down to 116 to give a sensible
    * limit in vec4s.
    */
   unsigned max_vgt_output_components = 116;

   *properties = (struct vk_properties){
      .apiVersion = hk_get_vk_version(),
      .driverVersion = vk_get_driver_version(),
      .vendorID = instance->drirc.debug.force_vk_vendor ?: VK_VENDOR_ID_MESA,
      .deviceID = 0,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,

      /* Vulkan 1.0 limits */
      .maxImageDimension1D = 16384,
      .maxImageDimension2D = 16384,
      .maxImageDimension3D = 16384,
      .maxImageDimensionCube = 16384,
      .maxImageArrayLayers = 2048,
      .maxTexelBufferElements = AGX_TEXTURE_BUFFER_MAX_SIZE,
      .maxUniformBufferRange = 65536,

      /* From a hardware perspective, storage buffers are lowered to global
       * address arithmetic so there is no hard limit. However, making efficient
       * use of the hardware addressing modes depends on no signed wrapping in
       * any `amul` operations, which are themselves bounded by
       * maxStorageBufferRange. Therefore, limit storage buffers to INT32_MAX
       * bytes instead of UINT32_MAX. This is believed to be acceptable for
       * Direct3D.
       */
      .maxStorageBufferRange = INT32_MAX,
      .maxPushConstantsSize = HK_MAX_PUSH_SIZE,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = 4000,
      .bufferImageGranularity = 0x400,
      .sparseAddressSpaceSize = HK_SPARSE_ADDR_SPACE_SIZE,
      .maxBoundDescriptorSets = HK_MAX_SETS,
      .maxPerStageDescriptorSamplers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorSampledImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorInputAttachments = HK_MAX_DESCRIPTORS,
      .maxPerStageResources = UINT32_MAX,
      .maxDescriptorSetSamplers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetSampledImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetInputAttachments = HK_MAX_DESCRIPTORS,
      .maxVertexInputAttributes = AGX_MAX_VBUFS,
      .maxVertexInputBindings = AGX_MAX_ATTRIBS,
      .maxVertexInputAttributeOffset = 65535,
      .maxVertexInputBindingStride = 2048,

      .maxVertexOutputComponents = max_vgt_output_components,
      .maxGeometryShaderInvocations = 32,
      .maxGeometryInputComponents = 128,
      .maxGeometryOutputComponents = max_vgt_output_components,
      .maxGeometryOutputVertices = 1024,
      .maxGeometryTotalOutputComponents = 1024,
      .maxTessellationGenerationLevel = 64,
      .maxTessellationPatchSize = 32,
      .maxTessellationControlPerVertexInputComponents = 128,
      .maxTessellationControlPerVertexOutputComponents = 128,
      .maxTessellationControlPerPatchOutputComponents = 120,
      .maxTessellationControlTotalOutputComponents = 4216,
      .maxTessellationEvaluationInputComponents = 128,
      .maxTessellationEvaluationOutputComponents = max_vgt_output_components,

      /* Set to match maxVertexOutputComponents, hardware limit is higher. */
      .maxFragmentInputComponents = max_vgt_output_components,
      .maxFragmentOutputAttachments = HK_MAX_RTS,
      .maxFragmentDualSrcAttachments = 1,
      .maxFragmentCombinedOutputResources = HK_MAX_RTS + HK_MAX_DESCRIPTORS,
      .maxComputeSharedMemorySize = HK_MAX_SHARED_SIZE,
      .maxComputeWorkGroupCount = {0x7fffffff, 65535, 65535},
      .maxComputeWorkGroupInvocations = 1024,
      .maxComputeWorkGroupSize = {1024, 1024, 64},
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT16_MAX,
      .maxSamplerLodBias = 15,
      .maxSamplerAnisotropy = 16,
      .maxViewports = HK_MAX_VIEWPORTS,
      .maxViewportDimensions = {32768, 32768},
      .viewportBoundsRange = {-65536, 65536},
      .viewportSubPixelBits = 8,
      .minMemoryMapAlignment = os_page_size,
      .minTexelBufferOffsetAlignment = HK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .minUniformBufferOffsetAlignment = HK_MIN_UBO_ALIGNMENT,
      .minStorageBufferOffsetAlignment = HK_MIN_SSBO_ALIGNMENT,
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .minTexelGatherOffset = -8,
      .maxTexelGatherOffset = 7,
      .minInterpolationOffset = -0.5,
      .maxInterpolationOffset = 0.4375,
      .subPixelInterpolationOffsetBits = 4,
      .maxFramebufferHeight = 16384,
      .maxFramebufferWidth = 16384,
      .maxFramebufferLayers = 2048,
      .framebufferColorSampleCounts = sample_counts,
      .framebufferDepthSampleCounts = sample_counts,
      .framebufferNoAttachmentsSampleCounts = sample_counts,
      .framebufferStencilSampleCounts = sample_counts,
      .maxColorAttachments = HK_MAX_RTS,
      .sampledImageColorSampleCounts = sample_counts,
      .sampledImageIntegerSampleCounts = sample_counts,
      .sampledImageDepthSampleCounts = sample_counts,
      .sampledImageStencilSampleCounts = sample_counts,
      .storageImageSampleCounts = sample_counts,
      .maxSampleMaskWords = 1,
      .timestampComputeAndGraphics = true,
      /* FIXME: Is timestamp period actually 1? */
      .timestampPeriod = 1.0f,
      .maxClipDistances = 8,
      .maxCullDistances = 8,
      .maxCombinedClipAndCullDistances = 8,
      .discreteQueuePriorities = 2,
      .pointSizeRange = {1.0, 512.f - 0.0625f},
      .lineWidthRange = {1.0, 16.0f},
      .pointSizeGranularity = 0.0625,
      .lineWidthGranularity = 1.0f / 16.0f,
      .strictLines = false,
      .standardSampleLocations = true,
      .optimalBufferCopyOffsetAlignment = 1,
      .optimalBufferCopyRowPitchAlignment = 1,
      .nonCoherentAtomSize = 64,

      /* Vulkan 1.0 sparse properties */
      .sparseResidencyNonResidentStrict = true,
      .sparseResidencyAlignedMipSize = false,
      .sparseResidencyStandard2DBlockShape = true,

      /* We can implement the standard block size for MSAA 4x but maybe not MSAA
       * 2x?
       */
      .sparseResidencyStandard2DMultisampleBlockShape = false,

      /* As far as I can tell, there is no way to implement this on G13. This
       * is a shame because D3D12 requires it for FL12.2.
       */
      .sparseResidencyStandard3DBlockShape = false,

      /* Vulkan 1.1 properties */
      .subgroupSize = 32,
      .subgroupSupportedStages =
         VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS,
      .subgroupSupportedOperations =
         VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
         VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT |
         VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
         VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
         VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR |
         VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
         VK_SUBGROUP_FEATURE_CLUSTERED_BIT |
         VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT_KHR,
      .subgroupQuadOperationsInAllStages = true,
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY,
      .maxMultiviewViewCount = HK_MAX_MULTIVIEW_VIEW_COUNT,
      .maxMultiviewInstanceIndex = UINT32_MAX,
      .maxPerSetDescriptors = UINT32_MAX,
      .maxMemoryAllocationSize = (1ull << 37),

      /* Vulkan 1.2 properties */
      .supportedDepthResolveModes =
         VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_AVERAGE_BIT |
         VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT,
      .supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT |
                                      VK_RESOLVE_MODE_MIN_BIT |
                                      VK_RESOLVE_MODE_MAX_BIT,
      .independentResolveNone = true,
      .independentResolve = true,
      .driverID = VK_DRIVER_ID_MESA_HONEYKRISP,
      .conformanceVersion = (VkConformanceVersion){1, 4, 0, 0},
      .denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      .roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      .shaderSignedZeroInfNanPreserveFloat16 = true,
      .shaderSignedZeroInfNanPreserveFloat32 = true,
      .shaderSignedZeroInfNanPreserveFloat64 = false,
      .shaderDenormPreserveFloat16 = true,
      .shaderDenormPreserveFloat32 = false,
      .shaderDenormPreserveFloat64 = false,
      .shaderDenormFlushToZeroFloat16 = false,
      .shaderDenormFlushToZeroFloat32 = true,
      .shaderDenormFlushToZeroFloat64 = false,
      .shaderRoundingModeRTEFloat16 = true,
      .shaderRoundingModeRTEFloat32 = true,
      .shaderRoundingModeRTEFloat64 = false,
      .shaderRoundingModeRTZFloat16 = false,
      .shaderRoundingModeRTZFloat32 = false,
      .shaderRoundingModeRTZFloat64 = false,
      .maxUpdateAfterBindDescriptorsInAllPools = UINT32_MAX,
      .shaderUniformBufferArrayNonUniformIndexingNative = true,
      .shaderSampledImageArrayNonUniformIndexingNative = true,
      .shaderStorageBufferArrayNonUniformIndexingNative = true,
      .shaderStorageImageArrayNonUniformIndexingNative = true,
      .shaderInputAttachmentArrayNonUniformIndexingNative = true,
      .robustBufferAccessUpdateAfterBind = true,
      .quadDivergentImplicitLod = false,
      .maxPerStageDescriptorUpdateAfterBindSamplers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindSampledImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindStorageImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindInputAttachments =
         HK_MAX_DESCRIPTORS,
      .maxPerStageUpdateAfterBindResources = UINT32_MAX,
      .maxDescriptorSetUpdateAfterBindSamplers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindUniformBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindStorageBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindSampledImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindStorageImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindInputAttachments = HK_MAX_DESCRIPTORS,
      .filterMinmaxSingleComponentFormats = instance->drirc.misc.fake_minmax,
      .filterMinmaxImageComponentMapping = false,
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,
      .framebufferIntegerColorSampleCounts = sample_counts,

      /* Vulkan 1.3 properties */
      .minSubgroupSize = 32,
      .maxSubgroupSize = 32,
      .maxComputeWorkgroupSubgroups = 1024 / 32,
      /* Apple9 executes fixed 32-lane subgroups, including explicitly
       * required size 32 in compute pipelines. */
      .requiredSubgroupSizeStages = dev->params.gpu_generation >= 16
                                      ? VK_SHADER_STAGE_COMPUTE_BIT : 0,
      .maxInlineUniformBlockSize = HK_MAX_INLINE_UNIFORM_BLOCK_SIZE,
      .maxPerStageDescriptorInlineUniformBlocks = 32,
      .maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = 32,
      .maxDescriptorSetInlineUniformBlocks = 6 * 32,
      .maxDescriptorSetUpdateAfterBindInlineUniformBlocks = 6 * 32,
      .maxInlineUniformTotalSize = 1 << 16,
      .integerDotProduct4x8BitPackedUnsignedAccelerated = false,
      .integerDotProduct4x8BitPackedSignedAccelerated = false,
      .integerDotProduct4x8BitPackedMixedSignednessAccelerated = false,
      .storageTexelBufferOffsetAlignmentBytes = HK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .storageTexelBufferOffsetSingleTexelAlignment = true,
      .uniformTexelBufferOffsetAlignmentBytes = HK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .uniformTexelBufferOffsetSingleTexelAlignment = true,
      .maxBufferSize = HK_MAX_BUFFER_SIZE,

      /* Vulkan 1.4 properties */
      .dynamicRenderingLocalReadDepthStencilAttachments = false,
      .dynamicRenderingLocalReadMultisampledAttachments = true,

      /* VK_KHR_push_descriptor */
      .maxPushDescriptors = HK_MAX_PUSH_DESCRIPTORS,

      /* VK_EXT_custom_border_color */
      .maxCustomBorderColorSamplers = 4000,

      /* VK_EXT_extended_dynamic_state3 */
      .dynamicPrimitiveTopologyUnrestricted = true,

      /* VK_EXT_graphics_pipeline_library */
      .graphicsPipelineLibraryFastLinking = true,
      .graphicsPipelineLibraryIndependentInterpolationDecoration = true,

      /* VK_EXT_host_image_copy */

      /* VK_KHR_line_rasterization */
      .lineSubPixelPrecisionBits = 8,

      /* VK_KHR_maintenance5 */
      .earlyFragmentMultisampleCoverageAfterSampleCounting = false,
      .earlyFragmentSampleMaskTestBeforeSampleCounting = true,
      .depthStencilSwizzleOneSupport = true,
      .polygonModePointSize = false,
      .nonStrictSinglePixelWideLinesUseParallelogram = false,
      .nonStrictWideLinesUseParallelogram = false,

      /* VK_KHR_maintenance6 */
      .blockTexelViewCompatibleMultipleLayers = false,
      .maxCombinedImageSamplerDescriptorCount = 3,
      .fragmentShadingRateClampCombinerInputs = false,

      /* VK_KHR_maintenance7 */
      .robustFragmentShadingRateAttachmentAccess = false,
      .separateDepthStencilAttachmentAccess = false, /* TODO */
      .maxDescriptorSetTotalUniformBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetTotalStorageBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetTotalBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS,
      .maxDescriptorSetUpdateAfterBindTotalUniformBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindTotalStorageBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindTotalBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS,

      /* VK_KHR_maintenance9 */
      .image2DViewOf3DSparse = false,
      .defaultVertexAttributeValue =
         VK_DEFAULT_VERTEX_ATTRIBUTE_VALUE_ZERO_ZERO_ZERO_ONE_KHR,

      /* VK_EXT_map_memory_placed */
      .minPlacedMemoryMapAlignment = os_page_size,

      /* VK_EXT_multi_draw */
      .maxMultiDrawCount = UINT16_MAX,

      /* VK_KHR_pipeline_binary
       *
       * InternalCache properties are set by
       * hk_physical_device_init_pipeline_cache()
       */
      .pipelineBinaryCompressedData = false,

      /* VK_EXT_pipeline_robustness */
      .defaultRobustnessStorageBuffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .defaultRobustnessUniformBuffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .defaultRobustnessVertexInputs =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .defaultRobustnessImages =
         VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2_EXT,

      /* VK_EXT_physical_device_drm gets populated later */

      /* VK_EXT_provoking_vertex */
      .provokingVertexModePerPipeline = true,
      .transformFeedbackPreservesTriangleFanProvokingVertex = true,

      /* VK_KHR_robustness2 */
      .robustStorageBufferAccessSizeAlignment = HK_SSBO_BOUNDS_CHECK_ALIGNMENT,
      .robustUniformBufferAccessSizeAlignment = HK_MIN_UBO_ALIGNMENT,

      /* VK_EXT_sample_locations */
      .sampleLocationSampleCounts = sample_counts & ~VK_SAMPLE_COUNT_1_BIT,
      .maxSampleLocationGridSize = (VkExtent2D){1, 1},
      .sampleLocationCoordinateRange[0] = 0.0f,
      .sampleLocationCoordinateRange[1] = 0.9375f,
      .sampleLocationSubPixelBits = 4,
      .variableSampleLocations = false,

      /* VK_EXT_shader_object */
      .shaderBinaryVersion = 0,

      /* VK_EXT_transform_feedback */
      .maxTransformFeedbackStreams = 4,
      .maxTransformFeedbackBuffers = 4,
      .maxTransformFeedbackBufferSize = UINT32_MAX,
      .maxTransformFeedbackStreamDataSize = 2048,
      .maxTransformFeedbackBufferDataSize = 512,
      .maxTransformFeedbackBufferDataStride = 2048,
      .transformFeedbackQueries = true,
      .transformFeedbackStreamsLinesTriangles = true,
      .transformFeedbackRasterizationStreamSelect = true,
      .transformFeedbackDraw = false,

      /* VK_KHR_vertex_attribute_divisor */
      .maxVertexAttribDivisor = UINT32_MAX,
      .supportsNonZeroFirstInstance = true,

      /* VK_KHR_fragment_shader_barycentric */
      .triStripVertexOrderIndependentOfProvokingVertex = false,

      /* VK_EXT_blend_operation_advanced */
      .advancedBlendMaxColorAttachments = HK_MAX_RTS,
      .advancedBlendIndependentBlend = true,
      .advancedBlendNonPremultipliedSrcColor = true,
      .advancedBlendNonPremultipliedDstColor = true,
      .advancedBlendCorrelatedOverlap = true,
      .advancedBlendAllOperations = true,
   };

   snprintf(properties->deviceName, sizeof(properties->deviceName), "%s",
            (strlen(instance->drirc.debug.force_vk_devicename) > 0) ?
            instance->drirc.debug.force_vk_devicename : dev->name);

   /* VK_EXT_shader_module_identifier */
   static_assert(sizeof(vk_shaderModuleIdentifierAlgorithmUUID) ==
                 sizeof(properties->shaderModuleIdentifierAlgorithmUUID));
   memcpy(properties->shaderModuleIdentifierAlgorithmUUID,
          vk_shaderModuleIdentifierAlgorithmUUID,
          sizeof(properties->shaderModuleIdentifierAlgorithmUUID));

   uint8_t dev_uuid[VK_UUID_SIZE];
   agx_get_device_uuid(dev, &dev_uuid);
   static_assert(sizeof(dev_uuid) == VK_UUID_SIZE);
   memcpy(properties->deviceUUID, &dev_uuid, VK_UUID_SIZE);
   static_assert(sizeof(instance->driver_build_sha) >= VK_UUID_SIZE);
   memcpy(properties->driverUUID, instance->driver_build_sha, VK_UUID_SIZE);

   strncpy(properties->driverName, "Honeykrisp", VK_MAX_DRIVER_NAME_SIZE);
   snprintf(properties->driverInfo, VK_MAX_DRIVER_INFO_SIZE,
            "Mesa " PACKAGE_VERSION MESA_GIT_SHA1);

   /* We don't use the layouts ATM so just report all layouts from
    * extensions that we support as compatible.
    */
   static const VkImageLayout supported_layouts[] = {
      VK_IMAGE_LAYOUT_GENERAL, /* required by spec */
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PREINITIALIZED,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
      // VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
      VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
   };

   properties->pCopySrcLayouts = (VkImageLayout *)supported_layouts;
   properties->copySrcLayoutCount = ARRAY_SIZE(supported_layouts);
   properties->pCopyDstLayouts = (VkImageLayout *)supported_layouts;
   properties->copyDstLayoutCount = ARRAY_SIZE(supported_layouts);

   /* We're a UMR so we can always map every kind of memory */
   properties->identicalMemoryTypeRequirements = true;

   {
      blake3_hasher blake3_ctx;
      uint8_t blake3[BLAKE3_KEY_LEN];

      _mesa_blake3_init(&blake3_ctx);
      /* Make sure we don't match with other vendors */
      const char *driver = "honeykrisp-v1";
      _mesa_blake3_update(&blake3_ctx, driver, strlen(driver));
      _mesa_blake3_final(&blake3_ctx, blake3);

      memcpy(properties->optimalTilingLayoutUUID, blake3, VK_UUID_SIZE);
   }
}

static void
hk_physical_device_init_pipeline_cache(struct hk_physical_device *pdev)
{
   struct hk_instance *instance = hk_physical_device_instance(pdev);

   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   _mesa_blake3_update(&blake3_ctx, instance->driver_build_sha,
                     sizeof(instance->driver_build_sha));

   const uint64_t compiler_flags = hk_physical_device_compiler_flags(pdev);
   _mesa_blake3_update(&blake3_ctx, &compiler_flags, sizeof(compiler_flags));

   unsigned char blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   static_assert(BLAKE3_KEY_LEN >= VK_UUID_SIZE);
   memcpy(pdev->vk.properties.pipelineCacheUUID, blake3, VK_UUID_SIZE);
   memcpy(pdev->vk.properties.shaderBinaryUUID, blake3, VK_UUID_SIZE);

#ifdef ENABLE_SHADER_CACHE
   char renderer[10];
   ASSERTED int len =
      snprintf(renderer, sizeof(renderer), "HK_G%u%c_",
               pdev->dev.params.gpu_generation, pdev->dev.params.gpu_variant);

   assert(len == sizeof(renderer) - 2);

   char timestamp[BLAKE3_HEX_LEN];
   _mesa_blake3_format(timestamp, instance->driver_build_sha);

   const uint64_t driver_flags = hk_physical_device_compiler_flags(pdev);
   pdev->vk.disk_cache = disk_cache_create(renderer, timestamp, driver_flags);
   if (pdev->vk.disk_cache != NULL) {
      pdev->vk.properties.pipelineBinaryInternalCache = true;
      pdev->vk.properties.pipelineBinaryInternalCacheControl = true;
      pdev->vk.properties.pipelineBinaryPrefersInternalCache = true;
      pdev->vk.properties.pipelineBinaryPrecompiledInternalCache = true;
   }
#endif
}

static void
hk_physical_device_free_disk_cache(struct hk_physical_device *pdev)
{
#ifdef ENABLE_SHADER_CACHE
   if (pdev->vk.disk_cache) {
      disk_cache_destroy(pdev->vk.disk_cache);
      pdev->vk.disk_cache = NULL;
   }
#else
   assert(pdev->vk.disk_cache == NULL);
#endif
}

static uint64_t
hk_get_sysmem_heap_size(struct hk_physical_device *pdev)
{
   if (pdev->sysmem)
      return pdev->sysmem;

   struct hk_instance *instance = hk_physical_device_instance(pdev);
   return os_get_gpu_heap_size(instance->drirc.misc.heap_memory_percent,
                               &instance->drirc.misc.heap_memory_percent);
}

static uint64_t
hk_get_sysmem_heap_available(struct hk_physical_device *pdev)
{
   assert(pdev->sysmem);

   uint64_t total_used = 0;
   for (unsigned i = 0; i < pdev->mem_heap_count; i++) {
      const struct hk_memory_heap *heap = &pdev->mem_heaps[i];
      uint64_t used = p_atomic_read(&heap->used);
      total_used += used;
   }

   return pdev->sysmem - total_used;
}

/* The native Apple9 path implements a subset of Honeykrisp's Apple8 feature
 * set. Everything Apple9 advertises is listed here explicitly, so enabling a
 * new Apple8 capability never silently exposes it on Apple9.
 */
static void
hk_apple9_restrict_caps(struct vk_device_extension_table *ext,
                        struct vk_features *feat, struct vk_properties *props)
{
   const struct vk_device_extension_table all = *ext;
   const struct vk_features all_feat = *feat;

   *ext = (struct vk_device_extension_table){
      /* Vulkan 1.1 */
      .KHR_16bit_storage = true,
      .KHR_bind_memory2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_descriptor_update_template = true,
      .KHR_device_group = true,
      .KHR_external_fence = true,
      .KHR_external_fence_fd = true,
      .KHR_external_memory = true,
      .KHR_external_memory_fd = true,
      .KHR_external_semaphore = true,
      .KHR_external_semaphore_fd = true,
      .KHR_get_memory_requirements2 = true,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_multiview = true,
      .KHR_relaxed_block_layout = true,
      .KHR_sampler_ycbcr_conversion = true,
      .KHR_shader_draw_parameters = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_variable_pointers = true,

      /* Vulkan 1.2 */
      .KHR_8bit_storage = true,
      .EXT_descriptor_indexing = true,
      .KHR_draw_indirect_count = true,
      .KHR_shader_float16_int8 = true,
      .KHR_buffer_device_address = true,
      .KHR_create_renderpass2 = true,
      .KHR_depth_stencil_resolve = true,
      .KHR_driver_properties = true,
      .KHR_image_format_list = true,
      .KHR_imageless_framebuffer = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_separate_depth_stencil_layouts = true,
      .KHR_shader_float_controls = true,
      .KHR_spirv_1_4 = true,
      .KHR_timeline_semaphore = true,
      .KHR_uniform_buffer_standard_layout = true,
      .KHR_vulkan_memory_model = true,
      .EXT_host_query_reset = true,
      .EXT_scalar_block_layout = true,
      .EXT_separate_stencil_usage = true,
      .EXT_shader_viewport_index_layer = true,

      /* Vulkan 1.3 */
      .KHR_copy_commands2 = true,
      .KHR_dynamic_rendering = true,
      .KHR_format_feature_flags2 = true,
      .KHR_maintenance4 = true,
      .KHR_shader_integer_dot_product = true,
      .KHR_shader_non_semantic_info = true,
      .KHR_shader_terminate_invocation = true,
      .KHR_synchronization2 = true,
      .KHR_zero_initialize_workgroup_memory = true,
      .EXT_4444_formats = true,
      .EXT_extended_dynamic_state = true,
      .EXT_extended_dynamic_state2 = true,
      .EXT_image_robustness = true,
      .EXT_inline_uniform_block = true,
      .EXT_pipeline_creation_cache_control = true,
      .EXT_pipeline_creation_feedback = true,
      .EXT_private_data = true,
      .EXT_shader_demote_to_helper_invocation = true,
      .EXT_subgroup_size_control = true,
      .EXT_texel_buffer_alignment = true,
      .EXT_tooling_info = true,

      /* Vulkan 1.4 */
      .KHR_dynamic_rendering_local_read = true,
      .KHR_global_priority = true,
      .KHR_index_type_uint8 = true,
      .KHR_line_rasterization = true,
      .KHR_maintenance5 = true,
      .KHR_maintenance6 = true,
      .KHR_push_descriptor = true,
      .KHR_shader_float_controls2 = true,
      .KHR_shader_subgroup_rotate = true,
      .KHR_vertex_attribute_divisor = true,
      .EXT_global_priority = true,
      .EXT_global_priority_query = true,
      .EXT_index_type_uint8 = true,
      .EXT_line_rasterization = true,
      .EXT_pipeline_robustness = true,
      .EXT_robustness2 = true,
      .KHR_robustness2 = true,
      .EXT_transform_feedback = true,
      .EXT_vertex_attribute_divisor = true,

      /* Other extensions implemented by the common runtime or CPU paths */
      .KHR_calibrated_timestamps = true,
      .KHR_maintenance7 = true,
      .KHR_shader_maximal_reconvergence = true,
      .KHR_shader_quad_control = true,
      .KHR_workgroup_memory_explicit_layout = true,
      .KHR_maintenance8 = true,
      .KHR_maintenance9 = true,
      .KHR_pipeline_executable_properties = true,
      .KHR_shader_subgroup_extended_types = true,
      .EXT_color_write_enable = true,
      .EXT_debug_marker = true,
      .EXT_hdr_metadata = true,
      .EXT_map_memory_placed = true,
      .EXT_shader_replicated_composites = true,
      .EXT_depth_bias_control = true,
      .EXT_depth_clip_control = true,
      .EXT_image_2d_view_of_3d = true,
      .EXT_primitive_topology_list_restart = true,
      .EXT_provoking_vertex = true,
      .EXT_mutable_descriptor_type = true,
      .EXT_shader_subgroup_ballot = true,
      .EXT_shader_subgroup_vote = true,
      .VALVE_mutable_descriptor_type = true,
      .KHR_shader_relaxed_extended_instruction = true,
      .KHR_shader_subgroup_uniform_control_flow = true,
      .EXT_attachment_feedback_loop_layout = true,
      .EXT_calibrated_timestamps = true,
      .EXT_dynamic_rendering_unused_attachments = true,
      .EXT_multi_draw = true,
      .EXT_non_seamless_cube_map = true,
      .EXT_shader_module_identifier = true,
      .EXT_shader_stencil_export = true,
      .KHR_load_store_op_none = true,
      .KHR_map_memory2 = true,
      .KHR_shader_expect_assume = true,
      .EXT_buffer_device_address = true,
      .EXT_depth_clip_enable = true,
      .EXT_external_memory_dma_buf = true,
      .EXT_image_drm_format_modifier = true,
      .EXT_load_store_op_none = true,
      .EXT_physical_device_drm = true,
      .EXT_queue_family_foreign = true,
      .GOOGLE_decorate_string = true,
      .GOOGLE_hlsl_functionality1 = true,
      .GOOGLE_user_type = true,
      .EXT_pipeline_protected_access = true,
      .EXT_image_view_min_lod = true,
      .KHR_compute_shader_derivatives = true,
      .EXT_blend_operation_advanced = true,
      /* Deferred shaders compile vertex fetch for the bound input state. */
      .EXT_vertex_input_dynamic_state = true,
      .EXT_extended_dynamic_state3 = true,
      .EXT_graphics_pipeline_library = true,
      .KHR_pipeline_library = true,
      .EXT_shader_object = true,
      .EXT_sample_locations = true,
      .EXT_shader_uniform_buffer_unsized_array = true,
      .KHR_pipeline_binary = true,
      /* Host copies use the same ail tiling as GPU sampling; host transfer
       * images are never compressed. */
      .EXT_host_image_copy = true,
      .EXT_shader_atomic_float = true,
      /* EXT_mesh_shader stays hidden: its workgroups still run as compute
       * rather than on the native mesh pipeline. */
      .EXT_attachment_feedback_loop_dynamic_state = true,
      .EXT_depth_clamp_zero_one = true,
      .EXT_depth_clamp_control = true,
      /* Resolves are vk_meta blits, which honor the transfer function
       * flags. */
      .KHR_maintenance10 = true,

      /* WSI */
      .KHR_incremental_present = all.KHR_incremental_present,
      .KHR_present_id = all.KHR_present_id,
      .KHR_present_id2 = all.KHR_present_id2,
      .KHR_present_wait = all.KHR_present_wait,
      .KHR_present_wait2 = all.KHR_present_wait2,
      .KHR_swapchain = all.KHR_swapchain,
      .KHR_swapchain_mutable_format = all.KHR_swapchain_mutable_format,
      .KHR_swapchain_maintenance1 = all.KHR_swapchain_maintenance1,
      .EXT_swapchain_maintenance1 = all.EXT_swapchain_maintenance1,
      .EXT_present_timing = all.EXT_present_timing,
      .GOOGLE_display_timing = all.GOOGLE_display_timing,
   };

#define KEEP(x) .x = all_feat.x
   *feat = (struct vk_features){
      /* Vulkan 1.0 */
      KEEP(robustBufferAccess),
      KEEP(fullDrawIndexUint32),
      KEEP(independentBlend),
      KEEP(sampleRateShading),
      KEEP(logicOp),
      KEEP(multiDrawIndirect),
      KEEP(drawIndirectFirstInstance),
      KEEP(depthClamp),
      KEEP(depthBiasClamp),
      KEEP(alphaToOne),
      KEEP(samplerAnisotropy),
      KEEP(textureCompressionETC2),
      KEEP(textureCompressionBC),
      KEEP(textureCompressionASTC_LDR),
      KEEP(occlusionQueryPrecise),
      KEEP(fragmentStoresAndAtomics),
      KEEP(shaderStorageImageExtendedFormats),
      KEEP(shaderImageGatherExtended),
      KEEP(shaderUniformBufferArrayDynamicIndexing),
      KEEP(shaderSampledImageArrayDynamicIndexing),
      KEEP(shaderStorageBufferArrayDynamicIndexing),
      KEEP(shaderStorageImageArrayDynamicIndexing),
      KEEP(shaderClipDistance),
      KEEP(inheritedQueries),
      KEEP(imageCubeArray),
      KEEP(dualSrcBlend),
      KEEP(fillModeNonSolid),
      KEEP(geometryShader),
      KEEP(pipelineStatisticsQuery),
      KEEP(tessellationShader),
      KEEP(multiViewport),
      KEEP(shaderTessellationAndGeometryPointSize),
      KEEP(wideLines),
      KEEP(largePoints),
      KEEP(vertexPipelineStoresAndAtomics),
      KEEP(shaderStorageImageReadWithoutFormat),
      KEEP(shaderStorageImageWriteWithoutFormat),
      KEEP(shaderCullDistance),
      KEEP(shaderInt64),
      KEEP(shaderInt16),
      KEEP(shaderResourceMinLod),

      /* Vulkan 1.1 */
      KEEP(multiview),
      KEEP(storageBuffer16BitAccess),
      KEEP(uniformAndStorageBuffer16BitAccess),
      KEEP(storagePushConstant16),
      KEEP(variablePointersStorageBuffer),
      KEEP(variablePointers),
      KEEP(shaderDrawParameters),

      /* Vulkan 1.2 */
      KEEP(samplerMirrorClampToEdge),
      /* Only the hk_fake_minmax driconf for vkd3d enables it, as upstream */
      KEEP(samplerFilterMinmax),
      KEEP(drawIndirectCount),
      KEEP(storageBuffer8BitAccess),
      KEEP(uniformAndStorageBuffer8BitAccess),
      KEEP(storagePushConstant8),
      KEEP(shaderFloat16),
      KEEP(shaderInt8),
      KEEP(uniformBufferStandardLayout),
      KEEP(scalarBlockLayout),
      KEEP(imagelessFramebuffer),
      KEEP(separateDepthStencilLayouts),
      KEEP(hostQueryReset),
      KEEP(timelineSemaphore),
      KEEP(bufferDeviceAddress),
      KEEP(vulkanMemoryModel),
      KEEP(vulkanMemoryModelDeviceScope),
      KEEP(shaderUniformTexelBufferArrayDynamicIndexing),
      KEEP(shaderStorageTexelBufferArrayDynamicIndexing),
      /* Sampled images beyond the per-draw tables, dynamically indexed or
       * late bound are read through descriptor memory; other descriptor
       * types always are. */
      KEEP(descriptorIndexing),
      KEEP(shaderInputAttachmentArrayDynamicIndexing),
      KEEP(shaderUniformBufferArrayNonUniformIndexing),
      KEEP(shaderSampledImageArrayNonUniformIndexing),
      KEEP(shaderStorageBufferArrayNonUniformIndexing),
      KEEP(shaderStorageImageArrayNonUniformIndexing),
      KEEP(shaderInputAttachmentArrayNonUniformIndexing),
      KEEP(shaderUniformTexelBufferArrayNonUniformIndexing),
      KEEP(shaderStorageTexelBufferArrayNonUniformIndexing),
      KEEP(descriptorBindingUniformBufferUpdateAfterBind),
      KEEP(descriptorBindingSampledImageUpdateAfterBind),
      KEEP(descriptorBindingStorageImageUpdateAfterBind),
      KEEP(descriptorBindingStorageBufferUpdateAfterBind),
      KEEP(descriptorBindingUniformTexelBufferUpdateAfterBind),
      KEEP(descriptorBindingStorageTexelBufferUpdateAfterBind),
      KEEP(descriptorBindingUpdateUnusedWhilePending),
      KEEP(descriptorBindingPartiallyBound),
      KEEP(descriptorBindingVariableDescriptorCount),
      KEEP(runtimeDescriptorArray),
      /* Inline uniform data is read from descriptor memory. */
      KEEP(descriptorBindingInlineUniformBlockUpdateAfterBind),
      KEEP(shaderOutputLayer),
      KEEP(shaderOutputViewportIndex),
      KEEP(subgroupBroadcastDynamicId),
      /* No 8/16/64-bit shader types are exposed, so this holds trivially. */
      KEEP(shaderSubgroupExtendedTypes),

      /* Vulkan 1.3 */
      KEEP(robustImageAccess),
      KEEP(inlineUniformBlock),
      KEEP(pipelineCreationCacheControl),
      KEEP(privateData),
      KEEP(shaderDemoteToHelperInvocation),
      KEEP(shaderTerminateInvocation),
      KEEP(subgroupSizeControl),
      KEEP(computeFullSubgroups),
      .computeDerivativeGroupLinear = true,
      .computeDerivativeGroupQuads = false,
      KEEP(synchronization2),
      KEEP(shaderZeroInitializeWorkgroupMemory),
      KEEP(dynamicRendering),
      KEEP(shaderIntegerDotProduct),
      KEEP(maintenance4),

      /* Vulkan 1.4 */
      KEEP(globalPriorityQuery),
      KEEP(shaderFloatControls2),
      KEEP(bresenhamLines),
      KEEP(dynamicRenderingLocalRead),
      KEEP(samplerYcbcrConversion),
      KEEP(indexTypeUint8),
      KEEP(maintenance5),
      KEEP(maintenance6),
      KEEP(pushDescriptor),
      KEEP(pipelineRobustness),
      KEEP(robustBufferAccess2),
      KEEP(robustImageAccess2),
      KEEP(nullDescriptor),
      KEEP(transformFeedback),
      KEEP(geometryStreams),
      KEEP(shaderSubgroupRotate),
      KEEP(shaderSubgroupRotateClustered),
      KEEP(vertexAttributeInstanceRateDivisor),
      KEEP(vertexAttributeInstanceRateZeroDivisor),
      KEEP(extendedDynamicState2LogicOp),
      /* Tessellation reads the patch size from dynamic state at draw. */
      KEEP(extendedDynamicState2PatchControlPoints),
      KEEP(hostImageCopy),
      /* Device memory has native FP32 atomic addition, which image atomics
       * also use. Shared memory has exchange but no addition. */
      .shaderBufferFloat32Atomics = true,
      .shaderBufferFloat32AtomicAdd = true,
      .shaderSharedFloat32Atomics = true,
      .shaderImageFloat32Atomics = true,
      .shaderImageFloat32AtomicAdd = true,

      /* Extensions */
      KEEP(mutableDescriptorType),
      KEEP(pipelineExecutableInfo),
      KEEP(maintenance8),
      KEEP(maintenance9),
      KEEP(shaderReplicatedComposites),
      KEEP(memoryMapPlaced),
      KEEP(memoryUnmapReserve),
      KEEP(colorWriteEnable),
      /* Lowered in the last pre-rasterization stage. */
      .depthClipControl = true,
      KEEP(depthBiasControl),
      KEEP(leastRepresentableValueForceUnormRepresentation),
      KEEP(depthBiasExact),
      KEEP(image2DViewOf3D),
      KEEP(sampler2DViewOf3D),
      KEEP(primitiveTopologyListRestart),
      KEEP(provokingVertexLast),
      KEEP(transformFeedbackPreservesProvokingVertex),
      KEEP(maintenance7),
      KEEP(shaderMaximalReconvergence),
      KEEP(shaderQuadControl),
      KEEP(workgroupMemoryExplicitLayout),
      KEEP(workgroupMemoryExplicitLayoutScalarBlockLayout),
      KEEP(workgroupMemoryExplicitLayout8BitAccess),
      KEEP(workgroupMemoryExplicitLayout16BitAccess),
      KEEP(attachmentFeedbackLoopLayout),
      KEEP(dynamicRenderingUnusedAttachments),
      KEEP(multiDraw),
      KEEP(nonSeamlessCubeMap),
      KEEP(shaderModuleIdentifier),
      KEEP(shaderRelaxedExtendedInstruction),
      KEEP(shaderSubgroupUniformControlFlow),
      KEEP(formatA4R4G4B4),
      KEEP(formatA4B4G4R4),
      KEEP(depthClipEnable),
      KEEP(extendedDynamicState),
      KEEP(extendedDynamicState2),
      KEEP(texelBufferAlignment),
      KEEP(swapchainMaintenance1),
      KEEP(presentId),
      KEEP(presentId2),
      KEEP(presentWait),
      KEEP(presentWait2),
      KEEP(shaderExpectAssume),
      KEEP(pipelineProtectedAccess),
      /* Advanced blending reads the tile in the shader. */
      KEEP(advancedBlendCoherentOperations),
      KEEP(vertexInputDynamicState),
      KEEP(graphicsPipelineLibrary),
      KEEP(shaderObject),
      /* Blend, sample and vertex state specialise deferred shaders at draw
       * time; rasterizer state is emitted per draw. */
      KEEP(extendedDynamicState3TessellationDomainOrigin),
      KEEP(extendedDynamicState3DepthClampEnable),
      KEEP(extendedDynamicState3PolygonMode),
      KEEP(extendedDynamicState3RasterizationSamples),
      KEEP(extendedDynamicState3SampleMask),
      KEEP(extendedDynamicState3AlphaToCoverageEnable),
      KEEP(extendedDynamicState3AlphaToOneEnable),
      KEEP(extendedDynamicState3LogicOpEnable),
      KEEP(extendedDynamicState3ColorBlendEnable),
      KEEP(extendedDynamicState3ColorBlendEquation),
      KEEP(extendedDynamicState3ColorWriteMask),
      KEEP(extendedDynamicState3RasterizationStream),
      KEEP(extendedDynamicState3DepthClipEnable),
      KEEP(extendedDynamicState3SampleLocationsEnable),
      KEEP(extendedDynamicState3ProvokingVertexMode),
      KEEP(extendedDynamicState3LineRasterizationMode),
      .extendedDynamicState3DepthClipNegativeOneToOne = true,
      /* 2D views clamp gradient samples in hardware; other implicit samples
       * select the clamped LOD explicitly, without anisotropy. */
      .minLod = true,
      KEEP(shaderUniformBufferUnsizedArray),
      KEEP(pipelineBinaries),
      KEEP(presentTiming),
      KEEP(presentAtAbsoluteTime),
      KEEP(presentAtRelativeTime),
      /* Capture replay and sparse binding rebind memory into reserved
       * addresses. Shaders look up image residency in a table per image. */
      KEEP(bufferDeviceAddressCaptureReplay),
      KEEP(bufferDeviceAddressCaptureReplayEXT),
      KEEP(sparseBinding),
      KEEP(sparseResidencyBuffer),
      KEEP(sparseResidencyAliased),
      KEEP(sparseResidencyImage2D),
      KEEP(shaderResourceResidency),
      .attachmentFeedbackLoopDynamicState = true,
      .depthClampZeroOne = true,
      .depthClampControl = true,
      .maintenance10 = true,
   };
#undef KEEP

   /* HK_APPLE9_MESH_SHADER=1 exposes EXT_mesh_shader on the existing compute
    * path (task and mesh workgroups run as compute kernels, then a draw of
    * their output records); for testing until a native mesh launch exists.
    * Mesh-shader queries and multiview stay off. */
   if (debug_get_bool_option("HK_APPLE9_MESH_SHADER", false)) {
      ext->EXT_mesh_shader = true;
      feat->meshShader = true;
      feat->taskShader = true;
   }

   props->apiVersion = VK_MAKE_API_VERSION(0, 1, 4, VK_HEADER_VERSION);
   props->conformanceVersion = (VkConformanceVersion){0};

   /* Subgroup operations are lowered to native ballot and simd_shuffle. */
   props->subgroupSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT |
                                    VK_SHADER_STAGE_ALL_GRAPHICS;
   props->subgroupSupportedOperations =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
      VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
      VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_CLUSTERED_BIT |
      VK_SUBGROUP_FEATURE_QUAD_BIT | VK_SUBGROUP_FEATURE_ROTATE_BIT |
      VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT;
   props->subgroupQuadOperationsInAllStages = true;
   props->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;

   props->maxComputeSharedMemorySize = AGX_APPLE9_MAX_SHARED_BYTES;

   /* VK_KHR_maintenance10: the opaque black border of RGBA4 formats is
    * not established. */
   props->rgba4OpaqueBlackSwizzled = false;
   props->resolveSrgbFormatAppliesTransferFunction = true;
   props->resolveSrgbFormatSupportsTransferFunctionControl = true;

   /* VK_EXT_mesh_shader */
   props->maxTaskWorkGroupTotalCount = 1 << 22;
   for (unsigned i = 0; i < 3; ++i) {
      props->maxTaskWorkGroupCount[i] = 65535;
      props->maxTaskWorkGroupSize[i] = 128;
      props->maxMeshWorkGroupCount[i] = 65535;
      props->maxMeshWorkGroupSize[i] = 128;
   }
   props->maxTaskWorkGroupInvocations = 128;
   props->maxTaskPayloadSize = 16384;
   props->maxTaskSharedMemorySize = AGX_APPLE9_MAX_SHARED_BYTES;
   props->maxTaskPayloadAndSharedMemorySize = AGX_APPLE9_MAX_SHARED_BYTES;
   props->maxMeshWorkGroupTotalCount = 1 << 22;
   props->maxMeshWorkGroupInvocations = 128;
   props->maxMeshSharedMemorySize = AGX_APPLE9_MAX_SHARED_BYTES;
   props->maxMeshPayloadAndSharedMemorySize = AGX_APPLE9_MAX_SHARED_BYTES;
   props->maxMeshOutputMemorySize = 32768;
   props->maxMeshPayloadAndOutputMemorySize = 48128;
   props->maxMeshOutputComponents = 128;
   props->maxMeshOutputVertices = 256;
   props->maxMeshOutputPrimitives = 256;
   props->maxMeshOutputLayers = 8;
   props->maxMeshMultiviewViewCount = props->maxMultiviewViewCount;
   props->meshOutputPerVertexGranularity = 1;
   props->meshOutputPerPrimitiveGranularity = 1;
   props->maxPreferredTaskWorkGroupInvocations = 32;
   props->maxPreferredMeshWorkGroupInvocations = 32;
   props->prefersLocalInvocationVertexOutput = false;
   props->prefersLocalInvocationPrimitiveOutput = false;
   props->prefersCompactVertexOutput = true;
   props->prefersCompactPrimitiveOutput = true;
   /* Unbound sparse ranges map scratch pages, which reads observe. */
   props->sparseResidencyNonResidentStrict = false;
   /* With early fragment tests, occlusion counting precedes the sample
    * mask test. */
   props->earlyFragmentSampleMaskTestBeforeSampleCounting = false;
   /* FP32 arithmetic flushes natively. The Apple9 compiler canonicalizes
    * raw-result operations for explicit FTZ, including min/max, sign-bit
    * operations, subgroup reductions and floating atomic return values.
    * DenormPreserve shaders take guarded software paths around the
    * flushing arithmetic (agx_nir_lower_apple9_denorm_preserve). */
   props->shaderDenormFlushToZeroFloat32 = true;
   props->shaderDenormPreserveFloat32 = true;
   /* The rasterized interface carries at most this many user scalars. */
   props->maxVertexOutputComponents = AGX_APPLE9_MAX_VARYING_COMPONENTS;
   props->maxTessellationEvaluationOutputComponents =
      AGX_APPLE9_MAX_VARYING_COMPONENTS;
   props->maxGeometryOutputComponents = AGX_APPLE9_MAX_VARYING_COMPONENTS;
   props->maxFragmentInputComponents = AGX_APPLE9_MAX_VARYING_COMPONENTS;
   props->maxComputeWorkGroupCount[0] = 65535;

   /* The native 4x tile layout supports 64 bytes per pixel. Four RGBA32
    * targets fill it; eight targets require tile spilling or a smaller tile
    * layout that this path does not yet implement.
    */
   props->maxColorAttachments = 8;
   props->maxFragmentOutputAttachments = 8;


   fprintf(stderr, "WARNING: experimental Apple9 Honeykrisp; not conformant\n");
}

uint64_t hk_process_heap_used[VK_MAX_MEMORY_HEAPS];

VkResult
hk_create_drm_physical_device(struct vk_instance *_instance,
                              drmDevicePtr drm_device,
                              struct vk_physical_device **pdev_out)
{
   struct hk_instance *instance = (struct hk_instance *)_instance;
   VkResult result;

   if (!(drm_device->available_nodes & (1 << DRM_NODE_RENDER)) ||
       drm_device->bustype != DRM_BUS_PLATFORM)
      return VK_ERROR_INCOMPATIBLE_DRIVER;

   const char *path = drm_device->nodes[DRM_NODE_RENDER];
   int fd = open(path, O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      return vk_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                       "failed to open device %s", path);
   }

   drmVersionPtr version = drmGetVersion(fd);
   if (!version) {
      result =
         vk_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                   "failed to query kernel driver version for device %s", path);
      goto fail_fd;
   }

   bool is_asahi = (strcmp(version->name, "asahi") == 0);
   is_asahi |= strcmp(version->name, "virtio_gpu") == 0;
   drmFreeVersion(version);

   if (!is_asahi) {
      /* Fail silently */
      result = VK_ERROR_INCOMPATIBLE_DRIVER;
      goto fail_fd;
   }

   struct stat st;
   if (stat(drm_device->nodes[DRM_NODE_RENDER], &st)) {
      result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                         "fstat() failed on %s: %m",
                         drm_device->nodes[DRM_NODE_RENDER]);
      goto fail_fd;
   }
   const dev_t render_dev = st.st_rdev;

   struct hk_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);

   if (pdev == NULL) {
      result = vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_fd;
   }

   /* We're render-only */
   pdev->master_fd = -1;
   pdev->render_dev = render_dev;
   pdev->dev.fd = fd;

   if (!agx_open_device(NULL, &pdev->dev)) {
      /* Fail silently, for virtgpu */
      result = VK_ERROR_INCOMPATIBLE_DRIVER;
      goto fail_pdev_alloc;
   }

   if (pdev->dev.chip == AGX_CHIP_G16G &&
       !debug_get_bool_option("HK_EXPERIMENTAL_APPLE9", false)) {
      result = VK_ERROR_INCOMPATIBLE_DRIVER;
      goto fail_agx_device;
   }

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &hk_physical_device_entrypoints, true);
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_physical_device_entrypoints, false);

   struct vk_device_extension_table supported_extensions;
   hk_get_device_extensions(instance, &supported_extensions);

   struct vk_features supported_features;
   hk_get_device_features(&pdev->dev, instance, &supported_extensions,
                          &supported_features);

   struct vk_properties properties;
   hk_get_device_properties(&pdev->dev, instance, &properties);

   if (pdev->dev.chip == AGX_CHIP_G16G)
      hk_apple9_restrict_caps(&supported_extensions, &supported_features,
                              &properties);


   properties.drmHasRender = true;
   properties.drmRenderMajor = major(render_dev);
   properties.drmRenderMinor = minor(render_dev);

   result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                    &supported_extensions, &supported_features,
                                    &properties, &dispatch_table);
   if (result != VK_SUCCESS)
      goto fail_agx_device;

   /* Set by driconf for vkd3d-proton, see hk_apple9_arrayed_2d_views. */
   pdev->apple9_layered_2d = pdev->dev.chip == AGX_CHIP_G16G &&
      instance->drirc.misc.apple9_arrayed_2d_views;

   hk_physical_device_init_pipeline_cache(pdev);

   const char *hk_sysmem = os_get_option("HK_SYSMEM");
   if (hk_sysmem) {
      uint64_t sysmem = strtoll(hk_sysmem, NULL, 10);
      if (sysmem != LLONG_MIN && sysmem != LLONG_MAX) {
         pdev->sysmem = sysmem;
      }
   }

   uint64_t sysmem_size_B = hk_get_sysmem_heap_size(pdev);
   if (sysmem_size_B == 0) {
      result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to query total system memory");
      goto fail_disk_cache;
   }

   uint32_t sysmem_heap_idx = pdev->mem_heap_count++;
   pdev->mem_heaps[sysmem_heap_idx] = (struct hk_memory_heap){
      .size = sysmem_size_B,
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
   };

   pdev->mem_types[pdev->mem_type_count++] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .heapIndex = sysmem_heap_idx,
   };

   pdev->mem_types[pdev->mem_type_count++] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .heapIndex = sysmem_heap_idx,
   };

   assert(pdev->mem_heap_count <= ARRAY_SIZE(pdev->mem_heaps));
   assert(pdev->mem_type_count <= ARRAY_SIZE(pdev->mem_types));

   pdev->queue_families[pdev->queue_family_count++] = (struct hk_queue_family){
      .queue_flags = pdev->dev.chip == AGX_CHIP_G16G
                        ? VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                             VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT :
                     VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                     VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT,

      .queue_count = 1,
   };
   /* Vulkan 1.4 requires a second transfer-capable family without host image
    * copies. Its queues submit through their own kernel command queues. */
   if (pdev->dev.chip == AGX_CHIP_G16G) {
      pdev->queue_families[pdev->queue_family_count++] = (struct hk_queue_family){
         .queue_flags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
         .queue_count = 1,
      };
   }
   assert(pdev->queue_family_count <= ARRAY_SIZE(pdev->queue_families));

   unsigned st_idx = 0;
   pdev->syncobj_sync_type = vk_drm_syncobj_get_type(fd);
   pdev->sync_types[st_idx++] = &pdev->syncobj_sync_type;
   pdev->sync_types[st_idx++] = NULL;
   assert(st_idx <= ARRAY_SIZE(pdev->sync_types));
   pdev->vk.supported_sync_types = pdev->sync_types;

   result = hk_init_wsi(pdev);
   if (result != VK_SUCCESS)
      goto fail_disk_cache;

   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);
   *pdev_out = &pdev->vk;

   return VK_SUCCESS;

fail_disk_cache:
   hk_physical_device_free_disk_cache(pdev);
   vk_physical_device_finish(&pdev->vk);
fail_agx_device:
   agx_close_device(&pdev->dev);
fail_pdev_alloc:
   if (pdev->master_fd)
      close(pdev->master_fd);

   vk_free(&instance->vk.alloc, pdev);
fail_fd:
   close(fd);
   return result;
}

void
hk_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);

   hk_finish_wsi(pdev);

   if (pdev->master_fd >= 0)
      close(pdev->master_fd);

   simple_mtx_destroy(&pdev->debug_compile_lock);
   hk_physical_device_free_disk_cache(pdev);
   agx_close_device(&pdev->dev);
   vk_physical_device_finish(&pdev->vk);
   vk_free(&pdev->vk.instance->alloc, pdev);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceMemoryProperties2(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   struct hk_instance *instance = hk_physical_device_instance(pdev);

   pMemoryProperties->memoryProperties.memoryHeapCount = pdev->mem_heap_count;
   for (int i = 0; i < pdev->mem_heap_count; i++) {
      pMemoryProperties->memoryProperties.memoryHeaps[i] = (VkMemoryHeap){
         .size = pdev->mem_heaps[i].size,
         .flags = pdev->mem_heaps[i].flags,
      };
   }

   pMemoryProperties->memoryProperties.memoryTypeCount = pdev->mem_type_count;
   for (int i = 0; i < pdev->mem_type_count; i++) {
      pMemoryProperties->memoryProperties.memoryTypes[i] = pdev->mem_types[i];
   }

   vk_foreach_struct(ext, pMemoryProperties->pNext) {
      switch (ext->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: {
         VkPhysicalDeviceMemoryBudgetPropertiesEXT *p = (void *)ext;

         uint64_t sysmem_budget =
            pdev->sysmem ? hk_get_sysmem_heap_available(pdev) : 0;

         for (unsigned i = 0; i < pdev->mem_heap_count; i++) {
            const struct hk_memory_heap *heap = &pdev->mem_heaps[i];
            uint64_t used = p_atomic_read(&hk_process_heap_used[i]);

            /* From the Vulkan 1.3.278 spec:
             *
             *    "heapUsage is an array of VK_MAX_MEMORY_HEAPS VkDeviceSize
             *    values in which memory usages are returned, with one element
             *    for each memory heap. A heap’s usage is an estimate of how
             *    much memory the process is currently using in that heap."
             *
             * TODO: Include internal allocations?
             */
            p->heapUsage[i] = used;

            /* Set the budget at 90% to avoid thrashing. */
            float percent = 0.9f;

            uint64_t budget;
            if (sysmem_budget) {
               budget = MIN2(sysmem_budget + used, heap->size);
               budget = ROUND_DOWN_TO((uint64_t)(budget * percent), 1 << 20);
            } else {
               /* Scale the budget the same way the heap was scaled. */
               percent *= instance->drirc.misc.heap_memory_percent;

               budget = vk_physical_device_heap_budget_from_system(
                  &pdev->vk, percent, heap->size, used);
            }

            p->heapBudget[i] = budget;
         }

         /* From the Vulkan 1.3.278 spec:
          *
          *    "The heapBudget and heapUsage values must be zero for array
          *    elements greater than or equal to
          *    VkPhysicalDeviceMemoryProperties::memoryHeapCount. The
          *    heapBudget value must be non-zero for array elements less than
          *    VkPhysicalDeviceMemoryProperties::memoryHeapCount."
          */
         for (unsigned i = pdev->mem_heap_count; i < VK_MAX_MEMORY_HEAPS; i++) {
            p->heapBudget[i] = 0u;
            p->heapUsage[i] = 0u;
         }
         break;
      }
      default:
         vk_debug_ignored_stype(ext->sType);
         break;
      }
   }
}

static const VkQueueGlobalPriorityKHR hk_global_queue_priorities[] = {
   VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
   VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
   VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
   VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR,
};

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceQueueFamilyProperties2(
   VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
   VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties,
                          pQueueFamilyPropertyCount);

   for (uint8_t i = 0; i < pdev->queue_family_count; i++) {
      const struct hk_queue_family *queue_family = &pdev->queue_families[i];

      vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p)
      {
         p->queueFamilyProperties.queueFlags = queue_family->queue_flags;
         p->queueFamilyProperties.queueCount = queue_family->queue_count;
         p->queueFamilyProperties.timestampValidBits = 64;
         p->queueFamilyProperties.minImageTransferGranularity =
            (VkExtent3D){1, 1, 1};

         VkQueueFamilyGlobalPriorityPropertiesKHR *prio = vk_find_struct(
            p->pNext, QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR);
         if (prio) {
            STATIC_ASSERT(ARRAY_SIZE(hk_global_queue_priorities) <=
                          VK_MAX_GLOBAL_PRIORITY_SIZE_KHR);
            prio->priorityCount = ARRAY_SIZE(hk_global_queue_priorities);
            memcpy(&prio->priorities, hk_global_queue_priorities,
                   sizeof(hk_global_queue_priorities));
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceMultisamplePropertiesEXT(
   VkPhysicalDevice physicalDevice, VkSampleCountFlagBits samples,
   VkMultisamplePropertiesEXT *pMultisampleProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);

   if (samples & pdev->vk.properties.sampleLocationSampleCounts) {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){1, 1};
   } else {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){0, 0};
   }
}
