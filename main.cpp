#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#define WINDOW_NAME "Vulkan Tutorial"
#define APP_NAME WINDOW_NAME
#define WAYLAND_APP_ID "vulkan_tutorial"

const uint32_t HEIGHT = 480;
const uint32_t WIDTH = 640;
const int MAX_FRAMES_IN_FLIGHT = 2;
const uint32_t PARTICLE_COUNT = 1024;

const std::vector<const char *> VALIDATION_LAYERS = {"VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> REQUIRED_EXTENSIONS = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_MAINTENANCE_1_EXTENSION_NAME,
};

const VmaVulkanFunctions VULKAN_FUNCTIONS{
	.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
	.vkGetDeviceProcAddr = vkGetDeviceProcAddr,
};

const VkApplicationInfo APP_INFO{
	.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	.pApplicationName = "Vulkan Tutorial App",
	.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
	.pEngineName = "No Engine",
	.engineVersion = VK_MAKE_VERSION(1, 0, 0),
	.apiVersion = VK_API_VERSION_1_0,
};

struct UniformBufferObject {
	float deltaTime;
};

struct Particle {
	alignas(8) glm::vec2 position;
	alignas(8) glm::vec2 velocity;
	alignas(16) glm::vec4 color;

	static constexpr VkVertexInputBindingDescription getBindingDescription() {
		return {.binding = 0,
				.stride = sizeof(Particle),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
	}

	static constexpr std::array<VkVertexInputAttributeDescription, 2>
	getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescription{};
		attributeDescription[0] = {
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Particle, position),
		};
		attributeDescription[1] = {
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(Particle, color),
		};
		return attributeDescription;
	}
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {
	std::cerr << "[Validation layer] " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

const VkDebugUtilsMessengerCreateInfoEXT DEFAULT_DEBUG_UTIL_MESSENGER_CREATE_INFO{
	.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
	.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
					   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
					   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
	.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
	.pfnUserCallback = debugCallback,
	.pUserData = nullptr,
};

VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
	const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
								   VkDebugUtilsMessengerEXT debugMessenger,
								   const VkAllocationCallbacks *pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

const std::vector<VkPresentModeKHR> PRESENT_MODE_ORDER = {
	VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR,
	VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR};

static std::vector<char> readFile(const std::string &f) {
	std::ifstream file(f, std::ios::ate | std::ios::binary);

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

static void checkError(int result, const std::string &message) {
	const auto throwErr = [&](const std::string &errMsg) {
		throw std::runtime_error(message + " - " + errMsg);
	};
	switch (result) {
	case VK_SUCCESS:
	case VK_INCOMPLETE:
	case VK_SUBOPTIMAL_KHR:
		break;
	case VK_ERROR_INITIALIZATION_FAILED:
		throwErr("Vulkan loader not found!");
	case VK_ERROR_EXTENSION_NOT_PRESENT:
		throwErr("Extension specified is not present in device!");
	case VK_ERROR_FEATURE_NOT_PRESENT:
		throwErr("Feature(s) specified was/were not found!");
	case VK_ERROR_OUT_OF_HOST_MEMORY:
		throwErr("Ran out of host memory!");
	case VK_ERROR_OUT_OF_DEVICE_MEMORY:
		throwErr("Ran out of graphics device memory!");
	case VK_ERROR_OUT_OF_POOL_MEMORY:
		throwErr("Ran out of memory in the descriptor/command pool!");
	default:
		throwErr("Unknown error occured!");
	}
}

class ParticleApplication {
  public:
	bool frameBufferResized;
	uint32_t currFrame = 0;

	void run() { mainLoop(); }

	ParticleApplication() {
		initWindow();
		initVulkan();
	}

	~ParticleApplication() {
		// Swap chain dependents
		cleanupSwapChainDependents();
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);

		// Sync dependents
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vmaDestroyBuffer(allocator, uniformBuffers[i],
							 uniformBuffersMemories[i]);
			vmaDestroyBuffer(allocator, storageBuffers[i], storageBufferMemories[i]);
		}

		// Device dependents
		vkDestroyDescriptorPool(device, renderDescriptorPool, nullptr);
		vkDestroyDescriptorPool(device, computeDescriptorPool, nullptr);
		vkDestroyCommandPool(device, graphicsPool, nullptr);
		vkDestroyCommandPool(device, transferPool, nullptr);
		vkDestroyCommandPool(device, computePool, nullptr);
		vkDestroyDescriptorSetLayout(device, graphicsDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
		vmaDestroyAllocator(allocator);
		destroySyncObjects();

		// Instance dependents
		vkDestroyDevice(device, nullptr);
#if !(NDEBUG)
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
#endif
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}

  private:
	GLFWwindow *window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;

	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VmaAllocator allocator;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue transferQueue;
	VkQueue computeQueue;

	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	VkRenderPass renderPass;
	VkDescriptorSetLayout graphicsDescriptorSetLayout;
	VkDescriptorSetLayout computeDescriptorSetLayout;
	VkPipelineLayout graphicsPipelineLayout;
	VkPipeline graphicsPipeline;
	VkPipelineLayout computePipelineLayout;
	VkPipeline computePipeline;

	VkViewport viewport;
	VkRect2D scissor;

	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool graphicsPool;
	VkCommandPool transferPool;
	VkCommandPool computePool;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkSemaphore> computeFinishedSemaphores;
	std::vector<VkFence> graphicsInFlightFences;
	std::vector<VkFence> computeInFlightFences;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VmaAllocation> uniformBuffersMemories;
	std::vector<void *> uniformBuffersMapped;
	VkDescriptorPool renderDescriptorPool;
	std::vector<VkDescriptorSet> renderDescriptorSets;

	std::vector<VkBuffer> storageBuffers;
	std::vector<VmaAllocation> storageBufferMemories;
	VkDescriptorPool computeDescriptorPool;
	std::vector<VkDescriptorSet> computeDescriptorSets;

	VkImage depthImage;
	VmaAllocation depthImageMemory;
	VkImageView depthImageView;

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
	VkImage colorImage;
	VmaAllocation colorImageMemory;
	VkImageView colorImageView;

	void initWindow() {
		glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHintString(GLFW_X11_CLASS_NAME, "vulkan_tutorial");
		glfwWindowHintString(GLFW_WAYLAND_APP_ID, "vulkan_tutorial");
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan ComputeTutorial", nullptr,
								  nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, [](GLFWwindow *win, int w, int h) {
			auto app = (ParticleApplication *)glfwGetWindowUserPointer(win);
			app->frameBufferResized = true;
		});
	}

	void initVulkan() {
		createInstance();
		// Instance dependents
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSyncObjects();

		// Device dependents
		createGraphicsDescriptorSetLayout();
		createCommandPools();
		createComputeDescriptorSetLayout();
		createGraphicsPool();
		createComputePool();
		createGraphicsCommandBuffers();
		createAllocator();

		// Allocator dependents
		createUniformBuffers();
		createStorageBuffers();

		// Buffer/image/swapchain dependent
		createGraphicsDescriptorSets();
		createComputeDescriptorSets();

		// Swap chain dependents
		createSwapChainDependents();

		// Render pass dependents
		createGraphicsPipeline();
	}

	void createInstance() {
		auto glfwExtensions = getRequiredExtensions();

		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
											   extensions.data());
#if !(NDEBUG)
		checkError(checkValidationLayerSupport(), "Failed to create instance");
#endif

		VkInstanceCreateInfo instanceInfo {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#if !(NDEBUG)
			.pNext = &DEFAULT_DEBUG_UTIL_MESSENGER_CREATE_INFO,
#endif
			.pApplicationInfo = &APP_INFO,
#if !(NDEBUG)
			.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size(),
			.ppEnabledLayerNames = VALIDATION_LAYERS.data(),
#endif
			.enabledExtensionCount = (uint32_t)glfwExtensions.size(),
			.ppEnabledExtensionNames = glfwExtensions.data(),
		};
		VkResult result = vkCreateInstance(&instanceInfo, nullptr, &instance);
		checkError(result, "Failed to create instance");

#if !(NDEBUG)
		result = CreateDebugUtilsMessengerEXT(
			instance, &DEFAULT_DEBUG_UTIL_MESSENGER_CREATE_INFO, nullptr,
			&debugMessenger);
		checkError(result, "Failed to create debug messenger extension");
#endif
	}

	std::vector<const char *> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char **glfwExtensions =
			glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char *> extensions(glfwExtensions,
											 glfwExtensions + glfwExtensionCount);
#if !(NDEBUG)
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
		return extensions;
	}

	VkResult checkValidationLayerSupport() {
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char *layerName : VALIDATION_LAYERS) {
			bool supported = [&]() {
				for (const auto &layer : availableLayers) {
					if (strcmp(layer.layerName, layerName) == 0)
						return true;
				}
				return false;
			}();
			if (!supported)
				return VK_ERROR_LAYER_NOT_PRESENT;
		}
		return VK_SUCCESS;
	}

	void createSurface() {
		checkError(glfwCreateWindowSurface(instance, window, nullptr, &surface),
				   "Failed to create surface");
	}

	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		checkError(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr),
				   "Failed to pick physical devices");
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		std::optional<VkPhysicalDevice> candidate;
		for (const auto &dev : devices) {
			if (isDeviceSuitable(dev) == VK_SUCCESS) {
				candidate = dev;
				break;
			}
		}
		if (!candidate.has_value()) {
			throw std::runtime_error(
				"Failed to pick physical devices - no suitable device found!");
		}
		physicalDevice = *candidate;

		// Pick multisampling properties
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(*candidate, &props);
		VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts &
									props.limits.framebufferDepthSampleCounts;
		for (size_t i = 6; i >= 1; i--) {
			if (counts & (1 << i)) {
				msaaSamples = (VkSampleCountFlagBits)(1 << i);
				break;
			}
		}
	}

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;
		std::optional<uint32_t> transferFamily;
		std::optional<uint32_t> computeFamily;

		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value() &&
				   transferFamily.has_value() && computeFamily.has_value();
		}
	};

	QueueFamilyIndices pickQueueFamilies(VkPhysicalDevice device) {
		uint32_t familyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(familyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount,
												 queueFamilies.data());

		auto pickFamily = [&](std::function<bool(uint32_t)> check) {
			for (size_t i = 0; i < queueFamilies.size(); i++) {
				if (check(i)) {
					return std::optional<uint32_t>(i);
				}
			}
			return std::optional<uint32_t>{};
		};
		return {
			.graphicsFamily = pickFamily([&](uint32_t i) {
				return queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
			}),
			.presentFamily = pickFamily([&](uint32_t i) {
				if (!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
					return false;
				}
				VkBool32 presentSupport = false;
				int result = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
																  &presentSupport);
				checkError(result, "Failed to pick present queue family");
				return (bool)presentSupport;
			}),
			.transferFamily = pickFamily([&](uint32_t i) {
				return queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT;
			}),
			.computeFamily = pickFamily([&](uint32_t i) {
				return queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
			}),
		};
	}

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	VkResult querySwapChainSupport(VkPhysicalDevice dev,
								   SwapChainSupportDetails &details) {
		const std::string ERR_MSG = "Failed to query swapchain support for device";
		// Get capabilities
		VkResult res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			dev, surface, &details.capabilities);
		if (res != VK_SUCCESS)
			return res;

		// Get supported surface formats
		uint32_t c;
		res = vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &c, nullptr);
		if (res != VK_SUCCESS && res != VK_INCOMPLETE)
			return res;
		details.formats.resize(c);
		vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &c,
											 details.formats.data());

		// Get supported present modes
		res = vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &c, nullptr);
		if (res != VK_SUCCESS)
			return res;
		details.presentModes.resize(c);
		vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &c,
												  details.presentModes.data());
		return VK_SUCCESS;
	}

	int isDeviceSuitable(VkPhysicalDevice dev) {
		const std::string ERR_MSG = "Failed to check device suitability";

		// Get device features and properties
		VkPhysicalDeviceProperties props;
		VkPhysicalDeviceFeatures features;
		vkGetPhysicalDeviceProperties(dev, &props);
		vkGetPhysicalDeviceFeatures(dev, &features);

		SwapChainSupportDetails support;
		int res = querySwapChainSupport(dev, support);
		checkError(res, "Failed to query swap chain support");

		// Get supported extensions
		uint32_t count;
		res = vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
		checkError(res, ERR_MSG);
		std::vector<VkExtensionProperties> availableExtensions(count);
		vkEnumerateDeviceExtensionProperties(dev, nullptr, &count,
											 availableExtensions.data());
		std::set<std::string> unavailableExtensions(REQUIRED_EXTENSIONS.begin(),
													REQUIRED_EXTENSIONS.end());
		for (const auto &ext : availableExtensions) {
			unavailableExtensions.erase(ext.extensionName);
		}
		if (!features.geometryShader || !features.samplerAnisotropy)
			return VK_ERROR_FEATURE_NOT_PRESENT;
		if (!pickQueueFamilies(dev).isComplete())
			return VK_ERROR_INCOMPATIBLE_DRIVER;
		if (!unavailableExtensions.empty() || support.presentModes.empty())
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		if (support.formats.empty())
			return VK_ERROR_FORMAT_NOT_SUPPORTED;
		return VK_SUCCESS;
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = pickQueueFamilies(physicalDevice);
		float queuePriority = 1.0f;

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {
			indices.graphicsFamily.value(),
			indices.presentFamily.value(),
			indices.transferFamily.value(),
			indices.computeFamily.value(),
		};

		for (uint32_t queueFamily : uniqueQueueFamilies) {
			queueCreateInfos.push_back({
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queueFamily,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority,
			});
		}
		VkPhysicalDeviceFeatures features;
		vkGetPhysicalDeviceFeatures(physicalDevice, &features);

		VkDeviceCreateInfo deviceInfo {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = (uint32_t)queueCreateInfos.size(),
			.pQueueCreateInfos = queueCreateInfos.data(),
#if !(NDEBUG)
			.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size(),
			.ppEnabledLayerNames = VALIDATION_LAYERS.data(),
#else
			.enabledLayerCount = 0,
#endif
			.enabledExtensionCount = (uint32_t)REQUIRED_EXTENSIONS.size(),
			.ppEnabledExtensionNames = REQUIRED_EXTENSIONS.data(),
			.pEnabledFeatures = &features,
		};
		int result = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
		checkError(result, "Failed to create logical device");
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
		vkGetDeviceQueue(device, indices.transferFamily.value(), 0, &transferQueue);
		vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
	}

	void createAllocator() {
		VmaAllocatorCreateInfo createInfo{
			.flags = 0,
			.physicalDevice = physicalDevice,
			.device = device,
			.pVulkanFunctions = &VULKAN_FUNCTIONS,
			.instance = instance,
			.vulkanApiVersion = APP_INFO.apiVersion,
		};
		checkError(vmaCreateAllocator(&createInfo, &allocator),
				   "Failed to create allocator");
	}

	struct BufferCreateInfo {
		VkDeviceSize size;
		VkBufferCreateFlags bufferFlags;
		VkBufferUsageFlags usage;
		VmaAllocationCreateFlags allocFlags;
	};
	void createBuffer(BufferCreateInfo &info, VkBuffer &buffer,
					  VmaAllocation &allocation, VmaAllocationInfo *allocRes) {
		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.flags = info.bufferFlags,
			.size = info.size,
			.usage = info.usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		QueueFamilyIndices indices = pickQueueFamilies(physicalDevice);
		uint32_t queueFamilies[] = {indices.graphicsFamily.value(),
									indices.presentFamily.value()};

		if (indices.graphicsFamily != indices.transferFamily) {
			bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
			bufferInfo.queueFamilyIndexCount = 2;
			bufferInfo.pQueueFamilyIndices = queueFamilies;
		}

		VmaAllocationCreateInfo allocInfo{
			.flags = info.allocFlags,
			.usage = VMA_MEMORY_USAGE_AUTO,
		};
		int res = vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer,
								  &allocation, allocRes);
		checkError(res, "Failed to create buffer");
	}

	struct ImageCreateInfo {
		VkExtent3D extent;
		VkFormat format;
		VkImageType imageType = VK_IMAGE_TYPE_2D;
		VkImageTiling tiling;
		VkImageCreateFlags imageFlags = 0;
		uint32_t mipLevels = 1;
		uint32_t arrayLayers = 1;
		VkSampleCountFlagBits numSamples = VK_SAMPLE_COUNT_1_BIT;
		VkImageUsageFlags usage = 0;
		VmaAllocationCreateFlags allocFlags = 0;
	};
	void createImage(ImageCreateInfo &info, VkImage &image,
					 VmaAllocation &allocation, VmaAllocationInfo *allocRes) {

		VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.flags = info.imageFlags,
			.imageType = info.imageType,
			.format = info.format,
			.extent = info.extent,
			.mipLevels = info.mipLevels,
			.arrayLayers = info.arrayLayers,
			.samples = info.numSamples,
			.tiling = info.tiling,
			.usage = info.usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		QueueFamilyIndices indices = pickQueueFamilies(physicalDevice);
		uint32_t queueFamilies[] = {indices.graphicsFamily.value(),
									indices.transferFamily.value()};
		if (indices.graphicsFamily.value() != indices.transferFamily.value()) {
			imageInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
			imageInfo.queueFamilyIndexCount = 2;
			imageInfo.pQueueFamilyIndices = queueFamilies;
		}

		VmaAllocationCreateInfo allocInfo{
			.flags = info.allocFlags,
			.usage = VMA_MEMORY_USAGE_AUTO,
		};
		int res = vmaCreateImage(allocator, &imageInfo, &allocInfo, &image,
								 &allocation, allocRes);
		checkError(res, "Failed to create image");
	}

	struct LayoutTransitionInfo {
		VkImage image;
		VkFormat format;
		VkImageSubresourceRange subresourceRange;
		VkImageLayout oldLayout;
		VkImageLayout newLayout;
	};
	void transitionImageLayout(LayoutTransitionInfo info, VkCommandBuffer &ctx) {
		VkImageMemoryBarrier2 barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.oldLayout = info.oldLayout,
			.newLayout = info.newLayout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = info.image,
			.subresourceRange = info.subresourceRange,

		};
		VkPipelineStageFlags srcFlags, dstFlags;
		switch (barrier.oldLayout) {
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			srcFlags = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
			break;
		case VK_IMAGE_LAYOUT_UNDEFINED:
			barrier.srcAccessMask = 0;
			srcFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		default:
			return;
		}
		switch (barrier.newLayout) {
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			dstFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
			dstFlags = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			dstFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		default:
			return;
		}

		VkDependencyInfo dependencyInfo{
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(ctx, &dependencyInfo);
	}

	void createSwapChain() {
		SwapChainSupportDetails details;
		int res = querySwapChainSupport(physicalDevice, details);
		checkError(res, "Failed to query swap chain support on swapchain creation");

		VkSurfaceFormatKHR format = details.formats[0];
		for (const auto &f : details.formats) {
			if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
				f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				format = f;
				break;
			}
		}

		VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
		std::set<VkPresentModeKHR> availablePresents(details.presentModes.begin(),
													 details.presentModes.end());
		for (const auto &pm : PRESENT_MODE_ORDER) {
			auto found = availablePresents.find(pm);
			if (found != availablePresents.end()) {
				presentMode = *found;
				break;
			}
		}

		VkExtent2D swapExtent = details.capabilities.currentExtent;
		if (details.capabilities.currentExtent.width == UINT32_MAX) {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			swapExtent = {
				.width = std::clamp((uint32_t)width,
									details.capabilities.minImageExtent.width,
									details.capabilities.maxImageExtent.width),
				.height = std::clamp((uint32_t)height,
									 details.capabilities.minImageExtent.height,
									 details.capabilities.maxImageExtent.height),
			};
		}

		uint32_t imageCount = details.capabilities.minImageCount + 1;
		if (details.capabilities.maxImageCount > 0 &&
			imageCount > details.capabilities.maxImageCount) {
			imageCount = details.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = surface,
			.minImageCount = imageCount,
			.imageFormat = format.format,
			.imageColorSpace = format.colorSpace,
			.imageExtent = swapExtent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.preTransform = details.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
			.oldSwapchain = VK_NULL_HANDLE,
		};

		QueueFamilyIndices indices = pickQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
										 indices.presentFamily.value()};

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		res = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);
		checkError(res, "Failed to create swapchain");
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		checkError(res, "Failed to get swap chain images");
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
								swapChainImages.data());
		swapChainFormat = format.format;
		swapChainExtent = swapExtent;
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);

		if (!width || !height) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		vkDeviceWaitIdle(device);
		cleanupSwapChainDependents();
		createSwapChainDependents();
	}

	void createSwapChainDependents() {
		createSwapChain();

		// Swapchain dependents
		viewport = {
			.width = (float)swapChainExtent.width,
			.height = (float)swapChainExtent.height,
			.maxDepth = 1.0f,
		};
		scissor = {
			.offset = {0, 0},
			.extent = swapChainExtent,
		};
		createSwapChainImageViews();
		createDepthResources();
		createColorResources();
		createRenderPass();

		// Render pass dependents
		createFrameBuffers();
	}

	void cleanupSwapChainDependents() {
		// Render pass dependents
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
		}

		// Swap chain dependents
		vkDestroyRenderPass(device, renderPass, nullptr);
		vkDestroyImageView(device, colorImageView, nullptr);
		vmaDestroyImage(allocator, colorImage, colorImageMemory);
		vkDestroyImageView(device, depthImageView, nullptr);
		vmaDestroyImage(allocator, depthImage, depthImageMemory);
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vkDestroyImageView(device, swapChainImageViews[i], nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	struct ImageViewInfo {
		VkImage image;
		VkFormat format;
		VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
		VkImageSubresourceRange subresourceRange;
	};

	VkImageView createImageView(ImageViewInfo &info) {
		VkImageViewCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = info.image,
			.viewType = info.viewType,
			.format = info.format,
			.components = {},
			.subresourceRange = info.subresourceRange,
		};

		VkImageView imageView;
		int res = vkCreateImageView(device, &createInfo, nullptr, &imageView);
		checkError(res, "Failed to create image view");
		return imageView;
	}

	void createSwapChainImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			ImageViewInfo info{.image = swapChainImages[i],
							   .format = swapChainFormat,
							   .viewType = VK_IMAGE_VIEW_TYPE_2D,
							   .subresourceRange = {
								   .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
								   .baseMipLevel = 0,
								   .levelCount = 1,
								   .baseArrayLayer = 0,
								   .layerCount = 1,
							   }};
			swapChainImageViews[i] = createImageView(info);
		}
	}

	void createRenderPass() {
		VkAttachmentDescription colorAttachment{
			.format = swapChainFormat,
			.samples = msaaSamples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference colorAttachmentRef{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentDescription depthAttachment{
			.format = pickDepthFormat(),
			.samples = msaaSamples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depthAttachmentRef{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentDescription colorResolveAttachment{
			.format = swapChainFormat,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};
		VkAttachmentReference colorResolveAttachmentRef{
			.attachment = 2,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachmentRef,
			.pResolveAttachments = &colorResolveAttachmentRef,
			.pDepthStencilAttachment = &depthAttachmentRef,
		};
		VkSubpassDependency dependency{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
							VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
							VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
							 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		};
		std::vector<VkAttachmentDescription> attachments = {
			colorAttachment,
			depthAttachment,
			colorResolveAttachment,
		};

		VkRenderPassCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = (uint32_t)attachments.size(),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dependency,
		};
		int res = vkCreateRenderPass(device, &info, nullptr, &renderPass);
		checkError(res, "Failed to create render pass");
	}

	VkFormat pickDepthFormat() {
		return pickSupportedFormat(
			{VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
			 VK_FORMAT_D32_SFLOAT},
			VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	VkFormat pickSupportedFormat(const std::vector<VkFormat> &candidates,
								 VkImageTiling tiling,
								 VkFormatFeatureFlags features) {
		for (auto &format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
			VkFormatFeatureFlags formatFeatures = (tiling == VK_IMAGE_TILING_OPTIMAL)
													  ? props.optimalTilingFeatures
													  : props.linearTilingFeatures;
			if (features & formatFeatures) {
				return format;
			}
		}
		checkError(VK_ERROR_FORMAT_NOT_SUPPORTED,
				   "Failed to pick a supported format");
		return VK_FORMAT_UNDEFINED;
	}

	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
			std::vector<VkImageView> imageViews = {colorImageView, depthImageView,
												   swapChainImageViews[i]};
			VkFramebufferCreateInfo framebufferInfo{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = renderPass,
				.attachmentCount = (uint32_t)imageViews.size(),
				.pAttachments = imageViews.data(),
				.width = swapChainExtent.width,
				.height = swapChainExtent.height,
				.layers = 1,
			};
			int res = vkCreateFramebuffer(device, &framebufferInfo, nullptr,
										  &swapChainFramebuffers[i]);
			checkError(res, "Failed to create framebuffer");
		}
	}

	void createGraphicsDescriptorSetLayout() {
		std::vector<VkDescriptorSetLayoutBinding> bindings = {{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.pImmutableSamplers = nullptr,
		}};
		VkDescriptorSetLayoutCreateInfo layoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32_t)bindings.size(),
			.pBindings = bindings.data(),
		};
		int res = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
											  &graphicsDescriptorSetLayout);
		checkError(res, "Failed to create graphics descriptor set layout");
	}

	void createComputeDescriptorSetLayout() {
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.pImmutableSamplers = nullptr,
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = nullptr,

			},
			{
				.binding = 2,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = nullptr,

			}};
		VkDescriptorSetLayoutCreateInfo layoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32_t)bindings.size(),
			.pBindings = bindings.data(),
		};
		int res = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
											  &computeDescriptorSetLayout);
		checkError(res, "Failed to create compute descriptor set layout");
	}

	void createCommandPools() {
		QueueFamilyIndices queueFamilyIndices = pickQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo graphicsPoolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
		};
		VkCommandPoolCreateInfo transferPoolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.transferFamily.value(),
		};
		VkCommandPoolCreateInfo computePoolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.computeFamily.value(),
		};
		int res =
			vkCreateCommandPool(device, &graphicsPoolInfo, nullptr, &graphicsPool);
		checkError(res, "Failed to create graphics command pool");
		res = vkCreateCommandPool(device, &transferPoolInfo, nullptr, &transferPool);
		checkError(res, "Failed to create transfer command pool");
		res = vkCreateCommandPool(device, &computePoolInfo, nullptr, &computePool);
		checkError(res, "Failed to create compute command pool");
	}

	void createSyncObjects() {
		graphicsInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

		const VkSemaphoreCreateInfo info{.sType =
											 VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
										 .flags = VK_SEMAPHORE_TYPE_BINARY};
		const VkFenceCreateInfo fenceInfo{.sType =
											  VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
										  .flags = VK_FENCE_CREATE_SIGNALED_BIT};
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			int res = vkCreateFence(device, &fenceInfo, nullptr,
									&graphicsInFlightFences[i]);
			checkError(res, "Failed to create graphics in-flight fence");
			res = vkCreateFence(device, &fenceInfo, nullptr,
								&computeInFlightFences[i]);
			checkError(res, "Failed to create compute in-flight fence");
			res = vkCreateSemaphore(device, &info, nullptr,
									&renderFinishedSemaphores[i]);
			checkError(res, "Failed to create render finished semaphore");
			res = vkCreateSemaphore(device, &info, nullptr,
									&computeFinishedSemaphores[i]);
			checkError(res, "Failed to create compute finished semaphore");
			res = vkCreateSemaphore(device, &info, nullptr,
									&imageAvailableSemaphores[i]);
			checkError(res, "Failed to create image available semaphore");
		}
	}

	void destroySyncObjects() {
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyFence(device, graphicsInFlightFences[i], nullptr);
			vkDestroyFence(device, computeInFlightFences[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
		}
	}

	struct SingleTimeCommand {
		VkCommandBuffer commandBuffer;
		std::function<void()> flush;
	};

	SingleTimeCommand beginSingleTimeCommand(VkCommandPool commandPool,
											 VkQueue commandQueue) {
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkCommandBuffer commandBuffer;
		int res = vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
		checkError(res, "Failed to allocate single time command buffer");

		VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		checkError(res, "Failed to begin single time command");

		return {
			.commandBuffer = commandBuffer,
			.flush =
				[=, this]() {
					endSingleTimeCommand(commandBuffer, commandPool, commandQueue);
				},
		};
	}

	void endSingleTimeCommand(VkCommandBuffer commandBuffer,
							  VkCommandPool commandPool, VkQueue commandQueue) {
		VkFence singleTimeFence;
		int res = vkEndCommandBuffer(commandBuffer);
		checkError(res, "Failed to end single time command");
		VkFenceCreateInfo fenceInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		vkCreateFence(device, &fenceInfo, nullptr, &singleTimeFence);
		vkResetFences(device, 1, &singleTimeFence);

		VkSubmitInfo submitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffer,
		};
		res = vkQueueSubmit(commandQueue, 1, &submitInfo, singleTimeFence);
		checkError(res, "Failed to submit single time command");
		res = vkWaitForFences(device, 1, &singleTimeFence, VK_TRUE, UINT64_MAX);
		checkError(res, "Failed to wait for single time command fence");
		vkDestroyFence(device, singleTimeFence, nullptr);
	}

	void createUniformBuffers() {
		VkDeviceSize bufferc = sizeof(UniformBufferObject);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemories.resize(MAX_FRAMES_IN_FLIGHT);

		BufferCreateInfo bufferInfo{
			.size = bufferc,
			.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			.allocFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
						  VMA_ALLOCATION_CREATE_MAPPED_BIT,
		};
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VmaAllocationInfo allocRes;
			createBuffer(bufferInfo, uniformBuffers[i], uniformBuffersMemories[i],
						 &allocRes);
			uniformBuffersMapped[i] = allocRes.pMappedData;
		}
	}

	void createStorageBuffers() {
		VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;
		storageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		storageBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);

		std::default_random_engine rndEngine((unsigned)time(nullptr));
		std::uniform_real_distribution<float> rndDist(0, 1);

		// Initialise particles in a circle, going towards the outside
		std::vector<Particle> particles(PARTICLE_COUNT);
		for (auto &particle : particles) {
			float r = 0.25 * sqrt(rndDist(rndEngine));
			float theta = rndDist(rndEngine) * 2 * 3.14159265358979323846;
			float x = r * cos(theta) * HEIGHT / WIDTH, y = r * sin(theta);
			particle.position = {x, y};
			particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
			particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
									   rndDist(rndEngine), 1);
		}

		// Create staging buffer
		VkBuffer stagingBuffer;
		VmaAllocation stagingBufferAlloc;
		VmaAllocationInfo allocRes;
		BufferCreateInfo stagingInfo{
			.size = bufferSize,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			.allocFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
						  VMA_ALLOCATION_CREATE_MAPPED_BIT,
		};
		createBuffer(stagingInfo, stagingBuffer, stagingBufferAlloc, &allocRes);
		memcpy(allocRes.pMappedData, particles.data(), bufferSize);

		// Create buffers
		BufferCreateInfo storageInfo{
			.size = bufferSize,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
					 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
					 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			.allocFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
		};
		VkBufferCopy copyRegion{.size = bufferSize};
		SingleTimeCommand cmd = beginSingleTimeCommand(transferPool, transferQueue);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(storageInfo, storageBuffers[i], storageBufferMemories[i],
						 nullptr);
			vkCmdCopyBuffer(cmd.commandBuffer, stagingBuffer, storageBuffers[i], 1,
							&copyRegion);
		}
		cmd.flush();
		vmaDestroyBuffer(allocator, stagingBuffer, stagingBufferAlloc);
	}

	void createGraphicsPool() {
		std::vector<VkDescriptorPoolSize> poolSizes{
			{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			 .descriptorCount = (uint32_t)MAX_FRAMES_IN_FLIGHT}};

		VkDescriptorPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = (uint32_t)MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = (uint32_t)poolSizes.size(),
			.pPoolSizes = poolSizes.data(),
		};
		int res =
			vkCreateDescriptorPool(device, &info, nullptr, &renderDescriptorPool);
		checkError(res, "Failed to create graphics descriptor pool");
	}

	void createGraphicsCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = graphicsPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t)commandBuffers.size(),
		};

		int res =
			vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
		checkError(res, "Failed to create graphics command buffers");
	}

	void createComputePool() {
		std::vector<VkDescriptorPoolSize> poolSizes{
			{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			 .descriptorCount = (uint32_t)MAX_FRAMES_IN_FLIGHT},
			{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 .descriptorCount = (uint32_t)MAX_FRAMES_IN_FLIGHT * 2}};

		VkDescriptorPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = (uint32_t)MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = (uint32_t)poolSizes.size(),
			.pPoolSizes = poolSizes.data(),
		};

		int res =
			vkCreateDescriptorPool(device, &info, nullptr, &computeDescriptorPool);
		checkError(res, "Failed to create compute descriptor pool");
	}

	inline bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
			   format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void createDepthResources() {
		VkFormat format = pickDepthFormat();
		ImageCreateInfo depthInfo{
			.extent = {.width = swapChainExtent.width,
					   .height = swapChainExtent.height,
					   .depth = 1},
			.format = format,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.numSamples = msaaSamples,
			.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			.allocFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
		createImage(depthInfo, depthImage, depthImageMemory, nullptr);

		ImageViewInfo viewInfo{
			.image = depthImage,
			.format = format,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
								 .baseMipLevel = 0,
								 .levelCount = 1,
								 .baseArrayLayer = 0,
								 .layerCount = 1}};
		if (hasStencilComponent(viewInfo.format)) {
			viewInfo.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
		depthImageView = createImageView(viewInfo);

		SingleTimeCommand cmd = beginSingleTimeCommand(graphicsPool, graphicsQueue);
		LayoutTransitionInfo transitionInfo{
			.image = depthImage,
			.format = format,
			.subresourceRange = viewInfo.subresourceRange,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
		transitionImageLayout(transitionInfo, cmd.commandBuffer);
		cmd.flush();
	}

	void createColorResources() {
		ImageCreateInfo msaaInfo{
			.extent = {.width = swapChainExtent.width,
					   .height = swapChainExtent.height,
					   .depth = 1},
			.format = swapChainFormat,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.numSamples = msaaSamples,
			.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
					 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.allocFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
		};
		createImage(msaaInfo, colorImage, colorImageMemory, nullptr);

		ImageViewInfo viewInfo{
			.image = colorImage,
			.format = msaaInfo.format,
			.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
								 .levelCount = 1,
								 .layerCount = 1}};
		colorImageView = createImageView(viewInfo);
	}

	void createGraphicsDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
												   graphicsDescriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = renderDescriptorPool,
			.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT,
			.pSetLayouts = layouts.data(),
		};
		renderDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		int res = vkAllocateDescriptorSets(device, &allocInfo,
										   renderDescriptorSets.data());
		checkError(res, "Failed to allocate render descriptor sets");

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorBufferInfo buffInfo{
				.buffer = uniformBuffers[i],
				.range = sizeof(UniformBufferObject),
			};
			std::vector<VkWriteDescriptorSet> descriptorWrites = {
				{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				 .dstSet = renderDescriptorSets[i],
				 .dstBinding = 0,
				 .descriptorCount = 1,
				 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				 .pBufferInfo = &buffInfo}};
			vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(),
								   descriptorWrites.data(), 0, nullptr);
		}
	}

	void createComputeDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
												   computeDescriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = computeDescriptorPool,
			.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT,
			.pSetLayouts = layouts.data(),
		};
		computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		int res = vkAllocateDescriptorSets(device, &allocInfo,
										   computeDescriptorSets.data());
		checkError(res, "Failed to allocate compute descriptor sets");

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorBufferInfo uniformBufferInfo{
				.buffer = uniformBuffers[i],
				.range = sizeof(UniformBufferObject),
			};
			VkDescriptorBufferInfo particlesInInfo{
				.buffer = storageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT],
				.range = sizeof(Particle) * PARTICLE_COUNT,
			};
			VkDescriptorBufferInfo particlesOutInfo{
				.buffer = storageBuffers[i],
				.range = sizeof(Particle) * PARTICLE_COUNT,
			};

			std::vector<VkWriteDescriptorSet> descriptorWrites = {
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = computeDescriptorSets[i],
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &uniformBufferInfo,
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = computeDescriptorSets[i],
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &particlesInInfo,
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = computeDescriptorSets[i],
					.dstBinding = 2,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &particlesOutInfo,
				},
			};
			vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(),
								   descriptorWrites.data(), 0, nullptr);
		}
	}

	VkResult createShaderModule(const std::vector<char> &code,
								VkShaderModule &module) {
		VkShaderModuleCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code.size(),
			.pCode = (const uint32_t *)code.data(),
		};
		return vkCreateShaderModule(device, &createInfo, nullptr, &module);
	}

	void createGraphicsPipeline() {
		VkShaderModule vertShader, fragShader;
		int res =
			createShaderModule(readFile("shaders/tutorial.vert.spv"), vertShader);
		checkError(res, "Failed to load vertex shader");
		res = createShaderModule(readFile("shaders/tutorial.frag.spv"), fragShader);
		checkError(res, "Failed to load fragment shader");

		std::vector<VkPipelineShaderStageCreateInfo> shaderStages{
			{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			 .stage = VK_SHADER_STAGE_VERTEX_BIT,
			 .module = vertShader,
			 .pName = "main"},
			{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			 .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			 .module = fragShader,
			 .pName = "main"}};

		auto bindingDescription = Particle::getBindingDescription();
		auto attributeDescriptions = Particle::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount =
				(uint32_t)attributeDescriptions.size(),
			.pVertexAttributeDescriptions = attributeDescriptions.data()};
		VkPipelineViewportStateCreateInfo viewportInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1};
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &graphicsDescriptorSetLayout};
		res = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
									 &graphicsPipelineLayout);
		checkError(res, "Failed to create graphics pipeline layout");

		// Fixed functions
		std::vector<VkDynamicState> dynamicStateVec{VK_DYNAMIC_STATE_VIEWPORT,
													VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamicState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = (uint32_t)dynamicStateVec.size(),
			.pDynamicStates = dynamicStateVec.data()};
		VkPipelineInputAssemblyStateCreateInfo assemblyState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
			.primitiveRestartEnable = VK_FALSE};
		VkPipelineRasterizationStateCreateInfo rasterization{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE};
		VkPipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable = VK_TRUE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
							  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
		VkPipelineColorBlendStateCreateInfo colorBlend{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment};
		VkPipelineDepthStencilStateCreateInfo depthStencil{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
			.stencilTestEnable = VK_FALSE};
		VkPipelineMultisampleStateCreateInfo multisampling{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = msaaSamples,
			.sampleShadingEnable = VK_FALSE};

		VkGraphicsPipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = (uint32_t)shaderStages.size(),
			.pStages = shaderStages.data(),
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &assemblyState,
			.pViewportState = &viewportInfo,
			.pRasterizationState = &rasterization,
			.pMultisampleState = &multisampling,
			//.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlend,
			.pDynamicState = &dynamicState,
			.layout = graphicsPipelineLayout,
			.renderPass = renderPass,
			.subpass = 0};
		res = vkCreateGraphicsPipelines(device, nullptr, 1, &pipelineInfo, nullptr,
										&graphicsPipeline);
		checkError(res, "Failed to create graphics pipeline");
		vkDestroyShaderModule(device, fragShader, nullptr);
		vkDestroyShaderModule(device, vertShader, nullptr);
	}

	void createComputePipeline() {}

	void createFrameBuffers() {
		swapChainFramebuffers.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
			std::vector<VkImageView> attachments = {colorImageView, depthImageView,
													swapChainImageViews[i]};
			VkFramebufferCreateInfo framebufferInfo{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = renderPass,
				.attachmentCount = (uint32_t)attachments.size(),
				.pAttachments = attachments.data(),
				.width = swapChainExtent.width,
				.height = swapChainExtent.height,
				.layers = 1,
			};
			int res = vkCreateFramebuffer(device, &framebufferInfo, nullptr,
										  &swapChainFramebuffers[i]);
			checkError(res, "Failed to create swap chain framebuffer");
		}
	}

	void drawFrame() {
		int res = vkWaitForFences(device, 1, &graphicsInFlightFences[currFrame],
								  VK_TRUE, UINT64_MAX);
		checkError(res, "Failed to wait for graphics in-flight fence");

		uint32_t imageIndex;
		res = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
									imageAvailableSemaphores[currFrame],
									VK_NULL_HANDLE, &imageIndex);
		if (res == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		checkError(res, "Failed to acquire next swap chain image");
		res = vkResetFences(device, 1, &graphicsInFlightFences[currFrame]);
		checkError(res, "Failed to reset graphics in-flight fence");

		res = vkResetCommandBuffer(commandBuffers[currFrame], 0);
		checkError(res, "Failed to reset current frame command buffer");
		recordCommandBuffer(commandBuffers[currFrame], imageIndex);

		VkPipelineStageFlags waitStages[] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		};
		VkSubmitInfo submitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &imageAvailableSemaphores[currFrame],
			.pWaitDstStageMask = waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffers[currFrame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &renderFinishedSemaphores[currFrame],
		};
		res = vkQueueSubmit(graphicsQueue, 1, &submitInfo,
							graphicsInFlightFences[currFrame]);
		checkError(res, "Failed to submit draw command buffer");

		VkPresentInfoKHR presentInfo{
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &renderFinishedSemaphores[currFrame],
			.swapchainCount = 1,
			.pSwapchains = &swapChain,
			.pImageIndices = &imageIndex,
		};
		res = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR ||
			frameBufferResized) {
			frameBufferResized = false;
			recreateSwapChain();
		}
		checkError(res, "Failed to present swap chain");
		currFrame = (currFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t image_i) {
		VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		int res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		checkError(res, "Failed to begin draw command buffer recording");

		std::vector<VkClearValue> clearValues{{.color = {{0, 0, 0}}},
											  {.depthStencil = {1, 0}}};
		VkRenderPassBeginInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = renderPass,
			.framebuffer = swapChainFramebuffers[image_i],
			.renderArea = {.extent = swapChainExtent},
			.clearValueCount = (uint32_t)clearValues.size(),
			.pClearValues = clearValues.data(),
		};
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
							 VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
						  graphicsPipeline);
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		VkDeviceSize offsets[] = {0};
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storageBuffers[currFrame],
							   offsets);
		vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);
		vkCmdEndRenderPass(commandBuffer);
		res = vkEndCommandBuffer(commandBuffer);
		checkError(res, "Failed to end draw command buffer recording");
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}
};

int main() {
	ParticleApplication app;
	try {
		app.run();
	} catch (const std::exception &e) {
		std::cerr << "Error occured: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
