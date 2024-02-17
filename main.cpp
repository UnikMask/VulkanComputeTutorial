#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#define WINDOW_NAME "Vulkan Tutorial"
#define APP_NAME WINDOW_NAME
#define WAYLAND_APP_ID "vulkan_tutorial"

const uint32_t HEIGHT = 480;
const uint32_t WIDTH = 640;
const int MAX_FRAMES_IN_FLIGHT = 2;

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
	default:
		throwErr("Unknown error occured!");
	}
}

class ParticleApplication {
  public:
	bool frameBufferResized;

	void run() { mainLoop(); }

	ParticleApplication() {
		initWindow();
		initVulkan();
	}

	~ParticleApplication() {
		for (auto &imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
		vmaDestroyAllocator(allocator);
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
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createAllocator();
		createSwapChain();
		createSwapChainImageViews();
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
		for (size_t i = VK_SAMPLE_COUNT_64_BIT; i >= VK_SAMPLE_COUNT_1_BIT; i--) {
			if (counts & (1 << i)) {
				msaaSamples = (VkSampleCountFlagBits)(1 << i);
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
	struct ImageCreateInfo {
		VkExtent3D extent;
		VkFormat format;
		VkImageType imageType = VK_IMAGE_TYPE_2D;
		VkImageTiling tiling;
		VkImageCreateFlags imageFlags = 0;
		uint32_t mipLevels = 1;
		uint32_t arrayLayers = 0;
		VkSampleCountFlagBits numSamples = VK_SAMPLE_COUNT_1_BIT;
		VkImageUsageFlags usage = 0;
		VmaAllocationCreateFlags allocFlags = 0;
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

	void createImage(ImageCreateInfo &info, VkImage &image,
					 VmaAllocation &allocation, VmaAllocationInfo *allocRes) {

		VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.flags = info.imageFlags,
			.imageType = info.imageType,
			.format = info.format,
			.extent = info.extent,
			.mipLevels = info.mipLevels,
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
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		createSwapChain();
	}

	struct ImageViewInfo {
		VkImage image;
		VkFormat format;
		VkImageViewType viewType;
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
			.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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

	void createGraphicsDescriptorSetLayout() {}
	void createComputeDescritporSetLayout() {}

	void createUniformBuffers() {}
	void createStorageBuffers() {}

	void createGraphicsPool() {}
	void createComputePool() {}

	void createDepthResources() {}
	void createColorResources() {}

	void createGraphicsDescriptorSets() {}
	void createComputeDescriptorSets() {}

	void createGraphicsPipeline() {}
	void createComputePipeline() {}

	void createFrameBuffers() {}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
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
