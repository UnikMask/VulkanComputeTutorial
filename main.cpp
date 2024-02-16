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
	std::vector<VkImageView> swapChainImageView;

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

	int isDeviceSuitable(VkPhysicalDevice dev) {
		const std::string ERR_MSG = "Failed to check device suitability";

		// Get device features and properties
		VkPhysicalDeviceProperties props;
		VkPhysicalDeviceFeatures features;
		vkGetPhysicalDeviceProperties(dev, &props);
		vkGetPhysicalDeviceFeatures(dev, &features);

		// Get capabilities
		VkSurfaceCapabilitiesKHR caps;
		int res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &caps);
		checkError(res, ERR_MSG);

		// Get supported surface formats
		uint32_t count;
		res = vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, nullptr);
		checkError(res, ERR_MSG);
		std::vector<VkSurfaceFormatKHR> formats(count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, formats.data());

		// Get supported present modes
		res =
			vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count, nullptr);
		checkError(res, ERR_MSG);
		std::vector<VkPresentModeKHR> presents(count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count,
												  presents.data());

		// Get supported extensions
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
		if (!unavailableExtensions.empty() || presents.empty())
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		if (formats.empty())
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

	void createBuffer() {}
	void createImage() {}

	void createSwapChain() {}
	void recreateSwapChain() {}
	void createSwapChainImageViews() {}

	void createRenderPass() {}
	void createFramebuffers() {}

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

	void mainLoop() {}
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
