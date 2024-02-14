#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>
#include <vulkan/vulkan_core.h>

#define VMA_VULKAN_VERSION 1000000 // Vulkan 1.0
#include "vk_mem_alloc.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define WINDOW_NAME "Vulkan Tutorial"
#define APP_NAME WINDOW_NAME
#define WAYLAND_APP_ID "vulkan_tutorial"

#ifdef NODEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const uint32_t HEIGHT = 480;
const uint32_t WIDTH = 640;
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char *> validationLayers = {
	"VK_LAYER_KHRONOS_validation"};

VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
	const VkAllocationCallbacks *pAllocator,
	VkDebugUtilsMessengerEXT *pMessenger) {
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

class ParticleApplication {
  public:
	bool frameBufferResized;

	void run() { mainLoop(); }

	ParticleApplication() {
		initWindow();
		initVulkan();
	}

	~ParticleApplication() { cleanup(); }

  private:
	GLFWwindow *window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;

	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VmaAllocator allocator;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
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

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan ComputeTutorial",
								  nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(
			window, [](GLFWwindow *win, int w, int h) {
				auto app = (ParticleApplication *)glfwGetWindowUserPointer(win);
				app->frameBufferResized = true;
			});
	}

	void initVulkan() {
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
	}

	void createInstance() {}

	void setupDebugMessenger() {}
	void createAllocator() {}
	void createSurface() {}
	void pickPhysicalDevice() {}

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

	QueueFamilyIndices pickQueueFamilies() { return {}; }

	bool isDeviceSuitable(VkPhysicalDevice device) { return false; }

	void createLogicalDevice() {}

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

	void cleanup() {}
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
