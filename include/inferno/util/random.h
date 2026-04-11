#pragma once
#include <cassert>
#include <random>


namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class RandomGenerator 
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class RandomGenerator {

	private:
		// Static instance of random engine
		static std::mt19937 gen;
		static bool isInitialized;

	public:
		// Initialize with a random seed (non-deterministic)
		static void initialize();

		// Initialize with a specific seed (deterministic)
		static void initializeWithSeed(unsigned int seed);

		// Generate a vector of random floats
		static std::vector<float> generateRandomFloatVector(size_t size, float minValue = 0.0f, float maxValue = 1.0f);

		// Generate a vector of random doubles
		static std::vector<double> generateRandomDoubleVector(size_t size, double minValue = 0.0, double maxValue = 1.0);

		// Generate a vector of random doubles
		static std::vector<int> generateRandomIntVector(size_t size, int minValue = -1, int maxValue = 1);

		// Generate a single random float
		static float generateRandomFloat(float minValue = 0.0f, float maxValue = 1.0f);

		// Generate a single random double
		static double generateRandomDouble(double minValue = 0.0, double maxValue = 1.0);

		// Generate a single random int
		static int generateRandomInt(int minValue = -1, int maxValue = 1);
	};
}