#include <vector>
#include <fstream>
#include <sstream>

class TrainingData
{
public:
	TrainingData(const std::string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }

	unsigned Inputs(std::vector<float> &input);
	unsigned TargetOutputs(std::vector<float> &targetOutputs);

private:
	std::ifstream m_trainingDataFile;
};

