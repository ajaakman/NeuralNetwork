#include "TrainingData.h"


TrainingData::TrainingData(const std::string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::Inputs(std::vector<float> &input)
{
	input.clear();

	std::string line;
	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		float value;
		while (ss >> value) 
			input.push_back(value);		
	}

	return input.size();
}

unsigned TrainingData::TargetOutputs(std::vector<float> &targetOutputs)
{
	targetOutputs.clear();

	std::string line;
	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		float value;
		while (ss >> value) 
			targetOutputs.push_back(value);		
	}

	return targetOutputs.size();
}