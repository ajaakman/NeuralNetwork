#include <vector>
#include <iostream>

#include "TrainingData.h"
#include "NeuralNetwork.h"

int main()
{
	TrainingData trainingData("XORTraining.txt");

	NeuralNetwork neuralNet(std::vector<unsigned>{2, 3, 1}, 0.15f, 0.5f, 100);
	
	for (auto & layer : neuralNet.GetNeuronData())
	{
		for (auto & neuron : layer)
		{
			std::cout << "<";
			for (auto & data : neuron)
			{
				std::cout << "|" << data << "|";
			}
			std::cout << "> ";
		}
		std::cout << std::endl;
	}

	std::vector<float> inputs, targets;
	unsigned trainingPass = 0;

	while (!trainingData.isEof()) 
	{
		++trainingPass;

		trainingData.Inputs(inputs);
		if (!neuralNet.ForwardPropagate(inputs))
			break;
					
		if (trainingPass > 9995)
		{
			std::cout << "\nPass " << trainingPass << "\nInputs:" << " ";
			for (auto & input : inputs)
				std::cout << input << " ";
		
			std::cout << "\nOutputs:" << " ";
			for (auto & output : neuralNet.GetOutputs())
				std::cout << output << " ";
		}		
		
		trainingData.TargetOutputs(targets);
		if (!neuralNet.BackPropagate(targets))
			break;

		if (trainingPass > 9995)
		{
			std::cout << "\nExpected:" << " ";
			for (auto & target : targets)
				std::cout << target << " ";
			std::cout << "\n";
		}
	}

	for (auto & layer : neuralNet.GetNeuronData())
	{
		for (auto & neuron : layer)
		{
			std::cout << "<";
			for (auto & data : neuron)
			{
				std::cout << "|" << data << "|";
			}
			std::cout << "> ";
		}
		std::cout << std::endl;
	}

	while (1)
	{
		std::cout << "\nPlease insert First Value: ";
		std::cin >> inputs[0];
		std::cout << "Please insert Second Value: ";
		std::cin >> inputs[1];

		if (!neuralNet.ForwardPropagate(inputs))
			break;

		std::cout << "Neural Net Outputs:" << " ";
		for (auto & output : neuralNet.GetOutputs())
			std::cout << (int)round(output) << " ";
	}
}

