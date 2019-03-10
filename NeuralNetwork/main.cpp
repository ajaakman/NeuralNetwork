#include <vector>
#include <iostream>
#include <time.h>

#include "TrainingData.h"
#include "NeuralNetwork.h"

#define RPS

int main()
{
	srand(time(NULL));
#if defined(XOR)
	TrainingData trainingData("XORTraining.txt");
	NeuralNetwork neuralNet(std::vector<unsigned>{2, 2, 1}, 0.15f, 0.5f, 100);
#elif defined(RPS)
	TrainingData trainingData("RPSTraining.txt");
	NeuralNetwork neuralNet(std::vector<unsigned>{1, 2, 1}, 0.15f, 0.5f, 100);
#endif
	
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
#if defined(XOR)
		std::cout << "\nPlease insert First Value: ";
		std::cin >> inputs[0];
		std::cout << "Please insert Second Value: ";
		std::cin >> inputs[1];
		if (!neuralNet.ForwardPropagate(inputs))
			break;

		std::cout << "Neural Net Outputs:" << " ";
		for (auto & output : neuralNet.GetOutputs())
			std::cout << (int)round(output) << " ";
#elif defined(RPS)
		std::cout << "\nType r for rock, p for paper, s for scissors: ";
		char in; std::cin >> in;		
		if (in == 'r') inputs[0] = -1.f;
		else if (in == 'p') inputs[0] = 0.f;
		else if (in == 's') inputs[0] = 1.f;
		else { std::cout << "\nInvalid input try again"; continue; }

		if (!neuralNet.ForwardPropagate(inputs))
			break;

		std::cout << "Neural Net Outputs:" << " ";
		if ((int)(round(neuralNet.GetOutputs()[0])) == -1) std::cout << "Rock";
		if ((int)(round(neuralNet.GetOutputs()[0])) == -0) std::cout << "Paper";
		if ((int)(round(neuralNet.GetOutputs()[0])) ==  1) std::cout << "Scissors";
#endif
	}
	std::cin.get();
}

