#include "NeuralNetwork.h"

#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<unsigned> &topology, const float & trainingRate, const float & trainingMomentum, const unsigned & smoothingFactor)
	:m_Topology(topology), m_fRecentAverageError(0.f), m_fEta(trainingRate), m_fAlpha(trainingMomentum), m_nRecentAverageSmoothingFactor(smoothingFactor)
{
	m_Outputs.resize(m_Topology.back(), 2.f);
	// Create Each Layer.
	for (unsigned layer = 0; layer < m_Topology.size(); ++layer) {
		m_Layers.push_back(std::vector<Neuron>());
		// Create Each Neuron in Layer + Bias Neuron.
		for (unsigned neuron = 0; neuron <= topology[layer]; ++neuron)
			m_Layers.back().push_back(Neuron(layer == topology.size() - 1 ? 0 : topology[layer + 1], neuron, this));
		// Set Bias Neuron to 1.0.
		m_Layers.back().back().SetOutput(1.f);
	}
}

bool NeuralNetwork::ForwardPropagate(const std::vector<float> &inputs)
{	
	if (inputs.size() != m_Layers[0].size() - 1) // Size should be (size - Bias Neuron).
	{
		std::cout << "ERROR! Invalid Neural Net Input parameter count!!!" << std::endl;
		return false;
	}
	// Set the Inputs for the first Layer.
	for (unsigned neuron = 0; neuron < inputs.size(); ++neuron) 
		m_Layers[0][neuron].SetOutput(inputs[neuron]);
	// Feed Forward each Neuron in each Layer except first
	for (unsigned layer = 1; layer < m_Layers.size(); ++layer)
		for (unsigned neuron = 0; neuron < m_Layers[layer].size() - 1; ++neuron)
			m_Layers[layer][neuron].FeedForward(m_Layers[layer - 1]);
	// Set Outputs
	for (unsigned output = 0; output < m_Outputs.size(); ++output)
		m_Outputs[output] = m_Layers.back()[output].GetOutput();

	return true;
}

bool NeuralNetwork::BackPropagate(const std::vector<float> &targets)
{
	if (targets.size() != m_Layers.back().size() - 1) // Size should be (size - Bias Neuron).
	{
		std::cout << "ERROR! Invalid Neural Net Target parameter count!!!" << std::endl;
		return false;
	}
	// Calculate net error
	float fError = 0.f;
	// Each Neuron in last Layer except Bias.
	for (unsigned neuron = 0; neuron < m_Layers.back().size() - 1; ++neuron) {
		float delta = targets[neuron] - m_Layers.back()[neuron].GetOutput();
		fError += delta * delta;
	}
	fError /= m_Layers.back().size() - 1;
	fError = sqrt(fError); // Error RMS

	// Recent average measurement
	m_fRecentAverageError =
		(m_fRecentAverageError * (float)m_nRecentAverageSmoothingFactor + fError)
		/ ((float)m_nRecentAverageSmoothingFactor + 1.f);

	// Output layer gradients
	for (unsigned neuron = 0; neuron < m_Layers.back().size() - 1; ++neuron)
		m_Layers.back()[neuron].CalcOutputGradients(targets[neuron]);

	// Hidden layer gradients
	for (unsigned layer = m_Layers.size() - 2; layer > 0; --layer) 
		for (unsigned neuron = 0; neuron < m_Layers[layer].size(); ++neuron)
			m_Layers[layer][neuron].CalcHiddenGradients(m_Layers[layer + 1]);	

	// Update weights for all Layers, back to front, except first.
	for (unsigned layer = m_Layers.size() - 1; layer > 0; --layer)
		for (unsigned neuron = 0; neuron < m_Layers[layer].size() - 1; ++neuron)
			m_Layers[layer][neuron].UpdateInputWeights(m_Layers[layer - 1]);

	return true;
}

const std::vector<std::vector<std::vector<float>>> NeuralNetwork::GetNeuronData() const
{
	std::vector<std::vector<std::vector<float>>> neuronData;
	for (unsigned layer = 0; layer < m_Layers.size(); ++layer)
	{
		neuronData.push_back(std::vector<std::vector<float>>());
		if (layer != m_Layers.size() - 1)
		{
			for (unsigned neuron = 0; neuron < m_Layers[layer].size(); ++neuron)
			{
				neuronData[layer].push_back(std::vector<float>(2));
				neuronData[layer].back()[0] = m_Layers[layer][neuron].GetOutput();
				neuronData[layer].back()[1] = m_Layers[layer][neuron].GetGradient();
				for (auto & connection : m_Layers[layer][neuron].Connections)
				{
					neuronData[layer].back().push_back(connection.weight);
					neuronData[layer].back().push_back(connection.deltaWeight);
				}
			}
		}
		else
		{
			for (unsigned neuron = 0; neuron < m_Layers[layer].size() - 1; ++neuron)
			{
				neuronData[layer].push_back(std::vector<float>(2));
				neuronData[layer].back()[0] = m_Layers[layer][neuron].GetOutput();
				neuronData[layer].back()[1] = m_Layers[layer][neuron].GetGradient();
			}
		}
	}
	return neuronData;
}

NeuralNetwork::Neuron::Neuron(unsigned outputCount, unsigned index, NeuralNetwork* neuralNetwork)
	:m_pNeuralNetwork(neuralNetwork), m_nIndex(index), m_fOutput(rand() / float(RAND_MAX)), m_fGradient(rand() / float(RAND_MAX))
{
	for (unsigned connection = 0; connection < outputCount; ++connection) {
		Connections.push_back(Connection());
		Connections.back().weight = rand() / float(RAND_MAX);
		Connections.back().deltaWeight = rand() / float(RAND_MAX);
	}
}

void NeuralNetwork::Neuron::UpdateInputWeights(std::vector<Neuron> &previousLayer)
{
	// Update Weights of connection in previous layer.
	for (unsigned neuron = 0; neuron < previousLayer.size(); ++neuron) {

		float newDeltaWeight =
			m_pNeuralNetwork->GetTrainingRate() * previousLayer[neuron].GetOutput() * m_fGradient	// Output * gradient * train rate:	
			+ m_pNeuralNetwork->GetTrainingMomentum() * previousLayer[neuron].Connections[m_nIndex].deltaWeight; // Add momentum, fraction of previous deltaWeight;

		previousLayer[neuron].Connections[m_nIndex].deltaWeight = newDeltaWeight;
		previousLayer[neuron].Connections[m_nIndex].weight += newDeltaWeight;
	}
}

void NeuralNetwork::Neuron::CalcHiddenGradients(const std::vector<Neuron> &nextLayer)
{
	float sum = 0.f;

	for (unsigned neuron = 0; neuron < nextLayer.size() - 1; ++neuron)
		sum += Connections[neuron].weight * nextLayer[neuron].m_fGradient;

	m_fGradient = sum * (1.f - m_fOutput * m_fOutput);
}

void NeuralNetwork::Neuron::CalcOutputGradients(float taget)
{
	m_fGradient = (taget - m_fOutput) * (1.f - m_fOutput * m_fOutput); // Delta * Transfer function derivative;
}

void NeuralNetwork::Neuron::FeedForward(const std::vector<Neuron> &previousLayer)
{
	float sum = 0.f;
	// Sum of Previous Layers Outputs + Bias.
	for (unsigned neuron = 0; neuron < previousLayer.size(); ++neuron)
		sum += previousLayer[neuron].GetOutput() * previousLayer[neuron].Connections[m_nIndex].weight;

	m_fOutput = tanh(sum); // Transfer Function output range (-1.0..1.0)
}