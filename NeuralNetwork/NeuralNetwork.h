#pragma once

#include <vector>

class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<unsigned> &topology, const float & trainingRate = 0.15f, const float & trainingMomentum = 0.5f, const unsigned & smoothingFactor = 100);
	bool ForwardPropagate(const std::vector<float> &inputs);
	bool BackPropagate(const std::vector<float> &targets);

	const std::vector<std::vector<std::vector<float>>> GetNeuronData()const;

	inline const std::vector<unsigned> & GetTopology() const { return m_Topology; }
	inline const std::vector<float> & GetOutputs() const { return m_Outputs; }
	inline const float & GetRecentAverageError(void) const { return m_fRecentAverageError; }

	// Range (0.0..1.0).
	inline void SetTrainingRate(const float & newRate) { m_fEta = newRate; }
	inline const float & GetTrainingRate() { return m_fEta; }
	// Range (0.0..1.0).
	inline void SetTrainingMomentum(const float & newMomentum) { m_fAlpha = newMomentum; }
	inline const float & GetTrainingMomentum() { return m_fAlpha; }

	inline void SetSmoothingFactor(const unsigned & newSmoothing) { m_nRecentAverageSmoothingFactor = newSmoothing; }
	inline const unsigned & GetSmoothingFactor() { return m_nRecentAverageSmoothingFactor; }

private:

	class Neuron
	{
	public:
		Neuron(unsigned outputCount, unsigned index, NeuralNetwork* neuralNetwork);
		void FeedForward(const std::vector<Neuron> &previousLayer);
		void CalcOutputGradients(float taget);
		void CalcHiddenGradients(const std::vector<Neuron> &nextLayer);
		void UpdateInputWeights(std::vector<Neuron> &previousLayer);

		inline void SetOutput(float newOutput) { m_fOutput = newOutput; }
		inline const float & GetOutput() const { return m_fOutput; }
		inline const float & GetGradient() const { return m_fGradient; }

	private:

		struct Connection
		{
			float weight;
			float deltaWeight;
		};

		NeuralNetwork* m_pNeuralNetwork;
		
		unsigned m_nIndex;
		float m_fOutput;
		float m_fGradient;
	public:
		std::vector<Connection> Connections;
	};

	std::vector<float> m_Outputs;
	std::vector<std::vector<Neuron>> m_Layers;
	std::vector<unsigned> m_Topology;

	float m_fRecentAverageError;
	float m_fEta;
	float m_fAlpha;
	unsigned m_nRecentAverageSmoothingFactor;	
};
