//=============================//
//Neural network script, handles the neurons of the neural network
//=============================//
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;
using System;

using Random = UnityEngine.Random;

public class NNet : MonoBehaviour
{
  //create matricies for the layers of the neural networks, and the weights
  public Matrix<float> inputLayer =Matrix<float>.Build.Dense(1,3);
  
  public List<Matrix<float>> hiddenLayers = new List<Matrix<float>>();

  public Matrix<float> outputLayer =Matrix<float>.Build.Dense(1,2);

  public List<Matrix<float>> weights = new List<Matrix<float>>();

  public List<float> biases = new List<float>();
  
  public float fitness;

  public void Initialise (int hiddenLayerCount, int hiddenNeuronCount)
  {
    //initialize the layers
    inputLayer.Clear();
    hiddenLayers.Clear();
    outputLayer.Clear();
    weights.Clear();
    biases.Clear();
    
    for (int i = 0; i<hiddenLayerCount + 1; i++)
    {
      //create matrix from hiddenneuroncount
      Matrix<float> f = Matrix<float>.Build.Dense(1,hiddenNeuronCount);
      //add matrix to hiddenlayers
      hiddenLayers.Add(f);
      //create biases
      biases.Add(Random.Range(-1f,1f));

      //WEIGHTS
       if (i==0)
       {
        //create the weights matricies
        Matrix<float> inputToH1 = Matrix<float>.Build.Dense(3,hiddenNeuronCount);
        weights.Add(inputToH1);

       }
      
      Matrix<float> HiddenToHidden = Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount);
      weights.Add(HiddenToHidden);


    }
    //initialize with random weights
    Matrix<float> OutputWeight = Matrix<float>.Build.Dense(hiddenNeuronCount, 2);
    weights.Add(OutputWeight);
    RandomiseWeights();


  }

  public NNet InitialiseCopy (int hiddenLayerCount, int hiddenNeuronCount)
  {
    NNet n = new NNet();

    List<Matrix<float>> newWeights = new List<Matrix<float>>();

    for (int i = 0; i < this.weights.Count; i++)
    {
      Matrix<float> currentWeight = Matrix<float>.Build.Dense(weights[i].RowCount,weights[i].ColumnCount);
      for (int x = 0; x < currentWeight.RowCount; x++)
      {
        for (int y = 0; y < currentWeight.ColumnCount; y++)
        {
          currentWeight[x, y] = weights[i][x, y];
        }
      }
      newWeights.Add(currentWeight);

    }
    List<float> newBiases = new List<float>();

    newBiases.AddRange(biases);

    n.weights=newWeights;
    n.biases=newBiases;

    n.InitializeHidden(hiddenLayerCount,hiddenNeuronCount);
    return n;
  }

  public void InitializeHidden (int hiddenLayerCount, int hiddenNeuronCount)
  {
    //initialize hidden layers
    inputLayer.Clear();
    hiddenLayers.Clear();
    outputLayer.Clear();

    for (int i=0; i<hiddenLayerCount+1; i++)
    {

      Matrix<float> newHiddenLayer = Matrix<float>.Build.Dense(1,hiddenNeuronCount);
      hiddenLayers.Add(newHiddenLayer);

    }
  } 

  public void RandomiseWeights()
  {
    //randomise weights
    for (int i = 0; i < weights.Count; i++)
    {
      for (int x = 0; x < weights[i].RowCount; x++)
      {
        for (int y = 0; y < weights[i].ColumnCount; y++)
        {

          weights[i][x,y]=Random.Range(-1f,1f);

        }
      }
    }

  }
  public (float,float) RunNetwork (float right, float middle, float left)
  {
    //run netwrok, send right middle and left sensors to the input layers
    inputLayer[0,0] = right;
    inputLayer[0,1] = middle;
    inputLayer[0,2] = left;
    //run input layer through function to get value between -1 and 1
    inputLayer = inputLayer.PointwiseTanh();

    hiddenLayers[0] = ((inputLayer*weights[0])+biases[0]).PointwiseTanh();

    for (int i = 1; i < hiddenLayers.Count; i++)
    {
      //multiply layers by weights
      hiddenLayers[i] = ((hiddenLayers[i-1]*weights[i]) + biases[i]).PointwiseTanh();
    }

    outputLayer = ((hiddenLayers[hiddenLayers.Count-1]*weights[weights.Count-1])+biases[biases.Count-1]).PointwiseTanh();

    //get outputs
    //first output is ACC second is TURN
    return (Sigmoid(outputLayer[0,0]) , (float)Math.Tanh(outputLayer[0,1]));
  }
  private float Sigmoid (float s)
  {
    return (1/(1+Mathf.Exp(-1)));
  }
}
