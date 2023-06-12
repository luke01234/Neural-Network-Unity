//==========================//
//algorithm that iterates upon generations of networks to create a most fit network
//==========================//
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;

public class GeneticAlg : MonoBehaviour
{
  [Header("References")]
  public raycasting2 controller;

  [Header("Controls")]
  public int initialPopulation=85;
  [Range(0.0f,1.0f)]
  public float mutationRate = 0.055f;

  [Header("Crossover Controls")]
  public int bestAgentSelection = 8;
  public int worstAgentSelection =3;
  public int numberToCross;

  private List<int> genePool = new List<int>();

  private int naturallySelected;

  private NNet[] population;

  [Header("Public View")]
  public int currentGeneration;
  public int currentGenome=0;

  private void Start()
  {
    //on start create a population
    CreatePop();
  }

  private void CreatePop()
  {
    //create a population of networks fill with random values
    population = new NNet[initialPopulation];
    FillPopulationWithRandomValues(population,0);
    ResetToCurrentGenome();

  }

  private void ResetToCurrentGenome()
  {
    //reset network
    controller.ResetWithNetwork(population[currentGenome]);
  }

  private void FillPopulationWithRandomValues(NNet[] newPopulation, int startingIndex)
  {
    //fill population with random variable networks
    while (startingIndex <initialPopulation)
    {
      newPopulation[startingIndex]= new NNet();
      newPopulation[startingIndex].Initialise(controller.LAYERS, controller.NEURONS);
      startingIndex++;
    }
  }
  public void Death(float fitness, NNet network)
  {
    //on death iterate genome, if population has finished, repopulate another generation, else, track the fitness and reset current genome
    if (currentGenome < population.Length-1)
    {
      population[currentGenome].fitness=fitness;
      currentGenome++;
      ResetToCurrentGenome();
    }
    else
    {
      Repopulate();
    }
  }
  private void Repopulate()
  {
    //clear genepool and iterate generation
    genePool.Clear();
    currentGeneration++;
    naturallySelected=0;
    //crossover the population and mutate it, use this new network in the next generation
    NNet[] newPopulation = PickBestPopulation();

    Crossover(newPopulation);
    Mutate(newPopulation);
    FillPopulationWithRandomValues(newPopulation, naturallySelected);

    population=newPopulation;
    currentGenome=0;
    ResetToCurrentGenome();

  }
  private void Mutate (NNet[] newPopulation)
  {
    //mutate the genome slightly by randomly tweaking weights between neurons to make each generation differ slightl more towards perfection
    for (int i =0; i<naturallySelected; i++)
    {
      for (int c=0; c <newPopulation[i].weights.Count; c++)
      {
        if (Random.Range(0.0f,1.0f) < mutationRate)
        {
          newPopulation[i].weights[c]=MutateMatrix(newPopulation[i].weights[c]);
        }
      }
    }
  }
  Matrix<float> MutateMatrix (Matrix<float> A)
  {
    int randomPoints = Random.Range(1,(A.RowCount * A.ColumnCount)/7);

    Matrix<float> C=A;
    for (int i = 0; i <randomPoints; i++)
    {
      int randomColumn =Random.Range(0,C.ColumnCount);
      int randomRow = Random.Range(0,C.RowCount);

      C[randomRow,randomColumn] = Mathf.Clamp(C[randomRow,randomColumn]+Random.Range(-1f,1f),-1f,1f);
    }
    return C;
  }
  private void Crossover (NNet[] newPopulation)
  {
    //crossover takes the two best networks of a generation and takes performs a crossover of matrix values to create a more perfect network
    for (int i =0; i<numberToCross;i+=2)
    {
      int AIndex = i;
      int BIndex = i+1;
      if (genePool.Count >= 1)
      {
        for (int l =0; l <100; l++)
        {
          AIndex = genePool[Random.Range(0,genePool.Count)];
          BIndex = genePool[Random.Range(0,genePool.Count)];

          if (AIndex !=BIndex)
          {
            break;
          }

        }
        NNet Child1 = new NNet();
        NNet Child2 = new NNet();

        Child1.Initialise(controller.LAYERS, controller.NEURONS);
        Child2.Initialise(controller.LAYERS, controller.NEURONS);

        Child1.fitness = 0;
        Child2.fitness = 0;

        for (int w = 0; w<Child1.weights.Count;w++)
        {
          if (Random.Range(0.0f,1.0f)<0.5f)
          {
            Child1.weights[w] = population[AIndex].weights[w];
            Child2.weights[w] = population[BIndex].weights[w];
          }
          else
          {
            Child2.weights[w] = population[AIndex].weights[w];
            Child2.weights[w]= population[BIndex].weights[w];
          }
        }
        for (int b =0; b<Child1.biases.Count;b++)
        {
          if (Random.Range(0.0f,1.0f)<0.5f)
          {
            Child1.biases[b] = population[AIndex].biases[b];
            Child2.biases[b] = population[BIndex].biases[b];
          }
          else
          {
            Child2.biases[b] = population[AIndex].biases[b];
            Child2.biases[b]= population[BIndex].biases[b];
          }
        }
        newPopulation[naturallySelected]=Child1;
        naturallySelected++;

        newPopulation[naturallySelected]=Child2;
        naturallySelected++;


      }
    }
  }

  private NNet [] PickBestPopulation()
  {
    //read the fitness of a population in order to select the best networks
    NNet [] newPopulation = new NNet[initialPopulation];

    for (int i= 0; i<bestAgentSelection; i++)
    {
      newPopulation[naturallySelected] = population[i].InitialiseCopy(controller.LAYERS,controller.NEURONS);
      newPopulation[naturallySelected].fitness = 0; 
      naturallySelected++;

      int f =Mathf.RoundToInt(population[i].fitness * 10);
      for (int c =0; c<f; c++)
      {
        genePool.Add(i);
      }
    }
    for (int i =0; i<worstAgentSelection; i++)
    {
      int last = population.Length -1;
      last -=i;

      int f =Mathf.RoundToInt(population[last].fitness * 10);
      for (int c =0; c<f; c++)
      {
        genePool.Add(i);
      } 
    }
    return newPopulation;
  }
  private void SortPopulation()
  {
    //sort populations by fitness
    for (int i = 0; i < population.Length; i++)
    {
      for (int j = i; j <population.Length; j++)
      {
        if (population[i].fitness<population[j].fitness)
        {
          NNet temp = population[i];
          population[i]=population[j];
          population[j]=temp;

        }
      }
    }
  }
}
