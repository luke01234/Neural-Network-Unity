//==========================//
//raycasting script to get info for the NeuralNet
//==========================//
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(NNet))]

public class raycasting2 : MonoBehaviour
{
  public Transform rayStart; //transform for raystart location
  public float distance; //distance flaot
  
  Color leftColor, rightColor, middleColor; //colors for rays
  
  RaycastHit hit; //hit of raycast
  
  private Vector3 right, middle, left; //vectors for each ray
  
  public float rightSensor; //florats for each sensor
  public float middleSensor;
  public float leftSensor;
  
  private Vector3 startPosition, startRotation; //start position and rotation vectors
  private NNet network; //NeuralNetwork variable

  [Range (-1f,1f)]
  public float a,t; //acceleration and turning sliders
  private Vector3 inp; //input vector

  [Header ("Fitness")]
  //variables that determin fitness of a network to determine if it moves on to the next generation
  private Vector3 lastPosition; 
  
  public float overallFitness;
  public float sensorMultiplier=.1f, distanceMultiplier=1.4f, avgSpeedMultiplier = .2f;
  private float totalDistanceTravelled;
  private float avgSpeed;
  private float timeSinceStart;

  [Header("Network Options")]
  //network options
  public int LAYERS =1;
  public int NEURONS =10;


    private void Awake()
    {
      //initialize on awake
      startPosition = transform.position;
      startRotation = transform.eulerAngles;
      network = GetComponent<NNet>();

    }

    public void ResetWithNetwork(NNet net)
    {
      //reset network and position and fitness
      network = net;
      Reset();
    }

    public void Reset()
    {
      //reset data collection and fitness
      timeSinceStart = 0f;
      totalDistanceTravelled=0f;
      avgSpeed=0f;
      lastPosition=startPosition;
      overallFitness=0f;
      transform.position=startPosition;
      transform.eulerAngles=startRotation;

    }

    private void OnCollisionEnter (Collision collision) 
    {
      //on collision, kill network
      Death();
    }

    public void MoveCar (float v, float h) 
    {
      //move car using inputs from neural network
      inp = Vector3.Lerp(Vector3.zero,new Vector3(0,0,v*25f),.02f);
      inp = transform.TransformDirection(inp);
      transform.position +=inp;

      transform.eulerAngles += new Vector3 (0, (h*90)*.02f,0);
    }

    private void CalculateFitness()
    {
      //calculate the fitness as a function of total distance, account for speed and distance from the walls
      totalDistanceTravelled += Vector3.Distance(transform.position,lastPosition);
      avgSpeed = totalDistanceTravelled/timeSinceStart;

      overallFitness =(totalDistanceTravelled*distanceMultiplier)+(avgSpeed*avgSpeedMultiplier)+(((rightSensor+middleSensor+leftSensor)/3)*sensorMultiplier); 

      if (timeSinceStart > 20 && overallFitness < 40) 
      {
        //if the network is not moving kill it
        Death();
      }
      if (overallFitness >= 10000)
      {
        //if a network wins
        //save to JSON (UNFINISHED)
        //then kill
        Death();
      }
    }

    private void InputSensors()
    {
      //distance the rays check is 5 units
      distance = 5f;

      //make the rays and draw them
      //make vectors for rays
      Vector3 right = rayStart.transform.forward+rayStart.transform.right;
      Vector3 middle = rayStart.transform.forward;
      Vector3 left = rayStart.transform.forward-rayStart.transform.right;
      
      //set colors
      rightColor=Color.green;
      middleColor=Color.green;
      leftColor=Color.green;

      //==========================//
      //check all raycasts
      Ray r = new Ray(rayStart.transform.position,right);
      RaycastHit hit;
      
      if (Physics.Raycast(r, out hit))
      {
        Debug.DrawLine(r.origin,hit.point,Color.red);
        rightSensor=hit.distance/25;
        //print ("Right : "+ rightSensor);

      }

      r.direction = middle;

      if (Physics.Raycast(r, out hit))
      {
        Debug.DrawLine(r.origin,hit.point,Color.red);
        middleSensor=hit.distance/25;
        //print ("Middle : "+ middleSensor);

      }

      r.direction = left;

      if (Physics.Raycast(r, out hit))
      {     
        Debug.DrawLine(r.origin,hit.point,Color.red);
        leftSensor=hit.distance/25;
        //print ("Left : "+ leftSensor);
        //==========================//
      }

      



    }
    private void FixedUpdate()
    {
      //on update, check sensors, update last position, change the acceleration and turn based on network output
      InputSensors();
      lastPosition=transform.position;

      
      (a,t) =network.RunNetwork(rightSensor,middleSensor,leftSensor);

      //move the car, increment time, and calculate fitness
      MoveCar(a,t);

      timeSinceStart+= Time.deltaTime;

      CalculateFitness();
 
      //a=0;
      //t=0;

    }
    private void Death()
    {
      //upon death, call genetic algorithm death, send the overall fitness and neural network
      GameObject.FindObjectOfType<GeneticAlg>().Death(overallFitness,network);
    }
}
