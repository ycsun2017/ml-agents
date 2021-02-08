using System.Net.Sockets;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class TankAgent : Agent
{
    TankAgent m_Opponent;
    Rigidbody m_Rb;
    BehaviorParameters m_Bp;

    public Transform FireTransform;

    const int k_AreaSize = 100;

    public override void Initialize()
    {
        m_Rb = GetComponent<Rigidbody>();
        m_Bp = GetComponent<BehaviorParameters>();
    }

    public void SetOpponent(GameObject tank)
    {
        m_Opponent = tank.GetComponent<TankAgent>();
    }

    public void SetTeam(int team)
    {
        m_Bp.TeamId = team;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        AddReward(-1f / MaxStep);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // relative vector to opponent
        if (ReferenceEquals(m_Opponent, null))
        {
            return;
        }

        // Vector to opponent
        var opponentPos = m_Opponent.transform.position;
        var myPos = transform.position;
        sensor.AddObservation(Vector3.Normalize(myPos - opponentPos));
        sensor.AddObservation(Vector3.Dot((FireTransform.position - opponentPos).normalized, FireTransform.forward));
        sensor.AddObservation(Vector3.Distance(myPos, opponentPos) / k_AreaSize);
        sensor.AddObservation(m_Rb.velocity);
        sensor.AddObservation(transform.localRotation.eulerAngles.normalized.y);
    }
}
