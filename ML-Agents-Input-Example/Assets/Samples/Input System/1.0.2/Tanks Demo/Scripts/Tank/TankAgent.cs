using System;
using System.Net.Sockets;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class TankAgent : Agent
{
    TankAgent m_Opponent;
    TankShooting m_Shooting;
    BehaviorParameters m_Bp;
    Rigidbody m_Rb;
    TankHealth m_Th;

    public Transform FireTransform;

    const int k_AreaSize = 100;

    void Awake()
    {
        m_Bp = GetComponent<BehaviorParameters>();
        m_Shooting = GetComponent<TankShooting>();
        m_Rb = GetComponent<Rigidbody>();
        m_Th = GetComponent<TankHealth>();
    }

    public override void Initialize()
    {
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
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // relative vector to opponent
        if (ReferenceEquals(m_Opponent, null))
        {
            return;
        }

        // Vector to opponent
        sensor.AddObservation(transform.localRotation.normalized.y);
        var opponentPos = m_Opponent.FireTransform.position;
        var position = FireTransform.position;
        // charge time
        sensor.AddObservation(m_Shooting.m_CurrentLaunchForce / m_Shooting.maxLaunchForce);
        sensor.AddObservation(m_Rb.velocity.normalized);
        sensor.AddObservation(m_Th.currentHealth / m_Th.startingHealth);
        sensor.AddObservation(Vector3.Dot((position - opponentPos).normalized, FireTransform.forward));
        sensor.AddObservation(Vector3.Distance(position, opponentPos) / k_AreaSize);
    }
}
