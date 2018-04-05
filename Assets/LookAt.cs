using Unity.Mathematics;

using Unity.Entities;
using Unity.Rendering;
using UnityEngine;
using Unity.Transforms;
using System;

public class LookAt : MonoBehaviour
{

    public Transform target;
    public Transform simulation;

    // Use this for initialization
    void Start()
    {
        target.parent = simulation;
        transform.parent = target;
    }

    // Update is called once per frame
    void Update()
    {
        transform.LookAt(target);
    }
}
