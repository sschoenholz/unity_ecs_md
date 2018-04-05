using System;
using Unity.Entities;
using Unity.Rendering;
using UnityEngine;
using Unity.Transforms;
using Unity.Mathematics;

namespace ParticleSimulator
{
    public struct Particle : IComponentData { }
    
    public struct Velocity : IComponentData { public float3 Value; }

    public struct Force : IComponentData { public float3 Value; }

    public class Simulator : MonoBehaviour
    {
        [SerializeField]
        public int particle_count = 10000;

        [SerializeField]
        public float particle_radius = 1.0f;

        [SerializeField]
        public float cell_size = 2.0f;

        [SerializeField]
        public float epsilon = 1f;

        [SerializeField]
        public Vector3 size = new Vector3(10f, 10f, 10f);

        [SerializeField]
        public GameObject particle_object;

        [SerializeField]
        public SimulateFrame.SimulationMode mode;

        [SerializeField]
        public float nose_hoover_frequency;

        [SerializeField]
        public float temperature;

        [SerializeField]
        public float goal_temperature;

        [SerializeField]
        public float temperature_steps;

        EntityArchetype particle_archetype;

        MeshInstanceRenderer particle_renderer;

        bool quenching;
        float temperature_rate = 0f;

        // Use this for initialization
        void Start()
        {
            var entity_manager = World.Active.GetOrCreateManager<EntityManager>();

            particle_archetype = entity_manager.CreateArchetype(
                typeof(Particle), typeof(Position), typeof(Velocity), typeof(Force), typeof(TransformMatrix));

            var particle_look = GameObject.Instantiate(particle_object);
            particle_renderer = particle_look.GetComponent<MeshInstanceRendererComponent>().Value;

            particle_renderer.receiveShadows = true;
            particle_renderer.castShadows = UnityEngine.Rendering.ShadowCastingMode.On;

            float velocity_rescale = Mathf.Sqrt(3f * temperature);
            for (int i = 0; i < particle_count; i++)
            {
                Entity particle = entity_manager.CreateEntity(particle_archetype);

                entity_manager.SetComponentData(particle, new Position
                {
                    Value = new float3(UnityEngine.Random.Range(0f, size.x),
                                       UnityEngine.Random.Range(0f, size.y),
                                       UnityEngine.Random.Range(0f, size.z))
                });

                
                entity_manager.SetComponentData(particle, new Velocity
                {
                    Value = velocity_rescale * (new float3(UnityEngine.Random.Range(-1f, 1f),
                                       UnityEngine.Random.Range(-1f, 1f),
                                       UnityEngine.Random.Range(-1f, 1f)))

                });

                entity_manager.SetComponentData(particle, new Force
                {
                    Value = new float3(0f, 0f, 0f)

                });

                entity_manager.AddSharedComponentData(particle, particle_renderer);
            }

            GameObject.Destroy(particle_look);


            SimulateFrame.system_size = size;
            SimulateFrame.particle_count = particle_count;
            SimulateFrame.cell_size = cell_size;
            SimulateFrame.mode = mode;
            SimulateFrame.particle_size = particle_radius;
            SimulateFrame.epsilon = epsilon;

            float freq = nose_hoover_frequency * Time.deltaTime;
            SimulateFrame.nose_hoover_mass =
                nose_hoover_frequency * nose_hoover_frequency * 3f * particle_count * temperature;

            SimulateFrame.temperature = temperature;

            current_mode = mode;

            vertical_rotation = 0f;

            style = new GUIStyle();
            style.normal.background = new Texture2D(2, 2);

            Color background = new Color(0.0f, 0.0f, 0.0f);
            style.normal.background.SetPixel(0, 0, background);
            style.normal.background.SetPixel(1, 0, background);
            style.normal.background.SetPixel(0, 1, background);
            style.normal.background.SetPixel(1, 1, background);

            style.active.background = style.normal.background;

            quenching = false;
        }

        GUIStyle style;

        private void OnGUI()
        {

            float ke_per_atom = SimulateFrame.kinetic_energy / SimulateFrame.particle_count;
            float pe_per_atom = SimulateFrame.potential_energy / SimulateFrame.particle_count;
            float total_energy = ke_per_atom + pe_per_atom;

            float measured_temperature = 2f * SimulateFrame.kinetic_energy / (3f * SimulateFrame.particle_count + 1f);



            GUILayout.BeginArea(new Rect(10f, 10f, 200f, 400f), style);
            GUILayout.Label("Total Energy = " + total_energy);
            GUILayout.Label("Kinetic Energy = " + ke_per_atom);
            GUILayout.Label("Potential Energy = " + pe_per_atom);
            GUILayout.Label("Temperature = " + measured_temperature);
            GUILayout.Label("Goal Temperature = " + SimulateFrame.temperature);
            GUILayout.Label("Nose-Hoover Zeta = " + SimulateFrame.nose_hoover_zeta);

            if (GUILayout.Button("Gradient Descent"))
                SimulateFrame.mode = SimulateFrame.SimulationMode.GD;
            if (GUILayout.Button("NVE Dynamics"))
                SimulateFrame.mode = SimulateFrame.SimulationMode.NVE;
            if (GUILayout.Button("NVT Dynamics"))
                SimulateFrame.mode = SimulateFrame.SimulationMode.NVT;

            if (GUILayout.Button("Start Quench"))
            {
                quenching = true;
                temperature_rate = (goal_temperature - temperature) / temperature_steps;
            }

            GUILayout.TextArea("Epsilon:");
            string epsilon = GUILayout.TextField("1.0");
            SimulateFrame.epsilon = float.Parse(epsilon);

            GUILayout.EndArea();
        }

        SimulateFrame.SimulationMode current_mode;
        float vertical_rotation;

        // Update is called once per frame
        void Update()
        {
            Transform rotor = transform.Find("Rotate");
            Transform camera_transform = rotor.Find("Main Camera");

            if (Math.Abs(Input.GetAxis("LeftJoystickHorizontal")) > 0.6f)
            {
                transform.Rotate(
                    new Vector3(0f, 1f, 0f),
                    -15f * Time.deltaTime * Math.Sign(Input.GetAxis("LeftJoystickHorizontal")));
            }

            if (Math.Abs(Input.GetAxis("LeftJoystickVertical")) > 0.6f)
            {
                Quaternion rot = Quaternion.AngleAxis(
                    15f * Time.deltaTime * Math.Sign(Input.GetAxis("LeftJoystickVertical")),
                    new Vector3(1f, 0f, -1f));
                rotor.localRotation *= rot;
            }

            if (Input.GetButton("RTButton"))
            {
                camera_transform.localPosition -= new Vector3(10f, 10f, 10f) * Time.deltaTime;
            }

            if (Input.GetButton("LTButton"))
            {
                camera_transform.localPosition += new Vector3(10f, 10f, 10f) * Time.deltaTime;
            }

            if(quenching)
            {
                temperature += temperature_rate;
                SimulateFrame.temperature = temperature;

                if (Mathf.Abs(temperature - goal_temperature) < 0.001f)
                    quenching = false;
            }

        }

        private void OnDestroy()
        {
        }
    }

}