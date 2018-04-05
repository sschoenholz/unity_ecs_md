using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Entities;
using UnityEngine;
using UnityEngine.Jobs;
using Unity.Transforms;
using Unity.Mathematics;

namespace ParticleSimulator {

    public class SimulateFrame : JobComponentSystem {

        ComponentGroup particle_group;

        [ComputeJobOptimization]
        struct ComputeVelocityVerletHalfStepJob : IJobParallelFor {

            [ReadOnly] public ComponentDataArray<Force> forces;
            [ReadOnly] public float dt;
            [ReadOnly] public Vector3 size;

            public ComponentDataArray<Velocity> velocities;
            public ComponentDataArray<Position> positions;

            public void Execute(int i)
            {
                velocities[i] = new Velocity
                {
                    Value = velocities[i].Value + 0.5f * dt * forces[i].Value
                };

                float3 pos = positions[i].Value;
                pos += velocities[i].Value * dt;

                if (pos.x < 0.0f)
                    pos.x += size.x;
                if (pos.y < 0.0f)
                    pos.y += size.y;
                if (pos.z < 0.0f)
                    pos.z += size.z;

                if (pos.x > size.x)
                    pos.x -= size.x;
                if (pos.y > size.y)
                    pos.y -= size.y;
                if (pos.z > size.z)
                    pos.z -= size.z;

                positions[i] = new Position { Value = pos };
            }
        }

        [ComputeJobOptimization]
        struct ComputeVelocityVerletNoseHooverHalfStepJob : IJobParallelFor
        {
            [ReadOnly] public ComponentDataArray<Velocity> velocities;
            [ReadOnly] public ComponentDataArray<Force> forces;
            [ReadOnly] public float dt;
            [ReadOnly] public Vector3 size;
            [ReadOnly] public float nose_hoover_zeta;

            public ComponentDataArray<Position> positions;
            public NativeArray<Velocity> velocities_half_step;

            // TODO(schsam): Add variable masses. Both here and above.
            public void Execute(int i)
            {
                float3 pos = positions[i].Value;
                pos += velocities[i].Value * dt + 
                    (forces[i].Value - nose_hoover_zeta * velocities[i].Value) * dt * dt / 2f;

                if (pos.x < 0.0f)
                    pos.x += size.x;
                if (pos.y < 0.0f)
                    pos.y += size.y;
                if (pos.z < 0.0f)
                    pos.z += size.z;

                if (pos.x > size.x)
                    pos.x -= size.x;
                if (pos.y > size.y)
                    pos.y -= size.y;
                if (pos.z > size.z)
                    pos.z -= size.z;

                positions[i] = new Position { Value = pos };

                velocities_half_step[i] = new Velocity
                {
                    Value = (1f - 0.5f * dt * nose_hoover_zeta) * velocities[i].Value + 
                            0.5f * dt * forces[i].Value
                };

            }
        }

        [ComputeJobOptimization]
        struct ComputeNoseHooverZetaHalfStepJob : IJob
        {
            [ReadOnly] public ComponentDataArray<Velocity> velocities;
            [ReadOnly] public float nose_hoover_zeta;
            [ReadOnly] public float dt;
            [ReadOnly] public float nose_hoover_inverse_mass;
            [ReadOnly] public float temperature;
            [ReadOnly] public float particle_count;

            public NativeArray<float> nose_hoover_zeta_half_step;

            // TODO(schsam): Have to add variable masses here as well.
            public void Execute()
            {
                float kinetic_energy = 0f;

                for (int i = 0; i < velocities.Length; i++)
                {
                    kinetic_energy += 0.5f * Vector3.SqrMagnitude(velocities[i].Value);
                }
 
                nose_hoover_zeta_half_step[0] = nose_hoover_zeta +
                    0.5f * dt * nose_hoover_inverse_mass * (
                    kinetic_energy - 0.5f * (3f * particle_count + 1f) * temperature);
            }
        }

        [ComputeJobOptimization]
        struct ComputeNoseHooverZetaFinalizeJob : IJob
        {
            [ReadOnly] public NativeArray<Velocity> velocities_half_step;
            [ReadOnly] public NativeArray<float> nose_hoover_zeta_half_step;
            [ReadOnly] public float dt;
            [ReadOnly] public float nose_hoover_inverse_mass;
            [ReadOnly] public float temperature;
            [ReadOnly] public float particle_count;

            public NativeArray<float> nose_hoover_zeta;

            // TODO(schsam): Have to add variable masses here as well.
            public void Execute()
            {
                float kinetic_energy = 0f;

                for (int i = 0; i < velocities_half_step.Length; i++)
                {
                    kinetic_energy += 0.5f * Vector3.SqrMagnitude(velocities_half_step[i].Value);
                }

                nose_hoover_zeta[0] = nose_hoover_zeta_half_step[0] +
                    0.5f * dt * nose_hoover_inverse_mass * (
                    kinetic_energy - 0.5f * (3f * particle_count + 1f) * temperature);
            }
        }

        [ComputeJobOptimization]
        struct HashParticlesJob : IJobParallelFor
        {
            // NOTE(schsam): We only hash the velocities so that we can reduce
            // over them to efficiently compute PE. If we're not interested
            // in the PE computation we can just hash the positions.
            [ReadOnly] public ComponentDataArray<Position> positions;
            [ReadOnly] public ComponentDataArray<Velocity> velocities;

            public NativeArray<int> hashes;
            public NativeMultiHashMap<int, int>.Concurrent hash_map;
            public NativeMultiHashMap<int, Position>.Concurrent hash_positions;
            public NativeMultiHashMap<int, Velocity>.Concurrent hash_velocities;

            public float cell_size;
            public int3 cells_per_side;

            public void Execute(int i)
            {
                int x_index = (int)Math.Floor(positions[i].Value.x / cell_size);
                int y_index = (int)Math.Floor(positions[i].Value.y / cell_size);
                int z_index = (int)Math.Floor(positions[i].Value.z / cell_size);

                int hash = (
                    z_index * cells_per_side.x * cells_per_side.y +
                    y_index * cells_per_side.x + x_index);

                hashes[i] = hash;
                hash_map.Add(hash, i);
                hash_positions.Add(hash, positions[i]);
                hash_velocities.Add(hash, velocities[i]);
            }
        }

        [ComputeJobOptimization]
        struct ComputeForcesJob : IJobParallelFor
        {
            [ReadOnly] public NativeMultiHashMap<int, int> hash_map;
            [ReadOnly] public NativeMultiHashMap<int, Position> cell_positions;

            [ReadOnly] public float epsilon;
            [ReadOnly] public float size;
            [ReadOnly] public int3 cells_per_side;
            [ReadOnly] public int num_cells;

            public NativeMultiHashMap<int, Force>.Concurrent cell_forces;
            public NativeMultiHashMap<int, int>.Concurrent force_hash_map;

            public void ComputeForce(Position p_i, Position p_j, ref float3 force)
            { 
                float3 dr = p_j.Value - p_i.Value;
                float r = Mathf.Sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

                if (r < size)
                {
                    force += -epsilon / size * (1.0f - r / size) * dr / r;
                }
                
            }

            public float3 SumForcesCell(int i, Position p_i,  int cell)
            { 
                if (cell < 0)
                    cell += num_cells;
                if (cell >= num_cells)
                    cell -= num_cells;

                int j;
                Position p_j;

                float3 force = new float3(0f, 0f, 0f);

                NativeMultiHashMapIterator<int> pos_it;
                NativeMultiHashMapIterator<int> hash_it;

                if (cell_positions.TryGetFirstValue(cell, out p_j, out pos_it) &&
                    hash_map.TryGetFirstValue(cell, out j, out hash_it))
                {
                    if (i != j)
                    {
                        ComputeForce(p_i, p_j, ref force);
                    }
                    
                    while (cell_positions.TryGetNextValue(out p_j, ref pos_it) &&
                           hash_map.TryGetNextValue(out j, ref hash_it))
                    {
                        if (i != j)
                        {
                            ComputeForce(p_i, p_j, ref force);
                        }
                    }
                }


                return force;
            }

            public Force SumForces(int i, Position p_i, int cell)
            {
                Force force = new Force { Value = new float3(0f, 0f, 0f) };

                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int current_cell = cell +
                                dz * cells_per_side.x * cells_per_side.y +
                                dy * cells_per_side.x +
                                dx;

                            force.Value += SumForcesCell(i, p_i, current_cell);
                        }
                    }
                }

                return force;
            }

            public void Execute(int cell)
            {
                int i;
                Position p_i;

                NativeMultiHashMapIterator<int> pos_it;
                NativeMultiHashMapIterator<int> hash_it;

                if (cell_positions.TryGetFirstValue(cell, out p_i, out pos_it) &&
                    hash_map.TryGetFirstValue(cell, out i, out hash_it))
                {

                    Force force = SumForces(i, p_i, cell);
                    cell_forces.Add(cell, force);
                    force_hash_map.Add(cell, i);

                    while (cell_positions.TryGetNextValue(out p_i, ref pos_it) &&
                           hash_map.TryGetNextValue(out i, ref hash_it))
                    {
                        force = SumForces(i, p_i, cell);
                        cell_forces.Add(cell, force);
                        force_hash_map.Add(cell, i);
                    }
                }

            }
        }

        [ComputeJobOptimization]
        struct ComputeForcesAndEnergiesJob : IJobParallelFor
        {
            [ReadOnly] public NativeMultiHashMap<int, int> hash_map;
            [ReadOnly] public NativeMultiHashMap<int, Position> cell_positions;

            [ReadOnly] public float epsilon;
            [ReadOnly] public float size;
            [ReadOnly] public float3 system_size;
            [ReadOnly] public int3 cells_per_side;
            [ReadOnly] public int num_cells;

            public NativeMultiHashMap<int, Force>.Concurrent cell_forces;
            public NativeMultiHashMap<int, float>.Concurrent cell_energies;
            public NativeMultiHashMap<int, int>.Concurrent force_hash_map;

            public void ComputeForceAndEnergy(
                Position p_i, Position p_j, ref float energy, ref float3 force)
            {
                float3 dr = p_j.Value - p_i.Value;

                if (Mathf.Abs(dr.x) > system_size.x / 2f)
                    dr.x -= Mathf.Sign(dr.x) * system_size.x;

                if (Mathf.Abs(dr.y) > system_size.y / 2f)
                    dr.y -= Mathf.Sign(dr.y) * system_size.y;

                if (Mathf.Abs(dr.z) > system_size.z / 2f)
                    dr.z -= Mathf.Sign(dr.z) * system_size.z;

                float r = Mathf.Sqrt(dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

                if (r < size)
                {
                    force += -epsilon / size * (1.0f - r / size) * dr / r;
                    energy += epsilon / 2f * (1f - r / size) * (1f - r / size);
                }
            }

            public float4 SumForcesAndEnergiesCell(int i, Position p_i, int cell)
            {
                int j;
                Position p_j;

                float3 force = new float3(0f, 0f, 0f);
                float energy = 0f;

                NativeMultiHashMapIterator<int> pos_it;
                NativeMultiHashMapIterator<int> hash_it;

                if (cell_positions.TryGetFirstValue(cell, out p_j, out pos_it) &&
                    hash_map.TryGetFirstValue(cell, out j, out hash_it))
                {
                    if (i != j)
                    {
                        ComputeForceAndEnergy(p_i, p_j, ref energy, ref force);
                    }

                    while (cell_positions.TryGetNextValue(out p_j, ref pos_it) &&
                           hash_map.TryGetNextValue(out j, ref hash_it))
                    {
                        if (i != j)
                        {
                            ComputeForceAndEnergy(p_i, p_j, ref energy, ref force);
                        }
                    }
                }


                return new float4(force, energy);
            }

            public float4 SumForcesAndEnergies(int i, Position p_i, int3 cell)
            {
                float4 result = new float4(0f, 0f, 0f, 0f);          

                for (int dz = -1; dz <= 1; dz++)
                {
                    int cell_index_z = cell.z + dz;

                    if (cell_index_z < 0)
                        cell_index_z += cells_per_side.z;
                    if (cell_index_z >= cells_per_side.z)
                        cell_index_z -= cells_per_side.z;

                    for (int dy = -1; dy <= 1; dy++)
                    {

                        int cell_index_y = cell.y + dy;

                        if (cell_index_y < 0)
                            cell_index_y += cells_per_side.y;
                        if (cell_index_y >= cells_per_side.y)
                            cell_index_y -= cells_per_side.y;

                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int cell_index_x = cell.x + dx;

                            if (cell_index_x < 0)
                                cell_index_x += cells_per_side.x;
                            if (cell_index_x >= cells_per_side.x)
                                cell_index_x -= cells_per_side.x;

                            int current_cell = cell_index_z * cells_per_side.x * cells_per_side.y +
                                cell_index_y * cells_per_side.x +
                                cell_index_x;

                            result += SumForcesAndEnergiesCell(i, p_i, current_cell);
                        }
                    }
                }

                return result;
            }

            public void Execute(int cell)
            {
                int i;
                Position p_i;

                NativeMultiHashMapIterator<int> pos_it;
                NativeMultiHashMapIterator<int> hash_it;

                int cell_index_x = cell % cells_per_side.x;
                int cell_index_y = cell % (cells_per_side.x * cells_per_side.y);
                cell_index_y /= cells_per_side.x;
                int cell_index_z = cell / (cells_per_side.x * cells_per_side.y);

                int3 cell_index = new int3(
                    cell_index_x,
                    cell_index_y,
                    cell_index_z);

                if (cell_positions.TryGetFirstValue(cell, out p_i, out pos_it) &&
                    hash_map.TryGetFirstValue(cell, out i, out hash_it))
                {

                    float4 result = SumForcesAndEnergies(i, p_i, cell_index);

                    Force force = new Force
                    {
                        Value = new float3(result.x, result.y, result.z)
                    };

                    cell_forces.Add(cell, force);
                    cell_energies.Add(cell, result.w);
                    force_hash_map.Add(cell, i);

                    while (cell_positions.TryGetNextValue(out p_i, ref pos_it) &&
                           hash_map.TryGetNextValue(out i, ref hash_it))
                    {
                        result = SumForcesAndEnergies(i, p_i, cell_index);

                        force = new Force
                        {
                            Value = new float3(result.x, result.y, result.z)
                        };

                        cell_forces.Add(cell, force);
                        cell_energies.Add(cell, result.w);
                        force_hash_map.Add(cell, i);
                    }
                }

            }
        }

        [ComputeJobOptimization]
        struct UnhashForcesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int> hashes;
            [ReadOnly] public NativeMultiHashMap<int, int> hash_map;
            [ReadOnly] public NativeMultiHashMap<int, Force> cell_forces;

            public NativeArray<Force> forces;

            public void Execute(int i)
            {
                NativeMultiHashMapIterator<int> hash_it;
                NativeMultiHashMapIterator<int> force_it;

                int j;
                Force force;
                int cell = hashes[i];

                if (hash_map.TryGetFirstValue(cell, out j, out hash_it) &&
                    cell_forces.TryGetFirstValue(cell, out force, out force_it))
                {

                    if (i == j)
                    {
                        forces[i] = force;
                        return;
                    }

                    while (hash_map.TryGetNextValue(out j, ref hash_it) &&
                           cell_forces.TryGetNextValue(out force, ref force_it))
                    {
                        if (i == j)
                        {
                            forces[i] = force;
                            return;
                        }
                    }
                }

            }
        }

        [ComputeJobOptimization]
        struct GradientDescentUpdateJob : IJobParallelFor
        {
            public ComponentDataArray<Position> positions;

            [ReadOnly] public NativeArray<Force> forces;
            [ReadOnly] public Vector3 size;
            [ReadOnly] public float dt;

            public void Execute(int i)
            {
                float3 pos = positions[i].Value;
                pos += forces[i].Value * dt;

                if (pos.x < 0.0f)
                    pos.x += size.x;
                if (pos.y < 0.0f)
                    pos.y += size.y;
                if (pos.z < 0.0f)
                    pos.z += size.z;

                if (pos.x > size.x)
                    pos.x -= size.x;
                if (pos.y > size.y)
                    pos.y -= size.y;
                if (pos.z > size.z)
                    pos.z -= size.z;

                positions[i] = new Position { Value = pos };
            }
        }

        [ComputeJobOptimization]
        struct FinalizeVelocityVerletStepJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Force> new_forces;
            [ReadOnly] public float dt;

            public ComponentDataArray<Velocity> velocities;
            public ComponentDataArray<Force> forces;

            public void Execute(int i)
            {
                velocities[i] = new Velocity
                {
                    Value = velocities[i].Value + 0.5f * dt * new_forces[i].Value
                };

                forces[i] = new_forces[i];
            }
        }

        [ComputeJobOptimization]
        struct FinalizeVelocityVerletNoseHooverStepJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Force> new_forces;
            [ReadOnly] public NativeArray<Velocity> velocities_half_step;
            [ReadOnly] public float dt;
            [ReadOnly] public NativeArray<float> nose_hoover_zeta;

            public ComponentDataArray<Velocity> velocities;
            public ComponentDataArray<Force> forces;

            // TODO(schsam): Add masses (and above).
            public void Execute(int i)
            {
                float3 v_update = velocities_half_step[i].Value + 0.5f * dt * new_forces[i].Value;

                velocities[i] = new Velocity
                {
                    Value = v_update / (1f + 0.5f * dt * nose_hoover_zeta[0])
                };

                forces[i] = new_forces[i];
            }
        }

        [ComputeJobOptimization]
        struct SumEnergiesInCellJob : IJobParallelFor
        {
            // NOTE(schsam): At the moment there is 1/2 dt difference between
            // the kinetic energies that we are summing and the potential
            // energies. This is almost certainly not a big deal, however we
            // could avoid this by hashing the velocities post-finalization.
            // This gives the scheduler less flexibility though.
            [ReadOnly] public NativeMultiHashMap<int, Velocity> cell_velocities;
            [ReadOnly] public NativeMultiHashMap<int, float> cell_energies;

            public NativeArray<float> kinetic_energy_per_cell;
            public NativeArray<float> potential_energy_per_cell;

            public void Execute(int cell)
            {
                float kinetic_energy = 0f;
                float potential_energy = 0f;

                Velocity velocity;
                float energy;

                NativeMultiHashMapIterator<int> v_it;
                NativeMultiHashMapIterator<int> e_it;

                if (cell_velocities.TryGetFirstValue(cell, out velocity, out v_it) &&
                    cell_energies.TryGetFirstValue(cell, out energy, out e_it))
                {
                    potential_energy += energy;
                    float speed_sq = velocity.Value.x * velocity.Value.x +
                        velocity.Value.y * velocity.Value.y +
                        velocity.Value.z * velocity.Value.z;

                    // NOTE(schsam): This currently assumes equal masses.
                    kinetic_energy += 0.5f * speed_sq;

                    while (cell_velocities.TryGetNextValue(out velocity, ref v_it) &&
                           cell_energies.TryGetNextValue(out energy, ref e_it))
                    {
                        potential_energy += energy;
                        speed_sq = velocity.Value.x * velocity.Value.x +
                                   velocity.Value.y * velocity.Value.y +
                                   velocity.Value.z * velocity.Value.z;

                        // NOTE(schsam): This currently assumes equal masses.
                        kinetic_energy += 0.5f * speed_sq;
                    }
                }

                kinetic_energy_per_cell[cell] = kinetic_energy;
                potential_energy_per_cell[cell] = potential_energy;
            }
        }

        [ComputeJobOptimization]
        struct SumEnergiesOverCellsJob : IJob
        {
            [ReadOnly] public NativeArray<float> kinetic_energy_per_cell;
            [ReadOnly] public NativeArray<float> potential_energy_per_cell;

            public NativeArray<float> energies;

            public void Execute()
            {
                energies[0] = 0f;
                energies[1] = 0f;

                for(int i = 0; i < kinetic_energy_per_cell.Length; i++)
                {
                    energies[0] += kinetic_energy_per_cell[i];
                    energies[1] += potential_energy_per_cell[i];
                }
            }
        }

        public enum SimulationMode { NVT, NVE, GD };

        static public Vector3 system_size;
        static public int particle_count;
        static public float cell_size;
        static public SimulationMode mode;
        static public float particle_size;

        static public float potential_energy = 0f;
        static public float kinetic_energy = 0f;
        static public float epsilon = 1f;

        static public float nose_hoover_mass = 1f;
        static public float nose_hoover_zeta = 0f;
        static public float temperature = 1f;

        protected override JobHandle OnUpdate(JobHandle inputDeps)
        {
            float system_volume = system_size.x * system_size.y * system_size.z;
            float step_size = 1f;

            int3 cells_per_side = new int3(
                (int)Math.Ceiling(system_size.x / cell_size),
                (int)Math.Ceiling(system_size.y / cell_size),
                (int)Math.Ceiling(system_size.z / cell_size));
            float cell_volume = cell_size * cell_size * cell_size;

            int num_cells = (int)Math.Ceiling(system_volume / cell_volume);

            var particle_forces = 
                new NativeArray<Force>(particle_count, Allocator.TempJob);

            // Map from particle to hashes.
            var particle_hashes = 
                new NativeArray<int>(particle_count, Allocator.TempJob);

            // NOTE(schsam): We need the extra hash map because the order of
            // forces and energies in the bucket will not be the same as the
            // order of positions and velocities.

            // Maps from hashes to particle indices.
            var force_hash_map = 
                new NativeMultiHashMap<int, int>(particle_count, Allocator.TempJob);
            var hash_map = 
                new NativeMultiHashMap<int, int>(particle_count, Allocator.TempJob);

            // All hashed quantities.
            var cell_positions = 
                new NativeMultiHashMap<int, Position>(particle_count, Allocator.TempJob);
            var cell_velocities = 
                new NativeMultiHashMap<int, Velocity>(particle_count, Allocator.TempJob);
            var cell_forces = 
                new NativeMultiHashMap<int, Force>(particle_count, Allocator.TempJob);
            var cell_energies = 
                new NativeMultiHashMap<int, float>(particle_count, Allocator.TempJob);

            // Cell statistics
            var kinetic_energy_per_cell =
                new NativeArray<float>(num_cells, Allocator.TempJob);
            var potential_energy_per_cell =
                new NativeArray<float>(num_cells, Allocator.TempJob);

            // Nose Hoover quantities
            var velocities_half_step =
                new NativeArray<Velocity>(particle_count, Allocator.TempJob);
            float nose_hoover_inverse_mass = 1f / nose_hoover_mass;
            NativeArray<float> nose_hoover_zeta_step =
                new NativeArray<float>(1, Allocator.TempJob);
            NativeArray<float> nose_hoover_zeta_half_step =
                new NativeArray<float>(1, Allocator.TempJob);

            float dt = step_size * Time.deltaTime;

            JobHandle current_deps = inputDeps;
            JobHandle compute_nose_hoover_zeta_finalize_handle = inputDeps;

            if (mode == SimulationMode.NVE)
            {
                var velocity_verlet_half_step_job = new ComputeVelocityVerletHalfStepJob()
                {
                    forces = particle_group.GetComponentDataArray<Force>(),
                    dt = dt,
                    size = system_size,
                    velocities = particle_group.GetComponentDataArray<Velocity>(),
                    positions = particle_group.GetComponentDataArray<Position>()
                };

                current_deps = 
                    velocity_verlet_half_step_job.Schedule(
                        particle_count, 64, current_deps);
            } else if(mode == SimulationMode.NVT)
            {
                var velocity_verlet_half_step_job =
                    new ComputeVelocityVerletNoseHooverHalfStepJob()
                    {
                        forces = particle_group.GetComponentDataArray<Force>(),
                        dt = dt,
                        size = system_size,
                        velocities = particle_group.GetComponentDataArray<Velocity>(),
                        velocities_half_step = velocities_half_step,
                        nose_hoover_zeta = nose_hoover_zeta,
                        positions = particle_group.GetComponentDataArray<Position>()
                    };

                var velocity_verlet_half_step_handle =
                    velocity_verlet_half_step_job.Schedule(
                        particle_count, 64, current_deps);

                float f_particle_count = (float)particle_count;

                var compute_nose_hoover_zeta_half_step_job =
                    new ComputeNoseHooverZetaHalfStepJob()
                    {
                        velocities = particle_group.GetComponentDataArray<Velocity>(),
                        nose_hoover_zeta = nose_hoover_zeta,
                        dt = dt,
                        nose_hoover_inverse_mass = nose_hoover_inverse_mass,
                        temperature = temperature,
                        particle_count = f_particle_count,
                        nose_hoover_zeta_half_step = nose_hoover_zeta_half_step
                    };

                var compute_nose_hoover_zeta_half_step_handle =
                    compute_nose_hoover_zeta_half_step_job.Schedule(current_deps);

                var compute_nose_hoover_zeta_full_step_barrier =
                    JobHandle.CombineDependencies(
                        compute_nose_hoover_zeta_half_step_handle,
                        velocity_verlet_half_step_handle);

                var compute_nose_hoover_zeta_finalize_job =
                    new ComputeNoseHooverZetaFinalizeJob()
                    {
                        velocities_half_step = velocities_half_step,
                        nose_hoover_zeta = nose_hoover_zeta_step,
                        dt = dt,
                        nose_hoover_inverse_mass = nose_hoover_inverse_mass,
                        temperature = temperature,
                        particle_count = (float)particle_count,
                        nose_hoover_zeta_half_step = nose_hoover_zeta_half_step
                    };

                compute_nose_hoover_zeta_finalize_handle =
                    compute_nose_hoover_zeta_finalize_job.Schedule(compute_nose_hoover_zeta_full_step_barrier);

                current_deps = velocity_verlet_half_step_handle;
            }

            var hash_particle_job = new HashParticlesJob()
            {
                positions = particle_group.GetComponentDataArray<Position>(),
                velocities = particle_group.GetComponentDataArray<Velocity>(),
                hash_map = hash_map,
                hash_positions = cell_positions,
                hash_velocities = cell_velocities,
                hashes = particle_hashes,
                cell_size = cell_size,
                cells_per_side = cells_per_side
            };

            var hash_particle_handle = 
                hash_particle_job.Schedule(particle_count, 64, current_deps);

            var compute_forces_job = new ComputeForcesAndEnergiesJob() {
                hash_map = hash_map,
                cell_positions = cell_positions,
                epsilon = epsilon,
                size = 1f,
                cells_per_side = cells_per_side,
                num_cells = num_cells,
                cell_forces = cell_forces,
                cell_energies = cell_energies,
                force_hash_map = force_hash_map,
                system_size = system_size
            };

            var compute_forces_handle =
                compute_forces_job.Schedule(num_cells, 64, hash_particle_handle);

            var unhash_forces_job = new UnhashForcesJob()
            {
                hashes = particle_hashes,
                hash_map = force_hash_map,
                cell_forces = cell_forces,
                forces = particle_forces
            };

            var unhash_forces_handle =
                unhash_forces_job.Schedule(particle_count, 64, compute_forces_handle);

            JobHandle step_finalize_handle = unhash_forces_handle;

            if (mode == SimulationMode.NVE)
            {
                var velocity_verlet_finalize_job = new FinalizeVelocityVerletStepJob()
                {
                    new_forces = particle_forces,
                    dt = dt,
                    velocities = particle_group.GetComponentDataArray<Velocity>(),
                    forces = particle_group.GetComponentDataArray<Force>()
                };

                step_finalize_handle = 
                    velocity_verlet_finalize_job.Schedule(particle_count, 64, unhash_forces_handle);

            }
            else if (mode == SimulationMode.NVT)
            {
                var finalize_step_deps = 
                    JobHandle.CombineDependencies(
                        unhash_forces_handle, 
                        compute_nose_hoover_zeta_finalize_handle);

                var velocity_verlet_finalize_job = new FinalizeVelocityVerletNoseHooverStepJob()
                {
                    new_forces = particle_forces,
                    dt = dt,
                    velocities_half_step = velocities_half_step,
                    velocities = particle_group.GetComponentDataArray<Velocity>(),
                    forces = particle_group.GetComponentDataArray<Force>(),
                    nose_hoover_zeta = nose_hoover_zeta_step
                };

                step_finalize_handle =
                    velocity_verlet_finalize_job.Schedule(particle_count, 64, finalize_step_deps);

            }
            else if (mode == SimulationMode.GD)
            {
                var position_job = new GradientDescentUpdateJob()
                {
                    dt = dt,
                    positions = particle_group.GetComponentDataArray<Position>(),
                    forces = particle_forces,
                    size = system_size
                };

                step_finalize_handle =
                    position_job.Schedule(particle_count, 64, unhash_forces_handle);
            }

            var sum_energies_in_cell_job = new SumEnergiesInCellJob()
            {
                cell_velocities = cell_velocities,
                cell_energies = cell_energies,
                kinetic_energy_per_cell = kinetic_energy_per_cell,
                potential_energy_per_cell = potential_energy_per_cell
            };

            var sum_energies_in_cell_handle =
                sum_energies_in_cell_job.Schedule(num_cells, 64, compute_forces_handle);

            var temp_energies = new NativeArray<float>(2, Allocator.TempJob);

            var sum_energies_over_cells_job = new SumEnergiesOverCellsJob()
            {
                kinetic_energy_per_cell = kinetic_energy_per_cell,
                potential_energy_per_cell = potential_energy_per_cell,
                energies = temp_energies
            };

            var sum_energies_over_cells_handle =
                sum_energies_over_cells_job.Schedule(sum_energies_in_cell_handle);

            var step_complete_barrier = 
                JobHandle.CombineDependencies(sum_energies_over_cells_handle, step_finalize_handle);

            step_complete_barrier.Complete();

            nose_hoover_zeta = nose_hoover_zeta_step[0];
            kinetic_energy = temp_energies[0];
            potential_energy = temp_energies[1];

            nose_hoover_zeta_step.Dispose();
            nose_hoover_zeta_half_step.Dispose();
            cell_velocities.Dispose();
            cell_energies.Dispose();
            force_hash_map.Dispose();
            particle_forces.Dispose();
            particle_hashes.Dispose();
            hash_map.Dispose();
            cell_positions.Dispose();
            cell_forces.Dispose();
            kinetic_energy_per_cell.Dispose();
            potential_energy_per_cell.Dispose();
            temp_energies.Dispose();
            velocities_half_step.Dispose();

            return hash_particle_handle;
        }

        protected override void OnCreateManager(int capacity)
        {
            particle_group = GetComponentGroup(
                ComponentType.ReadOnly(typeof(Particle)),
                typeof(Position),
                typeof(Velocity),
                typeof(Force));
        }

    }
}