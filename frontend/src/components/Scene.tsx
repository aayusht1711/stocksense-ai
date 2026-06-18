"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

const GRID_SIZE = 20;
const SPACING = 2;

function CandlestickCity() {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // Generate matrix data for the grid
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  const particles = useMemo(() => {
    const temp = [];
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let z = 0; z < GRID_SIZE; z++) {
        // Random height to simulate market volatility
        const height = 1 + Math.random() * 5;
        // Determine color based on height (taller = green, shorter = red/dark)
        const isUp = height > 3;
        temp.push({
          x: (x - GRID_SIZE / 2) * SPACING,
          y: height / 2, // Shift up so base is at 0
          z: (z - GRID_SIZE / 2) * SPACING,
          height,
          color: isUp ? new THREE.Color("#00FF41") : new THREE.Color("#0A5C22")
        });
      }
    }
    return temp;
  }, []);

  useFrame((state) => {
    if (!meshRef.current) return;
    
    // Slowly move the city towards the camera to simulate scrolling time
    const time = state.clock.getElapsedTime();
    const speed = 2;
    const offsetZ = (time * speed) % SPACING;

    let i = 0;
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let z = 0; z < GRID_SIZE; z++) {
        const particle = particles[i];
        
        // Calculate new Z position with looping
        let currentZ = particle.z + offsetZ;
        if (currentZ > (GRID_SIZE / 2) * SPACING) {
           currentZ -= GRID_SIZE * SPACING;
        }

        // Add a slight wave effect to heights
        const waveHeight = particle.height + Math.sin(time + x * 0.5) * 0.5;

        dummy.position.set(particle.x, waveHeight / 2 - 2, currentZ);
        dummy.scale.set(0.8, waveHeight, 0.8);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
        meshRef.current.setColorAt(i, particle.color);
        i++;
      }
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, GRID_SIZE * GRID_SIZE]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.1} metalness={0.8} />
    </instancedMesh>
  );
}

export default function Scene() {
  return (
    <div className="absolute inset-0 pointer-events-none z-0 opacity-40">
      <Canvas camera={{ position: [0, 8, 15], fov: 50 }}>
        {/* Matrix Theme Lighting */}
        <ambientLight intensity={0.2} color="#00FF41" />
        <directionalLight position={[10, 10, 5]} intensity={1.5} color="#ffffff" />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} color="#00FF41" />
        <pointLight position={[0, 5, 0]} intensity={3} color="#00FF41" distance={20} />
        
        {/* Fog to hide the grid edges */}
        <fog attach="fog" args={['#0D1117', 5, 25]} />
        
        <CandlestickCity />
        
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.2} target={[0, 0, 0]} />
      </Canvas>
    </div>
  );
}
