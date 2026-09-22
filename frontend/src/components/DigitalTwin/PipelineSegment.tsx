import { useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh, ShaderMaterial, Vector3, Quaternion, Euler } from 'three';
import { Edges } from '@react-three/drei';
import { usePipelineStore } from '../../store/usePipelineStore';
import { vertexShader, fragmentShader } from '../../shaders/integrityShader';

interface PipelineSegmentProps {
    segmentId: string;
    position: [number, number, number];
    direction?: [number, number, number];
    integrity: number;
}

/**
 * Individual pipeline segment with industrial details (flanges, bolt rings, support saddles)
 * and custom integrity shader mapping PINN physics outputs.
 */
export default function PipelineSegment({ segmentId, position, direction, integrity }: PipelineSegmentProps) {
    const meshRef = useRef<Mesh>(null);
    const materialRef = useRef<ShaderMaterial>(null);
    const [hovered, setHovered] = useState(false);

    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);
    const selectSegment = usePipelineStore(state => state.selectSegment);

    const isSelected = selectedSegmentId === segmentId;

    // Calculate quaternion from direction (cylinder is aligned along Y axis)
    const quaternion = useMemo(() => {
        if (!direction) {
            const q = new Quaternion();
            q.setFromEuler(new Euler(0, 0, Math.PI / 2));
            return q;
        }
        const dir = new Vector3(direction[0], direction[1], direction[2]).normalize();
        const up = new Vector3(0, 1, 0);
        return new Quaternion().setFromUnitVectors(up, dir);
    }, [direction]);

    // Bolt coordinates for 8 bolts around flange collar (radius 0.52)
    const boltPositions = useMemo(() => {
        const coords: [number, number, number][] = [];
        const numBolts = 8;
        const radius = 0.52;
        for (let i = 0; i < numBolts; i++) {
            const angle = (i / numBolts) * Math.PI * 2;
            coords.push([Math.cos(angle) * radius, 0, Math.sin(angle) * radius]);
        }
        return coords;
    }, []);

    // Update shader uniforms every frame
    useFrame((state) => {
        if (materialRef.current) {
            materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
            materialRef.current.uniforms.uIntegrity.value = integrity;
            materialRef.current.uniforms.uSelected.value = isSelected;
        }
    });

    // Handle segment click
    const handleClick = (e: any) => {
        e.stopPropagation();
        selectSegment(segmentId);
    };

    return (
        <group position={position} quaternion={quaternion}>
            {/* Main Segment Cylinder */}
            <mesh
                ref={meshRef}
                onClick={handleClick}
                onPointerOver={(e) => {
                    e.stopPropagation();
                    setHovered(true);
                    document.body.style.cursor = 'pointer';
                }}
                onPointerOut={() => {
                    setHovered(false);
                    document.body.style.cursor = 'default';
                }}
            >
                {/* Cylinder geometry: 2m long, 0.5m radius */}
                <cylinderGeometry args={[0.5, 0.5, 2, 32]} />

                {/* Custom integrity PBR shader material */}
                <shaderMaterial
                    ref={materialRef}
                    vertexShader={vertexShader}
                    fragmentShader={fragmentShader}
                    uniforms={{
                        uIntegrity: { value: integrity },
                        uTime: { value: 0 },
                        uSelected: { value: isSelected }
                    }}
                />

                {/* Structural Geometry Edges */}
                <Edges color={isSelected ? '#FE4C40' : '#88AABB'} threshold={25} transparent opacity={isSelected ? 0.8 : 0.25} />
            </mesh>

            {/* Industrial Flange Collar at Top (+0.96m) */}
            <mesh position={[0, 0.96, 0]}>
                <cylinderGeometry args={[0.55, 0.55, 0.08, 32]} />
                <meshStandardMaterial color="#2E3845" roughness={0.3} metalness={0.8} />
                <Edges color="#7A94A8" threshold={15} transparent opacity={0.4} />

                {/* Bolt Studs on Top Flange */}
                {boltPositions.map((pos, idx) => (
                    <mesh key={`top-bolt-${idx}`} position={pos}>
                        <cylinderGeometry args={[0.025, 0.025, 0.1, 8]} />
                        <meshStandardMaterial color="#8899AA" roughness={0.2} metalness={0.9} />
                    </mesh>
                ))}
            </mesh>

            {/* Industrial Flange Collar at Bottom (-0.96m) */}
            <mesh position={[0, -0.96, 0]}>
                <cylinderGeometry args={[0.55, 0.55, 0.08, 32]} />
                <meshStandardMaterial color="#2E3845" roughness={0.3} metalness={0.8} />
                <Edges color="#7A94A8" threshold={15} transparent opacity={0.4} />

                {/* Bolt Studs on Bottom Flange */}
                {boltPositions.map((pos, idx) => (
                    <mesh key={`bot-bolt-${idx}`} position={pos}>
                        <cylinderGeometry args={[0.025, 0.025, 0.1, 8]} />
                        <meshStandardMaterial color="#8899AA" roughness={0.2} metalness={0.9} />
                    </mesh>
                ))}
            </mesh>

            {/* Ground Support Cradle resting under bottom joint */}
            <group position={[0, -1.05, -0.3]}>
                {/* Cradle Support Base Plate */}
                <mesh position={[0, -0.2, 0]}>
                    <boxGeometry args={[1.2, 0.1, 0.5]} />
                    <meshStandardMaterial color="#1E2630" roughness={0.6} metalness={0.7} />
                    <Edges color="#4A5868" transparent opacity={0.4} />
                </mesh>
                {/* Vertical Cradle Support Uprights */}
                <mesh position={[-0.45, -0.05, 0]}>
                    <boxGeometry args={[0.1, 0.25, 0.35]} />
                    <meshStandardMaterial color="#283340" roughness={0.5} metalness={0.7} />
                </mesh>
                <mesh position={[0.45, -0.05, 0]}>
                    <boxGeometry args={[0.1, 0.25, 0.35]} />
                    <meshStandardMaterial color="#283340" roughness={0.5} metalness={0.7} />
                </mesh>
            </group>

            {/* Hover Outline Glow Effect */}
            {hovered && (
                <mesh scale={1.04}>
                    <cylinderGeometry args={[0.5, 0.5, 2, 32]} />
                    <meshBasicMaterial color="#ffffff" transparent opacity={0.12} />
                </mesh>
            )}
        </group>
    );
}

