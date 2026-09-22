import { useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh, ShaderMaterial, Vector3, Quaternion, Euler } from 'three';
import { usePipelineStore } from '../../store/usePipelineStore';
import { vertexShader, fragmentShader } from '../../shaders/integrityShader';

interface PipelineSegmentProps {
    segmentId: string;
    position: [number, number, number];
    direction?: [number, number, number];
    integrity: number;
}

/**
 * Individual pipeline segment with custom integrity shader
 * Handles click selection and hover effects
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
        <mesh
            ref={meshRef}
            position={position}
            quaternion={quaternion}
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

            {/* Custom shader material */}
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

            {/* Hover outline effect */}
            {hovered && (
                <mesh scale={1.05}>
                    <cylinderGeometry args={[0.5, 0.5, 2, 32]} />
                    <meshBasicMaterial color="#ffffff" transparent opacity={0.1} />
                </mesh>
            )}
        </mesh>
    );
}
