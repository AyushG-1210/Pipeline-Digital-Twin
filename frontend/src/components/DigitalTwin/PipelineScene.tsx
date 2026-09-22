import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import Pipeline from './Pipeline';
import SeverityScale from './SeverityScale';

/**
 * Main 3D scene containing the pipeline digital twin
 * Features orbital camera controls, industrial grid background, and severity scale legend
 */
export default function PipelineScene() {
    return (
        <div className="w-full h-full bg-[#0B1330] flex flex-col overflow-hidden">
            {/* 3D Canvas area */}
            <div className="flex-1 w-full min-h-0 relative bg-[#0B1330]">
                <Canvas>
                    {/* Background color — dark navy, tying the viewport to the primary ink color */}
                    <color attach="background" args={['#0B1330']} />

                    {/* Camera setup */}
                    <PerspectiveCamera makeDefault position={[15, 8, 15]} fov={50} />

                    {/* Orbital controls: Left-click rotates, Right-click pans (moves) */}
                    <OrbitControls
                        enableDamping
                        dampingFactor={0.05}
                        minDistance={1}
                        maxDistance={150}
                        maxPolarAngle={Math.PI / 2}
                    />

                    {/* Lighting */}
                    <ambientLight intensity={0.4} />
                    <directionalLight position={[10, 10, 5]} intensity={0.8} castShadow />
                    <directionalLight position={[-10, 5, -5]} intensity={0.3} />

                    {/* White grid floor for spatial reference - scaled up for the longer pipeline */}
                    <Grid
                        args={[200, 200]}
                        position={[0, -2, 0]}
                        cellColor="#2A3560"
                        sectionColor="#68B0AB"
                        fadeDistance={150}
                        fadeStrength={1}
                    />

                    {/* Pipeline */}
                    <Pipeline />

                    {/* Fog for depth perception, matched to the viewport background */}
                    <fog attach="fog" args={['#0B1330', 50, 200]} />
                </Canvas>
            </div>

            {/* Severity scale legend bar directly below the 3D representation */}
            <SeverityScale />
        </div>
    );
}

